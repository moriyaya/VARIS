"""VARIS inference, safeguarded repair, and lower-solver fallback."""
from __future__ import annotations
import csv,time
import numpy as np
import torch
from .physics import N_MODES,lower_loss,reflect,initial_field
def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def upper_values(y):
    b = y[..., N_MODES]
    return torch.stack(((b + 0.5).square(), (b - 0.5).square()), -1)


def upper_score(y, preference):
    f = upper_values(y)
    return (1 - preference) * f[..., 0] + preference * f[..., 1]


class LowerCounter:
    def __init__(self, nu, observation):
        self.nu, self.observation = nu, observation
        self.calls = 0
        self.response_evaluations = 0

    def __call__(self, y):
        self.calls += 1
        self.response_evaluations += len(y)
        return lower_loss(y, self.nu, self.observation)


def make_starts(tasks, starts, seed, device):
    rng = torch.Generator().manual_seed(71_000 + seed)
    y = 0.2 * torch.randn(tasks, starts, 2 * N_MODES, generator=rng)
    y[..., 0] += 0.5
    signs = torch.tensor([1 if k % 2 == 0 else -1 for k in range(starts)])
    y[..., N_MODES] = y[..., N_MODES].abs().clamp_min(0.05) * signs
    return y.to(device)


def rank_pool(pool, residual, preferences, threshold):
    """pool [task,pref or 1,start,D]; validity first, no sign oracle."""
    pool = pool.expand(-1, len(preferences), -1, -1)
    residual = residual.expand(-1, len(preferences), -1)
    preference = preferences[None, :, None]
    scores = upper_score(pool, preference)
    admitted = residual <= threshold
    valid_choice = scores.masked_fill(~admitted, torch.inf).argmin(-1)
    finite_loss = torch.nan_to_num(residual, nan=torch.inf, posinf=torch.inf)
    choice = torch.where(admitted.any(-1), valid_choice, finite_loss.argmin(-1))
    picked = pool.gather(2, choice[..., None, None].expand(-1, -1, 1, pool.shape[-1])).squeeze(2)
    return picked, choice


def preference_repair_plan(residual, scores, threshold, repair_band):
    """Plan preference-conditioned local repair.

    residual: [task,candidate], scores: [task,preference,candidate].
    Repair only when the best repair-band candidate has a strictly better
    upper score than the best currently valid response, or no valid response
    exists. Final validity is still enforced after repair.
    """
    valid = residual <= threshold
    eligible = (residual > threshold) & (residual <= repair_band)
    valid_mask = valid[:, None].expand_as(scores)
    eligible_mask = eligible[:, None].expand_as(scores)
    valid_index = scores.masked_fill(~valid_mask, torch.inf).argmin(-1)
    repair_index = scores.masked_fill(~eligible_mask, torch.inf).argmin(-1)
    valid_score = scores.gather(-1, valid_index[..., None]).squeeze(-1)
    repair_score = scores.gather(-1, repair_index[..., None]).squeeze(-1)
    has_valid = valid.any(-1)[:, None].expand_as(valid_index)
    has_eligible = eligible.any(-1)[:, None].expand_as(repair_index)
    should_repair = has_eligible & ((~has_valid) | (repair_score < valid_score))
    return valid_index, repair_index, should_repair, has_valid


def preference_repair_varis(model, starts, nu, obs, preferences, args):
    """Preference-conditioned trigger with bounded repair and incumbent retention."""
    tasks, _, dim = starts.shape
    pcount = len(preferences)
    codes = torch.tensor([-1.0, -0.5, 0, 0.5, 1.0], device=nu.device)
    sync(nu.device)
    begun = time.perf_counter()
    with torch.no_grad():
        pool = model(obs.repeat_interleave(5, 0), nu.repeat_interleave(5),
                     codes.repeat(tasks)).reshape(tasks, 5, dim)
        residual = lower_loss(pool.reshape(-1, dim), nu.repeat_interleave(5),
                              obs.repeat_interleave(5, 0)).reshape(tasks, 5)
        scores = upper_score(pool[:, None], preferences[None, :, None])
        valid_index, repair_index, should_repair, has_valid = preference_repair_plan(
            residual, scores, args.threshold, args.repair_band)
        expanded = pool[:, None].expand(-1, pcount, -1, -1)
        incumbent = expanded.gather(
            2, valid_index[..., None, None].expand(-1, -1, 1, dim)).squeeze(2).clone()
        repair_seed = expanded.gather(
            2, repair_index[..., None, None].expand(-1, -1, 1, dim)).squeeze(2).clone()
        selected = torch.where(should_repair[..., None], repair_seed, incumbent)

    y = selected.reshape(-1, dim)
    seeds = y.clone()
    incumbent = incumbent.reshape(-1, dim)
    has_incumbent = has_valid.reshape(-1)
    should_repair_flat = should_repair.reshape(-1)
    nus = nu.repeat_interleave(pcount)
    observations = obs.repeat_interleave(pcount, 0)
    repeated_preferences = preferences[None].expand(tasks, -1).reshape(-1)
    pending = should_repair_flat.clone()
    stages = ["repair_pending" if bool(should_repair_flat[i]) else
              ("direct" if bool(has_incumbent[i]) else "fallback") for i in range(len(y))]
    step_size = torch.full((len(y),), args.repair_lr, device=y.device)
    accepted = torch.zeros(len(y), dtype=torch.long, device=y.device)
    rejected = accepted.clone()
    repair_calls = 0
    repair_responses = 0
    history = []
    for trial in range(1, args.repair_trials + 1):
        indexes = torch.nonzero(pending).flatten()
        if not len(indexes):
            break
        req = y[indexes].detach().requires_grad_(True)
        old_loss = lower_loss(req, nus[indexes], observations[indexes])
        gradient = torch.autograd.grad(old_loss.sum(), req)[0]
        proposed = req.detach() - step_size[indexes, None] * gradient.detach()
        displacement = proposed - seeds[indexes]
        scale = (args.trust_radius / displacement.norm(dim=-1).clamp_min(1e-20)).clamp_max(1)
        proposed = seeds[indexes] + displacement * scale[:, None]
        with torch.no_grad():
            new_loss = lower_loss(proposed, nus[indexes], observations[indexes])
            sufficient = old_loss.detach() - args.sigma * (proposed - req.detach()).square().sum(-1)
            good = torch.isfinite(new_loss) & (new_loss <= sufficient)
            y[indexes] = torch.where(good[:, None], proposed, req.detach())
            accepted[indexes] += good.long()
            rejected[indexes] += (~good).long()
            step_size[indexes] = torch.where(good, torch.full_like(new_loss, args.repair_lr),
                                              0.5 * step_size[indexes])
            done = good & (new_loss <= args.threshold)
            pending[indexes[done]] = False
        repair_calls += 2
        repair_responses += 2 * len(indexes)

    with torch.no_grad():
        repaired_loss = lower_loss(y, nus, observations)
        repaired_score = upper_score(y, repeated_preferences)
        incumbent_score = upper_score(incumbent, repeated_preferences)
        use_repair = should_repair_flat & (repaired_loss <= args.threshold)
        use_repair &= (~has_incumbent) | (repaired_score < incumbent_score)
        restore = should_repair_flat & has_incumbent & ~use_repair
        y[restore] = incumbent[restore]
        after_restore = lower_loss(y, nus, observations)
        needs_fallback = after_restore > args.threshold
        needed_tasks = torch.nonzero(needs_fallback.reshape(tasks, -1).any(-1)).flatten()
    for i in range(len(y)):
        if bool(use_repair[i]):
            stages[i] = "repaired"
        elif bool(restore[i]):
            stages[i] = "incumbent"

    fallback_meta = {"online_s": 0.0, "lower_calls": 0, "lower_response_evaluations": 0}
    if len(needed_tasks):
        fallback, fallback_meta, _, _ = fallback_pool(
            starts[needed_tasks], nu[needed_tasks], obs[needed_tasks],
            preferences, args)
        reshaped = y.reshape(tasks, -1, dim)
        masks = needs_fallback.reshape(tasks, -1)[needed_tasks]
        reshaped[needed_tasks] = torch.where(masks[..., None], fallback, reshaped[needed_tasks])
        y = reshaped.reshape(-1, dim)
    with torch.no_grad():
        final_loss = lower_loss(y, nus, observations)
    for i in range(len(y)):
        if bool(needs_fallback[i]):
            stages[i] = "fallback" if bool(final_loss[i] <= args.threshold) else "unresolved"
    sync(y.device)
    meta = {"online_s": time.perf_counter() - begun, "repair_calls": repair_calls,
            "repair_response_evaluations": repair_responses, "fallback": fallback_meta,
            "fallback_contexts": len(needed_tasks), "triggered_repairs": int(should_repair_flat.sum()),
            "incumbent_restores": int(restore.sum()),
            "stage_counts": {stage: stages.count(stage) for stage in set(stages)},
            "accepted_mean": float(accepted.float().mean()),
            "rejected_mean": float(rejected.float().mean()),
            "generation_and_check_response_evaluations": 5 * tasks,
            "note": "Preference-conditioned local repair with validity recheck."}
    return y.reshape(tasks, -1, dim), meta, history, stages


def score_rows(y, teacher, nu, obs, preferences, method, split, seed, threshold, stages):
    tasks, pcount, dim = y.shape
    flat = y.reshape(-1, dim)
    with torch.no_grad():
        losses = lower_loss(flat, nu.repeat_interleave(pcount), obs.repeat_interleave(pcount, 0)).reshape(tasks, pcount)
        fs = upper_values(y)
        weighted = upper_score(y, preferences[None])
        desired = torch.where(preferences > 0.5, 1.0, -1.0)
        targets = torch.where((preferences > 0.5)[None, :, None], teacher[:, None], reflect(teacher)[:, None])
        fields = initial_field(y)
        true_fields = initial_field(targets)
        relative = (fields - true_fields).norm(dim=-1) / true_fields.norm(dim=-1).clamp_min(1e-12)
        valid = losses <= threshold
        correct = y[..., N_MODES] * desired > 0
    rows = []
    for i in range(tasks):
        for j, p in enumerate(preferences.tolist()):
            rows.append({"method": method, "seed": seed, "split": split, "task_index": i,
                         "preference": p, "nu": float(nu[i]), "lower_residual": float(losses[i, j]),
                         "upper_minus": float(fs[i, j, 0]), "upper_plus": float(fs[i, j, 1]),
                         "upper_scalar": float(weighted[i, j]), "valid": int(valid[i, j]),
                         "branch_correct": int(correct[i, j]), "joint": int(valid[i, j] & correct[i, j]),
                         "relative_reference_field_error": float(relative[i, j]),
                         "stage": stages[i * pcount + j] if isinstance(stages, list) else "solver",
                         "finite": int(torch.isfinite(y[i, j]).all())})
    return rows


def save_csv(path, rows):
    if rows:
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def fallback_pool(starts,nu,obs,preferences,args):
    count,k,dim=starts.shape
    y=starts.reshape(-1,dim).clone()
    nus=nu.repeat_interleave(k);observations=obs.repeat_interleave(k,0)
    counter=LowerCounter(nus,observations)
    moments=torch.zeros_like(y);variances=torch.zeros_like(y)
    sync(y.device);begun=time.perf_counter()
    for step in range(args.steps):
        req=y.detach().requires_grad_(True)
        loss=counter(req);g=torch.autograd.grad(loss.sum(),req)[0].detach()
        if not torch.isfinite(g).all():raise FloatingPointError('Nonfinite fallback gradient')
        moments=.9*moments+.1*g;variances=.999*variances+.001*g.square()
        y=(y-args.adam_lr*(moments/(1-.9**(step+1)))/((variances/(1-.999**(step+1))).sqrt()+1e-8)).detach()
    with torch.no_grad():
        residual=lower_loss(y,nus,observations).reshape(count,1,k)
        picked,choice=rank_pool(y.reshape(count,1,k,dim),residual,preferences,args.threshold)
    sync(y.device)
    return picked,dict(online_s=time.perf_counter()-begun,lower_calls=counter.calls,lower_response_evaluations=counter.response_evaluations),[],choice
