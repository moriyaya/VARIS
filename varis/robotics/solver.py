"""VARIS Panda oracle and shared upper-level optimization."""
from dataclasses import asdict
import hashlib,math,time,json
from pathlib import Path
import numpy as np
import torch
from .backend import sync
from .. import optimization as outer
CFG=json.loads((Path(__file__).resolve().parents[2]/"configs/panda.json").read_text())
OUT=Path("outputs/panda")
def seed_for(dataset_seed,target_id,inference_seed,stream):
    key=f"{dataset_seed}:{target_id}:{inference_seed}:{stream}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:4],"little")
def optimizer_config():
    return outer.OuterConfig(outer_steps=CFG["outer_steps"],outer_lr=CFG["outer_lr"],
        cg_steps=CFG["cg_steps"],cg_damping=CFG["cg_damping"],u=CFG["preference_scale"],
        direction_clip_norm=CFG["direction_clip_norm"],log_every=CFG["log_every"])

class Problem:
    num_objectives = 2

    def __init__(self, backend, goal, start, initial):
        self.backend = backend
        self.device = backend.device
        self.goal = goal.detach()
        self.start = start.detach()
        self.initial = initial.detach()
        from jrl.math_utils import quaternion_to_rotation_matrix
        self.rotation = quaternion_to_rotation_matrix(self.goal[None, 3:])[0].detach()
        self.archive = []

    def physical_x(self, x):
        return .1 * torch.tanh(x.reshape(()))

    def decode(self, y):
        return self.backend.mid + .5 * self.backend.span * torch.tanh(y)

    def encode(self, q):
        z = (2 * (q - self.backend.mid) / self.backend.span).clamp(-.99999, .99999)
        return torch.atanh(z)

    def initial_x(self, seed):
        return torch.zeros(1, device=self.device)

    def initial_y(self, seed):
        return self.encode(self.initial).clone()

    def lower_loss(self, x, y):
        q = self.decode(y).reshape(1, 7)
        transform = self.backend.robot.forward_kinematics(q, return_quaternion=False)[0]
        shift = torch.stack([self.physical_x(x), x.sum() * 0, x.sum() * 0])
        position = (transform[:3, 3] + shift - self.goal[:3]) / .1
        return .5 * position.square().sum() + .25 * (transform[:3, :3] - self.rotation).square().sum()

    def values(self, x, y):
        q = self.decode(y)
        px = self.physical_x(x)
        f1 = ((q-self.start)/self.backend.span).square().mean() + .05*(px/.1).square()
        f2 = 4*((q-self.backend.mid)/self.backend.span).square().mean()
        return torch.stack([f1, f2])

    def upper_losses(self):
        return [lambda x, y, j=j: self.values(x, y)[j] for j in range(2)]

    def record(self, x, y, origin):
        with torch.no_grad():
            q = self.decode(y).reshape(1, 7)
            px = float(self.physical_x(x))
            target = self.goal[None].clone()
            target[:, 0] -= px
            check = self.backend.evaluate(q, target)
            values = self.values(x, y)
            row = {"x": px, "q": q[0].cpu().tolist(), "valid": bool(check["valid"][0]),
                   "F": values.cpu().tolist(), "position_mm": 1000*float(check["position_m"][0]),
                   "orientation_deg": math.degrees(float(check["orientation_rad"][0])), "origin": origin}
            if row["valid"]:
                self.archive.append(row)
            return row

    def metrics(self, x, y):
        return self.record(x, y, "iterate")


def warm_pool(backend, goal, tid, seed, dataset_seed, persist=True):
    path = OUT / "warm" / f"t{tid:04d}_s{seed}.pt"
    if persist and path.exists():
        return torch.load(path, map_location="cpu", weights_only=False)
    with torch.no_grad():
        torch.manual_seed(seed_for(dataset_seed, tid, seed, "bilevel_shared_exact"))
        sync(); started = time.perf_counter()
        q, _ = backend.solver.generate_exact_ik_solutions(
            goal[None].expand(CFG["shared_initial_solutions"], -1), pos_error_threshold=.001,
            rot_error_threshold=float(np.pi/180), run_lma_on_cpu=True)
        check = backend.evaluate(q, goal[None].expand(len(q), -1))
        sync()
        payload = {"q": q.detach().cpu(), "valid": check["valid"].cpu(),
                   "seconds": time.perf_counter()-started}
    if persist:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        torch.save(payload, temporary); temporary.replace(path)
    return payload


def response(problem, x, r, latent_seed, counts):
    backend = problem.backend
    with torch.no_grad():
        target = problem.goal[None].clone()
        px = float(problem.physical_x(x)); target[:, 0] -= px
        q, _ = backend.generate(target, CFG["varis_candidates"], latent_seed)
        counts["generator_calls"] += 1
        q = q[0]
        ev = backend.evaluate(q, target.expand(len(q), -1))
        f1, f2 = backend.objectives(q, problem.start, px)
        score = r*f1+(1-r)*f2
        if bool(ev["valid"].any()):
            return problem.encode(q[score.masked_fill(~ev["valid"], torch.inf).argmin()])
        eligible = (ev["error"] > 1) & (ev["error"] <= 25)
        if bool(eligible.any()):
            index = score.masked_fill(~eligible, torch.inf).argmin()
            fixed, repair = backend.repair_all(q[index:index+1], target)
            counts["repair_trials"] += int(repair["attempts"].sum())
            if bool(repair["final"]["valid"][0]):
                return problem.encode(fixed[0])
        counts["fallback_calls"] += 1
        # Same released exact primitive as common initialization, new target.
        torch.manual_seed(latent_seed)
        exact, _ = backend.solver.generate_exact_ik_solutions(
            target.expand(16, -1), pos_error_threshold=.001,
            rot_error_threshold=float(np.pi/180), run_lma_on_cpu=True)
        check = backend.evaluate(exact, target.expand(len(exact), -1))
        a, b = backend.objectives(exact, problem.start, px)
        if bool(check["valid"].any()):
            index = (r*a+(1-r)*b).masked_fill(~check["valid"], torch.inf).argmin()
        else:
            # Keep a pose-nearest finite seed for the outer step; never mark it valid.
            index = check["pose_error"].argmin()
        return problem.encode(exact[index])


def run_varis(problem, preference, cfg, seed, callback, counts, dataset_seed, tid):
    x, y = problem.initial_x(seed), problem.initial_y(seed)
    cache = None
    latent_seed = seed_for(dataset_seed, tid, seed, "bilevel_varis")
    for step in range(cfg.outer_steps):
        y = response(problem, x, float(preference[0]), latent_seed, counts)
        values, matrix, cache = outer._stack_hypergradients(
            problem.upper_losses(), problem.lower_loss, x, y,
            cg_steps=cfg.cg_steps, damping=cfg.cg_damping, v_cache=cache)
        qp = outer._solve_preference_qp((matrix.T@matrix).cpu().numpy(), values.cpu().numpy(), preference, cfg.u)
        weights = torch.as_tensor(preference*qp.lam, device=x.device, dtype=x.dtype)
        direction, _, _, _ = outer._clip_vector(matrix@weights, cfg.direction_clip_norm)
        x = (x-cfg.outer_lr*direction).detach()
        if step % cfg.log_every == 0 or step == cfg.outer_steps-1:
            callback({"step": step, **problem.metrics(x, y)})
    y = response(problem, x, float(preference[0]), latent_seed, counts)
    return x, y


def solve(backend, goal, start, warm, tid, seed, r, dataset_seed):
    cfg = optimizer_config()
    valid_q = warm["q"][warm["valid"]].to(backend.device)
    if not len(valid_q):
        return {"failed_initialization": True, "archive": [], "solve_seconds": 0., "history": []}
    balance = .5 if r is None else r
    f1, f2 = backend.objectives(valid_q, start, 0.)
    initial = valid_q[(balance*f1+(1-balance)*f2).argmin()]
    problem = Problem(backend, goal, start, initial)
    # Keep valid initial responses in the return archive.
    for q in valid_q:
        problem.record(problem.initial_x(seed), problem.encode(q), "shared_initial")
    preference = np.asarray([balance, 1-balance], dtype=float)
    history, counts = [], {"generator_calls": 0, "repair_trials": 0, "fallback_calls": 0}
    torch.set_grad_enabled(True)
    sync(); started = time.perf_counter()
    x, y = run_varis(problem, preference, cfg, seed, history.append, counts, dataset_seed, tid)
    if not torch.isfinite(x).all() or not torch.isfinite(y).all():
        raise FloatingPointError(f"Nonfinite state VARIS/{tid}/{seed}")
    native = problem.record(x, y, "native_final")
    polish_seconds = 0.
    if not native["valid"]:
        with torch.no_grad():
            target = goal[None].clone(); target[:, 0] -= native["x"]
            sync(); p0 = time.perf_counter()
            fixed, check = backend.solve_pool(problem.decode(y).reshape(1, 7), target, CFG["final_polish_steps"])
            sync(); polish_seconds = time.perf_counter()-p0
            if bool(check["valid"][0]):
                problem.record(x, problem.encode(fixed[0]), "common_final_polish")
    sync(); elapsed = time.perf_counter()-started
    return {"failed_initialization": False, "archive": problem.archive, "native_final": native,
            "history": history, "solve_seconds": elapsed, "polish_seconds": polish_seconds,
            "counts": counts, "optimizer_config": asdict(cfg), "upper_coordinate": x.detach().cpu().tolist()}

