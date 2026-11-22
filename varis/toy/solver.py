"""VARIS response oracle and outer optimization for the triple-well study."""
from __future__ import annotations
import numpy as np
import torch
from .geometry import VALID_RADIUS, branch_centers, upper_targets
from ..optimization import OuterConfig, _clip_vector, _solve_preference_qp, _stack_hypergradients
class ContextualTripleWellProblem:
    num_objectives = 3

    def __init__(self, context: tuple[float, float], initial_seed: int):
        self.context = torch.tensor(context, dtype=torch.float64)
        self.initial_seed = initial_seed

    @property
    def split(self) -> str:
        return "context_shift" if torch.max(torch.abs(self.context)).item() > 0.8 else "interpolation"

    def centers(self, x: torch.Tensor) -> torch.Tensor:
        context = self.context[None].to(dtype=x.dtype, device=x.device)
        return branch_centers(context, x.reshape(1, 1))[0]

    def initial_x(self, seed: int) -> torch.Tensor:
        del seed
        return torch.tensor([0.0], dtype=torch.float64)

    def initial_y(self, seed: int) -> torch.Tensor:
        rng = np.random.default_rng(1907 + self.initial_seed * 101 + seed)
        x0 = self.initial_x(seed)
        center = self.centers(x0).mean(dim=0)
        perturbation = torch.tensor(rng.normal(0.0, 0.08, size=2), dtype=torch.float64)
        return center + perturbation

    def pair_scale(self, x: torch.Tensor) -> torch.Tensor:
        centers = self.centers(x)
        pair = torch.stack(
            (
                (centers[0] - centers[1]).square().sum(),
                (centers[0] - centers[2]).square().sum(),
                (centers[1] - centers[2]).square().sum(),
            )
        )
        return pair.mean().clamp_min(1e-8)

    def lower_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        centers = self.centers(x)
        distance_sq = (centers - y[None]).square().sum(dim=1)
        scale = self.pair_scale(x)
        return torch.prod(distance_sq / scale)

    def lower_gap(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        distance_sq = (self.centers(x) - y[None]).square().sum(dim=1)
        return distance_sq.min(dim=0)

    def upper_losses(self):
        losses = []
        for objective in range(3):
            def upper(x: torch.Tensor, y: torch.Tensor, objective=objective) -> torch.Tensor:
                centers = self.centers(x)
                context = self.context[None].to(dtype=x.dtype, device=x.device)
                target_x = upper_targets(context)[0, objective]
                return (y - centers[objective]).square().sum() + 0.15 * (x[0] - target_x).square()
            losses.append(upper)
        return losses

    def scalar_upper(self, x: torch.Tensor, y: torch.Tensor, preference: np.ndarray) -> torch.Tensor:
        values = torch.stack([loss(x, y) for loss in self.upper_losses()])
        weights = torch.as_tensor(preference, dtype=values.dtype, device=values.device)
        return torch.dot(weights, values)

    def exact_oracle(self, preference: np.ndarray) -> dict:
        context = self.context[None]
        targets = upper_targets(context)[0]
        weights = torch.tensor(preference, dtype=torch.float64)
        x_star = torch.dot(weights, targets).reshape(1)
        centers = self.centers(x_star)
        values = torch.stack([self.scalar_upper(x_star, centers[j], preference) for j in range(3)])
        value, branch = values.min(dim=0)
        preferred_branches = torch.nonzero(
            values <= value + 1e-10,
            as_tuple=False,
        ).flatten().tolist()
        return {
            "x": x_star,
            "y": centers[int(branch.item())],
            "branch": int(branch.item()),
            "preferred_branches": [int(index) for index in preferred_branches],
            "value": float(value.item()),
        }

    def metrics(self, x: torch.Tensor, y: torch.Tensor) -> dict:
        gap, branch = self.lower_gap(x, y)
        return {
            "x": float(x[0].detach().item()),
            "y0": float(y[0].detach().item()),
            "y1": float(y[1].detach().item()),
            "lower_gap": float(gap.detach().item()),
            "lower_smooth": float(self.lower_loss(x, y).detach().item()),
            "nearest_branch": int(branch.detach().item()),
        }


def candidate_upper_scores(
    problem: ContextualTripleWellProblem,
    x: torch.Tensor,
    candidates: torch.Tensor,
    preference: np.ndarray,
) -> torch.Tensor:
    return torch.stack([problem.scalar_upper(x, candidate, preference) for candidate in candidates])


def stable_feedback(
    problem: ContextualTripleWellProblem,
    x: torch.Tensor,
    initial: torch.Tensor,
    steps: int = 5,
) -> tuple[torch.Tensor, int]:
    state = initial.detach()
    _, selected_branch = problem.lower_gap(x, state)
    selected_branch = int(selected_branch.item())
    accepted_steps = 0
    for _ in range(steps):
        state_req = state.detach().requires_grad_(True)
        loss = problem.lower_loss(x, state_req)
        (gradient,) = torch.autograd.grad(loss, state_req)
        previous_gap, _ = problem.lower_gap(x, state)
        step_size = 0.8
        accepted = False
        for _ in range(12):
            proposal = (state - step_size * gradient).detach()
            proposal_gap, proposal_branch = problem.lower_gap(x, proposal)
            if int(proposal_branch.item()) == selected_branch and proposal_gap < previous_gap - 1e-12:
                state = proposal
                accepted_steps += 1
                accepted = True
                break
            step_size *= 0.5
        if not accepted:
            break
    return state, accepted_steps


def select_amortized_response(
    problem: ContextualTripleWellProblem,
    x: torch.Tensor,
    preference: np.ndarray,
    model: torch.nn.Module,
) -> tuple[torch.Tensor, int]:
    context = problem.context[None]
    common = torch.cat((context, x.reshape(1, 1)), dim=1)
    with torch.no_grad():
        candidates = model(common).reshape(3, 2)
    scores = candidate_upper_scores(problem, x, candidates, preference)
    gaps = torch.stack([problem.lower_gap(x, candidate)[0] for candidate in candidates])
    valid = gaps <= VALID_RADIUS**2
    if valid.any():
        selected = scores.masked_fill(~valid, torch.inf).argmin()
        return candidates[int(selected.item())], 0
    selected = gaps.argmin()
    return stable_feedback(problem, x, candidates[int(selected.item())], steps=5)


def run_amortized(
    problem: ContextualTripleWellProblem,
    preference: np.ndarray,
    model: torch.nn.Module,
    config: OuterConfig,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    x = problem.initial_x(0)
    v_cache = None
    total_feedback_steps = 0
    upper_losses = problem.upper_losses()
    for _ in range(config.outer_steps):
        y, feedback_steps = select_amortized_response(problem, x, preference, model)
        total_feedback_steps += feedback_steps
        values, grad_matrix, v_cache = _stack_hypergradients(
            upper_losses,
            problem.lower_loss,
            x,
            y,
            cg_steps=config.cg_steps,
            damping=config.cg_damping,
            v_cache=v_cache,
        )
        gram = (grad_matrix.T @ grad_matrix).detach().cpu().numpy()
        qp = _solve_preference_qp(gram, values.detach().cpu().numpy(), preference, config.u)
        weights_np = preference * qp.lam
        weights = torch.as_tensor(weights_np, dtype=x.dtype)
        direction = grad_matrix @ weights
        direction, _, _, _ = _clip_vector(direction, config.direction_clip_norm)
        x = (x - config.outer_lr * direction).detach()
    y, feedback_steps = select_amortized_response(problem, x, preference, model)
    total_feedback_steps += feedback_steps
    return x, y.detach(), total_feedback_steps


def evaluate_result(
    problem: ContextualTripleWellProblem,
    preference: np.ndarray,
    x: torch.Tensor,
    y: torch.Tensor,
) -> dict:
    oracle = problem.exact_oracle(preference)
    gap, nearest = problem.lower_gap(x, y)
    centers = problem.centers(x)
    projected_value = problem.scalar_upper(x, centers[int(nearest.item())], preference)
    raw_value = problem.scalar_upper(x, y, preference)
    valid = float(gap.item()) <= VALID_RADIUS**2
    correct = int(nearest.item()) in oracle["preferred_branches"]
    return {
        "x": float(x[0].item()),
        "y0": float(y[0].item()),
        "y1": float(y[1].item()),
        "oracle_x": float(oracle["x"][0].item()),
        "oracle_branch": oracle["branch"],
        "nearest_branch": int(nearest.item()),
        "valid": int(valid),
        "correct_branch": int(correct),
        "preferred_valid": int(valid and correct),
        "lower_gap": float(gap.item()),
        "lower_smooth": float(problem.lower_loss(x, y).item()),
        "raw_upper": float(raw_value.item()),
        "projected_regret": float(projected_value.item() - oracle["value"]),
    }
