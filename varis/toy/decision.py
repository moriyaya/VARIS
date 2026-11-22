"""Projected Adam through the selected learned response, as in the decision study."""
import torch
import numpy as np
from . import solver
from . import geometry as mechanism

def select_response(method, problem, x, preference, models):
    centers=problem.centers(x)
    common=torch.cat((problem.context[None],x.reshape(1,1)),dim=1)
    candidates=models['multi'](common).reshape(3,2)
    scores=torch.stack([problem.scalar_upper(x,y,preference) for y in candidates])
    gaps=torch.stack([problem.lower_gap(x,y)[0] for y in candidates])
    valid=gaps.detach() <= mechanism.VALID_RADIUS**2
    index=scores.detach().masked_fill(~valid,torch.inf).argmin() if valid.any() else gaps.detach().argmin()
    return candidates[int(index)]

def optimize_query(
    method: str,
    problem: solver.ContextualTripleWellProblem,
    preference: np.ndarray,
    models: dict[str, torch.nn.Module],
    steps: int,
    learning_rate: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.tensor([0.0], dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.Adam([x], lr=learning_rate)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        response = select_response(method, problem, x, preference, models)
        objective = problem.scalar_upper(x, response, preference)
        objective.backward()
        optimizer.step()
        with torch.no_grad():
            x.clamp_(-1.0, 1.0)
    with torch.no_grad():
        response = select_response(method, problem, x, preference, models)
    return x.detach(), response.detach()


def evaluate(
    problem: solver.ContextualTripleWellProblem,
    preference: np.ndarray,
    x: torch.Tensor,
    response: torch.Tensor,
) -> dict:
    oracle = problem.exact_oracle(preference)
    gap, nearest = problem.lower_gap(x, response)
    valid = float(gap.item()) <= mechanism.VALID_RADIUS**2
    correct = int(nearest.item()) in oracle["preferred_branches"]
    projected = problem.centers(x)[int(nearest.item())]
    projected_regret = problem.scalar_upper(x, projected, preference) - oracle["value"]
    raw_regret = problem.scalar_upper(x, response, preference) - oracle["value"]
    return {
        "x": float(x.item()),
        "oracle_x": float(oracle["x"].item()),
        "upper_decision_error": abs(float(x.item()) - float(oracle["x"].item())),
        "oracle_branch": int(oracle["branch"]),
        "nearest_branch": int(nearest.item()),
        "valid": int(valid),
        "correct_branch": int(correct),
        "preferred_valid": int(valid and correct),
        "lower_gap": float(gap.item()),
        "projected_regret": float(projected_regret.item()),
        "raw_regret": float(raw_regret.item()),
    }
