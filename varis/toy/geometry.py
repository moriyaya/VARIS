"""Contextual triple-well geometry and permutation-invariant set loss."""
from __future__ import annotations
import itertools, math, random
import numpy as np
import torch
from torch import nn
PERMUTATIONS=tuple(itertools.permutations(range(3)))
VALID_RADIUS=0.18
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def branch_centers(context: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Return the three exact lower responses, shape [batch, 3, 2]."""
    c1, c2 = context[:, 0], context[:, 1]
    xv = x[:, 0]
    theta = 0.65 * c1 - 0.45 * c2 + 0.15 * xv
    sx = 1.05 + 0.25 * c1 + 0.08 * c2.square()
    sy = 0.90 + 0.22 * c2 + 0.08 * torch.sin(math.pi * c1)
    center = torch.stack(
        (0.45 * xv + 0.25 * torch.sin(math.pi * c1), -0.35 * xv + 0.20 * c2),
        dim=1,
    )
    base = context.new_tensor(
        [[1.0, 0.0], [-0.5, math.sqrt(3.0) / 2.0], [-0.5, -math.sqrt(3.0) / 2.0]]
    )
    scaled = base[None] * torch.stack((sx, sy), dim=1)[:, None]
    cos_t, sin_t = torch.cos(theta), torch.sin(theta)
    rx = cos_t[:, None] * scaled[:, :, 0] - sin_t[:, None] * scaled[:, :, 1]
    ry = sin_t[:, None] * scaled[:, :, 0] + cos_t[:, None] * scaled[:, :, 1]
    return center[:, None] + torch.stack((rx, ry), dim=2)


def upper_targets(context: torch.Tensor) -> torch.Tensor:
    c1, c2 = context[:, 0], context[:, 1]
    return torch.stack(
        (
            0.55 + 0.25 * c1,
            -0.40 + 0.20 * c2,
            0.10 - 0.20 * c1 + 0.15 * c2,
        ),
        dim=1,
    )


def upper_scores(
    responses: torch.Tensor,
    centers: torch.Tensor,
    context: torch.Tensor,
    x: torch.Tensor,
    preference: torch.Tensor,
) -> torch.Tensor:
    """Preference scalarization for candidates, shape [batch, candidates]."""
    distances = (responses[:, :, None] - centers[:, None]).square().sum(dim=3)
    x_penalty = 0.15 * (x - upper_targets(context)).square()
    objectives = distances + x_penalty[:, None]
    return (objectives * preference[:, None]).sum(dim=2)


def lower_gaps(responses: torch.Tensor, centers: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    distances = (responses[:, :, None] - centers[:, None]).square().sum(dim=3)
    return distances.min(dim=2)


def sample_preferences(batch: int, device: torch.device) -> torch.Tensor:
    exponential = -torch.log(torch.rand(batch, 3, device=device).clamp_min(1e-7))
    return exponential / exponential.sum(dim=1, keepdim=True)


def sample_problem(
    batch: int,
    device: torch.device,
    split: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if split in {"train", "interpolation"}:
        context = 1.6 * torch.rand(batch, 2, device=device) - 0.8
        x = 1.6 * torch.rand(batch, 1, device=device) - 0.8
    elif split == "context_shift":
        magnitude = 0.85 + 0.30 * torch.rand(batch, 2, device=device)
        sign = torch.where(torch.rand(batch, 2, device=device) < 0.5, -1.0, 1.0)
        context = magnitude * sign
        x = 2.0 * torch.rand(batch, 1, device=device) - 1.0
    else:
        raise ValueError(split)
    preference = sample_preferences(batch, device)
    return context, x, preference, branch_centers(context, x)


class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, width: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, width),
            nn.Tanh(),
            nn.Linear(width, width),
            nn.Tanh(),
            nn.Linear(width, output_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


def permutation_set_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    losses = []
    for permutation in PERMUTATIONS:
        losses.append((prediction - target[:, permutation]).square().mean(dim=(1, 2)))
    return torch.stack(losses, dim=1).min(dim=1).values.mean()
