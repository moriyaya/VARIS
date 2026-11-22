#!/usr/bin/env python3
"""Shared geometry for the decision-coupled triple-well experiment."""

from __future__ import annotations

import math

import numpy as np
import torch


SCALE_X_COUPLING = 0.015
SCALE_Y_COUPLING = -0.012


def branch_centers(context: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Return three exact lower responses with non-rigid x-dependent scaling."""
    c1, c2 = context[:, 0], context[:, 1]
    xv = x[:, 0]
    theta = 0.65 * c1 - 0.45 * c2 + 0.15 * xv
    sx = 1.05 + 0.25 * c1 + 0.08 * c2.square() + SCALE_X_COUPLING * xv
    sy = (
        0.90
        + 0.22 * c2
        + 0.08 * torch.sin(math.pi * c1)
        + SCALE_Y_COUPLING * xv
    )
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


def exact_oracle(problem, preference: np.ndarray) -> dict:
    """Enumerate branch-conditioned quadratic optima for the coupled task."""
    preference = np.asarray(preference, dtype=float)

    def values_at(x_value: float) -> torch.Tensor:
        x = torch.tensor([x_value], dtype=torch.float64)
        centers = problem.centers(x)
        return torch.stack(
            [problem.scalar_upper(x, centers[branch], preference) for branch in range(3)]
        )

    value_zero = values_at(0.0)
    value_plus = values_at(1.0)
    value_minus = values_at(-1.0)
    quadratic = 0.5 * (value_plus + value_minus - 2.0 * value_zero)
    linear = 0.5 * (value_plus - value_minus)
    x_by_branch = -linear / (2.0 * quadratic)
    minimum_by_branch = quadratic * x_by_branch.square() + linear * x_by_branch + value_zero
    value, branch = minimum_by_branch.min(dim=0)
    preferred_branches = torch.nonzero(
        minimum_by_branch <= value + 1e-10,
        as_tuple=False,
    ).flatten().tolist()
    x_star = x_by_branch[int(branch.item())].reshape(1)
    centers = problem.centers(x_star)
    return {
        "x": x_star,
        "y": centers[int(branch.item())],
        "branch": int(branch.item()),
        "preferred_branches": [int(index) for index in preferred_branches],
        "value": float(value.item()),
        "x_by_branch": [float(value.item()) for value in x_by_branch],
    }
