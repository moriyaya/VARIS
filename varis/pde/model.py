"""Preference-independent conditional response generator and simulated training data."""
from __future__ import annotations
import math
import torch
from torch import nn
from .physics import N_MODES, sample_teacher, mirror_observation, solve_pde, reflect
OBS_DIM=62
LOG_MIN=math.log(.01)
LOG_MAX=math.log(.10)
class ResponseGenerator(nn.Module):
    def __init__(self, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(OBS_DIM + 2, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 2 * N_MODES),
        )

    def forward(self, observation: torch.Tensor, nu: torch.Tensor, code: torch.Tensor) -> torch.Tensor:
        log_nu = 2.0 * (torch.log(nu) - LOG_MIN) / (LOG_MAX - LOG_MIN) - 1.0
        features = torch.cat([observation, log_nu[:, None], code[:, None]], dim=-1)
        return self.net(features)


def make_dataset(
    count: int, seed: int, device: torch.device, split: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    coefficients, nu = sample_teacher(count, seed, device)
    if split != "train":
        generator = torch.Generator(device="cpu").manual_seed(seed + 31)
        if split == "low_extrapolation":
            nu = (0.01 + 0.01 * torch.rand(count, generator=generator)).to(device)
        elif split == "high_extrapolation":
            nu = (0.08 + 0.02 * torch.rand(count, generator=generator)).to(device)
        elif split != "interpolation":
            raise ValueError(split)
    with torch.no_grad():
        observation = mirror_observation(solve_pde(coefficients, nu))
    return coefficients, nu, observation


def branch_batch(
    coefficients: torch.Tensor, nu: torch.Tensor, observation: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    n = len(coefficients)
    repeated_nu = nu.repeat(2)
    repeated_obs = observation.repeat(2, 1)
    codes = torch.cat([-torch.ones(n, device=nu.device), torch.ones(n, device=nu.device)])
    # The canonical teacher has positive sine-1; align code +1 with it.
    targets = torch.cat([reflect(coefficients), coefficients], dim=0)
    return targets, repeated_nu, repeated_obs, codes
