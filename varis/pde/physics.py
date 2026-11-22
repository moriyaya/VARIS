"""Differentiable periodic Allen-Cahn forward and observation operators."""
from __future__ import annotations
import math
import torch
N_SPACE=64
N_MODES=4
N_STEPS=24
FINAL_TIME=0.20
REACTION=1.5
def initial_field(coefficients: torch.Tensor) -> torch.Tensor:
    x = -1.0 + 2.0 * torch.arange(N_SPACE, device=coefficients.device) / N_SPACE
    modes = torch.arange(1, N_MODES + 1, device=coefficients.device)
    angles = math.pi * modes[:, None] * x[None, :]
    cosine = torch.cos(angles)
    sine = torch.sin(angles)
    return coefficients[..., :N_MODES] @ cosine + coefficients[..., N_MODES:] @ sine


def solve_pde(coefficients: torch.Tensor, nu: torch.Tensor) -> torch.Tensor:
    """Differentiable semi-implicit spectral integration of u_t=nu*u_xx+rho(u-u^3)."""
    u = initial_field(coefficients)
    wave_numbers = 2.0 * math.pi * torch.fft.fftfreq(
        N_SPACE, d=2.0 / N_SPACE, device=coefficients.device
    )
    dt = FINAL_TIME / N_STEPS
    denominator = 1.0 + dt * nu[..., None] * wave_numbers.square()
    for _ in range(N_STEPS):
        reaction = REACTION * (u - u.pow(3))
        u = torch.fft.ifft(torch.fft.fft(u + dt * reaction) / denominator).real
    return u


def mirror_observation(final_field: torch.Tensor) -> torch.Tensor:
    indexes = torch.arange(1, N_SPACE // 2, device=final_field.device)
    first = final_field[..., indexes]
    second = final_field[..., N_SPACE - indexes]
    return torch.stack([torch.minimum(first, second), torch.maximum(first, second)], dim=-1).flatten(-2)


def reflect(coefficients: torch.Tensor) -> torch.Tensor:
    result = coefficients.clone()
    result[..., N_MODES:] *= -1.0
    return result


def sample_teacher(count: int, seed: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    coefficients = 0.12 * torch.randn(count, 2 * N_MODES, generator=generator)
    coefficients[:, 0] += 0.65
    coefficients[:, N_MODES] = 0.35 + 0.25 * torch.rand(count, generator=generator)
    nu = 0.02 + 0.06 * torch.rand(count, generator=generator)
    return coefficients.to(device), nu.to(device)


def lower_loss(coefficients: torch.Tensor, nu: torch.Tensor, observation: torch.Tensor) -> torch.Tensor:
    prediction = mirror_observation(solve_pde(coefficients, nu))
    return torch.mean((prediction - observation).square(), dim=-1)
