"""Implicit-gradient and preference-QP primitives used by VARIS outer updates."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import numpy as np
import torch
TensorClosure = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

@dataclass
class OuterConfig:
    outer_steps: int = 80
    outer_lr: float = 0.04
    cg_steps: int = 5
    cg_damping: float = 1e-3
    u: float = 10.0
    direction_clip_norm: float | None = None
    log_every: int = 10

@dataclass(frozen=True)
class _QPResult:
    lam: np.ndarray
    objective: float
    success: bool
    message: str


def _grad_or_zeros(
    outputs: torch.Tensor,
    inputs: tuple[torch.Tensor, ...],
    *,
    grad_outputs: torch.Tensor | None = None,
    create_graph: bool = False,
    retain_graph: bool = True,
) -> tuple[torch.Tensor, ...]:
    grads = torch.autograd.grad(
        outputs,
        inputs,
        grad_outputs=grad_outputs,
        create_graph=create_graph,
        retain_graph=retain_graph,
        allow_unused=True,
    )
    return tuple(torch.zeros_like(inp) if grad is None else grad for inp, grad in zip(inputs, grads))


def _project_to_simplex(v: np.ndarray) -> np.ndarray:
    z = np.asarray(v, dtype=float)
    if z.ndim != 1:
        raise ValueError("simplex projection expects a 1-D vector")
    n = z.size
    if n == 0:
        raise ValueError("cannot project an empty vector")
    u = np.sort(z)[::-1]
    cssv = np.cumsum(u) - 1.0
    ind = np.arange(1, n + 1)
    cond = u - cssv / ind > 0
    if not np.any(cond):
        return np.full(n, 1.0 / n)
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / rho
    return np.maximum(z - theta, 0.0)


def _preference_objective_and_grad(
    lam: np.ndarray,
    gram: np.ndarray,
    values: np.ndarray,
    preference: np.ndarray,
    u: float,
) -> tuple[float, np.ndarray]:
    scaled = preference * lam
    obj = float(scaled @ gram @ scaled - u * lam @ (preference * values))
    grad = 2.0 * preference * (gram @ scaled) - u * preference * values
    return obj, grad


def _solve_preference_qp(
    gram: np.ndarray,
    values: np.ndarray,
    preference: np.ndarray,
    u: float,
    *,
    max_iter: int = 1000,
    tol: float = 1e-10,
) -> _QPResult:
    gram = np.asarray(gram, dtype=float)
    values = np.asarray(values, dtype=float)
    preference = np.asarray(preference, dtype=float)
    if gram.ndim != 2 or gram.shape[0] != gram.shape[1]:
        raise ValueError("gram must be square")
    s = gram.shape[0]
    if values.shape != (s,) or preference.shape != (s,):
        raise ValueError("values and preference must match gram size")
    if np.any(preference <= 0):
        raise ValueError("Preference QP requires a strictly positive preference")

    if not np.all(np.isfinite(gram)) or not np.all(np.isfinite(values)):
        gram = np.nan_to_num(gram, nan=0.0, posinf=1e12, neginf=-1e12)
        values = np.nan_to_num(values, nan=0.0, posinf=1e6, neginf=-1e6)

    # Divide the whole objective by one positive constant. This improves
    # conditioning while preserving the preference-QP minimizer.
    objective_scale = max(float(np.max(np.abs(gram))), float(np.max(np.abs(values))), 1.0)
    gram = gram / objective_scale
    values = values / objective_scale
    lam0 = np.full(s, 1.0 / s)

    try:
        from scipy.optimize import minimize

        result = minimize(
            lambda lam: _preference_objective_and_grad(lam, gram, values, preference, u)[0],
            lam0,
            jac=lambda lam: _preference_objective_and_grad(lam, gram, values, preference, u)[1],
            method="SLSQP",
            bounds=[(0.0, 1.0)] * s,
            constraints={"type": "eq", "fun": lambda z: np.sum(z) - 1.0},
            options={"maxiter": max_iter, "ftol": tol},
        )
        lam = _project_to_simplex(result.x)
        obj = _preference_objective_and_grad(lam, gram, values, preference, u)[0]
        if result.success:
            return _QPResult(lam=lam, objective=obj, success=True, message=result.message)
        message = f"scipy failed: {result.message}"
    except Exception as exc:  # pragma: no cover - depends on scipy availability.
        message = f"scipy unavailable or failed: {exc}"

    h = np.diag(preference) @ gram @ np.diag(preference)
    h = np.nan_to_num(h, nan=0.0, posinf=1e12, neginf=-1e12)
    step = 1.0 / max(2.0 * float(np.linalg.norm(h, ord="fro")), 1e-12)
    lam = lam0.copy()
    for _ in range(max_iter):
        _, grad = _preference_objective_and_grad(lam, gram, values, preference, u)
        new_lam = _project_to_simplex(lam - step * grad)
        if np.linalg.norm(new_lam - lam) <= tol:
            lam = new_lam
            break
        lam = new_lam
    obj = _preference_objective_and_grad(lam, gram, values, preference, u)[0]
    return _QPResult(lam=lam, objective=obj, success=False, message=message)


def _conjugate_gradient(
    matvec: Callable[[torch.Tensor], torch.Tensor],
    rhs: torch.Tensor,
    *,
    x0: torch.Tensor | None,
    max_iter: int,
    tol: float = 1e-10,
    damping: float,
) -> torch.Tensor:
    x = torch.zeros_like(rhs) if x0 is None else x0.clone()
    residual = rhs - (matvec(x) + damping * x)
    direction = residual.clone()
    rs_old = torch.dot(residual, residual)
    for _ in range(max_iter):
        ad = matvec(direction) + damping * direction
        alpha = rs_old / (torch.dot(direction, ad) + 1e-20)
        x = x + alpha * direction
        residual = residual - alpha * ad
        rs_new = torch.dot(residual, residual)
        if torch.sqrt(rs_new).item() <= tol:
            break
        direction = residual + (rs_new / (rs_old + 1e-20)) * direction
        rs_old = rs_new
    return x


def _implicit_hypergradient_cg(
    upper_loss: TensorClosure,
    lower_loss: TensorClosure,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    cg_steps: int,
    damping: float,
    v0: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x_req = x.detach().requires_grad_(True)
    y_req = y.detach().requires_grad_(True)
    f_val = upper_loss(x_req, y_req)
    grad_x_f, grad_y_f = _grad_or_zeros(f_val, (x_req, y_req), create_graph=True)

    g_val = lower_loss(x_req, y_req)
    (grad_y_g,) = torch.autograd.grad(g_val, y_req, create_graph=True)

    def hy_matvec(vec: torch.Tensor) -> torch.Tensor:
        (hvp,) = torch.autograd.grad(
            grad_y_g,
            y_req,
            grad_outputs=vec,
            retain_graph=True,
            create_graph=True,
        )
        return hvp

    v = _conjugate_gradient(
        hy_matvec,
        grad_y_f.detach(),
        x0=v0,
        max_iter=cg_steps,
        damping=damping,
    )
    (cross_term,) = _grad_or_zeros(
        grad_y_g,
        (x_req,),
        grad_outputs=v.detach(),
        retain_graph=False,
    )
    hypergrad = grad_x_f.detach() - cross_term.detach()
    return f_val.detach(), hypergrad, v.detach()


def _stack_hypergradients(
    upper_losses: Iterable[TensorClosure],
    lower_loss: TensorClosure,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    cg_steps: int,
    damping: float,
    v_cache: list[torch.Tensor] | None,
) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
    values = []
    grads = []
    next_cache = []
    for idx, upper_loss in enumerate(upper_losses):
        v0 = None if v_cache is None else v_cache[idx]
        value, grad, v = _implicit_hypergradient_cg(
            upper_loss,
            lower_loss,
            x,
            y,
            cg_steps=cg_steps,
            damping=damping,
            v0=v0,
        )
        values.append(value)
        grads.append(grad)
        next_cache.append(v)
    return torch.stack(values), torch.stack(grads, dim=1), next_cache


def _clip_vector(vec: torch.Tensor, max_norm: float | None) -> tuple[torch.Tensor, float, float, bool]:
    vec = torch.nan_to_num(vec, nan=0.0, posinf=max_norm or 1e6, neginf=-(max_norm or 1e6))
    raw_norm = torch.linalg.norm(vec.detach())
    clipped = False
    if max_norm is not None and torch.isfinite(raw_norm) and raw_norm.item() > max_norm:
        vec = vec * (max_norm / (raw_norm + 1e-12))
        clipped = True
    norm = torch.linalg.norm(vec.detach())
    return vec, float(raw_norm.item()), float(norm.item()), clipped
