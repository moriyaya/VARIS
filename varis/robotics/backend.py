"""Shared model, physical checker and deterministic repair for the Panda study."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2] / "assets" / "robotics"
sys.path.insert(0, str(ROOT / "vendor" / "jrl"))
sys.path.insert(0, str(ROOT / "vendor" / "ikflow"))


def read_config() -> dict:
    return json.loads((Path(__file__).resolve().parents[2] / "configs" / "panda.json").read_text())


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class PandaBackend:
    def __init__(self, load_generator: bool = True):
        torch.set_num_threads(2)
        torch.set_grad_enabled(False)
        from jrl.robots import get_robot
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.robot = get_robot("panda")
        self.limits = torch.tensor(self.robot.actuated_joints_limits, device=self.device, dtype=torch.float32)
        self.lower, self.upper = self.limits[:, 0], self.limits[:, 1]
        self.span = self.upper - self.lower
        self.mid = (self.upper + self.lower) / 2
        self.cfg = read_config()
        self.solver = None
        if load_generator:
            from ikflow.model import IkflowModelParameters
            from ikflow.ikflow_solver import IKFlowSolver
            import yaml
            descriptions = yaml.safe_load((ROOT / "vendor" / "ikflow" / "ikflow" / "model_descriptions.yaml").read_text())
            parameters = IkflowModelParameters()
            parameters.__dict__.update(descriptions[self.cfg["generator_checkpoint"]])
            self.solver = IKFlowSolver(parameters, self.robot)
            self.solver.load_state_dict(str(ROOT / "downloads" / "panda_checkpoint.pkl"))
        self.collision_calls = 0

    def tensor(self, value) -> torch.Tensor:
        return torch.as_tensor(value, dtype=torch.float32, device=self.device)

    def clamp(self, q: torch.Tensor) -> torch.Tensor:
        return torch.maximum(torch.minimum(q, self.upper), self.lower)

    def collision_free(self, q: torch.Tensor) -> torch.Tensor:
        # Klampt collision checks mutate the robot model; execute serially.
        arrays = q.detach().cpu().numpy()
        valid = np.asarray([not self.robot.config_self_collides(row) for row in arrays], dtype=bool)
        self.collision_calls += len(arrays)
        return torch.as_tensor(valid, device=self.device)

    def pose_errors(self, q: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        from jrl.math_utils import geodesic_distance_between_quaternions
        actual = self.robot.forward_kinematics(q)
        ep = (actual[:, :3] - target[:, :3]).norm(dim=-1)
        er = geodesic_distance_between_quaternions(actual[:, 3:], target[:, 3:])
        return ep, er

    def evaluate(self, q: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        ep, er = self.pose_errors(q, target)
        pose_error = torch.maximum((ep / 0.001).square(), (er / math.radians(1)).square())
        bounded = ((q >= self.lower - 1e-6) & (q <= self.upper + 1e-6)).all(dim=-1)
        finite = torch.isfinite(q).all(dim=-1) & torch.isfinite(pose_error)
        safe = torch.zeros(len(q), dtype=torch.bool, device=self.device)
        idx = torch.where(finite & bounded)[0]
        if len(idx):
            safe[idx] = self.collision_free(q[idx])
        error = torch.where(safe, pose_error, torch.inf)
        return {"position_m": ep, "orientation_rad": er, "safe": safe,
                "pose_error": pose_error, "error": error, "valid": error <= 1.0}

    def sample_safe(self, count: int, rng: np.random.Generator) -> torch.Tensor:
        accepted = []
        total = 0
        low, high = self.lower.cpu().numpy(), self.upper.cpu().numpy()
        for _ in range(100):
            batch = max(64, 2 * (count - total))
            q = self.tensor(rng.uniform(low + 1e-6, high - 1e-6, size=(batch, 7)))
            good = q[self.collision_free(q)]
            accepted.append(good)
            total += len(good)
            if total >= count:
                return torch.cat(accepted)[:count]
        raise RuntimeError("Safe-configuration sampling failed")

    def generate(self, target: torch.Tensor, count: int, seed: int) -> tuple[torch.Tensor, dict]:
        assert self.solver is not None
        generator = torch.Generator(device="cpu").manual_seed(seed)
        z = torch.randn(len(target), count, self.solver.network_width, generator=generator, device="cpu").to(self.device)
        repeated = target[:, None, :].expand(-1, count, -1).reshape(-1, 7)
        sync()
        start = time.perf_counter()
        raw = self.solver.generate_ik_solutions(repeated, n=None, latent=z.flatten(0, 1),
                                               clamp_to_joint_limits=False, refine_solutions=False)
        clamped = self.clamp(raw)
        sync()
        return clamped.reshape(len(target), count, 7), {
            "generation_s": time.perf_counter() - start,
            "pre_clamp_joint_violation_count": int(((raw < self.lower) | (raw > self.upper)).any(-1).sum()),
        }

    def lm_step(self, q: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Same LM residual and normal equations as JRL's public primitive.
        # Its vectorized Jacobian removes the serial numpy Jacobian loop.
        from jrl.math_utils import quaternion_inverse, quaternion_product, quaternion_to_rpy
        current = self.robot.forward_kinematics(q)
        rotation = quaternion_to_rpy(quaternion_product(target[:, 3:], quaternion_inverse(current[:, 3:])))
        residual = torch.cat((rotation, target[:, :3] - current[:, :3]), dim=-1)
        jacobian = self.robot.jacobian(q)
        transposed = jacobian.transpose(1, 2)
        matrix = transposed.bmm(jacobian) + 1e-4 * torch.eye(7, device=self.device)[None]
        step = torch.linalg.solve(matrix, transposed.bmm(residual[:, :, None])).squeeze(-1)
        return self.clamp(q + step)

    def repair_all(self, q: torch.Tensor, target: torch.Tensor, initial=None) -> tuple[torch.Tensor, dict]:
        current = q.clone()
        seed = q.clone()
        initial = self.evaluate(current, target) if initial is None else initial
        final = {key: value.clone() for key, value in initial.items()}
        error = initial["error"].clone()
        eligible = (error > 1) & (error <= 25)
        factors = torch.ones(len(q), device=self.device)
        attempts = torch.zeros(len(q), dtype=torch.int32, device=self.device)
        accepted = torch.zeros_like(attempts)
        traces = []
        for trial in range(self.cfg["repair"]["maximum_total_trials_including_rejections"]):
            ids = torch.where(eligible & (error > 1))[0]
            if not len(ids):
                break
            proposal = current[ids] + factors[ids, None] * (self.lm_step(current[ids], target[ids]) - current[ids])
            proposal = self.clamp(proposal)
            displacement = (proposal - seed[ids]) / self.span
            norm = displacement.square().mean(-1).sqrt()
            scale = torch.minimum(torch.ones_like(norm), 0.05 / norm.clamp_min(1e-12))
            proposal = seed[ids] + scale[:, None] * (proposal - seed[ids])
            result = self.evaluate(proposal, target[ids])
            step_norm = ((proposal - current[ids]) / self.span).square().mean(-1)
            improve = result["error"] <= error[ids] - 1e-4 * step_norm
            attempts[ids] += 1
            good_ids = ids[improve]
            current[good_ids] = proposal[improve]
            error[good_ids] = result["error"][improve]
            for key in final:
                final[key][good_ids] = result[key][improve]
            accepted[good_ids] += 1
            factors[ids[~improve]] *= 0.5
            traces.append({"trial": trial + 1, "attempted": len(ids), "accepted": int(improve.sum()),
                           "newly_valid": int((result["error"][improve] <= 1).sum())})
        return current, {"initial": initial, "final": final, "attempts": attempts, "accepted": accepted,
                         "eligible": eligible, "trace_summary": traces,
                         "displacement": ((current - seed) / self.span).square().mean(-1).sqrt()}

    def solve_pool(self, q: torch.Tensor, target: torch.Tensor, maximum_steps: int = 50) -> tuple[torch.Tensor, dict]:
        current = q.clone()
        attempts = torch.zeros(len(q), dtype=torch.int32, device=self.device)
        sync()
        start = time.perf_counter()
        for _ in range(maximum_steps):
            ep, er = self.pose_errors(current, target)
            active = (ep > 0.001) | (er > math.radians(1))
            ids = torch.where(active)[0]
            if not len(ids):
                break
            current[ids] = self.lm_step(current[ids], target[ids])
            attempts[ids] += 1
        final = self.evaluate(current, target)
        sync()
        final["attempts"] = attempts
        final["solve_s"] = time.perf_counter() - start
        return current, final

    def objectives(self, q: torch.Tensor, q_start: torch.Tensor, x: float) -> tuple[torch.Tensor, torch.Tensor]:
        return (((q - q_start) / self.span).square().mean(-1) + 0.05 * (x / 0.1) ** 2,
                4 * ((q - self.mid) / self.span).square().mean(-1))


def make_dataset(backend: PandaBackend, count: int, seed: int, output: Path) -> dict:
    rng = np.random.default_rng(seed)
    witness = backend.sample_safe(count, rng)
    starts = backend.sample_safe(count, rng)
    targets = backend.robot.forward_kinematics(witness)
    data = {"target_poses": targets.cpu(), "q_start": starts.cpu(), "withheld_witness": witness.cpu(),
            "target_ids": torch.arange(count, device="cpu"), "dataset_seed": seed}
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, output)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    write_json(output.with_suffix(".manifest.json"), {"targets": count, "seed": seed, "sha256": digest,
               "withheld_witness_used_by_solvers": False, "joint_names": backend.robot.actuated_joint_names,
               "tool_frame": backend.robot.end_effector_link_name, "limits": backend.limits.cpu().tolist()})
    return data
