"""Train the conditional generator or evaluate the VARIS response oracle."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from .model import ResponseGenerator, make_dataset, branch_batch
from .physics import lower_loss, N_MODES
from .inference import make_starts, preference_repair_varis, score_rows, save_csv

ROOT = Path(__file__).resolve().parents[2]

def train(cfg, seed, device, output, quick):
    torch.manual_seed(51000 + seed)
    count = 16 if quick else cfg.train_tasks
    data = make_dataset(count, 52000 + seed, device, "train")
    model = ResponseGenerator().to(device)
    phases = [(cfg.distill_steps, cfg.lower_steps, cfg.learning_rate),
              (0, cfg.finetune_steps, cfg.finetune_lr)]
    for phase, (distill_steps, lower_steps, lr) in enumerate(phases):
        if quick:
            distill_steps, lower_steps = (1, 1) if phase == 0 else (0, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        sampler = torch.Generator().manual_seed(54000 + seed)
        for step in range(distill_steps + lower_steps):
            ids = torch.randint(count, (cfg.batch_size,), generator=sampler).to(device)
            targets, nu, obs, codes = branch_batch(*(x[ids] for x in data))
            y = model(obs, nu, codes)
            loss = (y - targets).square().mean()
            if step >= distill_steps:
                loss = loss + cfg.lower_weight * lower_loss(y, nu, obs).mean()
                loss = loss + cfg.branch_weight * torch.relu(.10 - codes*y[:, N_MODES]).square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            if (step+1) % 250 == 0 or step+1 == distill_steps+lower_steps:
                print(json.dumps(dict(phase=phase+1, step=step+1, loss=float(loss.detach()))), flush=True)
    torch.save(dict(state_dict=model.cpu().state_dict(), seed=seed, config=vars(cfg)), output/"generator.pt")

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["train", "evaluate"])
    p.add_argument("--config", type=Path, default=ROOT/"configs/pde.json")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--output", type=Path, default=Path("outputs/pde"))
    p.add_argument("--tasks", type=int)
    p.add_argument("--quick", action="store_true", help="Smoke run; not a paper evaluation.")
    a = p.parse_args()
    torch.set_num_threads(2)
    cfg = SimpleNamespace(**json.loads(a.config.read_text()))
    a.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(a.device)
    (a.output/"run.json").write_text(json.dumps(dict(seed=a.seed, config=vars(cfg), quick=a.quick, device=a.device), indent=2))
    if a.action == "train":
        train(cfg, a.seed, device, a.output, a.quick)
        return
    model = ResponseGenerator().to(device)
    checkpoint = a.checkpoint or ROOT/f"checkpoints/pde/seed{a.seed}.pt"
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True)["state_dict"])
    model.eval()
    tasks = a.tasks or (4 if a.quick else cfg.test_tasks)
    prefs = torch.tensor(cfg.preferences, device=device)
    rows, summary = [], {}
    for index, split in enumerate(["interpolation", "low_extrapolation", "high_extrapolation"]):
        teacher, nu, observation = make_dataset(tasks, 53000 + 997*index+a.seed, device, split)
        starts = make_starts(tasks, cfg.starts, a.seed+997*index, device)
        y, meta, _, stages = preference_repair_varis(model, starts, nu, observation, prefs, cfg)
        records = score_rows(y, teacher, nu, observation, prefs, "VARIS", split, a.seed, cfg.threshold, stages)
        rows.extend(records)
        meta.update({key:float(np.mean([r[key] for r in records])) for key in ["valid", "joint", "branch_correct", "lower_residual"]})
        summary[split] = meta
        save_csv(a.output/"results.csv", rows)
        (a.output/"summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(dict(split=split, **meta)), flush=True)

if __name__ == "__main__":
    main()
