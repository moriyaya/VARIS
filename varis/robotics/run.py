"""Evaluate VARIS on the fixed Panda target protocol with a frozen IKFlow model."""
import argparse
import json
from pathlib import Path
import torch
from .backend import PandaBackend, make_dataset
from . import solver

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path("outputs/panda"))
    p.add_argument("--dataset", type=Path, default=Path("assets/robotics/data/targets.pt"))
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    solver.OUT = a.output
    backend = PandaBackend()
    cfg = solver.CFG
    if not a.dataset.exists():
        make_dataset(backend, cfg["dataset_size"], cfg["dataset_seed"], a.dataset)
    data = torch.load(a.dataset, map_location="cpu", weights_only=True)
    # Witnesses construct targets; the response oracle never receives them.
    data.pop("withheld_witness", None)
    for tid in range(cfg["formal_targets"]):
        goal = data["target_poses"][tid].to(backend.device)
        start = data["q_start"][tid].to(backend.device)
        warm = solver.warm_pool(backend, goal, tid, a.seed, data["dataset_seed"])
        for r in cfg["preferences_r"]:
            destination = a.output/f"target{tid:03d}_seed{a.seed}_r{r}.json"
            if destination.exists():
                continue
            result = solver.solve(backend, goal, start, warm, tid, a.seed, r, data["dataset_seed"])
            result.update(target=tid, seed=a.seed, preference=r)
            archive = result["archive"]
            result["selected"] = min(archive, key=lambda v:r*v["F"][0]+(1-r)*v["F"][1]) if archive else None
            destination.write_text(json.dumps(result, indent=2, allow_nan=False))
            print(json.dumps(dict(target=tid, preference=r, valid=bool(archive), seconds=result["solve_seconds"])), flush=True)

if __name__ == "__main__":
    main()
