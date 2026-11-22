"""Train/evaluate six-response segmentation using the frozen five-fold protocol."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from .model import seed_everything, QubiqDataset, qubiq_collate, CompactUNet, matched_loss
from .evaluate import evaluate

ROOT = Path(__file__).resolve().parents[2]

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["train", "evaluate"])
    p.add_argument("--data", type=Path, required=True, help="Directory containing case01/.../case48.")
    p.add_argument("--fold", type=int, choices=range(5), default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--config", type=Path, default=ROOT/"configs/qubiq.json")
    p.add_argument("--splits", type=Path, default=ROOT/"configs/qubiq_splits.json")
    p.add_argument("--device", default="cpu")
    p.add_argument("--epochs", type=int)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--output", type=Path, default=Path("outputs/qubiq"))
    a = p.parse_args()
    cfg = json.loads(a.config.read_text())
    seed_everything(a.seed)
    torch.set_num_threads(2)
    device = torch.device(a.device)
    a.output.mkdir(parents=True, exist_ok=True)
    split = json.loads(a.splits.read_text())["folds"][a.fold]
    all_cases = [case for key in ("train", "val", "test") for case in split[key]]
    assert len(all_cases) == len(set(all_cases)) == 48
    loaders = {}
    for key in ("train", "val", "test"):
        data = QubiqDataset([a.data/case for case in split[key]], cfg["task"], cfg["size"], key == "train")
        loaders[key] = DataLoader(data, batch_size=cfg["batch_size"] if key == "train" else 1,
                                  shuffle=key == "train", num_workers=0, collate_fn=qubiq_collate)
    model = CompactUNet(1, cfg["candidates"]).to(device)
    checkpoint = a.checkpoint or a.output/"best.pt"
    history = []
    if a.action == "train":
        epochs = a.epochs or cfg["epochs"]
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
        best = -np.inf
        for epoch in range(1, epochs+1):
            model.train()
            losses = []
            for _, image, masks, _ in loaders["train"]:
                logits = model(image.to(device))
                loss = torch.stack([matched_loss(logits[b:b+1], mask.to(device)[None]) for b, mask in enumerate(masks)]).mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
                optimizer.step()
                losses.append(float(loss.detach()))
            scheduler.step()
            if epoch == 1 or epoch % cfg["eval_interval"] == 0 or epoch == epochs:
                val = evaluate(model, loaders["val"], device)["summary"]
                record = dict(epoch=epoch, training_loss=float(np.mean(losses)), validation=val)
                history.append(record)
                print(json.dumps(record), flush=True)
                if val["annotation_coverage_dice"] > best:
                    best = val["annotation_coverage_dice"]
                    torch.save(dict(model=model.state_dict(), epoch=epoch, seed=a.seed, fold=a.fold, config=cfg), checkpoint)
        (a.output/"history.json").write_text(json.dumps(history, indent=2))
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state["model"])
    result = dict(seed=a.seed, fold=a.fold, epoch=state["epoch"], split=split, config=cfg,
                  validation=evaluate(model, loaders["val"], device), test=evaluate(model, loaders["test"], device))
    (a.output/"result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result["test"]["summary"]), flush=True)

if __name__ == "__main__":
    main()
