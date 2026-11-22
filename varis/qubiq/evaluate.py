"""Set and preference metrics; expert masks are used only for evaluation."""
import time
import numpy as np
import torch
from .model import preference_indices,iou_matrix,staged_q_dice

@torch.no_grad()
@torch.no_grad()
def evaluate(model, loader, device: torch.device) -> dict:
    model.eval()
    rows = []
    elapsed = 0.0
    for names, image, masks, pref in loader:
        image, pref = image.to(device), pref.to(device)
        masks = [item.to(device) for item in masks]
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        logits = model(image)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed += time.perf_counter() - start
        probabilities = torch.sigmoid(logits)
        pred = (probabilities >= 0.5).float()
        for b, name in enumerate(names):
            p, g = pred[b], masks[b]
            pg_iou = iou_matrix(p, g)
            pp_iou = iou_matrix(p, p)
            gg_iou = iou_matrix(g, g)
            pg_dice = 2 * pg_iou / (1 + pg_iou)
            selected = preference_indices(p).to(device)
            reference = preference_indices(g).to(device)
            rows.append({
                "preference_dice": float(pg_dice[selected, reference].mean()),
                "case": name,
                "annotation_coverage_dice": float(pg_dice.max(0).values.mean()),
                "prediction_precision_dice": float(pg_dice.max(1).values.mean()),
                "oracle_dice": float(pg_dice.max()),
                "prediction_diversity": float((1 - pp_iou)[~torch.eye(len(p), dtype=torch.bool, device=device)].mean()) if len(p) > 1 else 0.0,
                "ged_iou": float(2 * (1 - pg_iou).mean() - (1 - pp_iou).mean() - (1 - gg_iou).mean()),
                "q_dice": float(staged_q_dice(probabilities[b].mean(0), g.mean(0))),
            })
    keys = [key for key in rows[0] if key != "case"]
    summary = {key: float(np.mean([row[key] for row in rows])) for key in keys}
    summary["cases"] = len(rows)
    summary["seconds_per_case"] = elapsed / len(rows)
    return {"summary": summary, "cases": rows}

