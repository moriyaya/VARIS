"""Set and preference metrics; expert masks are used only for evaluation."""
import numpy as np
import torch
from .model import preference_indices,iou_matrix,staged_q_dice

@torch.no_grad()
def evaluate(model, loader, device: torch.device) -> dict:
    model.eval()
    rows = []
    for names, image, masks, pref in loader:
        image, pref = image.to(device), pref.to(device)
        masks = [item.to(device) for item in masks]
        logits = model(image)
        probabilities = torch.sigmoid(logits)
        pred = (probabilities >= 0.5).float()
        for b, name in enumerate(names):
            p, g = pred[b], masks[b]
            pg_iou = iou_matrix(p, g)
            pg_dice = 2 * pg_iou / (1 + pg_iou)
            selected = preference_indices(p).to(device)
            reference = preference_indices(g).to(device)
            preference_scores = pg_dice[selected, reference]
            rows.append({
                "conservative_dice": float(preference_scores[0]),
                "central_dice": float(preference_scores[1]),
                "sensitive_dice": float(preference_scores[2]),
                "preference_dice": float(preference_scores.mean()),
                "case": name,
                "annotation_coverage_dice": float(pg_dice.max(0).values.mean()),
                "prediction_precision_dice": float(pg_dice.max(1).values.mean()),
                "q_dice": float(staged_q_dice(probabilities[b].mean(0), g.mean(0))),
            })
    keys = [key for key in rows[0] if key != "case"]
    summary = {key: float(np.mean([row[key] for row in rows])) for key in keys}
    summary["cases"] = len(rows)
    return {"summary": summary, "cases": rows}

