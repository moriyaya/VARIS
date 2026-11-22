#!/usr/bin/env python3
"""Create deterministic five-fold QUBIQ splits balanced by expert disagreement."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np


def load_mask(path: Path) -> np.ndarray:
    return np.squeeze(np.asarray(nib.load(str(path)).get_fdata())) > 0.5


def mean_pairwise_iou(masks: list[np.ndarray]) -> float:
    scores = []
    for left in range(len(masks)):
        for right in range(left + 1, len(masks)):
            inter = np.logical_and(masks[left], masks[right]).sum()
            union = np.logical_or(masks[left], masks[right]).sum()
            scores.append(float(inter / union) if union else 1.0)
    return float(np.mean(scores))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", type=int, default=2)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    cases = sorted(path for path in args.root.glob("case*") if path.is_dir())
    diagnostics = []
    for case in cases:
        mask_paths = sorted(case.glob(f"task{args.task:02d}_seg*.nii.gz"))
        masks = [load_mask(path) for path in mask_paths]
        pair_iou = mean_pairwise_iou(masks)
        diagnostics.append({
            "case": case.name,
            "annotation_count": len(masks),
            "mean_pairwise_iou": pair_iou,
            "disagreement": 1 - pair_iou,
        })

    ordered = sorted(
        diagnostics,
        key=lambda item: (-item["disagreement"], item["case"]),
    )
    buckets = [[] for _ in range(args.folds)]
    for block_index, start in enumerate(range(0, len(ordered), args.folds)):
        block = ordered[start:start + args.folds]
        destinations = list(range(args.folds))
        if block_index % 2:
            destinations.reverse()
        for item, fold_index in zip(block, destinations):
            buckets[fold_index].append(item["case"])

    folds = []
    all_names = {item["case"] for item in diagnostics}
    for test_fold in range(args.folds):
        validation_fold = (test_fold + 1) % args.folds
        test = sorted(buckets[test_fold])
        validation = sorted(buckets[validation_fold])
        train = sorted(all_names - set(test) - set(validation))
        folds.append({
            "fold": test_fold,
            "train": train,
            "val": validation,
            "test": test,
        })

    report = {
        "task": args.task,
        "strategy": "descending expert disagreement with serpentine fold assignment",
        "validation_fold_rule": "(test_fold + 1) mod 5",
        "case_diagnostics": diagnostics,
        "fold_buckets": [sorted(bucket) for bucket in buckets],
        "folds": folds,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "fold_sizes": [len(bucket) for bucket in buckets],
        "annotation_counts": {
            str(count): sum(item["annotation_count"] == count for item in diagnostics)
            for count in sorted({item["annotation_count"] for item in diagnostics})
        },
    }, indent=2))


if __name__ == "__main__":
    main()
