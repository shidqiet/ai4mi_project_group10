#!/usr/bin/env python3.10

# MIT License

# Copyright (c) 2025 Hoel Kervadec, Caroline Magg

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import csv
import json
import argparse
from pathlib import Path

import numpy as np
import nibabel as nib

from ai4mi.utils import starmmap_, surface_distance_metrics_binary

ORGAN_NAMES = ("Esophagus", "Heart", "Trachea", "Aorta")
METRICS = ("dice", "hd95", "hd", "assd")


def evaluate_patient(pred_path: Path, gt_path: Path, K: int, background: bool = False) -> dict[str, list[float]]:
    pred_nib = nib.load(pred_path)
    gt_nib = nib.load(gt_path)
    pred = np.asarray(pred_nib.dataobj).astype(np.int64)
    gt = np.asarray(gt_nib.dataobj).astype(np.int64)
    assert pred.shape == gt.shape, (pred_path, pred.shape, gt.shape)
    assert set(np.unique(pred)) <= set(range(K)), (pred_path, np.unique(pred))
    assert set(np.unique(gt)) <= set(range(K)), (gt_path, np.unique(gt))
    spacing = tuple(float(s) for s in gt_nib.header.get_zooms()[:3])

    scores: dict[str, list[float]] = {m: [] for m in METRICS}
    for k in range(0 if background else 1, K):
        p, g = pred == k, gt == k
        denom = p.sum() + g.sum()
        scores["dice"].append(float(2 * (p & g).sum() / denom) if denom else 1.0)
        hd95, hd, assd = surface_distance_metrics_binary(p, g, spacing)
        scores["hd95"].append(hd95)
        scores["hd"].append(hd)
        scores["assd"].append(assd)

    return scores


def main(args) -> None:
    pred_paths: list[Path] = sorted(args.pred_folder.glob("*.nii.gz"))
    assert pred_paths, f"No volumes found in {args.pred_folder}"
    ids: list[str] = [p.name.removesuffix(".nii.gz") for p in pred_paths]
    gt_paths: list[Path] = [Path(args.gt_pattern.format(id_=id_)) for id_ in ids]
    print(f"Evaluating {len(ids)} patients from {args.pred_folder}")

    K: int = args.num_classes
    organs = [ORGAN_NAMES[k - 1] if k <= len(ORGAN_NAMES) else f"Class {k}" for k in range(1, K)]
    background = args.npz_dir is not None  # The .npz files hold all K classes
    results = starmmap_(evaluate_patient, zip(pred_paths, gt_paths, [K] * len(ids), [background] * len(ids)))
    if background:
        args.npz_dir.mkdir(parents=True, exist_ok=True)
        for m in METRICS:  # <metric>.npz: patient ID -> (K,) array, background first
            np.savez(args.npz_dir / f"{m}.npz", **{id_: np.asarray(s[m]) for id_, s in zip(ids, results)})
        print(f"Saved {', '.join(f'{m}.npz' for m in METRICS)} to {args.npz_dir}")
        results = [{m: s[m][1:] for m in METRICS} for s in results]  # Organs only from here on

    args.dest.mkdir(parents=True, exist_ok=True)
    with open(args.dest / "per_patient.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["patient", "organ", *METRICS])
        for id_, scores in zip(ids, results):
            for i, organ in enumerate(organs):
                writer.writerow([id_, organ, *(scores[m][i] for m in METRICS)])

    # Per-organ mean/std over patients, plus the mean over organs
    summary: dict[str, dict[str, dict[str, float]]] = {}
    for m in METRICS:
        table = np.asarray([scores[m] for scores in results])  # patients x organs
        summary[m] = {organ: {"mean": float(table[:, i].mean()), "std": float(table[:, i].std())}
                      for i, organ in enumerate(organs)}
        organ_means = table.mean(axis=1)
        summary[m]["Mean"] = {"mean": float(organ_means.mean()), "std": float(organ_means.std())}

    with open(args.dest / "summary.json", "w") as f:
        json.dump({"patients": ids, "metrics": summary}, f, indent=2)

    print(f"{'':<10}" + "".join(f"{m:>18}" for m in METRICS))
    for organ in [*organs, "Mean"]:
        row = "".join(f"{summary[m][organ]['mean']:>10.3f} ± {summary[m][organ]['std']:<5.3f}"
                      for m in METRICS)
        print(f"{organ:<10}{row}")
    print(f"Saved per-patient scores and summary to {args.dest}")


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Evaluate stitched volumes against the ground truth')
    parser.add_argument('--pred_folder', type=Path, required=True,
                        help="Folder of stitched <patient>.nii.gz volumes (output of stitch.py)")
    parser.add_argument('--gt_pattern', type=str, required=True,
                        help="Pattern to the ground truth, e.g. 'tools/data/segthor_train_full/train/{id_}/GT.nii.gz'")
    parser.add_argument('--dest', type=Path, required=True,
                        help="Folder to write per_patient.csv and summary.json to")
    parser.add_argument('--num_classes', type=int, default=5)
    parser.add_argument('--npz_dir', type=Path, default=None,
                        help="Also write <metric>.npz files here, mapping patient ID to a (K,) array "
                             "over all classes, background included (submission format)")

    args = parser.parse_args()

    print(args)

    return args


if __name__ == "__main__":
    main(get_args())
