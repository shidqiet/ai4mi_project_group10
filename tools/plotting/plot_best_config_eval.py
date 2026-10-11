#!/usr/bin/env python3

# MIT License
#
# Copyright (c) 2025 Hoel Kervadec
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Qualitative plots for the best_config_eval runs, drawn from the post-processed native-space volumes."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.ndimage import find_objects


ORGANS = ("Esophagus", "Heart", "Trachea", "Aorta")
RECIPES = ("k64", "k32")
RECIPE_LABEL = {"k64": "k64 (lr 2.5e-4)", "k32": "k32 (lr 5e-4)"}
RECIPE_COLOUR = {"k64": "#2a78d6", "k32": "#eb6834"}
ORGAN_COLOUR = {"Esophagus": "#80ae80", "Heart": "#f1d691", "Trachea": "#b17a65", "Aorta": "#6fb8d2"}
INK = "#0b0b0b"
INK_2 = "#52514e"
SURFACE = "#fcfcfb"
MASK_ALPHA = 0.8
WINDOW = (-160, 240)  # mediastinal window, HU

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": INK_2, "text.color": INK, "axes.titlecolor": INK,
    "font.size": 10, "axes.titlesize": 10, "axes.titleweight": "bold",
})


def save(fig, out: Path, name: str) -> None:
    fig.savefig(out / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def load(path: Path) -> np.ndarray:
    return np.asarray(nib.load(path).dataobj)


def per_patient(sweep: Path, run: str) -> dict[tuple[str, str], float]:
    with open(sweep / run / "eval_lcc50" / "per_patient.csv") as f:
        return {(r["patient"], r["organ"]): float(r["dice"]) for r in csv.DictReader(f)}


def median_seed(sweep: Path, recipe: str) -> str:
    # Median seed by mean Dice, so we don't pick the nicest-looking run.
    runs = sorted(p.name for p in sweep.glob(f"unet_{recipe}_*"))
    mean = {r: np.mean(list(per_patient(sweep, r).values())) for r in runs}
    return sorted(runs, key=mean.get)[len(runs) // 2]


def axial(vol: np.ndarray, z: int) -> np.ndarray:
    # LPS array -> anterior up, patient right on the image left.
    return vol[:, :, z].T


def crop_box(masks: list[np.ndarray], pad: int) -> tuple[slice, slice]:
    sl = find_objects(np.any(np.stack(masks), 0).astype(int))[0]
    return tuple(slice(max(0, s.start - pad), s.stop + pad) for s in sl)


def corner(ax, text):
    ax.text(0.03, 0.97, text, transform=ax.transAxes, ha="left", va="top", fontsize=8, color="white",
            bbox=dict(boxstyle="round,pad=0.2", fc="black", ec="none", alpha=0.55))


def legend(fig, y):
    handles = [Patch(fc=ORGAN_COLOUR[o], alpha=MASK_ALPHA, label=o) for o in ORGANS]
    handles.append(Line2D([], [], color="white", lw=1.8, label="ground truth (outline)",
                          path_effects=[pe.Stroke(linewidth=3.2, foreground="black"), pe.Normal()]))
    leg = fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, y), ncol=5, frameon=True, fontsize=9)
    leg.get_frame().set_facecolor("#9a9995")
    leg.get_frame().set_edgecolor("none")


def fig_overview(patients, vols, scores, picks, out):
    # One axial slice per patient: the middle of the slices showing the most organs.
    cols = 5
    blocks = [patients[i:i + cols] for i in range(0, len(patients), cols)]
    rows = ["Ground truth", *[RECIPE_LABEL[r] for r in RECIPES]]
    fig, axes = plt.subplots(len(blocks) * len(rows), cols, figsize=(2.5 * cols, 2.6 * len(blocks) * len(rows)))
    for b, block in enumerate(blocks):
        for c, p in enumerate(block):
            ct, gt = vols[p]["ct"], vols[p]["gt"]
            present = np.stack([(gt == k).any((0, 1)) for k in range(1, 5)]).sum(0)
            zs = np.flatnonzero(present == present.max())
            z = int(zs[len(zs) // 2])
            box = crop_box([axial(gt, z) > 0] + [axial(vols[p][r], z) > 0 for r in RECIPES], pad=25)
            img, g = axial(ct, z)[box], axial(gt, z)[box]
            overlay(axes[b * len(rows), c], img, g)
            axes[b * len(rows), c].set_title(f"{p.replace('Patient_', 'P')}  slice {z}")
            for i, r in enumerate(RECIPES, 1):
                ax = axes[b * len(rows) + i, c]
                overlay(ax, img, axial(vols[p][r], z)[box], gt_outline=g)
                corner(ax, f"Dice {np.mean([scores[r][p, o] for o in ORGANS]):.3f}")
        for i, label in enumerate(rows):
            axes[b * len(rows) + i, 0].set_ylabel(label, fontsize=9, fontweight="bold",
                                                  color=RECIPE_COLOUR.get(RECIPES[i - 1], INK) if i else INK)
        for c in range(len(block), cols):
            for i in range(len(rows)):
                axes[b * len(rows) + i, c].axis("off")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    legend(fig, -0.005)
    fig.text(0.5, -0.03, "All 10 validation patients, post-processed native-space volumes. Axial slice with the most "
             "organs present (anterior up, patient right on the left); the trachea appears only as main bronchi in P01 and P22. Filled = "
             "segmentation mask; white outline = ground truth. Corner = 3D Dice for that patient, mean of 4 organs. "
             f"Median seed of each recipe: {picks['k64']}, {picks['k32']}.",
             ha="center", color=INK_2, fontsize=9, wrap=True)
    save(fig, out, "1_val_overview")


def slice_dice(a: np.ndarray, b: np.ndarray) -> float:
    d = a.sum() + b.sum()
    return 2 * (a & b).sum() / d if d else 1.0


def overlay(ax, img, mask, gt_outline=None):
    ax.imshow(img, cmap="gray", vmin=WINDOW[0], vmax=WINDOW[1], interpolation="none")
    rgba = np.zeros((*mask.shape, 4))
    for k, organ in enumerate(ORGANS, 1):
        rgba[mask == k] = (*to_rgb(ORGAN_COLOUR[organ]), MASK_ALPHA)
    ax.imshow(rgba, interpolation="none")
    if gt_outline is not None:
        cs = ax.contour(gt_outline > 0, levels=[0.5], colors="white", linewidths=1.8)
        cs.set_path_effects([pe.Stroke(linewidth=3.2, foreground="black"), pe.Normal()])
    ax.set_xticks([]); ax.set_yticks([])


def fig_hard_cases(hard, vols, out):
    # Per patient: the axial slice where both recipes do worst on the esophagus, and a mid-esophagus slice.
    cols = ["CT", "Ground truth", *[RECIPE_LABEL[r] for r in RECIPES]]
    fig, axes = plt.subplots(2 * len(hard), len(cols), figsize=(2.7 * len(cols), 2.7 * 2 * len(hard)))
    for i, p in enumerate(hard):
        ct, gt = vols[p]["ct"], vols[p]["gt"]
        eso = gt == 1
        zs = np.flatnonzero(eso.any((0, 1)))
        worst = {z: np.mean([slice_dice(eso[:, :, z], vols[p][r][:, :, z] == 1) for r in RECIPES]) for z in zs}
        for row, (z, what) in enumerate([(min(worst, key=worst.get), "worst"), (int(zs[len(zs) // 2]), "mid")]):
            box = crop_box([axial(eso, z)] + [axial(vols[p][r], z) == 1 for r in RECIPES], pad=35)
            img, g = axial(ct, z)[box], axial(gt, z)[box]
            ax_row = axes[2 * i + row]
            ax_row[0].imshow(img, cmap="gray", vmin=WINDOW[0], vmax=WINDOW[1], interpolation="none")
            ax_row[0].set_xticks([]); ax_row[0].set_yticks([])
            overlay(ax_row[1], img, g)
            for j, r in enumerate(RECIPES):
                pred = axial(vols[p][r], z)
                overlay(ax_row[2 + j], img, pred[box], gt_outline=g)
                corner(ax_row[2 + j], f"Eso Dice {slice_dice(eso[:, :, z], vols[p][r][:, :, z] == 1):.2f}")
            ax_row[0].set_ylabel(f"{p.replace('Patient_', 'P')} slice {z}\n({what} esophagus)", fontsize=9)
    for j, c in enumerate(cols):
        axes[0, j].set_title(c, color=RECIPE_COLOUR.get(RECIPES[j - 2], INK) if j >= 2 else INK)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    legend(fig, -0.005)
    fig.text(0.5, -0.03, "The three patients with the lowest esophagus Dice (mean over all 6 runs). For each, the axial "
             "slice where the two recipes' mean esophagus Dice is lowest, and the slice at the middle of the esophagus "
             "(anterior up). Filled = segmentation mask; white outline = ground truth. Corner = that slice's 2D "
             "esophagus Dice.", ha="center", color=INK_2, fontsize=9, wrap=True)
    save(fig, out, "2_hard_cases")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sweep", type=Path, default=Path("results/segthor/best_config_eval"))
    p.add_argument("--volumes", type=Path, default=Path("volumes/lcc50/best_config_eval"))
    p.add_argument("--data", type=Path, default=Path("tools/data/segthor_train_full/train"))
    p.add_argument("--out", type=Path, default=None, help="default: docs/experiments/<sweep name>")
    args = p.parse_args()
    figs = (args.out or Path("docs/experiments") / args.sweep.name) / "figures"
    figs.mkdir(parents=True, exist_ok=True)

    picks = {r: median_seed(args.sweep, r) for r in RECIPES}
    scores = {r: per_patient(args.sweep, picks[r]) for r in RECIPES}
    patients = sorted({pt for pt, _ in scores["k64"]})
    print("median seeds:", picks)

    vols = {}
    for pt in patients:
        vols[pt] = {"ct": load(args.data / pt / f"{pt}.nii.gz"), "gt": load(args.data / pt / "GT.nii.gz")}
        for r in RECIPES:
            vols[pt][r] = load(args.volumes / picks[r] / f"{pt}.nii.gz")

    eso_all = defaultdict(list)
    for run in (d.name for d in args.sweep.glob("unet_*")):
        for (pt, organ), d in per_patient(args.sweep, run).items():
            if organ == "Esophagus":
                eso_all[pt].append(d)
    hard = sorted(patients, key=lambda pt: np.mean(eso_all[pt]))[:3]
    print("hard cases:", hard)

    fig_overview(patients, vols, scores, picks, figs)
    fig_hard_cases(hard, vols, figs)
    print(f"wrote {figs}")


if __name__ == "__main__":
    main()
