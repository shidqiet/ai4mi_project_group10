#!/usr/bin/env python3
"""Create figures and tables for the SegTHOR optimiser screen."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from plot_arch_sweep import INK, INK_2, ORGANS, SEQ_BLUE, SURFACE, curve, load_runs, save


COLOURS = {"adam": "#2a78d6", "adamw": "#1baf7a", "sgd_nesterov": "#eb6834"}
LABELS = {"adam": "Adam", "adamw": "AdamW", "sgd_nesterov": "Nesterov SGD"}
LRS = (2.5e-4, 5e-4, 1e-3)
SGD_LRS = (3e-3, 1e-2, 3e-2)
WDS = (0.0, 1e-4)


def score(run):
    return float(np.mean(run.summary["dice3d"]))


def optimiser(run):
    return run.summary["args"]["optimizer"]


def lr(run):
    return float(run.summary["args"]["lr"])


def wd(run):
    return float(run.summary["args"]["weight_decay"])


def selected(run):
    return run.summary["best_epoch"] + 1


def config_key(run):
    return optimiser(run), lr(run), wd(run)


def configs(runs):
    out = {}
    for run in runs:
        out.setdefault(config_key(run), []).append(run)
    return out


def mean_score(runs):
    return float(np.mean([score(run) for run in runs]))


def std_score(runs):
    return float(np.std([score(run) for run in runs]))


def best_config(groups, opt=None):
    candidates = [(key, runs) for key, runs in groups.items() if opt is None or key[0] == opt]
    return max(candidates, key=lambda item: mean_score(item[1]))


def fig_heatmaps(runs, out):
    groups = configs(runs)
    opts = ("adam", "adamw", "sgd_nesterov")
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.1), sharey=True)
    cmap = plt.matplotlib.colors.LinearSegmentedColormap.from_list("sweep", SEQ_BLUE)
    values = np.array([score(r) for r in runs])
    norm = plt.matplotlib.colors.Normalize(values.min(), values.max())
    for ax, opt in zip(axes, opts):
        lrs = SGD_LRS if opt == "sgd_nesterov" else LRS
        grid = np.full((len(WDS), len(lrs)), np.nan)
        counts = np.zeros_like(grid, dtype=int)
        for (group_opt, group_lr, group_wd), rs in groups.items():
            if group_opt == opt:
                i, j = WDS.index(group_wd), lrs.index(group_lr)
                grid[i, j], counts[i, j] = mean_score(rs), len(rs)
        im = ax.imshow(grid, cmap=cmap, norm=norm, aspect="auto", origin="lower")
        winner = np.unravel_index(np.nanargmax(grid), grid.shape)
        for (i, j), value in np.ndenumerate(grid):
            colour = "white" if norm(value) > 0.55 else INK
            suffix = f"\n±{std_score(groups[(opt, lrs[j], WDS[i])]):.3f}" if counts[i, j] > 1 else ""
            ax.text(j, i, f"{value:.3f}{suffix}", ha="center", va="center", color=colour,
                    fontweight="bold" if (i, j) == winner else "normal")
        ax.add_patch(plt.Rectangle((winner[1] - .5, winner[0] - .5), 1, 1,
                                   fill=False, ec=INK, lw=2))
        ax.set_title(LABELS[opt], loc="left")
        ax.set_xticks(range(len(lrs)), [f"{value:g}" for value in lrs])
        ax.set_xlabel("initial learning rate")
        ax.grid(False)
    axes[0].set_yticks(range(len(WDS)), ["0", "1e-4"])
    axes[0].set_ylabel("weight decay")
    fig.subplots_adjust(left=.08, right=.88, bottom=.20, top=.90, wspace=.06)
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=.025, pad=.02,
                 label="mean validation 3D Dice")
    fig.text(.5, -.03, "Mean ± std where three seeds are available; other cells are seed-0 screens. Box = best setting within optimiser.",
             ha="center", color=INK_2, fontsize=9)
    save(fig, out, "1_optimiser_heatmaps")


def fig_curves(runs, out):
    groups = configs(runs)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True)
    for opt in ("adam", "adamw", "sgd_nesterov"):
        rs = [r for r in runs if optimiser(r) == opt]
        _, champion = best_config(groups, opt)
        for r in rs:
            x = np.arange(1, len(curve(r, "dice3d_mean")) + 1)
            axes[0].plot(x, curve(r, "dice3d_mean"), color=COLOURS[opt], alpha=.25, lw=1)
            axes[1].plot(x, curve(r, "loss_val"), color=COLOURS[opt], alpha=.25, lw=1)
        x = np.arange(1, len(curve(champion[0], "dice3d_mean")) + 1)
        dice = np.stack([curve(run, "dice3d_mean") for run in champion])
        loss = np.stack([curve(run, "loss_val") for run in champion])
        label = f"{LABELS[opt]} best ({lr(champion[0]):g}, wd {wd(champion[0]):g})"
        axes[0].fill_between(x, dice.min(0), dice.max(0), color=COLOURS[opt], alpha=.15, lw=0)
        axes[0].plot(x, dice.mean(0), color=COLOURS[opt], lw=2.4, label=label)
        axes[0].plot([selected(run) for run in champion], [curve(run, "dice3d_mean")[selected(run) - 1] for run in champion], "o",
                     color=COLOURS[opt], mec=SURFACE, mew=1.3)
        axes[1].fill_between(x, loss.min(0), loss.max(0), color=COLOURS[opt], alpha=.15, lw=0)
        axes[1].plot(x, loss.mean(0), color=COLOURS[opt], lw=2.4)
    axes[0].set_title("Validation mean 3D Dice", loc="left")
    axes[1].set_title("Validation DiceCE loss", loc="left")
    axes[0].set_ylim(.75, .91)
    axes[1].set_yscale("log")
    for ax in axes:
        ax.set_xlabel("Epoch")
    axes[0].legend(fontsize=8, loc="lower right")
    fig.text(.5, -.03, "Faint lines = individual runs; strong line and band = the best configuration's seed mean and range. Markers = selected epochs.",
             ha="center", color=INK_2, fontsize=9)
    fig.tight_layout()
    save(fig, out, "2_epoch_curves")


def fig_best_configs(runs, out):
    groups = configs(runs)
    champions = [best_config(groups, opt)[1] for opt in ("adam", "adamw", "sgd_nesterov")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), gridspec_kw={"width_ratios": [5, 4, 4]})
    x = np.arange(len(ORGANS) + 1)
    width = .23
    for i, rs in enumerate(champions):
        opt = optimiser(rs[0])
        dice = np.array([run.summary["dice3d"] for run in rs])
        hd95 = np.array([run.summary["hd95"] for run in rs])
        assd = np.array([run.summary["assd"] for run in rs])
        values = np.r_[dice.mean(0), mean_score(rs)]
        axes[0].bar(x + (i - 1) * width, values, width, color=COLOURS[opt], edgecolor=SURFACE)
        axes[1].bar(np.arange(len(ORGANS)) + (i - 1) * width, hd95.mean(0), width,
                    color=COLOURS[opt], edgecolor=SURFACE)
        axes[2].bar(np.arange(len(ORGANS)) + (i - 1) * width, assd.mean(0), width,
                    color=COLOURS[opt], edgecolor=SURFACE)
    axes[0].set_xticks(x, [*ORGANS, "Mean"])
    axes[0].set_ylim(.55, 1)
    axes[0].set_title("3D Dice at selected epoch", loc="left")
    for ax, title in zip(axes[1:], ("HD95 [mm]", "ASSD [mm]")):
        ax.set_xticks(range(len(ORGANS)), ORGANS, rotation=20, ha="right")
        ax.set_title(title + " (lower is better)", loc="left")
    handles = [Patch(color=COLOURS[optimiser(rs[0])], label=f"{LABELS[optimiser(rs[0])]}: {mean_score(rs):.3f}")
               for rs in champions]
    fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(.5, 1.06))
    fig.tight_layout()
    save(fig, out, "3_best_configs")


def tables(runs):
    groups = configs(runs)
    ordered = sorted(groups.values(), key=mean_score, reverse=True)
    lines = ["## Ranking\n", "| Rank | Optimiser | LR | WD | 3D Dice | Eso. | Heart | Trach. | Aorta | HD95 mm | ASSD mm | Best epoch |",
             "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for i, rs in enumerate(ordered, 1):
        r = rs[0]
        d = np.array([run.summary["dice3d"] for run in rs]).mean(0)
        dice = f"{mean_score(rs):.3f} ± {std_score(rs):.3f}" if len(rs) > 1 else f"{mean_score(rs):.3f} (s0)"
        lines.append(f"| {i} | {LABELS[optimiser(r)]} | {lr(r):g} | {wd(r):g} | {dice} | "
                     f"{d[0]:.3f} | {d[1]:.3f} | {d[2]:.3f} | {d[3]:.3f} | "
                     f"{np.mean([np.mean(run.summary['hd95']) for run in rs]):.1f} | "
                     f"{np.mean([np.mean(run.summary['assd']) for run in rs]):.2f} | "
                     f"{','.join(str(selected(run)) for run in rs)} |")
    lines += ["\n## Best setting per optimiser\n", "| Optimiser | LR | WD | 3D Dice | Best epoch |",
              "|---|---:|---:|---:|---:|"]
    for opt in ("adam", "adamw", "sgd_nesterov"):
        _, rs = best_config(groups, opt)
        r = rs[0]
        value = f"{mean_score(rs):.3f} ± {std_score(rs):.3f}" if len(rs) > 1 else f"{mean_score(rs):.3f} (s0)"
        lines.append(f"| {LABELS[opt]} | {lr(r):g} | {wd(r):g} | {value} | {','.join(str(selected(run)) for run in rs)} |")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", type=Path, default=Path("results/segthor/optimiser_sweep"))
    parser.add_argument("--out", type=Path, default=Path("experiments/optimiser_sweep"))
    args = parser.parse_args()
    runs = load_runs(args.sweep)
    if len(runs) < 18:
        raise ValueError(f"Expected at least 18 runs, found {len(runs)}")
    figures = args.out / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    fig_heatmaps(runs, figures)
    fig_curves(runs, figures)
    fig_best_configs(runs, figures)
    (args.out / "tables.md").write_text(tables(runs))
    print(f"Wrote figures and tables for {len(runs)} runs to {args.out}")


if __name__ == "__main__":
    main()
