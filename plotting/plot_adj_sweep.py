#!/usr/bin/env python3

"""Plots and tables for the 2.5D sweep (adjacent slices as extra input channels)."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from plot_arch_sweep import (INK, INK_2, ORGANS, SURFACE, by_config, config_mean, curve,
                             load_runs, nsw, results_table, run_scores, save)


# Grey for the 2D baseline, distinct hues for n = 1..3 (shades of one hue were hard to tell apart).
# Size colours avoid those hues so figure 4 doesn't read as n. Both sets pass the colour-blind check.
N_COLOUR = {0: "#9e9d99", 1: "#2a78d6", 2: "#eb6834", 3: "#1baf7a"}
SIZE_COLOUR = {32: "#4a3aa7", 64: "#e87ba4"}


def n_adj(runs) -> int:
    return runs[0].summary["args"]["adjacent_slices"]


def label(n: int) -> str:
    return "2D (n=0)" if n == 0 else f"2.5D n={n} ({2 * n + 1} slices)"


def patients(data: Path) -> list[str]:
    return sorted({p.name.rsplit("_", 1)[0].replace("Patient_", "P") for p in (data / "gt").glob("*.png")})


def per_patient(runs) -> np.ndarray:
    """(seeds, patients, organs) val 3D Dice at each run's best epoch."""
    return np.stack([r.logs["dice3d_val"][r.best_epoch, :, 1:] for r in runs])


def fig_epoch_curves(configs, out):
    panels = [("dice3d_mean", "Val 3D Dice (mean over organs)"),
              ("loss", "DiceCE loss (solid = val, dashed = train)")]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, (metric, title) in zip(axes, panels):
        for c, runs in configs.items():
            col = N_COLOUR[n_adj(runs)]
            series = [("loss_val", "-"), ("loss_tra", "--")] if metric == "loss" else [(metric, "-")]
            for m, ls in series:
                ys = np.stack([curve(r, m) for r in runs])
                x = np.arange(1, ys.shape[1] + 1)
                ax.fill_between(x, ys.min(0), ys.max(0), color=col, alpha=0.12, lw=0)
                ax.plot(x, ys.mean(0), color=col, lw=1.8, ls=ls)
        ax.set_title(title, loc="left")
        ax.set_xlabel("Epoch")
    axes[0].set_ylim(0.75, 0.88)
    axes[1].set_yscale("log")
    handles = [Line2D([], [], color=N_COLOUR[n_adj(rs)], lw=2, label=label(n_adj(rs))) for rs in configs.values()]
    fig.legend(handles=handles, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.07))
    fig.text(0.5, -0.03, "Line = mean over 3 seeds, band = min–max over seeds. Validation set: 10 patients.",
             ha="center", color=INK_2, fontsize=9)
    fig.tight_layout()
    save(fig, out, "1_epoch_curves")


def bars(ax, groups, vals, width=0.2):
    """vals[n] has shape (n_seeds, n_groups). Bar = seed mean, dot = one seed."""
    k = len(vals)
    for i, (n, v) in enumerate(vals.items()):
        x = np.arange(len(groups)) + (i - (k - 1) / 2) * width
        ax.bar(x, v.mean(0), width * 0.88, color=N_COLOUR[n], edgecolor=SURFACE, lw=1)
        for s in v:
            ax.plot(x, s, "o", ms=3, color=INK, alpha=0.55, zorder=4)
    ax.set_xticks(np.arange(len(groups)), groups)
    ax.grid(axis="x", visible=False)


def fig_per_organ(configs, out):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), gridspec_kw={"width_ratios": [6, 4, 4]})
    vals = {n_adj(rs): np.array([[*r.at_best("dice3d"), r.at_best("dice3d").mean(), nsw(r.at_best("dice3d"))]
                                 for r in rs]) for rs in configs.values()}
    bars(axes[0], [*ORGANS, "Mean", "NSW"], vals)
    axes[0].axvline(3.5, color=INK_2, lw=0.8, ls=":")
    axes[0].set_ylim(0.65, 0.95)
    axes[0].set_title("3D Dice at best epoch (higher is better)", loc="left")
    for ax, metric, title in [(axes[1], "hd95", "HD95 [mm] (lower is better)"),
                              (axes[2], "assd", "ASSD [mm] (lower is better)")]:
        bars(ax, ORGANS, {n_adj(rs): np.array([r.at_best(metric) for r in rs]) for rs in configs.values()})
        ax.set_title(title, loc="left")
        ax.tick_params(axis="x", labelsize=8.5)
    handles = [Patch(color=N_COLOUR[n_adj(rs)], label=label(n_adj(rs))) for rs in configs.values()]
    handles.append(Line2D([], [], ls="", marker="o", ms=4, color=INK, alpha=0.55, label="individual seed"))
    fig.legend(handles=handles, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.07))
    fig.tight_layout()
    save(fig, out, "2_best_epoch_per_organ")


def fig_per_patient(configs, base, pats, out):
    """3D Dice change vs 2D per patient (seed means), esophagus and mean over organs."""
    ref = per_patient(configs[base]).mean(0)
    others = {n_adj(rs): per_patient(rs).mean(0) - ref for c, rs in configs.items() if c != base}
    fig, axes = plt.subplots(1, 2, figsize=(14, 4), sharey=True)
    w = 0.26
    for ax, k, title in [(axes[0], 0, "Esophagus"), (axes[1], None, "Mean over organs")]:
        for i, (n, d) in enumerate(others.items()):
            v = d[:, 0] if k == 0 else d.mean(1)
            x = np.arange(len(pats)) + (i - 1) * w
            ax.bar(x, v, w * 0.88, color=N_COLOUR[n], edgecolor=SURFACE, label=label(n))
        ax.axhline(0, color=INK_2, lw=0.8)
        ax.set_xticks(range(len(pats)), pats)
        ax.grid(axis="x", visible=False)
        ax.set_title(f"{title}: 3D Dice, 2.5D − 2D", loc="left")
    axes[0].legend(fontsize=8.5, loc="lower left")
    fig.text(0.5, -0.03, "Seed means at the best epoch; bars above 0 = 2.5D is better on that patient.",
             ha="center", color=INK_2, fontsize=9)
    fig.tight_layout()
    save(fig, out, "3_per_patient")


def fig_size_interaction(configs, k64, out):
    """Metric vs n for k32 (n=0..3) and k64 (n=0, 1)."""
    panels = [(lambda r: r.at_best("dice3d").mean(), "3D Dice (mean)"),
              (lambda r: r.at_best("dice3d")[0], "Esophagus 3D Dice"),
              (lambda r: r.at_best("hd95").mean(), "HD95 [mm] (lower is better)"),
              (lambda r: r.at_best("assd").mean(), "ASSD [mm] (lower is better)")]
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.8))
    for ax, (f, title) in zip(axes, panels):
        for k, cfgs in [(32, configs), (64, k64)]:
            ns = [n_adj(rs) for rs in cfgs.values()]
            v = [np.array([f(r) for r in rs]) for rs in cfgs.values()]
            ax.errorbar(ns, [x.mean() for x in v], yerr=[x.std() for x in v], color=SIZE_COLOUR[k],
                        marker="o", ms=6, lw=1.8, capsize=3, mec=SURFACE)
            for n, x in zip(ns, v):
                ax.plot([n] * len(x), x, "o", ms=3, color=SIZE_COLOUR[k], alpha=0.4)
        ax.set_xticks(range(4))
        ax.set_xlabel("adjacent slices per side (n)")
        ax.set_title(title, loc="left")
    handles = [Line2D([], [], color=SIZE_COLOUR[k], marker="o", lw=2, label=l)
               for k, l in [(32, "UNet k32, LR 5e-4 (7.8M)"), (64, "UNet k64, LR 2.5e-4 (31.0M)")]]
    fig.legend(handles=handles, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.08))
    fig.text(0.5, -0.04, "Seed mean ± std at the best epoch; small dots = individual seeds. "
             "k64 was only run at n = 0 and 1.", ha="center", color=INK_2, fontsize=9)
    fig.tight_layout()
    save(fig, out, "4_size_interaction")


def patient_table(configs, base, pats) -> str:
    ref = per_patient(configs[base]).mean(0)
    lines = ["| Config | " + " | ".join(pats) + " | Patients better |", "|---|" + "---|" * (len(pats) + 1)]
    for c, rs in configs.items():
        if c == base:
            continue
        d = (per_patient(rs).mean(0) - ref).mean(1)
        lines.append(f"| `{c}` | " + " | ".join(f"{x:+.3f}" for x in d) + f" | {(d > 0).sum()}/{len(d)} |")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sweep", type=Path, default=Path("results/segthor/adj_sweep"))
    p.add_argument("--k64", type=Path, default=Path("results/segthor/adj_k64_sweep"))
    p.add_argument("--data", type=Path, default=Path("data/SEGTHOR/val"))
    p.add_argument("--out", type=Path, default=None, help="default: experiments/<sweep name>")
    args = p.parse_args()
    out = args.out or Path("experiments") / args.sweep.name
    figs = out / "figures"
    figs.mkdir(parents=True, exist_ok=True)

    runs = load_runs(args.sweep)
    configs = dict(sorted(by_config(runs).items(), key=lambda kv: n_adj(kv[1])))
    base = next(c for c, rs in configs.items() if n_adj(rs) == 0)
    pats = patients(args.data)
    print(f"{len(runs)} runs, {len(configs)} configs")

    fig_epoch_curves(configs, figs)
    fig_per_organ(configs, figs)
    fig_per_patient(configs, base, pats, figs)

    tables = ("## All configs\n\n" + results_table(configs, list(configs))
              + "\n\n## Per patient (mean-over-organs 3D Dice, 2.5D − 2D)\n\n" + patient_table(configs, base, pats))
    if args.k64.exists():
        k64 = dict(sorted(by_config(load_runs(args.k64)).items(), key=lambda kv: n_adj(kv[1])))
        fig_size_interaction(configs, k64, figs)
        base64 = next(c for c, rs in k64.items() if n_adj(rs) == 0)
        both = {**configs, **k64}
        tables += ("\n\n## k32 vs k64, n = 0 and 1\n\n"
                   + results_table(both, [c for c, rs in both.items() if n_adj(rs) <= 1])
                   + "\n\n## k64 per patient (mean-over-organs 3D Dice, 2.5D − 2D)\n\n"
                   + patient_table(k64, base64, pats))
    (out / "tables.md").write_text(tables + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
