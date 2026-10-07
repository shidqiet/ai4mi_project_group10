#!/usr/bin/env python3

"""Plots and tables for the ENet width x LR sweep.

The UNet runs and the ENet baseline come from arch_sweep for comparison. Loading,
metrics and styling are common with plot_arch_sweep.py.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from plot_arch_sweep import (ARCH_COLOUR, INK, INK_2, ORGANS, SEQ_BLUE, SURFACE, by_config,
                        config_mean, curve, load_runs, results_table, run_scores, save)


ALIVE = 0.1  # organ counts as learned once its val 3D Dice goes above this
# Schedule plot: constant LR, poly 5e-4, poly 1e-3. Violet/aqua/orange pass the
# colour-blind check; blue is left out because it means UNet elsewhere in the report.
SCHED_COLOUR = ("#4a3aa7", "#1baf7a", "#eb6834")
# (UNet kernels, ENet kernels) pairs with similar parameter counts
SCALES = [(16, 20), (32, 40), (64, 80)]


def dead_organs(run) -> list[str]:
    """Organs the run never learned (val 3D Dice <= ALIVE at every epoch)."""
    d3 = run.logs["dice3d_val"][:, :, 1:].mean(1)
    return [o for k, o in enumerate(ORGANS) if not (d3[:, k] > ALIVE).any()]


def onset_epoch(run, organ: str) -> int | None:
    """First 1-indexed epoch the organ's val 3D Dice exceeds ALIVE."""
    d3 = run.logs["dice3d_val"][:, :, 1 + ORGANS.index(organ)].mean(1)
    hit = np.flatnonzero(d3 > ALIVE)
    return int(hit[0]) + 1 if hit.size else None


def poly_lr(lr0: float, epochs: int = 50, power: float = 0.9) -> np.ndarray:
    """LR used in each epoch, as stepped in main.py."""
    return lr0 * (1 - np.arange(epochs) / epochs) ** power


def best_at(configs, arch: str, kernels: int) -> str:
    cands = [c for c, rs in configs.items()
             if rs[0].arch == arch and rs[0].hparams.get("kernels") == kernels and "baseline" not in c]
    return max(cands, key=lambda c: config_mean(configs[c], "dice3d"))


def fig_heatmap(configs, out):
    cfgs = {c: rs for c, rs in configs.items() if rs[0].arch == "enet" and "baseline" not in c}
    rows = sorted({rs[0].hparams["kernels"] for rs in cfgs.values()})
    cols = sorted({rs[0].hparams["lr"] for rs in cfgs.values()})
    mean = np.full((len(rows), len(cols)), np.nan)
    std = mean.copy()
    dead = np.zeros_like(mean, bool)
    for rs in cfgs.values():
        i, j = rows.index(rs[0].hparams["kernels"]), cols.index(rs[0].hparams["lr"])
        v = [run_scores(r)["dice3d"] for r in rs]
        mean[i, j], std[i, j] = np.mean(v), np.std(v)
        dead[i, j] = any(dead_organs(r) for r in rs)
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    cmap = plt.matplotlib.colors.LinearSegmentedColormap.from_list("seq", SEQ_BLUE)
    ok = mean[~dead]
    im = ax.imshow(mean, cmap=cmap, aspect="auto", origin="lower", vmin=ok.min(), vmax=ok.max())
    bi = np.unravel_index(np.nanargmax(np.where(dead, -1, mean)), mean.shape)
    for (i, j), m in np.ndenumerate(mean):
        r, g, b, _ = cmap(np.clip(im.norm(m), 0, 1))
        light = 0.2126 * r + 0.7152 * g + 0.0722 * b < 0.45
        ax.text(j, i, f"{m:.3f}\n±{std[i, j]:.3f}{' ‡' if dead[i, j] else ''}", ha="center",
                va="center", fontsize=9.5, color="white" if light else INK,
                fontweight="bold" if (i, j) == bi else "normal")
    ax.add_patch(plt.Rectangle((bi[1] - 0.5, bi[0] - 0.5), 1, 1, fill=False, ec=INK, lw=2))
    params = {k: next(rs[0].params for rs in cfgs.values() if rs[0].hparams["kernels"] == k) for k in rows}
    ax.set_xticks(range(len(cols)), [f"{c:g}" for c in cols])
    ax.set_yticks(range(len(rows)), [f"{k}  ({params[k] / 1e6:.2g}M)" for k in rows])
    ax.set_xlabel("learning rate (peak, polynomial decay)")
    ax.set_ylabel("base kernels (params)")
    ax.grid(False)
    ax.set_title("ENet: base kernels × learning rate", loc="left")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="3D Dice (mean)")
    fig.text(0.5, -0.06, "Mean ± std over 3 seeds; box = best cell. ‡ one seed never learned the "
             "trachea (cell clipped to the lightest colour).\nENet baseline (k8, constant LR 5e-4): "
             f"{config_mean(configs['enet_baseline'], 'dice3d'):.3f}.", ha="center", color=INK_2, fontsize=9)
    fig.tight_layout()
    save(fig, out, "1_enet_heatmap")


def fig_matched_scale(configs, out):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), gridspec_kw={"width_ratios": [4, 4, 5]})
    pairs = [(best_at(configs, "unet", u), best_at(configs, "enet", e)) for u, e in SCALES]
    for ax, key, title in [(axes[0], "dice3d", "3D Dice (mean), best LR per width"),
                           (axes[1], "hd95", "HD95 [mm], same configs (lower is better)")]:
        for arch, idx in [("unet", 0), ("enet", 1)]:
            cs = [p[idx] for p in pairs]
            if arch == "enet":
                cs = [best_at(configs, "enet", 8)] + cs
            x = [configs[c][0].params for c in cs]
            v = np.array([[run_scores(r)[key] for r in configs[c]] for c in cs])
            ax.errorbar(x, v.mean(1), yerr=v.std(1), color=ARCH_COLOUR[arch], marker="o", ms=6,
                        lw=1.8, capsize=3, mec=SURFACE)
            for xi, c in zip(x, cs):
                ax.annotate(c.split("_")[1], (xi, v.mean(1)[cs.index(c)]), textcoords="offset points",
                            xytext=(5, -12 if arch == "enet" and key == "dice3d" else 6),
                            fontsize=8, color=INK_2)
        ax.set_xscale("log")
        ax.set_xlabel("Parameters")
        ax.set_title(title, loc="left")
    # ENet minus UNet per organ, at each scale
    w = 0.26
    for i, (u, e) in enumerate(pairs):
        gap = np.array([[run_scores(r)[f"d3_{o}"] for o in ORGANS] for r in configs[e]]).mean(0) \
            - np.array([[run_scores(r)[f"d3_{o}"] for o in ORGANS] for r in configs[u]]).mean(0)
        x = np.arange(len(ORGANS)) + (i - 1) * w
        axes[2].bar(x, gap, w * 0.88, color=ARCH_COLOUR["enet"], alpha=0.45 + 0.27 * i,
                    edgecolor=SURFACE, label=f"{e.split('_')[1]} vs {u.split('_')[1]}")
    axes[2].axhline(0, color=INK_2, lw=0.8)
    axes[2].set_xticks(range(len(ORGANS)), ORGANS)
    axes[2].grid(axis="x", visible=False)
    axes[2].set_title("3D Dice gap, ENet − UNet, per organ", loc="left")
    axes[2].legend(fontsize=8.5, loc="lower right")
    handles = [Line2D([], [], color=ARCH_COLOUR[a], marker="o", lw=2, label=l)
               for a, l in [("unet", "UNet (k16/32/64)"), ("enet", "ENet (k8/20/40/80)")]]
    fig.legend(handles=handles, loc="upper center", ncol=2, bbox_to_anchor=(0.36, 1.06))
    fig.text(0.5, -0.03, "Both arms: Adam + polynomial schedule, LR grid 2.5e-4/5e-4/1e-3, best LR per "
             "width by mean 3D Dice. Error bars = std over 3 seeds.", ha="center", color=INK_2, fontsize=9)
    fig.tight_layout()
    save(fig, out, "2_matched_scale")


def fig_schedule(configs, out):
    # (config, legend label, colour, LR per epoch)
    arms = [("enet_baseline", "constant LR 5e-4 (baseline)", SCHED_COLOUR[0], np.full(50, 5e-4)),
            ("enet_k8_lr5e-4", "poly 5e-4", SCHED_COLOUR[1], poly_lr(5e-4)),
            ("enet_k8_lr1e-3", "poly 1e-3", SCHED_COLOUR[2], poly_lr(1e-3))]
    runs = {c: {r.seed: r for r in configs[c]} for c, *_ in arms}
    x = np.arange(1, 51)
    t = 1 + ORGANS.index("Trachea")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    for cfg, _, c, lr in arms:
        # Left panel compares the schedule alone, so only the two runs with peak LR 5e-4.
        # Seed 1 is left out there because poly 5e-4 never learns the trachea on it (middle panel).
        if cfg != "enet_k8_lr1e-3":
            for s, ls in [(0, "-"), (2, "--")]:
                axes[0].plot(x, curve(runs[cfg][s], "dice3d_mean"), color=c, ls=ls, lw=1.6)
        run = runs[cfg][1]
        axes[1].plot(x, run.logs["dice3d_val"][:, :, t].mean(1), color=c, lw=1.8)
        axes[2].plot(x, lr, color=c, lw=1.8)
        on = onset_epoch(run, "Trachea")
        if on:
            axes[1].axvline(on, color=c, lw=0.8, ls=":")
            axes[2].plot(on, lr[on - 1], "o", color=c, ms=8, mec=SURFACE, mew=1.5, zorder=5)

    axes[0].set_ylim(0.6, 0.85)
    axes[0].set_title("Constant vs poly, same peak LR 5e-4\nMean 3D Dice, seeds 0 (solid) and 2 (dashed)",
                      loc="left")
    axes[1].set_ylim(-0.03, 1)
    axes[1].set_title("Trachea 3D Dice, seed 1", loc="left")
    axes[2].set_title("Learning rate (dot = trachea learned, seed 1)", loc="left")
    axes[2].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    for ax in axes:
        ax.set_xlabel("Epoch")
    handles = [Line2D([], [], color=c, lw=2, label=f"ENet k8, {lab}") for _, lab, c, _ in arms]
    fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.06))
    fig.text(0.5, -0.03, "Same seed = same initialisation and data order; the runs differ only in the "
             "LR schedule (and peak LR for poly 1e-3).", ha="center", color=INK_2, fontsize=9)
    fig.tight_layout()
    save(fig, out, "3_schedule")


def matched_table(configs) -> str:
    lines = ["| Scale | UNet | Params | 3D Dice | HD95 mm | ENet | Params | 3D Dice | HD95 mm | Δ Dice |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for u, e in SCALES:
        cu, ce = best_at(configs, "unet", u), best_at(configs, "enet", e)
        mu, me = config_mean(configs[cu], "dice3d"), config_mean(configs[ce], "dice3d")

        def cell(c):
            rs = configs[c]
            d = [run_scores(r)["dice3d"] for r in rs]
            return (f"`{c}` | {rs[0].params / 1e6:.1f}M | {np.mean(d):.3f} ± {np.std(d):.3f} | "
                    f"{config_mean(rs, 'hd95'):.1f}")
        lines.append(f"| ~{configs[ce][0].params / 1e6:.0f}M | {cell(cu)} | {cell(ce)} | {me - mu:+.3f} |")
    return "\n".join(lines)


def schedule_table(configs) -> str:
    lines = ["| Config | Schedule | Seed 0 | Seed 1 | Seed 2 | Trachea learned (epoch, per seed) | "
             "Late noise | Best − final |", "|---|---|---|---|---|---|---|---|"]
    for c, sched in [("enet_baseline", "constant 5e-4"), ("enet_k8_lr2.5e-4", "poly 2.5e-4"),
                     ("enet_k8_lr5e-4", "poly 5e-4"), ("enet_k8_lr1e-3", "poly 1e-3")]:
        rs = sorted(configs[c], key=lambda r: r.seed)
        d = " | ".join(f"{run_scores(r)['dice3d']:.3f}" for r in rs)
        on = "/".join(str(onset_epoch(r, "Trachea") or "never") for r in rs)
        late = np.mean([np.abs(np.diff(curve(r, "dice3d_mean")[35:])).mean() for r in rs])
        gap = np.mean([curve(r, "dice3d_mean").max() - curve(r, "dice3d_mean")[-1] for r in rs])
        lines.append(f"| `{c}` | {sched} | {d} | {on} | {late:.4f} | {gap:.3f} |")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sweep", type=Path, default=Path("results/segthor/enet_sweep"))
    p.add_argument("--ref", type=Path, default=Path("results/segthor/arch_sweep"))
    p.add_argument("--out", type=Path, default=None, help="default: experiments/<sweep name>")
    args = p.parse_args()
    out = args.out or Path("experiments") / args.sweep.name
    figs = out / "figures"
    figs.mkdir(parents=True, exist_ok=True)

    runs = load_runs(args.sweep) + [r for r in load_runs(args.ref) if r.arch in ("unet", "enet")]
    configs = by_config(runs)
    print(f"{len(runs)} runs, {len(configs)} configs")
    print("dead organs:", {r.name: dead_organs(r) for r in runs if dead_organs(r)})

    fig_heatmap(configs, figs)
    fig_matched_scale(configs, figs)
    fig_schedule(configs, figs)

    enet = sorted((c for c, rs in configs.items() if rs[0].arch == "enet"),
                  key=lambda c: -config_mean(configs[c], "dice3d"))
    (out / "tables.md").write_text(
        "## Matched scale\n\n" + matched_table(configs)
        + "\n\n## Schedule (k8)\n\n" + schedule_table(configs)
        + "\n\n## All ENet configs\n\n" + results_table(configs, enet) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
