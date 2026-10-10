#!/usr/bin/env python3

"""Plots and tables for the architecture sweep in results/segthor/arch_sweep."""

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from PIL import Image
from scipy.ndimage import find_objects
from scipy.stats import spearmanr


ORGANS = ("Esophagus", "Heart", "Trachea", "Aorta")
ARCHS = ("unet", "swin_unet", "enet")
ARCH_LABEL = {"unet": "UNet", "swin_unet": "SwinUNet", "enet": "ENet"}
ARCH_COLOUR = {"unet": "#2a78d6", "swin_unet": "#eb6834", "enet": "#1baf7a"}
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
COLLAPSE_EPOCH = 5  # best epoch earlier than this means the run collapsed

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": INK_2, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK_2, "ytick.color": INK_2, "axes.titlecolor": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "font.size": 10, "axes.titlesize": 11,
    "axes.titleweight": "bold", "legend.frameon": False,
})


@dataclass
class Run:
    name: str
    config: str
    arch: str
    seed: int
    hparams: dict
    summary: dict
    params: int
    logs: dict[str, np.ndarray] = field(repr=False)

    @property
    def best_epoch(self) -> int:
        return self.summary["best_epoch"]

    @property
    def collapsed(self) -> bool:
        return self.best_epoch < COLLAPSE_EPOCH

    def at_best(self, metric: str) -> np.ndarray:
        """Per-organ value at the best epoch, as in summary.json."""
        return np.asarray(self.summary[metric])


def load_runs(sweep: Path) -> list[Run]:
    runs = []
    for d in sorted(p for p in sweep.iterdir() if (p / "summary.json").exists()):
        summary = json.loads((d / "summary.json").read_text())
        args = summary["args"]
        m = re.fullmatch(r"(.+)_s(\d+)", d.name)
        assert m, d.name
        logs = {f.stem: np.load(f) for f in d.glob("*.npy")}
        weights = torch.load(d / "bestweights.pt", map_location="cpu", weights_only=True)
        params = sum(v.numel() for k, v in weights.items()
                     if v.is_floating_point() and "running" not in k)
        hparams = {"lr": args["lr"], **args["net_kwargs"]}
        runs.append(Run(d.name, m[1], args["net"], int(m[2]), hparams, summary, params, logs))
    return runs


# Same averaging as main.py: over patients first, then over organs.

def nsw(x: np.ndarray, axis: int = -1) -> np.ndarray:
    with np.errstate(divide="ignore"):
        return np.exp(np.log(x).mean(axis=axis))


def curve(run: Run, metric: str) -> np.ndarray:
    """Per-epoch validation/training curve for one run."""
    if metric == "dice3d_mean":
        return run.logs["dice3d_val"][:, :, 1:].mean(1).mean(1)
    if metric == "dice3d_nsw":
        return nsw(run.logs["dice3d_val"][:, :, 1:].mean(1))
    if metric == "dice2d_val":
        return run.logs["dice_val"][:, :, 1:].mean(1).mean(1)
    if metric == "dice2d_tra":
        return run.logs["dice_tra"][:, :, 1:].mean(1).mean(1)
    if metric == "loss_tra":
        return run.logs["loss_tra"].mean(1)
    if metric == "loss_val":
        return run.logs["loss_val"].mean(1)
    raise ValueError(metric)


def min_per_epoch(run: Run) -> float:
    """Minutes per epoch, training and validation included. Older summaries only
    have elapsed_s, which stops at the best epoch, so divide by best_epoch + 1."""
    s = run.summary
    if "total_elapsed_s" in s:
        return s["total_elapsed_s"] / 60 / s["epochs_run"]
    return s["elapsed_s"] / 60 / (run.best_epoch + 1)


def run_scores(run: Run) -> dict[str, float]:
    d3 = run.at_best("dice3d")
    return {
        "dice3d": d3.mean(), "nsw": float(nsw(d3)), "dice2d": run.at_best("dice2d").mean(),
        "hd95": run.at_best("hd95").mean(), "assd": run.at_best("assd").mean(),
        **{f"d3_{o}": v for o, v in zip(ORGANS, d3)},
        "best_epoch": run.best_epoch, "min_per_epoch": min_per_epoch(run),
        "params": run.params,
    }


def by_config(runs: list[Run]) -> dict[str, list[Run]]:
    out: dict[str, list[Run]] = {}
    for r in runs:
        out.setdefault(r.config, []).append(r)
    return out


def config_mean(runs: list[Run], key: str, include_collapsed: bool = True) -> float:
    vals = [run_scores(r)[key] for r in runs if include_collapsed or not r.collapsed]
    return float(np.mean(vals))


def best_configs(configs: dict[str, list[Run]]) -> dict[str, str]:
    """Best config per architecture by mean 3D Dice over seeds, skipping collapsed runs."""
    best = {}
    for arch in ARCHS:
        cands = {c: rs for c, rs in configs.items() if rs[0].arch == arch}
        best[arch] = max(cands, key=lambda c: config_mean(cands[c], "dice3d", False))
    return best


def model_label(cfg: str, runs: list[Run]) -> str:
    return f"{ARCH_LABEL[runs[0].arch]} ({cfg.split('_', 1)[1].replace('_', ', ')})" \
        if runs[0].arch != "enet" else "ENet (baseline)"


def save(fig, out: Path, name: str) -> None:
    fig.savefig(out / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_epoch_curves(configs, best, out):
    panels = [("dice3d_mean", "Val 3D Dice (mean over organs)"),
              ("dice3d_nsw", "Val 3D Dice (NSW over organs)"),
              ("dice2d_val", "Val 2D Dice (mean over organs)"),
              ("loss", "DiceCE loss (solid = val, dashed = train)")]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True)
    for ax, (metric, title) in zip(axes.flat, panels):
        for arch in ARCHS:
            runs = configs[best[arch]]
            c = ARCH_COLOUR[arch]
            series = [("loss_val", "-"), ("loss_tra", "--")] if metric == "loss" else [(metric, "-")]
            for m, ls in series:
                ys = np.stack([curve(r, m) for r in runs])
                x = np.arange(1, ys.shape[1] + 1)
                ax.fill_between(x, ys.min(0), ys.max(0), color=c, alpha=0.15, lw=0)
                ax.plot(x, ys.mean(0), color=c, lw=1.8, ls=ls)
                if metric == "dice3d_mean":
                    for r, y in zip(runs, ys):
                        ax.plot(r.best_epoch + 1, y[r.best_epoch], "o", ms=6, color=c,
                                mec=SURFACE, mew=1.2, zorder=5)
        ax.set_title(title, loc="left")
        if metric != "loss":
            lo = min(curve(r, metric)[5:].min() for a in ARCHS for r in configs[best[a]])
            ax.set_ylim(max(0, lo - 0.03), 1.0)
        else:
            ax.set_yscale("log")
    for ax in axes[1]:
        ax.set_xlabel("Epoch")
    handles = [Line2D([], [], color=ARCH_COLOUR[a], lw=2, label=model_label(best[a], configs[best[a]]))
               for a in ARCHS]
    handles.append(Line2D([], [], ls="", marker="o", color=INK_2, label="best epoch (per seed)"))
    fig.legend(handles=handles, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.02))
    fig.text(0.5, -0.01, "Line = mean over 3 seeds, band = min–max over seeds. Validation set: 5 patients.",
             ha="center", color=INK_2, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, out, "1_epoch_curves")


def grouped_bars(ax, groups, values_by_arch, archs, width=0.26):
    """values_by_arch[arch] has shape (n_seeds, n_groups). Bar = seed mean, dot = one seed."""
    n = len(archs)
    for i, arch in enumerate(archs):
        v = values_by_arch[arch]
        x = np.arange(len(groups)) + (i - (n - 1) / 2) * width
        ax.bar(x, v.mean(0), width * 0.88, color=ARCH_COLOUR[arch], edgecolor=SURFACE, lw=1)
        for s in v:
            ax.plot(x, s, "o", ms=3.5, color=INK, alpha=0.55, zorder=4)
    ax.set_xticks(np.arange(len(groups)), groups)
    ax.grid(axis="x", visible=False)


def fig_best_epoch_bars(configs, best, out):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), gridspec_kw={"width_ratios": [6, 4, 4]})
    vals = {a: np.array([[*r.at_best("dice3d"), r.at_best("dice3d").mean(), nsw(r.at_best("dice3d"))]
                         for r in configs[best[a]]]) for a in ARCHS}
    grouped_bars(axes[0], [*ORGANS, "Mean", "NSW"], vals, ARCHS)
    axes[0].axvline(3.5, color=INK_2, lw=0.8, ls=":")
    axes[0].set_ylim(0.2, 1.0)
    axes[0].set_title("3D Dice at best epoch (higher is better)", loc="left")
    for ax, metric, title in [(axes[1], "hd95", "HD95 [mm] (lower is better)"),
                              (axes[2], "assd", "ASSD [mm] (lower is better)")]:
        grouped_bars(ax, ORGANS, {a: np.array([r.at_best(metric) for r in configs[best[a]]])
                                  for a in ARCHS}, ARCHS)
        ax.set_title(title, loc="left")
        ax.tick_params(axis="x", labelsize=8.5)
    handles = [Patch(color=ARCH_COLOUR[a], label=model_label(best[a], configs[best[a]])) for a in ARCHS]
    handles.append(Line2D([], [], ls="", marker="o", ms=4, color=INK, alpha=0.55, label="individual seed"))
    fig.legend(handles=handles, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout()
    save(fig, out, "2_best_epoch_per_organ")


def fig_metric_correlation(runs, out):
    pairs = [("dice2d", "dice3d", "2D Dice", "3D Dice (mean)"),
             ("dice3d", "nsw", "3D Dice (mean)", "3D Dice (NSW)"),
             ("dice3d", "hd95", "3D Dice (mean)", "HD95 [mm]"),
             ("dice3d", "assd", "3D Dice (mean)", "ASSD [mm]")]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    scores = [(r, run_scores(r)) for r in runs]
    for ax, (kx, ky, lx, ly) in zip(axes, pairs):
        for arch in ARCHS:
            pts = [(s[kx], s[ky], r.collapsed) for r, s in scores if r.arch == arch]
            ok = np.array([(x, y) for x, y, c in pts if not c])
            ax.scatter(ok[:, 0], ok[:, 1], s=36, color=ARCH_COLOUR[arch], edgecolor=SURFACE, lw=1,
                       label=ARCH_LABEL[arch], zorder=3)
        if kx == "dice3d" and ky == "nsw":
            lim = [min(ax.get_xlim()[0], ax.get_ylim()[0]), 1]
            ax.plot(lim, lim, color=INK_2, lw=0.8, ls=":", zorder=1)
        ax.set_xlabel(lx)
        ax.set_ylabel(ly)
    handles = [Line2D([], [], ls="", marker="o", ms=7, color=ARCH_COLOUR[a], label=ARCH_LABEL[a]) for a in ARCHS]
    fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.07))
    n_bad = sum(r.collapsed for r in runs)
    fig.suptitle(f"Each point is one run at its best epoch ({len(runs) - n_bad} runs; "
                 f"{n_bad} collapsed run not shown, see table)", y=-0.02, fontsize=9,
                 color=INK_2, fontweight="normal")
    fig.tight_layout()
    save(fig, out, "3_metric_correlation")


def fig_heatmaps(configs, out):
    specs = [("unet", "kernels", "UNet: base kernels × learning rate"),
             ("swin_unet", "embed_dim", "SwinUNet: embed dim × learning rate")]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    cmap = plt.matplotlib.colors.LinearSegmentedColormap.from_list("seq", SEQ_BLUE)
    for ax, (arch, key, title) in zip(axes, specs):
        cfgs = {c: rs for c, rs in configs.items() if rs[0].arch == arch}
        rows = sorted({rs[0].hparams[key] for rs in cfgs.values()})
        cols = sorted({rs[0].hparams["lr"] for rs in cfgs.values()})
        mean = np.full((len(rows), len(cols)), np.nan)
        std = mean.copy()
        dagger = np.zeros_like(mean, bool)
        for rs in cfgs.values():
            i, j = rows.index(rs[0].hparams[key]), cols.index(rs[0].hparams["lr"])
            v = [run_scores(r)["dice3d"] for r in rs]
            mean[i, j], std[i, j] = np.mean(v), np.std(v)
            dagger[i, j] = any(r.collapsed for r in rs)
        ok = mean[~dagger]
        im = ax.imshow(mean, cmap=cmap, aspect="auto", origin="lower", vmin=ok.min(), vmax=ok.max())
        bi = np.unravel_index(np.nanargmax(np.where(dagger, -1, mean)), mean.shape)
        for (i, j), m in np.ndenumerate(mean):
            r, g, b, _ = cmap(np.clip(im.norm(m), 0, 1))
            light = 0.2126 * r + 0.7152 * g + 0.0722 * b < 0.45
            ax.text(j, i, f"{m:.3f}\n±{std[i, j]:.3f}{' †' if dagger[i, j] else ''}", ha="center",
                    va="center", fontsize=9.5, color="white" if light else INK,
                    fontweight="bold" if (i, j) == bi else "normal")
        ax.add_patch(plt.Rectangle((bi[1] - 0.5, bi[0] - 0.5), 1, 1, fill=False, ec=INK, lw=2))
        ax.set_xticks(range(len(cols)), [f"{c:g}" for c in cols])
        ax.set_yticks(range(len(rows)), rows)
        ax.set_xlabel("learning rate")
        ax.set_ylabel(key.replace("_", " "))
        ax.grid(False)
        ax.set_title(title, loc="left")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="3D Dice (mean)")
    fig.text(0.5, -0.04, "Mean ± std over 3 seeds; box = best cell; colour scales differ per panel. "
             "† includes a collapsed run (cell clipped to the lightest colour).", ha="center", color=INK_2, fontsize=9)
    fig.tight_layout()
    save(fig, out, "4_hparam_heatmaps")


def fig_per_patient(configs, best, sweep, out):
    patients = sorted({p.name.rsplit("_", 1)[0]
                       for p in (sweep / configs[best["unet"]][0].name / "best_epoch" / "val").glob("*.png")})
    fig, axes = plt.subplots(1, 5, figsize=(17, 3.8), sharey=True)
    for k, (ax, title) in enumerate(zip(axes, [*ORGANS, "Mean over organs"])):
        vals = {}
        for a in ARCHS:
            d = np.stack([r.logs["dice3d_val"][r.best_epoch, :, 1:] for r in configs[best[a]]])
            vals[a] = d.mean(2) if k == 4 else d[:, :, k]
        grouped_bars(ax, [p.replace("Patient_", "P") for p in patients], vals, ARCHS)
        ax.set_title(title, loc="left")
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("3D Dice at best epoch")
    handles = [Patch(color=ARCH_COLOUR[a], label=ARCH_LABEL[a]) for a in ARCHS]
    fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.07))
    fig.tight_layout()
    save(fig, out, "5_per_patient")
    return patients


def fig_cost(configs, out):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, key, label in [(axes[0], "min_per_epoch", "Wall-clock time per epoch [min]"),
                           (axes[1], "params", "Parameters")]:
        for c, rs in configs.items():
            a = rs[0].arch
            ax.scatter(config_mean(rs, key), config_mean(rs, "dice3d", False), s=42,
                       color=ARCH_COLOUR[a], edgecolor=SURFACE, lw=1, zorder=3)
        ax.set_xlabel(label)
    axes[0].set_xlim(left=0)
    axes[1].set_xscale("log")
    axes[0].set_ylabel("3D Dice (mean), seed mean")
    handles = [Line2D([], [], ls="", marker="o", ms=7, color=ARCH_COLOUR[a], label=ARCH_LABEL[a]) for a in ARCHS]
    fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.06))
    fig.text(0.5, -0.03, "One point per config (collapsed run excluded). Time per epoch includes data loading "
             "and validation; every run had one A100.", ha="center", color=INK_2, fontsize=9)
    fig.tight_layout()
    save(fig, out, "6_cost_vs_accuracy")


def fig_qualitative(configs, best, sweep, data, out):
    # Median seed, so we don't pick the nicest-looking run.
    picks = {a: sorted(configs[best[a]], key=lambda r: run_scores(r)["dice3d"])[1] for a in ARCHS}
    # Patient with the median esophagus Dice for UNet, sliced at three heights along the esophagus.
    u = picks["unet"]
    eso = u.logs["dice3d_val"][u.best_epoch, :, 1]
    patients = sorted({p.name.rsplit("_", 1)[0] for p in (data / "gt").glob("*.png")})
    patient = patients[int(np.argsort(eso)[len(eso) // 2])]
    stems = sorted(p.stem for p in (data / "gt").glob(f"{patient}_*.png"))
    gts = {s: np.array(Image.open(data / "gt" / f"{s}.png")) // 63 for s in stems}
    eso_stems = [s for s in stems if (gts[s] == 1).any()]
    rows = [eso_stems[int(q * (len(eso_stems) - 1))] for q in (0.2, 0.5, 0.8)]
    allfg = np.stack([gts[s] > 0 for s in rows]).any(0)
    sl = find_objects(allfg.astype(int))[0]
    pad = 12
    crop = tuple(slice(max(0, s.start - pad), s.stop + pad) for s in sl)

    cols = ["CT", "Ground truth", *[ARCH_LABEL[a] for a in ARCHS]]
    fig, axes = plt.subplots(len(rows), len(cols), figsize=(2.6 * len(cols), 2.6 * len(rows)))
    for i, s in enumerate(rows):
        img = np.array(Image.open(data / "img" / f"{s}.png"))[crop]
        gt = gts[s][crop]
        for j, ax in enumerate(axes[i]):
            ax.imshow(img, cmap="gray")
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            if j >= 1:
                ax.contour(gt, levels=[0.5, 1.5, 2.5, 3.5], colors="white", linewidths=1.0,
                           linestyles="--" if j >= 2 else "-")
            if j == 1:
                for k, name in enumerate(ORGANS, 1):
                    if (gt == k).any():
                        yy, xx = np.nonzero(gt == k)
                        ax.text(xx.mean(), yy.mean(), name[0], color="white", fontsize=9,
                                fontweight="bold", ha="center", va="center")
            if j >= 2:
                a = ARCHS[j - 2]
                pred = np.array(Image.open(sweep / picks[a].name / "best_epoch" / "val" / f"{s}.png"))[crop] // 63
                ax.contour(pred, levels=[0.5, 1.5, 2.5, 3.5], colors=ARCH_COLOUR[a], linewidths=1.6)
            if i == 0:
                ax.set_title(cols[j], fontsize=10)
        axes[i, 0].set_ylabel(s.replace("Patient_", "P").replace("_", " slice "), fontsize=9)
    fig.text(0.5, -0.02, f"{patient} (median esophagus Dice for UNet); slices at 20/50/80% of the "
             "esophagus extent. Coloured = prediction, dashed white = ground truth. "
             "E/H/T/A = esophagus/heart/trachea/aorta. Median seed of each model.",
             ha="center", color=INK_2, fontsize=9, wrap=True)
    fig.tight_layout()
    save(fig, out, "7_qualitative")


def fmt(rs, key, digits=3, include_collapsed=True):
    v = [run_scores(r)[key] for r in rs if include_collapsed or not r.collapsed]
    return f"{np.mean(v):.{digits}f} ± {np.std(v):.{digits}f}"


TABLE_COLS = [("dice3d", "3D Dice", 3, max), ("nsw", "3D NSW", 3, max),
              *[(f"d3_{o}", h, 3, max) for o, h in zip(ORGANS, ("Eso.", "Heart", "Trach.", "Aorta"))],
              ("dice2d", "2D Dice", 3, max), ("hd95", "HD95 mm", 1, min), ("assd", "ASSD mm", 2, min)]


def results_table(configs, cfg_order) -> str:
    best_val = {k: agg([config_mean(configs[c], k) for c in cfg_order]) for k, _, _, agg in TABLE_COLS}
    head = "| Config | Params | " + " | ".join(h for _, h, _, _ in TABLE_COLS) + " | Best ep. | Min/ep. |"
    lines = [head, "|" + "---|" * (len(TABLE_COLS) + 4)]
    for c in cfg_order:
        rs = configs[c]
        cells = []
        for k, _, d, _ in TABLE_COLS:
            s = fmt(rs, k, d)
            cells.append(f"**{s}**" if np.isclose(config_mean(rs, k), best_val[k]) else s)
        eps = "/".join(str(r.best_epoch + 1) for r in rs)
        dag = " †" if any(r.collapsed for r in rs) else ""
        lines.append(f"| `{c}`{dag} | {rs[0].params / 1e6:.1f}M | " + " | ".join(cells)
                     + f" | {eps} | {config_mean(rs, 'min_per_epoch'):.1f} |")
    return "\n".join(lines)


def rank_agreement(configs) -> str:
    keys = [("dice3d", "3D Dice", 1), ("nsw", "3D NSW", 1), ("dice2d", "2D Dice", 1),
            ("hd95", "HD95", -1), ("assd", "ASSD", -1)]
    lines = ["| Architecture | n configs | ρ(3D, NSW) | ρ(3D, 2D) | ρ(3D, −HD95) | ρ(3D, −ASSD) | "
             "Winner by 3D / NSW / 2D / HD95 / ASSD |", "|---|---|---|---|---|---|---|"]
    for arch in ("unet", "swin_unet"):
        cfgs = [c for c, rs in configs.items() if rs[0].arch == arch]
        m = {k: np.array([s * config_mean(configs[c], k, False) for c in cfgs]) for k, _, s in keys}
        rho = [spearmanr(m["dice3d"], m[k]).statistic for k, _, _ in keys[1:]]
        winners = " / ".join(cfgs[int(np.argmax(m[k]))].split("_", 1)[1] for k, _, _ in keys)
        lines.append(f"| {ARCH_LABEL[arch]} | {len(cfgs)} | " + " | ".join(f"{x:.2f}" for x in rho)
                     + f" | {winners} |")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sweep", type=Path, default=Path("results/segthor/arch_sweep"))
    p.add_argument("--data", type=Path, default=Path("data/SEGTHOR/val"))
    p.add_argument("--out", type=Path, default=None, help="default: experiments/<sweep name>")
    args = p.parse_args()
    out = args.out or Path("experiments") / args.sweep.name
    (out / "figures").mkdir(parents=True, exist_ok=True)
    figs = out / "figures"

    runs = load_runs(args.sweep)
    configs = by_config(runs)
    best = best_configs(configs)
    print(f"{len(runs)} runs, {len(configs)} configs; best: {best}")
    print("collapsed:", [r.name for r in runs if r.collapsed])

    fig_epoch_curves(configs, best, figs)
    fig_best_epoch_bars(configs, best, figs)
    fig_metric_correlation(runs, figs)
    fig_heatmaps(configs, figs)
    fig_per_patient(configs, best, args.sweep, figs)
    fig_cost(configs, figs)
    fig_qualitative(configs, best, args.sweep, args.data, figs)

    order = sorted(configs, key=lambda c: (ARCHS.index(configs[c][0].arch),
                                           -config_mean(configs[c], "dice3d")))
    headline = [best[a] for a in ARCHS]
    (out / "tables.md").write_text(
        "## Headline\n\n" + results_table(configs, headline)
        + "\n\n## All configs\n\n" + results_table(configs, order)
        + "\n\n## Rank agreement\n\n" + rank_agreement(configs) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
