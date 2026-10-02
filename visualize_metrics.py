#!/usr/bin/env python3

"""Generate metric plots and summaries from a training results directory."""

import argparse
import csv
import json
import re
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ORGAN_NAMES = ("Aorta", "Esophagus", "Heart", "Trachea")
COLOURS = plt.get_cmap("tab10")


def load_array(directory: Path, filename: str) -> np.ndarray | None:
    path = directory / filename
    return np.load(path) if path.exists() else None


def safe_mean(values: np.ndarray, axis: int | tuple[int, ...]) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(values, axis=axis)


def safe_std(values: np.ndarray, axis: int | tuple[int, ...]) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanstd(values, axis=axis)


def dice_aggregates(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return arithmetic and NSW foreground Dice per epoch."""
    class_means = safe_mean(values[:, :, 1:], axis=1)
    arithmetic = class_means.mean(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        nsw = np.exp(np.log(class_means).mean(axis=1))
    return arithmetic, nsw


def best_epoch(results_dir: Path) -> int | None:
    path = results_dir / "best_epoch.txt"
    if not path.exists():
        return None
    match = re.search(r"\bepoch\s+(\d+)", path.read_text(encoding="utf-8"))
    return int(match.group(1)) if match else None


def config(results_dir: Path) -> dict[str, object]:
    path = results_dir / "args.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_figure(figure: plt.Figure, path: Path) -> None:
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    figure.savefig(path.with_suffix(".svg"))
    plt.close(figure)


def mark_epoch(axis: plt.Axes, epoch: int | None) -> None:
    if epoch is not None:
        axis.axvline(epoch, color="0.4", linestyle=":", linewidth=1.3,
                     label=f"Selected epoch ({epoch})")


def plot_loss(output: Path, train: np.ndarray | None, validation: np.ndarray | None,
              selected: int | None) -> None:
    if train is None and validation is None:
        return
    figure, axis = plt.subplots(figsize=(9, 5))
    if train is not None:
        axis.plot(train.mean(axis=1), label="Training", color="tab:blue")
    if validation is not None:
        axis.plot(validation.mean(axis=1), label="Validation", color="tab:orange")
    mark_epoch(axis, selected)
    axis.set(title="Training and validation loss", xlabel="Epoch", ylabel="Loss")
    axis.grid(alpha=0.25)
    axis.legend()
    save_figure(figure, output)


def plot_per_class(output: Path, values: np.ndarray, title: str, ylabel: str,
                   selected: int | None, lower_is_better: bool = False) -> None:
    figure, axis = plt.subplots(figsize=(9, 5))
    class_means = safe_mean(values, axis=1)
    for class_index in range(1, values.shape[2]):
        label = ORGAN_NAMES[class_index - 1] if class_index <= len(ORGAN_NAMES) else f"Class {class_index}"
        axis.plot(class_means[:, class_index - 1], label=label,
                  color=COLOURS(class_index - 1))
    mark_epoch(axis, selected)
    axis.set(title=title, xlabel="Epoch", ylabel=ylabel)
    if lower_is_better:
        axis.text(0.01, 0.03, "Lower is better", transform=axis.transAxes,
                  color="0.4", fontsize=9)
    axis.grid(alpha=0.25)
    axis.legend()
    save_figure(figure, output)


def plot_foreground_dashboard(output: Path, metrics: dict[str, np.ndarray],
                              selected: int | None) -> None:
    specs = [
        ("dice2d", "2D Dice", "Dice", False, True),
        ("dice3d", "3D Dice", "Dice", False, True),
        ("hd95", "HD95", "Distance (mm)", True, False),
        ("assd", "ASSD", "Distance (mm)", True, False),
        ("hd", "HD", "Distance (mm)", True, False),
    ]
    available = [spec for spec in specs if spec[0] in metrics]
    if not available:
        return
    columns = 2
    rows = (len(available) + 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(12, 4 * rows), squeeze=False)
    for axis, (name, title, ylabel, lower, is_dice) in zip(axes.flat, available):
        values = metrics[name]
        if is_dice:
            mean, nsw = dice_aggregates(values)
            axis.plot(mean, label="Arithmetic mean", color="tab:blue")
            axis.plot(nsw, label="NSW", color="tab:orange")
        else:
            axis.plot(safe_mean(values[:, :, 1:], axis=(1, 2)),
                      label="Arithmetic mean", color="tab:blue")
        mark_epoch(axis, selected)
        axis.set(title=title, xlabel="Epoch", ylabel=ylabel)
        if lower:
            axis.text(0.01, 0.03, "Lower is better", transform=axis.transAxes,
                      color="0.4", fontsize=9)
        axis.grid(alpha=0.25)
        axis.legend()
    for axis in axes.flat[len(available):]:
        axis.set_visible(False)
    figure.suptitle("Foreground-average validation metrics")
    save_figure(figure, output)


def plot_per_organ_dashboard(output: Path, metrics: dict[str, np.ndarray],
                             selected: int | None) -> None:
    specs = [("dice3d", "3D Dice", "Dice", False),
             ("hd95", "HD95", "mm", True),
             ("assd", "ASSD", "mm", True),
             ("hd", "HD", "mm", True)]
    available = [spec for spec in specs if spec[0] in metrics]
    if not available:
        return
    class_count = max(metrics[name].shape[2] for name, *_ in available) - 1
    figure, axes = plt.subplots(class_count, len(available),
                                figsize=(4.5 * len(available), 2.8 * class_count),
                                squeeze=False)
    for row in range(class_count):
        for column, (name, title, ylabel, lower) in enumerate(available):
            axis = axes[row, column]
            values = safe_mean(metrics[name], axis=1)
            axis.plot(values[:, row + 1], color=COLOURS(row), linewidth=1.8)
            mark_epoch(axis, selected)
            if row == 0:
                axis.set_title(title)
            if column == 0:
                organ = ORGAN_NAMES[row] if row < len(ORGAN_NAMES) else f"Class {row + 1}"
                axis.set_ylabel(organ)
            axis.set_xlabel("Epoch")
            axis.grid(alpha=0.25)
            if lower:
                axis.text(0.02, 0.04, "lower is better",
                          transform=axis.transAxes, fontsize=8, color="0.4")
    figure.suptitle("Per-organ validation metrics (raw organ scores)")
    save_figure(figure, output)


def statistic(values: np.ndarray) -> tuple[float, float, float, float, float, int]:
    valid = values[np.isfinite(values)]
    if len(valid) == 0:
        return (float("nan"),) * 5 + (0,)
    return (float(valid.mean()), float(valid.std()), float(np.median(valid)),
            float(np.percentile(valid, 25)), float(np.percentile(valid, 75)), len(valid))


def write_csv(output: Path, metrics: dict[str, np.ndarray], selected: int | None,
              selection_metric: object, selection_aggregation: object) -> None:
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["epoch", "metric", "organ", "mean", "std", "median",
                         "q25", "q75", "valid_count", "selected_epoch",
                         "selection_metric", "selection_aggregation"])
        for name, values in metrics.items():
            for epoch in range(values.shape[0]):
                for class_index in range(1, values.shape[2]):
                    mean, std, median, q25, q75, count = statistic(values[epoch, :, class_index])
                    organ = ORGAN_NAMES[class_index - 1] if class_index <= len(ORGAN_NAMES) else f"Class {class_index}"
                    writer.writerow([epoch, name, organ, mean, std, median, q25,
                                     q75, count, epoch == selected,
                                     selection_metric, selection_aggregation])
                if name.startswith("dice"):
                    arithmetic, nsw = dice_aggregates(values)
                    writer.writerow([epoch, name, "foreground_mean", arithmetic[epoch],
                                     np.nan, np.nan, np.nan, np.nan, np.nan,
                                     epoch == selected, selection_metric,
                                     selection_aggregation])
                    writer.writerow([epoch, name, "foreground_nsw", nsw[epoch],
                                     np.nan, np.nan, np.nan, np.nan, np.nan,
                                     epoch == selected, selection_metric,
                                     selection_aggregation])


def plot_best_summary(output: Path, metrics: dict[str, np.ndarray], selected: int | None) -> None:
    if selected is None or "dice3d" not in metrics:
        return
    specs = [("dice3d", "3D Dice", "tab:blue", False),
             ("hd95", "HD95", "tab:orange", True),
             ("assd", "ASSD", "tab:green", True),
             ("hd", "HD", "tab:red", True)]
    available = [spec for spec in specs if spec[0] in metrics]
    figure, axes = plt.subplots(1, len(available), figsize=(4.5 * len(available), 4),
                                squeeze=False)
    for column, (name, title, colour, lower) in enumerate(available):
        axis = axes[0, column]
        means = safe_mean(metrics[name], axis=1)[selected, 1:]
        organs = [ORGAN_NAMES[i] if i < len(ORGAN_NAMES) else f"Class {i + 1}"
                  for i in range(len(means))]
        axis.bar(np.arange(len(means)), means, color=colour)
        axis.set_xticks(np.arange(len(means)), organs, rotation=35, ha="right")
        axis.set(title=title, ylabel="mm" if lower else "Dice")
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle(f"Selected epoch summary (epoch {selected})")
    save_figure(figure, output)


def plot_patient_distributions(output: Path, metrics: dict[str, np.ndarray],
                               selected: int | None) -> None:
    """Plot selected-epoch patient distributions grouped by organ."""
    if selected is None:
        return
    specs = [("dice3d", "3D Dice", "Dice", False),
             ("hd95", "HD95", "Distance (mm)", True),
             ("assd", "ASSD", "Distance (mm)", True),
             ("hd", "HD", "Distance (mm)", True)]
    available = [spec for spec in specs if spec[0] in metrics]
    if not available:
        return
    figure, axes = plt.subplots(1, len(available),
                                figsize=(4.5 * len(available), 5), squeeze=False)
    for column, (name, title, ylabel, lower) in enumerate(available):
        axis = axes[0, column]
        values = metrics[name][selected, :, 1:]
        distributions = [
            values[:, class_index][np.isfinite(values[:, class_index])]
            for class_index in range(values.shape[1])
        ]
        labels = [ORGAN_NAMES[index] if index < len(ORGAN_NAMES)
                  else f"Class {index + 1}" for index in range(values.shape[1])]
        axis.boxplot(distributions, tick_labels=labels, showmeans=True)
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.tick_params(axis="x", rotation=35)
        axis.grid(axis="y", alpha=0.25)
        if lower:
            axis.text(0.02, 0.04, "lower is better",
                      transform=axis.transAxes, fontsize=8, color="0.4")
    figure.suptitle(f"Patient-level distributions at epoch {selected}")
    save_figure(figure, output)


def plot_best_epoch_heatmap(output: Path, metrics: dict[str, np.ndarray],
                            selected: int | None) -> None:
    """Plot selected-epoch organ metrics with one independently scaled map per metric."""
    if selected is None:
        return
    specs = [("dice3d", "3D Dice", False),
             ("hd95", "HD95 (mm)", True),
             ("assd", "ASSD (mm)", True),
             ("hd", "HD (mm)", True)]
    available = [spec for spec in specs if spec[0] in metrics]
    if not available:
        return
    class_count = max(metrics[name].shape[2] for name, *_ in available) - 1
    figure, axes = plt.subplots(1, len(available),
                                figsize=(4.2 * len(available), 4.5), squeeze=False)
    for column, (name, title, lower) in enumerate(available):
        values = safe_mean(metrics[name], axis=1)[selected, 1:class_count + 1]
        axis = axes[0, column]
        image = axis.imshow(values[:, None], aspect="auto", cmap="viridis")
        axis.set_title(title)
        axis.set_xticks([0], [title])
        axis.set_yticks(np.arange(class_count), [
            ORGAN_NAMES[index] if index < len(ORGAN_NAMES) else f"Class {index + 1}"
            for index in range(class_count)
        ])
        axis.set_ylabel("Organ")
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        if lower:
            axis.text(0.02, -0.12, "lower is better", transform=axis.transAxes,
                      fontsize=8, color="0.4")
    figure.suptitle(f"Selected-epoch metric heatmaps (epoch {selected})")
    save_figure(figure, output)


def plot_metric_correlation(output: Path, metrics: dict[str, np.ndarray],
                            selected: int | None) -> None:
    """Plot selected-epoch Pearson correlations across valid patient-organ values."""
    if selected is None:
        return
    names = [name for name in ("dice3d", "hd95", "assd", "hd") if name in metrics]
    if len(names) < 2:
        return
    flattened = [metrics[name][selected, :, 1:].reshape(-1) for name in names]
    values = np.asarray(flattened, dtype=float)
    valid = np.all(np.isfinite(values), axis=0)
    if valid.sum() < 2:
        return
    values = values[:, valid]
    correlations = np.ones((len(names), len(names)), dtype=float)
    for row in range(len(names)):
        for column in range(row):
            if np.std(values[row]) == 0 or np.std(values[column]) == 0:
                correlation = 0.0
            else:
                correlation = float(np.corrcoef(values[row], values[column])[0, 1])
            correlations[row, column] = correlation
            correlations[column, row] = correlation

    figure, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(correlations, vmin=-1, vmax=1, cmap="coolwarm")
    labels = [name.upper() for name in names]
    axis.set_xticks(np.arange(len(labels)), labels, rotation=35, ha="right")
    axis.set_yticks(np.arange(len(labels)), labels)
    for row in range(len(names)):
        for column in range(len(names)):
            axis.text(column, row, f"{correlations[row, column]:.2f}",
                      ha="center", va="center", color="black")
    axis.set_title(f"Metric correlations at epoch {selected}")
    figure.colorbar(image, ax=axis, label="Pearson correlation")
    save_figure(figure, output)


def generate_report(results_dir: Path, output_dir: Path | None = None) -> None:
    output_dir = output_dir or results_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = best_epoch(results_dir)
    run_config = config(results_dir)
    metrics = {}
    for name, filename in {"dice2d": "dice_val.npy", "dice3d": "dice3d_val.npy",
                           "hd95": "hd95_val.npy", "assd": "assd_val.npy",
                           "hd": "hd_val.npy"}.items():
        values = load_array(results_dir, filename)
        if values is not None:
            metrics[name] = values
    plot_loss(output_dir / "loss.png", load_array(results_dir, "loss_tra.npy"),
              load_array(results_dir, "loss_val.npy"), selected)
    for name, values in metrics.items():
        plot_per_class(output_dir / f"{name}_by_organ.png", values, name.upper(),
                       "Dice" if name.startswith("dice") else "Distance (mm)",
                       selected, name in {"hd95", "assd", "hd"})
    plot_foreground_dashboard(output_dir / "foreground_dashboard.png", metrics, selected)
    plot_per_organ_dashboard(output_dir / "per_organ_dashboard.png", metrics, selected)
    plot_best_summary(output_dir / "best_epoch_summary.png", metrics, selected)
    plot_patient_distributions(output_dir / "patient_distributions.png", metrics, selected)
    plot_best_epoch_heatmap(output_dir / "best_epoch_heatmap.png", metrics, selected)
    plot_metric_correlation(output_dir / "metric_correlation.png", metrics, selected)
    write_csv(output_dir / "metrics_summary.csv", metrics, selected,
              run_config.get("selection_metric", "unknown"),
              run_config.get("selection_aggregation", "mean"))
    print(f"Saved metric visualisations to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot metrics from a training results directory.")
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    generate_report(args.results_dir, args.output_dir)


if __name__ == "__main__":
    main()
