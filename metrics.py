"""Utility helpers for logging training metrics to CSV and plotting them."""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _maybe_float(value):
    if value in (None, ""):
        return None
    if hasattr(value, "item"):
        try:
            value = value.item()
        except ValueError:
            pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _moving_average(values: Sequence[float], window: int) -> List[float]:
    if window <= 1:
        return list(values)
    smoothed: List[float] = []
    for idx in range(len(values)):
        start = max(0, idx - window + 1)
        window_vals = values[start : idx + 1]
        smoothed.append(sum(window_vals) / len(window_vals))
    return smoothed


@dataclass
class PlotConfig:
    metrics: Optional[Sequence[str]] = None
    x_axis: str = "step"
    smoothing: int = 1


class MetricsTracker:
    """Tracks metrics over time and stores them to disk."""

    def __init__(
        self,
        log_dir: str = "metrics",
        csv_filename: str = "training_metrics.csv",
        fieldnames: Optional[Sequence[str]] = None,
        plots_dir: Optional[str] = None,
        resume: bool = True,
    ) -> None:
        self.log_dir = log_dir
        self.csv_filename = csv_filename
        self.csv_path = os.path.join(log_dir, csv_filename)
        self.plots_dir = plots_dir or os.path.join(log_dir, "plots")
        self.fieldnames = list(fieldnames or ("epoch", "step", "loss", "lr"))

        _ensure_dir(self.log_dir)
        _ensure_dir(self.plots_dir)

        if not resume and os.path.exists(self.csv_path):
            os.remove(self.csv_path)

        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="") as fp:
                writer = csv.DictWriter(fp, fieldnames=self.fieldnames)
                writer.writeheader()

    def log(self, **metrics) -> None:
        """Append a metrics row to the CSV file."""

        unexpected = set(metrics) - set(self.fieldnames)
        if unexpected:
            raise ValueError(
                f"Unexpected metrics {unexpected}. Known fields: {self.fieldnames}"
            )

        row = {name: metrics.get(name, "") for name in self.fieldnames}
        with open(self.csv_path, "a", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=self.fieldnames)
            writer.writerow(row)

    def generate_plots(
        self,
        metrics: Optional[Sequence[str]] = None,
        x_axis: str = "step",
        smoothing: int = 1,
    ) -> None:
        """Create PNG plots for the selected metrics."""

        rows = self._load_rows()
        if not rows:
            return

        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:  # pragma: no cover - optional dependency
            print(f"[metrics] matplotlib requis pour tracer les courbes: {exc}")
            return

        metrics_to_plot = metrics or [n for n in self.fieldnames if n != x_axis]

        x_values: List[float] = []
        for row in rows:
            val = _maybe_float(row.get(x_axis))
            if val is not None:
                x_values.append(val)
            else:
                x_values.append(float(len(x_values)))

        for metric in metrics_to_plot:
            if metric == x_axis:
                continue
            y_vals = [
                _maybe_float(row.get(metric))
                for row in rows
            ]
            filtered_x: List[float] = []
            filtered_y: List[float] = []
            for x_val, y_val in zip(x_values, y_vals):
                if y_val is None:
                    continue
                filtered_x.append(x_val)
                filtered_y.append(y_val)

            if not filtered_x:
                continue

            smooth_y = _moving_average(filtered_y, smoothing)

            plt.figure(figsize=(8, 4))
            plt.plot(filtered_x, smooth_y, label=metric)
            plt.xlabel(x_axis)
            plt.ylabel(metric)
            plt.title(f"{metric} vs {x_axis}")
            plt.grid(True, linestyle="--", alpha=0.4)
            plt.legend()
            filename = os.path.join(self.plots_dir, f"{metric}_vs_{x_axis}.png")
            plt.tight_layout()
            plt.savefig(filename)
            plt.close()

    def _load_rows(self) -> List[dict]:
        if not os.path.exists(self.csv_path):
            return []
        with open(self.csv_path, "r", newline="") as fp:
            return list(csv.DictReader(fp))


def plot_from_csv(
    csv_path: str,
    output_dir: Optional[str] = None,
    metrics: Optional[Sequence[str]] = None,
    x_axis: str = "step",
    smoothing: int = 1,
) -> None:
    """Convenience helper to generate plots from an existing CSV file."""

    csv_dir = os.path.dirname(csv_path) or "."
    plots_dir = output_dir or os.path.join(csv_dir, "plots")
    tracker = MetricsTracker(
        log_dir=csv_dir,
        csv_filename=os.path.basename(csv_path),
        plots_dir=plots_dir,
        resume=True,
    )
    tracker.generate_plots(metrics=metrics, x_axis=x_axis, smoothing=smoothing)


__all__ = ["MetricsTracker", "PlotConfig", "plot_from_csv"]
