from __future__ import annotations

from pathlib import Path
from typing import Optional
from uuid import uuid4

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA_ART_STYLES: dict[str, str] = {
    "all": "Composite poster showing every artistic transformation.",
    "landscape": "Wave-like terrain generated from smoothed data.",
    "heatmap": "Textured pattern built from rolling numeric values.",
    "gradient": "Expressive bar field with dynamic colors.",
    "radial": "Circular bloom that turns a series into a rhythmic burst.",
}

COLORMAP_OPTIONS = ("magma", "viridis", "cividis", "plasma", "cubehelix")
DATASET_ROW_LIMIT = 5000


def _synthetic_dataset(rows: int = 320) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    t = np.linspace(0, 10 * np.pi, rows)

    return pd.DataFrame(
        {
            "date": pd.date_range(end=pd.Timestamp.utcnow().normalize(), periods=rows, freq="D"),
            "temperature": 17 + 8 * np.sin(t * 0.8) + rng.normal(0, 1.3, rows),
            "humidity": 58 + 16 * np.cos(t * 0.45) + rng.normal(0, 1.8, rows),
            "wind": 10 + 4 * np.sin(t * 1.3) + rng.normal(0, 0.8, rows),
            "traffic": 150 + 38 * np.cos(t * 0.62) + rng.normal(0, 5.4, rows),
            "energy": 230 + 30 * np.sin(t * 0.33 + 0.4) + rng.normal(0, 4.5, rows),
        }
    )


def load_and_preprocess_dataset(dataset_path: Optional[Path] = None) -> tuple[pd.DataFrame, dict]:
    source = "synthetic"
    original_rows = 0

    if dataset_path and dataset_path.exists():
        try:
            frame = pd.read_csv(dataset_path, nrows=DATASET_ROW_LIMIT)
            source = dataset_path.name
            if len(frame) == DATASET_ROW_LIMIT:
                source = f"{dataset_path.name} (first {DATASET_ROW_LIMIT} rows)"
        except Exception:
            frame = _synthetic_dataset()
            source = "synthetic (fallback)"
    else:
        frame = _synthetic_dataset()

    original_rows = int(len(frame))

    if "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")

    numeric = frame.select_dtypes(include=["number"]).copy()
    if numeric.empty:
        numeric = _synthetic_dataset().select_dtypes(include=["number"])
        source = f"{source} - numeric fallback"

    numeric.replace([np.inf, -np.inf], np.nan, inplace=True)
    numeric.interpolate(method="linear", axis=0, limit_direction="both", inplace=True)
    numeric.fillna(numeric.mean(numeric_only=True), inplace=True)
    numeric.fillna(0, inplace=True)

    if len(numeric) > 480:
        indices = np.linspace(0, len(numeric) - 1, 480, dtype=int)
        numeric = numeric.iloc[indices].reset_index(drop=True)

    metadata = {
        "source": source,
        "rows": int(len(numeric)),
        "original_rows": original_rows,
        "columns": list(numeric.columns),
    }

    return numeric, metadata


def _resolve_focus_series(frame: pd.DataFrame, focus_column: str) -> tuple[str, pd.Series]:
    if focus_column != "auto" and focus_column in frame.columns:
        return focus_column, frame[focus_column]
    fallback_name = frame.columns[0]
    return fallback_name, frame[fallback_name]


def _smoothed(series: pd.Series, window: int) -> np.ndarray:
    safe_window = max(1, min(30, int(window)))
    return series.rolling(window=safe_window, min_periods=1).mean().to_numpy()


def _plot_abstract_landscape(axis, series: pd.Series, cmap_name: str, smoothing_window: int) -> None:
    values = _smoothed(series, smoothing_window)
    x = np.arange(len(values))
    normalized = (values - values.min()) / (np.ptp(values) + 1e-9)
    colors = plt.get_cmap(cmap_name)(np.linspace(0.2, 0.95, 4))

    axis.plot(x, normalized, color=colors[-1], linewidth=2.3, alpha=0.95)
    for index, color in enumerate(colors):
        scale = 1 - index * 0.17
        axis.fill_between(x, 0, normalized * scale, color=color, alpha=0.16 + index * 0.06)

    axis.set_title("Abstract Landscape", color="#f7f1e3", fontsize=13, fontweight="bold")
    axis.set_facecolor("#132028")
    axis.set_xticks([])
    axis.set_yticks([])


def _plot_heatmap_pattern(axis, frame: pd.DataFrame, cmap_name: str, smoothing_window: int) -> None:
    subset = frame.iloc[:, : min(6, frame.shape[1])]
    rolling = subset.rolling(window=max(2, smoothing_window), min_periods=1).mean()
    matrix = rolling.to_numpy().T
    matrix = (matrix - matrix.min()) / (np.ptp(matrix) + 1e-9)

    axis.imshow(matrix, aspect="auto", cmap=cmap_name, interpolation="nearest")
    axis.set_title("Heatmap Pattern", color="#f7f1e3", fontsize=13, fontweight="bold")
    axis.set_facecolor("#132028")
    axis.set_xticks([])
    axis.set_yticks([])


def _plot_gradient_bars(axis, series: pd.Series, cmap_name: str, smoothing_window: int) -> None:
    sampled = _smoothed(series.tail(36), smoothing_window)
    normalized = (sampled - sampled.min()) / (np.ptp(sampled) + 1e-9)
    colors = plt.get_cmap(cmap_name)(normalized)
    x = np.arange(len(sampled))

    axis.bar(x, sampled, color=colors, width=0.82)
    axis.plot(x, sampled, color="#fff7ee", linewidth=1.2, alpha=0.45)
    axis.set_title("Gradient Bars", color="#f7f1e3", fontsize=13, fontweight="bold")
    axis.set_facecolor("#132028")
    axis.set_xticks([])
    axis.set_yticks([])


def _plot_radial_bloom(axis, series: pd.Series, cmap_name: str, smoothing_window: int) -> None:
    sampled = _smoothed(series.tail(48), smoothing_window)
    normalized = (sampled - sampled.min()) / (np.ptp(sampled) + 1e-9)
    theta = np.linspace(0, 2 * np.pi, len(normalized), endpoint=False)
    colors = plt.get_cmap(cmap_name)(np.linspace(0.15, 0.95, len(normalized)))

    axis.set_facecolor("#132028")
    axis.scatter([0], [0], s=140, color="#f7f1e3", alpha=0.9)
    for angle, radius, color in zip(theta, normalized, colors):
        x = np.cos(angle) * (0.18 + radius * 0.82)
        y = np.sin(angle) * (0.18 + radius * 0.82)
        axis.plot([0, x], [0, y], color=color, linewidth=2.0, alpha=0.7)
        axis.scatter([x], [y], s=30 + radius * 80, color=color, alpha=0.85, linewidths=0)

    axis.set_title("Radial Bloom", color="#f7f1e3", fontsize=13, fontweight="bold")
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_xlim(-1.1, 1.1)
    axis.set_ylim(-1.1, 1.1)


def create_data_art(
    output_dir: Path,
    dataset_path: Optional[Path] = None,
    frame: Optional[pd.DataFrame] = None,
    metadata: Optional[dict] = None,
    style: str = "all",
    focus_column: str = "auto",
    colormap: str = "magma",
    smoothing_window: int = 8,
) -> tuple[str, dict]:
    """Generate a data-driven artistic visualization and return filename + metadata."""

    output_dir.mkdir(parents=True, exist_ok=True)
    if frame is None:
        frame, metadata = load_and_preprocess_dataset(dataset_path)
    else:
        frame = frame.copy()
        metadata = dict(metadata or {})
        metadata.setdefault("source", "provided")
        metadata.setdefault("rows", int(len(frame)))
        metadata.setdefault("original_rows", int(len(frame)))
        metadata.setdefault("columns", list(frame.columns))

    safe_colormap = colormap if colormap in COLORMAP_OPTIONS else COLORMAP_OPTIONS[0]
    focus_name, focus_series = _resolve_focus_series(frame, focus_column)
    secondary_name = frame.columns[min(1, len(frame.columns) - 1)]
    secondary_series = frame[secondary_name]

    normalized_style = style.lower()
    if normalized_style not in DATA_ART_STYLES:
        normalized_style = "all"

    if normalized_style == "landscape":
        figure, axis = plt.subplots(figsize=(11.5, 5.6), dpi=120)
        figure.patch.set_facecolor("#0c1419")
        _plot_abstract_landscape(axis, focus_series, safe_colormap, smoothing_window)
    elif normalized_style == "heatmap":
        figure, axis = plt.subplots(figsize=(11.5, 5.6), dpi=120)
        figure.patch.set_facecolor("#0c1419")
        _plot_heatmap_pattern(axis, frame, safe_colormap, smoothing_window)
    elif normalized_style == "gradient":
        figure, axis = plt.subplots(figsize=(11.5, 5.6), dpi=120)
        figure.patch.set_facecolor("#0c1419")
        _plot_gradient_bars(axis, focus_series, safe_colormap, smoothing_window)
    elif normalized_style == "radial":
        figure, axis = plt.subplots(figsize=(8, 8), dpi=120)
        figure.patch.set_facecolor("#0c1419")
        _plot_radial_bloom(axis, focus_series, safe_colormap, smoothing_window)
    else:
        figure, axes = plt.subplots(2, 2, figsize=(15, 10), dpi=120)
        figure.patch.set_facecolor("#0c1419")
        _plot_abstract_landscape(axes[0, 0], focus_series, safe_colormap, smoothing_window)
        _plot_heatmap_pattern(axes[0, 1], frame, safe_colormap, smoothing_window)
        _plot_gradient_bars(axes[1, 0], secondary_series, safe_colormap, smoothing_window)
        _plot_radial_bloom(axes[1, 1], focus_series, safe_colormap, smoothing_window)
        figure.subplots_adjust(wspace=0.18, hspace=0.22)

    filename = f"data_art_{normalized_style}_{uuid4().hex[:12]}.png"
    output_path = output_dir / filename

    metadata.update(
        {
            "focus_column": focus_name,
            "secondary_column": secondary_name,
            "style": normalized_style,
            "colormap": safe_colormap,
            "smoothing_window": max(1, min(30, int(smoothing_window))),
        }
    )

    figure.savefig(output_path, dpi=120, facecolor=figure.get_facecolor(), bbox_inches="tight", pad_inches=0.14)
    plt.close(figure)

    return filename, metadata
