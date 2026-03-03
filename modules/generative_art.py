from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors
from matplotlib import patches
from matplotlib.collections import LineCollection

PALETTES: dict[str, list[str]] = {
    "sunset": ["#ff8c42", "#f4d35e", "#ee964b", "#f95738", "#0d3b66"],
    "ocean": ["#0b3954", "#087e8b", "#bfd7ea", "#ff5a5f", "#c81d25"],
    "forest": ["#1f4d3f", "#3d7a5c", "#80a96d", "#d1b97f", "#f3e8c8"],
    "mono": ["#101419", "#38404d", "#7a828d", "#d6d9de", "#f5f6f8"],
    "festival": ["#153243", "#e76f51", "#f4a261", "#2a9d8f", "#e9c46a"],
    "electric": ["#0d1b2a", "#1b263b", "#00b4d8", "#ffb703", "#fb5607"],
}

SERIES_INFO: dict[str, str] = {
    "constellation": "A star-field of connected points, nebula clouds, and luminous trails.",
    "mosaic": "A tile-based composition built from rectangles, triangles, and striped cells.",
    "kinetic": "An object-oriented scene of moving shapes rendered with layered motion trails.",
}

BACKGROUND_STYLES: dict[str, str] = {
    "aurora": "Soft gradient atmosphere with luminous blooms.",
    "paper": "Warm editorial background with subtle grain.",
    "night": "Dark backdrop with spectral texture and higher contrast.",
}

ALLOWED_SHAPES = {"circle", "square", "triangle"}


@dataclass
class Shape:
    x: float
    y: float
    size: float
    color: str
    velocity_x: float = 0.0
    velocity_y: float = 0.0

    def draw(self, axis, alpha: float = 0.75, edgecolor: str = "#f8f4ec") -> None:
        axis.add_patch(self.as_patch(alpha=alpha, edgecolor=edgecolor))

    def move(self, width: float, height: float) -> None:
        self.x += self.velocity_x
        self.y += self.velocity_y

        if self.x < 0 or self.x > width:
            self.velocity_x *= -1
            self.x = float(np.clip(self.x, 0, width))
        if self.y < 0 or self.y > height:
            self.velocity_y *= -1
            self.y = float(np.clip(self.y, 0, height))

    def set_color(self, color: str) -> None:
        self.color = color

    def as_patch(self, alpha: float = 0.75, edgecolor: str = "#f8f4ec"):
        raise NotImplementedError


class Circle(Shape):
    def as_patch(self, alpha: float = 0.75, edgecolor: str = "#f8f4ec"):
        return patches.Circle(
            (self.x, self.y),
            radius=self.size,
            facecolor=self.color,
            edgecolor=edgecolor,
            linewidth=0.4,
            alpha=alpha,
        )


class Square(Shape):
    def as_patch(self, alpha: float = 0.75, edgecolor: str = "#f8f4ec"):
        return patches.Rectangle(
            (self.x - self.size / 2, self.y - self.size / 2),
            width=self.size,
            height=self.size,
            facecolor=self.color,
            edgecolor=edgecolor,
            linewidth=0.4,
            alpha=alpha,
        )


class Triangle(Shape):
    def as_patch(self, alpha: float = 0.75, edgecolor: str = "#f8f4ec"):
        top = (self.x, self.y + self.size)
        left = (self.x - self.size * 0.85, self.y - self.size * 0.65)
        right = (self.x + self.size * 0.85, self.y - self.size * 0.65)
        return patches.Polygon(
            [top, left, right],
            closed=True,
            facecolor=self.color,
            edgecolor=edgecolor,
            linewidth=0.4,
            alpha=alpha,
        )


SHAPE_CLASSES = {
    "circle": Circle,
    "square": Square,
    "triangle": Triangle,
}


def _sanitize_palette(palette_name: str, custom_palette: str) -> list[str]:
    custom_colors = [token.strip() for token in custom_palette.split(",") if token.strip()]
    valid_custom = [color for color in custom_colors if mcolors.is_color_like(color)]
    if valid_custom:
        return valid_custom[:8]
    return PALETTES.get(palette_name, PALETTES["sunset"])


def _palette_cmap(palette: list[str]):
    return mcolors.LinearSegmentedColormap.from_list("studio_palette", palette)


def _normalize_overlay_shapes(overlay_shapes: list[dict] | None) -> list[dict]:
    normalized: list[dict] = []
    if not overlay_shapes:
        return normalized

    for item in overlay_shapes[:180]:
        if not isinstance(item, dict):
            continue
        shape_name = str(item.get("shape", "circle")).lower()
        if shape_name not in ALLOWED_SHAPES:
            continue
        try:
            x = float(item.get("x", 0.5))
            y = float(item.get("y", 0.5))
            size = float(item.get("size", 24))
        except (TypeError, ValueError):
            continue

        color = str(item.get("color", "#f8f4ec"))
        if not mcolors.is_color_like(color):
            color = "#f8f4ec"

        normalized.append(
            {
                "shape": shape_name,
                "x": max(0.0, min(1.0, x)),
                "y": max(0.0, min(1.0, y)),
                "size": max(6.0, min(96.0, size)),
                "color": color,
            }
        )
    return normalized


def _paint_background(axis, width: int, height: int, background: str, palette: list[str], rng: np.random.Generator) -> None:
    cmap = _palette_cmap(palette)
    grid_x = np.linspace(0.0, 1.0, 200)
    grid_y = np.linspace(0.0, 1.0, 200)
    xx, yy = np.meshgrid(grid_x, grid_y)

    if background == "paper":
        field = 0.55 + 0.2 * np.sin(xx * np.pi * 1.4) + 0.15 * np.cos(yy * np.pi * 1.8)
        noise = rng.normal(0.0, 0.04, size=field.shape)
        base = np.clip(field + noise, 0.0, 1.0)
        axis.imshow(base, extent=[0, width, 0, height], origin="lower", cmap="copper", alpha=0.22, aspect="auto")
        axis.set_facecolor("#f3eadc")
    elif background == "night":
        field = 0.25 + 0.35 * yy + 0.18 * np.sin((xx + yy) * np.pi * 1.7)
        axis.imshow(field, extent=[0, width, 0, height], origin="lower", cmap=cmap, alpha=0.35, aspect="auto")
        stars_x = rng.uniform(0, width, 150)
        stars_y = rng.uniform(0, height, 150)
        axis.scatter(stars_x, stars_y, s=rng.uniform(3, 20, 150), color="#f8f4ec", alpha=0.18, linewidths=0)
        axis.set_facecolor("#08131b")
    else:
        field = 0.45 + 0.3 * np.sin((xx * 1.6 + yy * 0.8) * np.pi) + 0.18 * np.cos(yy * np.pi * 2.4)
        axis.imshow(field, extent=[0, width, 0, height], origin="lower", cmap=cmap, alpha=0.42, aspect="auto")
        axis.set_facecolor("#0d1720")

    for _ in range(5):
        patch = patches.Ellipse(
            (rng.uniform(0, width), rng.uniform(0, height)),
            width=rng.uniform(width * 0.12, width * 0.42),
            height=rng.uniform(height * 0.08, height * 0.32),
            angle=rng.uniform(0, 180),
            facecolor=palette[int(rng.integers(0, len(palette)))],
            edgecolor="none",
            alpha=0.08 if background != "night" else 0.06,
        )
        axis.add_patch(patch)


def _clustered_points(count: int, width: int, height: int, density: float, rng: np.random.Generator) -> np.ndarray:
    spread_x = width * (0.32 - min(density, 1.6) * 0.09)
    spread_y = height * (0.3 - min(density, 1.6) * 0.08)
    anchor = np.array([width * rng.uniform(0.35, 0.65), height * rng.uniform(0.35, 0.65)])
    clustered = rng.normal(loc=anchor, scale=[max(30, spread_x), max(30, spread_y)], size=(count, 2))
    scattered = np.column_stack((rng.uniform(0, width, count // 3 + 1), rng.uniform(0, height, count // 3 + 1)))
    points = np.vstack((clustered, scattered))[:count]
    points[:, 0] = np.clip(points[:, 0], 0, width)
    points[:, 1] = np.clip(points[:, 1], 0, height)
    return points


def _draw_constellation(
    axis,
    number_of_shapes: int,
    palette: list[str],
    size_variation: float,
    density: float,
    width: int,
    height: int,
    line_density: float,
    rng: np.random.Generator,
) -> None:
    count = max(45, number_of_shapes)
    points = _clustered_points(count, width, height, density, rng)

    diff = points[:, None, :] - points[None, :, :]
    distances = np.sqrt(np.sum(diff**2, axis=2))
    threshold = max(width, height) * (0.07 + min(line_density, 2.0) * 0.045)
    indices = np.argwhere((distances > 0) & (distances < threshold))

    segments = []
    colors = []
    for first, second in indices:
        if first >= second or rng.random() > 0.28:
            continue
        segments.append([points[first], points[second]])
        colors.append(palette[int(rng.integers(0, len(palette)))])

    if segments:
        collection = LineCollection(
            segments,
            colors=colors,
            linewidths=rng.uniform(0.3, 1.2, len(segments)),
            alpha=0.22,
        )
        axis.add_collection(collection)

    sizes = rng.uniform(18, 220, len(points)) * size_variation
    axis.scatter(
        points[:, 0],
        points[:, 1],
        s=sizes,
        c=rng.choice(palette, len(points)),
        alpha=0.58,
        linewidths=0,
    )

    for _ in range(8):
        center_x, center_y = points[rng.integers(0, len(points))]
        cloud = patches.Circle(
            (center_x, center_y),
            radius=rng.uniform(30, 120) * size_variation,
            facecolor=palette[int(rng.integers(0, len(palette)))],
            edgecolor="none",
            alpha=0.06,
        )
        axis.add_patch(cloud)


def _draw_mosaic(
    axis,
    number_of_shapes: int,
    palette: list[str],
    size_variation: float,
    density: float,
    width: int,
    height: int,
    rng: np.random.Generator,
) -> None:
    cell = max(18, int(72 / max(0.45, size_variation)))
    columns = max(6, width // cell)
    rows = max(5, height // cell)
    max_tiles = max(60, number_of_shapes)

    tile_index = 0
    for row in range(rows):
        for col in range(columns):
            if tile_index >= max_tiles:
                return
            if rng.random() > min(0.95, 0.36 + density * 0.38):
                continue

            x = col * cell
            y = row * cell
            color = palette[int(rng.integers(0, len(palette)))]
            alpha = 0.92 if (row + col) % 5 == 0 else 0.74 if row % 2 == 0 else 0.58

            selector = (row * columns + col) % 4
            if selector == 0:
                axis.add_patch(
                    patches.Rectangle(
                        (x, y),
                        cell,
                        cell,
                        facecolor=color,
                        edgecolor="#f8f4ec",
                        linewidth=0.2,
                        alpha=alpha,
                    )
                )
            elif selector == 1:
                axis.add_patch(
                    patches.Polygon(
                        [(x, y), (x + cell, y), (x + cell, y + cell)],
                        closed=True,
                        facecolor=color,
                        edgecolor="#f8f4ec",
                        linewidth=0.2,
                        alpha=alpha,
                    )
                )
            elif selector == 2:
                axis.add_patch(
                    patches.Polygon(
                        [(x, y + cell), (x + cell, y), (x + cell, y + cell)],
                        closed=True,
                        facecolor=color,
                        edgecolor="#f8f4ec",
                        linewidth=0.2,
                        alpha=alpha,
                    )
                )
                axis.plot([x, x + cell], [y, y + cell], color="#f8f4ec", linewidth=0.45, alpha=0.26)
            else:
                for band in range(3):
                    offset = band * (cell / 3)
                    axis.add_patch(
                        patches.Rectangle(
                            (x, y + offset),
                            cell,
                            cell / 3,
                            facecolor=palette[(band + col) % len(palette)],
                            edgecolor="none",
                            alpha=0.65,
                        )
                    )
            tile_index += 1


def _build_shapes(
    number_of_shapes: int,
    palette: list[str],
    size_variation: float,
    width: int,
    height: int,
    py_rng: random.Random,
) -> list[Shape]:
    shape_types = [Circle, Square, Triangle]
    scene: list[Shape] = []

    for _ in range(number_of_shapes):
        shape_class = py_rng.choice(shape_types)
        scene.append(
            shape_class(
                x=py_rng.uniform(0, width),
                y=py_rng.uniform(0, height),
                size=py_rng.uniform(8, 42) * size_variation,
                color=py_rng.choice(palette),
                velocity_x=py_rng.uniform(-2.4, 2.4),
                velocity_y=py_rng.uniform(-2.4, 2.4),
            )
        )
    return scene


def _draw_kinetic(
    axis,
    number_of_shapes: int,
    palette: list[str],
    size_variation: float,
    width: int,
    height: int,
    animation: bool,
    py_rng: random.Random,
) -> None:
    scene = _build_shapes(max(18, number_of_shapes // 2), palette, size_variation, width, height, py_rng)
    frames = 5 if animation else 1

    for frame in range(frames):
        alpha = max(0.18, 0.72 - frame * 0.12)
        for shape in scene:
            if animation:
                shape.move(width, height)
                if py_rng.random() < 0.1:
                    shape.set_color(py_rng.choice(palette))
            shape.draw(axis, alpha=alpha)


def _draw_overlay_shapes(axis, overlay_shapes: list[dict], width: int, height: int) -> None:
    for item in overlay_shapes:
        shape_class = SHAPE_CLASSES[item["shape"]]
        shape = shape_class(
            x=item["x"] * width,
            y=(1 - item["y"]) * height,
            size=item["size"],
            color=item["color"],
        )
        shape.draw(axis, alpha=0.9, edgecolor="#fff7ee")


def create_generative_art(
    output_dir: Path,
    number_of_shapes: int = 120,
    palette_name: str = "sunset",
    custom_palette: str = "",
    size_variation: float = 1.0,
    density: float = 0.8,
    canvas_size: tuple[int, int] = (960, 640),
    animation: bool = False,
    mode: str = "constellation",
    seed: int | None = None,
    background: str = "aurora",
    line_density: float = 1.0,
    overlay_shapes: list[dict] | None = None,
    filename_stem: str | None = None,
) -> str:
    """Render one generative artwork and return the generated filename."""

    output_dir.mkdir(parents=True, exist_ok=True)

    width = max(320, min(2400, int(canvas_size[0])))
    height = max(240, min(1800, int(canvas_size[1])))

    normalized_mode = str(mode).lower()
    normalized_mode = {
        "random": "mosaic",
        "hybrid": "constellation",
        "oop": "kinetic",
    }.get(normalized_mode, normalized_mode)
    if normalized_mode not in SERIES_INFO:
        normalized_mode = "constellation"

    normalized_background = str(background).lower()
    if normalized_background not in BACKGROUND_STYLES:
        normalized_background = "aurora"

    palette = _sanitize_palette(palette_name, custom_palette)
    safe_seed = seed if isinstance(seed, int) else random.randint(1, 9_999_999)
    py_rng = random.Random(safe_seed)
    rng = np.random.default_rng(safe_seed)
    normalized_overlay = _normalize_overlay_shapes(overlay_shapes)

    figure = plt.figure(figsize=(width / 100, height / 100), dpi=100)
    axis = figure.add_axes([0, 0, 1, 1])
    figure.patch.set_facecolor("#0e1419")

    _paint_background(axis, width, height, normalized_background, palette, rng)

    if normalized_mode == "constellation":
        _draw_constellation(
            axis=axis,
            number_of_shapes=number_of_shapes,
            palette=palette,
            size_variation=size_variation,
            density=density,
            width=width,
            height=height,
            line_density=line_density,
            rng=rng,
        )
    elif normalized_mode == "mosaic":
        _draw_mosaic(
            axis=axis,
            number_of_shapes=number_of_shapes,
            palette=palette,
            size_variation=size_variation,
            density=density,
            width=width,
            height=height,
            rng=rng,
        )
    else:
        _draw_kinetic(
            axis=axis,
            number_of_shapes=number_of_shapes,
            palette=palette,
            size_variation=size_variation,
            width=width,
            height=height,
            animation=animation,
            py_rng=py_rng,
        )

    if normalized_overlay:
        _draw_overlay_shapes(axis, normalized_overlay, width, height)

    axis.set_xlim(0, width)
    axis.set_ylim(0, height)
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_aspect("auto")
    axis.axis("off")

    if filename_stem:
        filename = f"{filename_stem}.png"
    else:
        filename = f"generative_{normalized_mode}_{uuid4().hex[:12]}.png"
    output_path = output_dir / filename

    figure.savefig(output_path, dpi=100, facecolor=figure.get_facecolor())
    plt.close(figure)

    return filename
