from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFilter, ImageOps

Image.MAX_IMAGE_PIXELS = 25_000_000

try:
    from sklearn.cluster import KMeans
except Exception:  # pragma: no cover - optional dependency safeguard
    KMeans = None

IMAGE_EFFECTS: dict[str, str] = {
    "grayscale": "Classic monochrome conversion.",
    "sepia": "Warm cinematic color treatment.",
    "invert": "Invert the full color range.",
    "blur": "Soft atmospheric blur.",
    "edge": "Edge detection with high contrast lines.",
    "pixelate": "Blocky retro pixel treatment.",
    "mirror": "Horizontal mirrored reflection.",
    "rotate": "Rotates and reframes the image.",
    "neon": "Saturated base with glowing edge accents.",
    "glitch": "RGB channel displacement with digital offsets.",
    "watercolor": "Painterly smoothing and posterization.",
    "contour": "Contour overlay for graphic outlines.",
    "kmeans_palette": "Dominant color extraction using K-Means.",
}


def is_kmeans_available() -> bool:
    return KMeans is not None


def get_image_effects() -> dict[str, str]:
    if is_kmeans_available():
        return dict(IMAGE_EFFECTS)
    return {name: description for name, description in IMAGE_EFFECTS.items() if name != "kmeans_palette"}


def _apply_sepia(image: Image.Image) -> Image.Image:
    matrix = np.array(
        [
            [0.393, 0.769, 0.189],
            [0.349, 0.686, 0.168],
            [0.272, 0.534, 0.131],
        ]
    )
    raw = np.asarray(image, dtype=np.float32)
    transformed = raw @ matrix.T
    transformed = np.clip(transformed, 0, 255).astype(np.uint8)
    return Image.fromarray(transformed)


def _pixelate(image: Image.Image, block_size: int) -> Image.Image:
    block_size = max(2, block_size)
    small = image.resize(
        (max(1, image.width // block_size), max(1, image.height // block_size)),
        Image.Resampling.NEAREST,
    )
    return small.resize(image.size, Image.Resampling.NEAREST)


def _apply_neon(image: Image.Image) -> Image.Image:
    vivid = ImageEnhance.Color(image).enhance(1.9)
    vivid = ImageEnhance.Contrast(vivid).enhance(1.35)
    edges = image.filter(ImageFilter.FIND_EDGES).convert("L")
    neon_edges = ImageOps.colorize(edges, black="#09121c", white="#76f7ff")
    hot_edges = ImageOps.colorize(ImageOps.invert(edges), black="#ff4d8d", white="#0e1720")
    blended_edges = Image.blend(neon_edges, hot_edges, 0.35)
    return Image.blend(vivid, blended_edges, 0.42)


def _apply_glitch(image: Image.Image, shift: int) -> Image.Image:
    shift = max(4, min(48, int(shift)))
    red, green, blue = image.split()
    red = ImageChops.offset(red, shift, 0)
    blue = ImageChops.offset(blue, -shift, 0)
    merged = Image.merge("RGB", (red, green, blue))

    scan = Image.new("RGB", image.size, color=(0, 0, 0))
    draw = ImageDraw.Draw(scan)
    for y in range(0, image.height, 7):
        draw.line([(0, y), (image.width, y)], fill=(255, 255, 255), width=1)

    return Image.blend(merged, scan, 0.08)


def _apply_watercolor(image: Image.Image) -> Image.Image:
    softened = image.filter(ImageFilter.SMOOTH_MORE).filter(ImageFilter.SMOOTH_MORE)
    softened = softened.filter(ImageFilter.ModeFilter(size=7))
    softened = ImageOps.posterize(softened, bits=5)
    edges = image.filter(ImageFilter.CONTOUR).convert("L")
    edges = ImageOps.invert(edges)
    watercolor = Image.blend(softened, ImageEnhance.Color(softened).enhance(1.2), 0.5)
    return Image.composite(watercolor, softened, edges)


def _apply_contour_overlay(image: Image.Image) -> Image.Image:
    contour = image.filter(ImageFilter.CONTOUR)
    contour = ImageEnhance.Contrast(contour).enhance(1.5)
    return Image.blend(image, contour, 0.35)


def apply_image_filter(
    input_path: Path,
    effect: str,
    output_dir: Path,
    rotate_degrees: int = 45,
    pixel_size: int = 8,
    glitch_shift: int = 16,
) -> str:
    """Apply one of the configured image effects and return the generated filename."""

    output_dir.mkdir(parents=True, exist_ok=True)
    normalized_effect = effect.lower()

    with Image.open(input_path) as opened_image:
        image = opened_image.convert("RGB")

    if normalized_effect == "grayscale":
        processed = ImageOps.grayscale(image).convert("RGB")
    elif normalized_effect == "sepia":
        processed = _apply_sepia(image)
    elif normalized_effect == "invert":
        processed = ImageOps.invert(image)
    elif normalized_effect == "blur":
        processed = image.filter(ImageFilter.GaussianBlur(radius=2.8))
    elif normalized_effect == "edge":
        processed = image.filter(ImageFilter.FIND_EDGES).convert("RGB")
    elif normalized_effect == "pixelate":
        processed = _pixelate(image, pixel_size)
    elif normalized_effect == "mirror":
        processed = ImageOps.mirror(image)
    elif normalized_effect == "rotate":
        processed = image.rotate(rotate_degrees, expand=True, fillcolor=(16, 21, 30))
    elif normalized_effect == "neon":
        processed = _apply_neon(image)
    elif normalized_effect == "glitch":
        processed = _apply_glitch(image, glitch_shift)
    elif normalized_effect == "watercolor":
        processed = _apply_watercolor(image)
    elif normalized_effect == "contour":
        processed = _apply_contour_overlay(image)
    else:
        raise ValueError(f"Unsupported image effect: {effect}")

    filename = f"image_{normalized_effect}_{uuid4().hex[:12]}.png"
    output_path = output_dir / filename
    processed.save(output_path, format="PNG", optimize=True)
    return filename


def kmeans_color_palette(input_path: Path, output_dir: Path, n_colors: int = 5) -> str:
    """Create a palette strip using K-Means color extraction."""

    if not is_kmeans_available():
        raise RuntimeError("scikit-learn is not installed. Install it to use K-Means palette extraction.")

    output_dir.mkdir(parents=True, exist_ok=True)

    with Image.open(input_path) as opened_image:
        image = opened_image.convert("RGB").resize((240, 240))

    pixels = np.asarray(image).reshape(-1, 3)

    clamped_colors = max(2, min(10, int(n_colors)))
    model = KMeans(n_clusters=clamped_colors, random_state=42, n_init="auto")
    labels = model.fit_predict(pixels)

    centers = model.cluster_centers_.astype(np.uint8)
    counts = np.bincount(labels)
    order = np.argsort(counts)[::-1]
    centers = centers[order]
    shares = counts[order] / counts.sum()

    width, height = 860, 200
    palette = Image.new("RGB", (width, height), color=(16, 19, 24))
    draw = ImageDraw.Draw(palette)

    cursor = 0
    for color, share in zip(centers, shares):
        segment = max(1, int(width * float(share)))
        next_cursor = min(width, cursor + segment)
        rgb = tuple(int(channel) for channel in color)
        hex_code = "#%02x%02x%02x" % rgb

        draw.rectangle([(cursor, 0), (next_cursor, 138)], fill=rgb)
        draw.text((cursor + 12, 150), hex_code, fill=(242, 242, 240))
        cursor = next_cursor

    filename = f"kmeans_palette_{uuid4().hex[:12]}.png"
    output_path = output_dir / filename
    palette.save(output_path, format="PNG", optimize=True)
    return filename
