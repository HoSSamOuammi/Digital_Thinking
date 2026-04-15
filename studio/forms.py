from __future__ import annotations

import json
from typing import Any, Mapping, Optional

from modules.audio_processing import AUDIO_OPERATIONS
from modules.data_visualization import COLORMAP_OPTIONS, DATA_ART_STYLES
from modules.generative_art import BACKGROUND_STYLES, PALETTES, SERIES_INFO


def coerce_int(raw_value: Any, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(raw_value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, value))


def coerce_optional_int(
    raw_value: Any,
    default: Optional[int],
    minimum: int,
    maximum: int,
) -> Optional[int]:
    if raw_value is None or str(raw_value).strip() == "":
        return default
    try:
        value = int(str(raw_value).strip())
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, value))


def coerce_float(raw_value: Any, default: float, minimum: float, maximum: float) -> float:
    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, value))


def coerce_bool(raw_value: Any, default: bool = False) -> bool:
    if raw_value is None:
        return default
    if isinstance(raw_value, bool):
        return raw_value
    return str(raw_value).strip().lower() in {"1", "true", "yes", "on"}


def parse_overlay_shapes(raw_value: Any) -> list[dict]:
    if not raw_value:
        return []
    if isinstance(raw_value, list):
        return raw_value
    try:
        payload = json.loads(raw_value)
    except (TypeError, json.JSONDecodeError):
        return []
    return payload if isinstance(payload, list) else []


def default_generative_params() -> dict[str, Any]:
    return {
        "series": "constellation",
        "palette": "sunset",
        "custom_palette": "",
        "number_of_shapes": 140,
        "size_variation": 1.0,
        "density": 0.9,
        "line_density": 1.0,
        "canvas_width": 1120,
        "canvas_height": 720,
        "background": "aurora",
        "seed": "",
        "animation": True,
    }


def read_generative_params(
    source: Mapping[str, Any],
    *,
    seed_default: Optional[int],
) -> tuple[dict[str, Any], list[dict], Optional[int]]:
    params = default_generative_params()

    params["series"] = str(source.get("series", params["series"])).strip().lower()
    if params["series"] not in SERIES_INFO:
        params["series"] = "constellation"

    params["palette"] = str(source.get("palette", params["palette"])).strip().lower()
    if params["palette"] not in PALETTES:
        params["palette"] = "sunset"

    params["custom_palette"] = str(source.get("custom_palette", "")).strip()
    params["number_of_shapes"] = coerce_int(source.get("number_of_shapes"), 140, 20, 900)
    params["size_variation"] = coerce_float(source.get("size_variation"), 1.0, 0.4, 2.6)
    params["density"] = coerce_float(source.get("density"), 0.9, 0.2, 1.8)
    params["line_density"] = coerce_float(source.get("line_density"), 1.0, 0.4, 2.0)
    params["canvas_width"] = coerce_int(source.get("canvas_width"), 1120, 360, 2400)
    params["canvas_height"] = coerce_int(source.get("canvas_height"), 720, 280, 1800)

    params["background"] = str(source.get("background", params["background"])).strip().lower()
    if params["background"] not in BACKGROUND_STYLES:
        params["background"] = "aurora"

    params["animation"] = coerce_bool(source.get("animation"), default=True)
    params["seed"] = str(source.get("seed", "")).strip()

    overlay_shapes = parse_overlay_shapes(source.get("overlay_shapes"))
    seed_value = coerce_optional_int(params["seed"], seed_default, 1, 9_999_999)
    params["seed"] = str(seed_value) if seed_value is not None else ""
    return params, overlay_shapes, seed_value


def default_data_params() -> dict[str, Any]:
    return {
        "data_style": "all",
        "focus_column": "auto",
        "colormap": COLORMAP_OPTIONS[0],
        "smoothing_window": 8,
    }


def read_data_params(source: Mapping[str, Any]) -> dict[str, Any]:
    params = default_data_params()

    params["data_style"] = str(source.get("data_style", "all")).strip().lower()
    if params["data_style"] not in DATA_ART_STYLES:
        params["data_style"] = "all"

    params["focus_column"] = str(source.get("focus_column", "auto")).strip()
    params["colormap"] = str(source.get("colormap", COLORMAP_OPTIONS[0])).strip()
    if params["colormap"] not in COLORMAP_OPTIONS:
        params["colormap"] = COLORMAP_OPTIONS[0]

    params["smoothing_window"] = coerce_int(source.get("smoothing_window", "8"), 8, 1, 30)
    return params


def default_image_params() -> dict[str, Any]:
    return {
        "image_effect": "neon",
        "rotate_degrees": 45,
        "pixel_size": 8,
        "kmeans_colors": 5,
        "glitch_shift": 16,
    }


def read_image_params(source: Mapping[str, Any], available_effects: Mapping[str, str]) -> dict[str, Any]:
    params = default_image_params()
    fallback_effect = next(iter(available_effects), "grayscale")

    params["image_effect"] = str(source.get("image_effect", "neon")).strip().lower()
    if params["image_effect"] not in available_effects:
        params["image_effect"] = fallback_effect

    params["rotate_degrees"] = coerce_int(source.get("rotate_degrees", "45"), 45, -360, 360)
    params["pixel_size"] = coerce_int(source.get("pixel_size", "8"), 8, 2, 40)
    params["kmeans_colors"] = coerce_int(source.get("kmeans_colors", "5"), 5, 2, 10)
    params["glitch_shift"] = coerce_int(source.get("glitch_shift", "16"), 16, 4, 48)
    return params


def default_audio_params() -> dict[str, Any]:
    return {
        "audio_operation": "reverse",
        "speed_factor": 1.25,
        "echo_delay": 180,
        "pitch_steps": 4,
        "fade_duration": 900,
    }


def read_audio_params(source: Mapping[str, Any]) -> dict[str, Any]:
    params = default_audio_params()

    params["audio_operation"] = str(source.get("audio_operation", "reverse")).strip().lower()
    if params["audio_operation"] not in AUDIO_OPERATIONS:
        params["audio_operation"] = "reverse"

    params["speed_factor"] = coerce_float(source.get("speed_factor", "1.25"), 1.25, 0.5, 2.5)
    params["echo_delay"] = coerce_int(source.get("echo_delay", "180"), 180, 50, 2000)
    params["pitch_steps"] = coerce_int(source.get("pitch_steps", "4"), 4, -12, 12)
    params["fade_duration"] = coerce_int(source.get("fade_duration", "900"), 900, 100, 6000)
    return params
