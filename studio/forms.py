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
