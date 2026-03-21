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
