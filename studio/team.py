from __future__ import annotations

from pathlib import Path

from flask import url_for

from studio.config import IMAGE_EXTENSIONS, TEAM_MEMBERS
from studio.storage import find_named_image


def build_team_profiles(directory: Path) -> list[dict[str, str | None]]:
    profiles: list[dict[str, str | None]] = []

