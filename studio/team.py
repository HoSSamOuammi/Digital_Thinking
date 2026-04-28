from __future__ import annotations

from pathlib import Path

from flask import url_for

from studio.config import IMAGE_EXTENSIONS, TEAM_MEMBERS
from studio.storage import find_named_image


def build_team_profiles(directory: Path) -> list[dict[str, str | None]]:
    profiles: list[dict[str, str | None]] = []

    for member in TEAM_MEMBERS:
        photo_name = find_named_image(directory, member["slug"], IMAGE_EXTENSIONS)
        profiles.append(
            {
                "slug": member["slug"],
                "name": member["name"],
                "email": member["email"],
                "role": member["role"],
                "photo_name": photo_name,
                "photo_url": url_for("static", filename=f"Admins/{photo_name}") if photo_name else None,
            }
        )
    return profiles
