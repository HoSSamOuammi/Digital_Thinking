from __future__ import annotations

from pathlib import Path

from flask import abort, flash, redirect, render_template, request, send_from_directory, url_for

from modules.generative_art import PALETTES
from studio.config import (
    AUDIO_EXTENSIONS,
    DATA_EXTENSIONS,
    GALLERY_AUDIO_PAGE_SIZE,
    GALLERY_IMAGE_PAGE_SIZE,
    IMAGE_EXTENSIONS,
)
from studio.forms import coerce_int
from studio.storage import cleanup_directory_files, list_generated_files, paginate_items, save_uploaded_file
from studio.team import build_team_profiles


def register_page_routes(app) -> None:
    @app.route("/")
    def home():
        generated_images, generated_audio = _generated_files(app)
        return render_template(
            "home.html",
            image_count=len(generated_images),
            audio_count=len(generated_audio),
            palette_names=sorted(PALETTES.keys()),
            featured_images=generated_images[:4],
        )

    @app.route("/gallery")
    def gallery():
        generated_images, generated_audio = _generated_files(app)
        image_page = coerce_int(request.args.get("image_page"), 1, 1, 10_000)
        audio_page = coerce_int(request.args.get("audio_page"), 1, 1, 10_000)

        image_pagination = paginate_items(generated_images, image_page, GALLERY_IMAGE_PAGE_SIZE)
        audio_pagination = paginate_items(generated_audio, audio_page, GALLERY_AUDIO_PAGE_SIZE)

        return render_template(
            "gallery.html",
            generated_images=image_pagination["items"],
            generated_audio=audio_pagination["items"],
            image_pagination=image_pagination,
            audio_pagination=audio_pagination,
            total_image_count=len(generated_images),
            total_audio_count=len(generated_audio),
        )

