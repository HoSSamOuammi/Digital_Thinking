from __future__ import annotations

import hashlib
import json
import secrets
from typing import Any, Mapping
from uuid import uuid4

from flask import flash, jsonify, render_template, request, url_for

from modules.generative_art import BACKGROUND_STYLES, PALETTES, SERIES_INFO, create_generative_art
from studio.config import AUDIO_EXTENSIONS, IMAGE_EXTENSIONS, PREVIEW_CACHE_LIMIT
from studio.forms import default_generative_params, read_generative_params
from studio.labels import BACKGROUND_LABELS, PALETTE_LABELS, SERIES_LABELS
from studio.storage import cleanup_directory_files, cleanup_preview_files


def register_generative_routes(app) -> None:
    @app.route("/generative", methods=["GET", "POST"])
    def generative():
        generated_image = None
        overlay_shapes: list[dict] = []
        params = default_generative_params()

        if request.method == "POST":
            params, overlay_shapes, seed_value = read_generative_params(
                request.form,
                seed_default=100000 + secrets.randbelow(900000),
            )
            generated_image = _create_final_artwork(app, params, overlay_shapes, seed_value)

        return render_template(
            "generative.html",
            generated_image=generated_image,
            palette_names=sorted(PALETTES.keys()),
            palette_map=PALETTES,
            series_info=SERIES_INFO,
            series_labels=SERIES_LABELS,
            background_styles=BACKGROUND_STYLES,
            background_labels=BACKGROUND_LABELS,
            palette_labels=PALETTE_LABELS,
            params=params,
            overlay_shapes=overlay_shapes,
            overlay_shape_count=len(overlay_shapes),
        )

    @app.post("/api/generative-preview")
    def generative_preview():
        payload = request.get_json(silent=True) or {}
        params, _overlay_shapes, seed_value = read_generative_params(payload, seed_default=424242)
        preview_width, preview_height = preview_canvas_size(params["canvas_width"], params["canvas_height"])
        filename_stem = preview_filename(params)

        try:
            filename = create_generative_art(
                output_dir=app.config["PREVIEW_FOLDER"],
                number_of_shapes=params["number_of_shapes"],
                palette_name=params["palette"],
                custom_palette=params["custom_palette"],
                size_variation=params["size_variation"],
                density=params["density"],
                canvas_size=(preview_width, preview_height),
                animation=params["animation"],
                mode=params["series"],
                seed=seed_value,
                background=params["background"],
                line_density=params["line_density"],
                overlay_shapes=None,
                filename_stem=filename_stem,
            )
            cleanup_preview_files(app.config["PREVIEW_FOLDER"], keep=PREVIEW_CACHE_LIMIT)
            preview_path = app.config["PREVIEW_FOLDER"] / filename
            cache_bust = preview_path.stat().st_mtime_ns if preview_path.exists() else uuid4().hex

            return jsonify(
                {
                    "preview_url": url_for("static", filename=f"generated/previews/{filename}") + f"?v={cache_bust}",
                    "width": preview_width,
                    "height": preview_height,
                }
            )
        except Exception:  # pragma: no cover - defensive path
            app.logger.exception("Generative preview rendering failed.")
            return jsonify({"error": "La prévisualisation n’a pas pu être générée."}), 500


def _create_final_artwork(app, params: Mapping[str, Any], overlay_shapes: list[dict], seed_value: int | None) -> str | None:
    try:
        filename = create_generative_art(
            output_dir=app.config["GENERATED_FOLDER"],
            number_of_shapes=params["number_of_shapes"],
            palette_name=params["palette"],
            custom_palette=params["custom_palette"],
            size_variation=params["size_variation"],
            density=params["density"],
            canvas_size=(params["canvas_width"], params["canvas_height"]),
            animation=params["animation"],
            mode=params["series"],
            seed=seed_value,
            background=params["background"],
            line_density=params["line_density"],
            overlay_shapes=overlay_shapes,
        )
        cleanup_directory_files(
            app.config["GENERATED_FOLDER"],
            keep=app.config["MAX_SAVED_GENERATED_FILES"],
            allowed_extensions=IMAGE_EXTENSIONS | AUDIO_EXTENSIONS,
        )
        flash("Visuel généré avec succès.", "success")
        return filename
    except Exception:  # pragma: no cover - defensive path
        app.logger.exception("Generative artwork rendering failed.")
        flash("Le visuel n’a pas pu être généré pour le moment.", "error")
        return None


def preview_canvas_size(width: int, height: int) -> tuple[int, int]:
    safe_width = max(360, width)
    safe_height = max(280, height)
    scale = min(1.0, 900 / safe_width)
    return max(360, int(safe_width * scale)), max(240, int(safe_height * scale))


def preview_filename(params: Mapping[str, Any]) -> str:
    payload = {
        "series": params["series"],
        "palette": params["palette"],
        "custom_palette": params["custom_palette"],
        "number_of_shapes": params["number_of_shapes"],
        "size_variation": params["size_variation"],
        "density": params["density"],
        "line_density": params["line_density"],
        "canvas_width": params["canvas_width"],
        "canvas_height": params["canvas_height"],
        "background": params["background"],
        "seed": params["seed"],
        "animation": params["animation"],
    }
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    return f"preview_{digest}"
