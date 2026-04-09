from __future__ import annotations

from flask import flash, render_template, request

from modules.audio_processing import AUDIO_OPERATIONS, get_audio_status, process_audio
from modules.image_processing import apply_image_filter, get_image_effects, is_kmeans_available, kmeans_color_palette
from studio.config import AUDIO_EXTENSIONS, DATA_EXTENSIONS, IMAGE_EXTENSIONS
from studio.forms import default_audio_params, default_image_params, read_audio_params, read_image_params
from studio.labels import AUDIO_OPERATION_LABELS, IMAGE_EFFECT_LABELS
from studio.storage import cleanup_directory_files, delete_file_if_exists, save_uploaded_file


def register_media_routes(app) -> None:
    @app.route("/media-tools", methods=["GET", "POST"])
    def media_tools():
        processed_image = None
        processed_audio = None
        active_panel = "image"
        image_effects = get_image_effects()
        kmeans_available = is_kmeans_available()
        audio_status = get_audio_status()
        audio_available = bool(audio_status["available"])
        image_params = default_image_params()
        audio_params = default_audio_params()

        if request.method == "POST":
            active_panel = request.form.get("panel", "image")
            if active_panel == "image":
                image_params = read_image_params(request.form, image_effects)
                processed_image = _process_image(app, image_params, kmeans_available)
            elif active_panel == "audio":
                audio_params = read_audio_params(request.form)
                processed_audio = _process_audio(app, audio_params, audio_available, audio_status)

        return render_template(
            "media_tools.html",
            processed_image=processed_image,
            processed_audio=processed_audio,
            audio_available=audio_available,
            audio_status=audio_status,
            active_panel=active_panel,
            image_effects=image_effects,
            image_effect_labels=IMAGE_EFFECT_LABELS,
            audio_operations=AUDIO_OPERATIONS,
            audio_operation_labels=AUDIO_OPERATION_LABELS,
            image_params=image_params,
            audio_params=audio_params,
            kmeans_available=kmeans_available,
        )


def _process_image(app, image_params: dict, kmeans_available: bool) -> str | None:
    image_name = save_uploaded_file(
        file_storage=request.files.get("image_file"),
        target_dir=app.config["UPLOAD_FOLDER"],
        allowed_extensions=IMAGE_EXTENSIONS,
    )
    if not image_name:
        flash("Veuillez importer une image valide.", "error")
        return None

    source_path = app.config["UPLOAD_FOLDER"] / image_name
    try:
        if image_params["image_effect"] == "kmeans_palette":
            if not kmeans_available:
                raise RuntimeError("K-Means palette extraction is unavailable on this environment.")
            filename = kmeans_color_palette(
                input_path=source_path,
                output_dir=app.config["GENERATED_FOLDER"],
                n_colors=image_params["kmeans_colors"],
            )
        else:
            filename = apply_image_filter(
                input_path=source_path,
                effect=image_params["image_effect"],
                output_dir=app.config["GENERATED_FOLDER"],
                rotate_degrees=image_params["rotate_degrees"],
                pixel_size=image_params["pixel_size"],
                glitch_shift=image_params["glitch_shift"],
            )
        _cleanup_media_folders(app)
        flash("Image traitée avec succès.", "success")
        return filename
    except Exception:  # pragma: no cover - defensive path
        app.logger.exception("Image processing failed.")
        flash("Le traitement de l’image a échoué.", "error")
        return None
    finally:
        delete_file_if_exists(source_path)
        _cleanup_upload_folder(app)


def _process_audio(app, audio_params: dict, audio_available: bool, audio_status: dict) -> str | None:
    if not audio_available:
        flash(f"Les outils audio sont désactivés. {audio_status['reason']}", "error")
        return None

    audio_name = save_uploaded_file(
        file_storage=request.files.get("audio_file"),
        target_dir=app.config["UPLOAD_FOLDER"],
        allowed_extensions=AUDIO_EXTENSIONS,
    )
    if not audio_name:
        flash("Veuillez importer un fichier audio valide.", "error")
        return None

    audio_path = app.config["UPLOAD_FOLDER"] / audio_name
    merge_path = _save_optional_merge_file(app)
    try:
        filename = process_audio(
            input_path=audio_path,
            output_dir=app.config["GENERATED_FOLDER"],
            operation=audio_params["audio_operation"],
            speed=audio_params["speed_factor"],
            echo_delay=audio_params["echo_delay"],
            merge_path=merge_path,
            pitch_steps=audio_params["pitch_steps"],
            fade_duration=audio_params["fade_duration"],
        )
        _cleanup_media_folders(app)
        flash("Audio traité avec succès.", "success")
        return filename
    except Exception:  # pragma: no cover - defensive path
        app.logger.exception("Audio processing failed.")
        flash("Le traitement audio a échoué.", "error")
        return None
    finally:
        delete_file_if_exists(audio_path)
        delete_file_if_exists(merge_path)
        _cleanup_upload_folder(app)


def _save_optional_merge_file(app):
