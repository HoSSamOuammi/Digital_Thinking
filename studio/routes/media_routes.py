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
