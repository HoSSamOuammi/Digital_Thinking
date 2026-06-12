from __future__ import annotations

from pathlib import Path

from flask import flash, render_template, request

from modules.data_visualization import COLORMAP_OPTIONS, DATA_ART_STYLES, create_data_art, load_and_preprocess_dataset
from studio.config import AUDIO_EXTENSIONS, DATA_EXTENSIONS, IMAGE_EXTENSIONS
from studio.forms import default_data_params, read_data_params
from studio.labels import DATA_STYLE_LABELS
from studio.storage import cleanup_directory_files, delete_file_if_exists, save_uploaded_file


def register_data_routes(app) -> None:
    @app.route("/data-art", methods=["GET", "POST"])
    def data_art():
        generated_image = None
        metadata = None
        _, sample_metadata = load_and_preprocess_dataset()
        available_columns = sample_metadata["columns"]
        params = default_data_params()

        if request.method == "POST":
            params = read_data_params(request.form)
            dataset_path = _save_dataset_file(app)
            if dataset_path is False:
                return _render_data_art_page(generated_image, metadata, available_columns, params)

            generated_image, metadata, available_columns = _create_data_art_output(app, dataset_path, params)

        return _render_data_art_page(generated_image, metadata, available_columns, params)


def _save_dataset_file(app) -> Path | bool | None:
    uploaded_csv = request.files.get("dataset_file")
    if not uploaded_csv or not uploaded_csv.filename:
        return None

    saved_csv = save_uploaded_file(
        file_storage=uploaded_csv,
        target_dir=app.config["UPLOAD_FOLDER"],
        allowed_extensions=DATA_EXTENSIONS,
    )
    if not saved_csv:
        flash("Le jeu de données doit être un fichier CSV.", "error")
        return False
    return app.config["UPLOAD_FOLDER"] / saved_csv


def _create_data_art_output(app, dataset_path: Path | None, params: dict) -> tuple[str | None, dict | None, list[str]]:
    metadata = None
    generated_image = None
    available_columns: list[str] = []

    try:
        frame, preview_metadata = load_and_preprocess_dataset(dataset_path)
        available_columns = preview_metadata["columns"]
        if params["focus_column"] != "auto" and params["focus_column"] not in available_columns:
            params["focus_column"] = "auto"

        generated_image, metadata = create_data_art(
            output_dir=app.config["GENERATED_FOLDER"],
            dataset_path=dataset_path,
            frame=frame,
            metadata=preview_metadata,
            style=params["data_style"],
            focus_column=params["focus_column"],
            colormap=params["colormap"],
            smoothing_window=params["smoothing_window"],
        )
        cleanup_directory_files(
            app.config["GENERATED_FOLDER"],
            keep=app.config["MAX_SAVED_GENERATED_FILES"],
            allowed_extensions=IMAGE_EXTENSIONS | AUDIO_EXTENSIONS,
        )
        flash("Visualisation générée avec succès.", "success")
    except Exception:  # pragma: no cover - defensive path
        app.logger.exception("Data-art rendering failed.")
        flash("La visualisation n’a pas pu être générée pour le moment.", "error")
    finally:
        delete_file_if_exists(dataset_path)
        cleanup_directory_files(
            app.config["UPLOAD_FOLDER"],
            keep=app.config["MAX_SAVED_UPLOADS"],
            allowed_extensions=IMAGE_EXTENSIONS | AUDIO_EXTENSIONS | DATA_EXTENSIONS,
        )

    return generated_image, metadata, available_columns


def _render_data_art_page(generated_image, metadata, available_columns, params):
    return render_template(
        "data_art.html",
        generated_image=generated_image,
        metadata=metadata,
        available_columns=available_columns,
        params=params,
        data_art_styles=DATA_ART_STYLES,
        data_style_labels=DATA_STYLE_LABELS,
        colormap_options=COLORMAP_OPTIONS,
    )
