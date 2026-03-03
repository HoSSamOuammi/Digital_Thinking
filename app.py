from __future__ import annotations

import hashlib
import json
import os
import secrets
from pathlib import Path
from typing import Any, Mapping, Optional
from uuid import uuid4

from flask import Flask, abort, flash, jsonify, redirect, render_template, request, send_from_directory, session, url_for
from werkzeug.utils import secure_filename

from modules.audio_processing import AUDIO_OPERATIONS, get_audio_status, process_audio
from modules.data_visualization import COLORMAP_OPTIONS, DATA_ART_STYLES, create_data_art, load_and_preprocess_dataset
from modules.generative_art import BACKGROUND_STYLES, PALETTES, SERIES_INFO, create_generative_art
from modules.image_processing import apply_image_filter, get_image_effects, is_kmeans_available, kmeans_color_palette

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / "static" / "uploads"
GENERATED_FOLDER = BASE_DIR / "static" / "generated"
PREVIEW_FOLDER = GENERATED_FOLDER / "previews"

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}
AUDIO_EXTENSIONS = {".wav", ".mp3", ".ogg", ".flac", ".m4a"}
DATA_EXTENSIONS = {".csv"}

GALLERY_IMAGE_PAGE_SIZE = 12
GALLERY_AUDIO_PAGE_SIZE = 8
PREVIEW_CACHE_LIMIT = 24
UPLOAD_CACHE_LIMIT = 40
GENERATED_CACHE_LIMIT = 120


def _is_truthy_env(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _resolve_secret_key() -> tuple[str, bool]:
    configured_key = os.getenv("FLASK_SECRET_KEY", "").strip()
    if configured_key:
        return configured_key, False
    return secrets.token_hex(32), True


def _coerce_int(raw_value: Any, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(raw_value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, value))


def _coerce_optional_int(
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


def _coerce_float(raw_value: Any, default: float, minimum: float, maximum: float) -> float:
    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, value))


def _coerce_bool(raw_value: Any, default: bool = False) -> bool:
    if raw_value is None:
        return default
    if isinstance(raw_value, bool):
        return raw_value
    return str(raw_value).strip().lower() in {"1", "true", "yes", "on"}


def _save_uploaded_file(file_storage, target_dir: Path, allowed_extensions: set[str]) -> Optional[str]:
    if not file_storage or not file_storage.filename:
        return None

    raw_filename = secure_filename(file_storage.filename)
    extension = Path(raw_filename).suffix.lower()
    if extension not in allowed_extensions:
        return None

    final_name = f"{Path(raw_filename).stem}_{uuid4().hex[:10]}{extension}"
    final_path = target_dir / final_name
    file_storage.save(final_path)
    return final_name


def _delete_file_if_exists(path: Optional[Path]) -> None:
    if not path:
        return
    try:
        path.unlink(missing_ok=True)
    except OSError:
        return


def _list_generated_files(directory: Path) -> tuple[list[str], list[str]]:
    images: list[str] = []
    audios: list[str] = []
    items: list[tuple[float, str, str]] = []

    for path in directory.glob("*"):
        if not path.is_file():
            continue
        try:
            items.append((path.stat().st_mtime, path.name, path.suffix.lower()))
        except OSError:
            continue

    items.sort(key=lambda item: item[0], reverse=True)
    for _, name, extension in items:
        if extension in IMAGE_EXTENSIONS:
            images.append(name)
        elif extension in AUDIO_EXTENSIONS:
            audios.append(name)
    return images, audios


def _paginate_items(items: list[str], page: int, per_page: int) -> dict[str, int | bool | list[str]]:
    total_items = len(items)
    total_pages = max(1, (total_items + per_page - 1) // per_page)
    safe_page = max(1, min(total_pages, page))
    start = (safe_page - 1) * per_page
    end = start + per_page

    return {
        "items": items[start:end],
        "page": safe_page,
        "per_page": per_page,
        "total_items": total_items,
        "total_pages": total_pages,
        "has_prev": safe_page > 1,
        "has_next": safe_page < total_pages,
        "prev_page": max(1, safe_page - 1),
        "next_page": min(total_pages, safe_page + 1),
    }


def _parse_overlay_shapes(raw_value: Any) -> list[dict]:
    if not raw_value:
        return []
    if isinstance(raw_value, list):
        return raw_value
    try:
        payload = json.loads(raw_value)
    except (TypeError, json.JSONDecodeError):
        return []
    return payload if isinstance(payload, list) else []


def _default_generative_params() -> dict[str, Any]:
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


def _read_generative_params(
    source: Mapping[str, Any],
    *,
    seed_default: Optional[int],
) -> tuple[dict[str, Any], list[dict], Optional[int]]:
    params = _default_generative_params()

    params["series"] = str(source.get("series", params["series"])).strip().lower()
    if params["series"] not in SERIES_INFO:
        params["series"] = "constellation"

    params["palette"] = str(source.get("palette", params["palette"])).strip().lower()
    if params["palette"] not in PALETTES:
        params["palette"] = "sunset"

    params["custom_palette"] = str(source.get("custom_palette", "")).strip()
    params["number_of_shapes"] = _coerce_int(source.get("number_of_shapes"), 140, 20, 900)
    params["size_variation"] = _coerce_float(source.get("size_variation"), 1.0, 0.4, 2.6)
    params["density"] = _coerce_float(source.get("density"), 0.9, 0.2, 1.8)
    params["line_density"] = _coerce_float(source.get("line_density"), 1.0, 0.4, 2.0)
    params["canvas_width"] = _coerce_int(source.get("canvas_width"), 1120, 360, 2400)
    params["canvas_height"] = _coerce_int(source.get("canvas_height"), 720, 280, 1800)
    params["background"] = str(source.get("background", params["background"])).strip().lower()
    if params["background"] not in BACKGROUND_STYLES:
        params["background"] = "aurora"

    params["animation"] = _coerce_bool(source.get("animation"), default=True)
    params["seed"] = str(source.get("seed", "")).strip()
    overlay_shapes = _parse_overlay_shapes(source.get("overlay_shapes"))

    seed_value = _coerce_optional_int(params["seed"], seed_default, 1, 9_999_999)
    params["seed"] = str(seed_value) if seed_value is not None else ""
    return params, overlay_shapes, seed_value


def _default_data_params() -> dict[str, Any]:
    return {
        "data_style": "all",
        "focus_column": "auto",
        "colormap": COLORMAP_OPTIONS[0],
        "smoothing_window": 8,
    }


def _preview_canvas_size(width: int, height: int) -> tuple[int, int]:
    safe_width = max(360, width)
    safe_height = max(280, height)
    scale = min(1.0, 900 / safe_width)
    return max(360, int(safe_width * scale)), max(240, int(safe_height * scale))


def _preview_filename(params: Mapping[str, Any]) -> str:
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


def _cleanup_preview_files(directory: Path, keep: int = PREVIEW_CACHE_LIMIT) -> None:
    preview_files = [path for path in directory.glob("*.png") if path.is_file()]
    if len(preview_files) <= keep:
        return
    preview_files.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    for path in preview_files[keep:]:
        try:
            path.unlink()
        except OSError:
            continue


def _cleanup_directory_files(
    directory: Path,
    *,
    keep: int,
    allowed_extensions: Optional[set[str]] = None,
) -> None:
    if keep < 1:
        return

    files = []
    for path in directory.glob("*"):
        if not path.is_file():
            continue
        if allowed_extensions and path.suffix.lower() not in allowed_extensions:
            continue
        try:
            files.append((path.stat().st_mtime, path))
        except OSError:
            continue

    if len(files) <= keep:
        return

    files.sort(key=lambda item: item[0], reverse=True)
    for _, path in files[keep:]:
        _delete_file_if_exists(path)


def create_app() -> Flask:
    app = Flask(__name__)
    secret_key, generated_secret = _resolve_secret_key()
    app.config["SECRET_KEY"] = secret_key
    app.config["MAX_CONTENT_LENGTH"] = 30 * 1024 * 1024
    app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
    app.config["GENERATED_FOLDER"] = GENERATED_FOLDER
    app.config["PREVIEW_FOLDER"] = PREVIEW_FOLDER
    app.config["MAX_SAVED_UPLOADS"] = UPLOAD_CACHE_LIMIT
    app.config["MAX_SAVED_GENERATED_FILES"] = GENERATED_CACHE_LIMIT
    app.config["SESSION_COOKIE_HTTPONLY"] = True
    app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
    app.config["SESSION_COOKIE_SECURE"] = _is_truthy_env("FLASK_SESSION_SECURE")

    app.config["UPLOAD_FOLDER"].mkdir(parents=True, exist_ok=True)
    app.config["GENERATED_FOLDER"].mkdir(parents=True, exist_ok=True)
    app.config["PREVIEW_FOLDER"].mkdir(parents=True, exist_ok=True)

    if generated_secret:
        app.logger.warning("FLASK_SECRET_KEY is not set; using an ephemeral secret key for this process.")

    def _get_csrf_token() -> str:
        token = session.get("_csrf_token")
        if not token:
            token = secrets.token_urlsafe(24)
            session["_csrf_token"] = token
        return token

    @app.context_processor
    def inject_csrf_token():
        return {"csrf_token": _get_csrf_token}

    @app.before_request
    def protect_against_csrf():
        if request.method not in {"POST", "PUT", "PATCH", "DELETE"}:
            return None

        sent_token = request.headers.get("X-CSRF-Token") or request.form.get("csrf_token")
        expected_token = session.get("_csrf_token")
        if expected_token and secrets.compare_digest(sent_token or "", expected_token):
            return None

        message = "The request could not be verified. Refresh the page and try again."
        if request.is_json or request.path.startswith("/api/"):
            return jsonify({"error": message}), 400

        flash(message, "error")
        return redirect(request.url)

    @app.route("/")
    def home():
        generated_images, generated_audio = _list_generated_files(app.config["GENERATED_FOLDER"])
        return render_template(
            "home.html",
            image_count=len(generated_images),
            audio_count=len(generated_audio),
            palette_names=sorted(PALETTES.keys()),
            featured_images=generated_images[:4],
        )

    @app.route("/gallery")
    def gallery():
        generated_images, generated_audio = _list_generated_files(app.config["GENERATED_FOLDER"])
        image_page = _coerce_int(request.args.get("image_page"), 1, 1, 10_000)
        audio_page = _coerce_int(request.args.get("audio_page"), 1, 1, 10_000)
        image_pagination = _paginate_items(generated_images, image_page, GALLERY_IMAGE_PAGE_SIZE)
        audio_pagination = _paginate_items(generated_audio, audio_page, GALLERY_AUDIO_PAGE_SIZE)
        return render_template(
            "gallery.html",
            generated_images=image_pagination["items"],
            generated_audio=audio_pagination["items"],
            image_pagination=image_pagination,
            audio_pagination=audio_pagination,
            total_image_count=len(generated_images),
            total_audio_count=len(generated_audio),
        )

    @app.route("/generative", methods=["GET", "POST"])
    def generative():
        generated_image = None
        overlay_shapes: list[dict] = []
        params = _default_generative_params()

        if request.method == "POST":
            params, overlay_shapes, seed_value = _read_generative_params(
                request.form,
                seed_default=100000 + secrets.randbelow(900000),
            )

            try:
                generated_image = create_generative_art(
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
                _cleanup_directory_files(
                    app.config["GENERATED_FOLDER"],
                    keep=app.config["MAX_SAVED_GENERATED_FILES"],
                    allowed_extensions=IMAGE_EXTENSIONS | AUDIO_EXTENSIONS,
                )
                flash("Artwork generated successfully.", "success")
            except Exception:  # pragma: no cover - defensive path
                app.logger.exception("Generative artwork rendering failed.")
                flash("Could not generate artwork right now.", "error")

        return render_template(
            "generative.html",
            generated_image=generated_image,
            palette_names=sorted(PALETTES.keys()),
            palette_map=PALETTES,
            series_info=SERIES_INFO,
            background_styles=BACKGROUND_STYLES,
            params=params,
            overlay_shapes=overlay_shapes,
            overlay_shape_count=len(overlay_shapes),
        )

    @app.post("/api/generative-preview")
    def generative_preview():
        payload = request.get_json(silent=True) or {}
        params, _overlay_shapes, seed_value = _read_generative_params(payload, seed_default=424242)
        preview_width, preview_height = _preview_canvas_size(
            params["canvas_width"],
            params["canvas_height"],
        )
        filename_stem = _preview_filename(params)

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
            _cleanup_preview_files(app.config["PREVIEW_FOLDER"])

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
            return jsonify({"error": "Preview rendering failed."}), 500

    @app.route("/data-art", methods=["GET", "POST"])
    def data_art():
        generated_image = None
        metadata = None
        _, sample_metadata = load_and_preprocess_dataset()
        available_columns = sample_metadata["columns"]
        params = _default_data_params()

        if request.method == "POST":
            params["data_style"] = str(request.form.get("data_style", "all")).strip().lower()
            if params["data_style"] not in DATA_ART_STYLES:
                params["data_style"] = "all"

            params["focus_column"] = str(request.form.get("focus_column", "auto")).strip()
            params["colormap"] = str(request.form.get("colormap", COLORMAP_OPTIONS[0])).strip()
            if params["colormap"] not in COLORMAP_OPTIONS:
                params["colormap"] = COLORMAP_OPTIONS[0]
            params["smoothing_window"] = _coerce_int(
                request.form.get("smoothing_window", "8"),
                8,
                1,
                30,
            )

            uploaded_csv = request.files.get("dataset_file")
            dataset_path = None

            if uploaded_csv and uploaded_csv.filename:
                saved_csv = _save_uploaded_file(
                    file_storage=uploaded_csv,
                    target_dir=app.config["UPLOAD_FOLDER"],
                    allowed_extensions=DATA_EXTENSIONS,
                )
                if not saved_csv:
                    flash("Dataset must be a CSV file.", "error")
                    return render_template(
                        "data_art.html",
                        generated_image=generated_image,
                        metadata=metadata,
                        available_columns=available_columns,
                        params=params,
                        data_art_styles=DATA_ART_STYLES,
                        colormap_options=COLORMAP_OPTIONS,
                    )
                dataset_path = app.config["UPLOAD_FOLDER"] / saved_csv

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
                _cleanup_directory_files(
                    app.config["GENERATED_FOLDER"],
                    keep=app.config["MAX_SAVED_GENERATED_FILES"],
                    allowed_extensions=IMAGE_EXTENSIONS | AUDIO_EXTENSIONS,
                )
                flash("Data artwork generated successfully.", "success")
            except Exception:  # pragma: no cover - defensive path
                app.logger.exception("Data-art rendering failed.")
                flash("Could not generate data artwork right now.", "error")
            finally:
                _delete_file_if_exists(dataset_path)
                _cleanup_directory_files(
                    app.config["UPLOAD_FOLDER"],
                    keep=app.config["MAX_SAVED_UPLOADS"],
                    allowed_extensions=IMAGE_EXTENSIONS | AUDIO_EXTENSIONS | DATA_EXTENSIONS,
                )

        return render_template(
            "data_art.html",
            generated_image=generated_image,
            metadata=metadata,
            available_columns=available_columns,
            params=params,
            data_art_styles=DATA_ART_STYLES,
            colormap_options=COLORMAP_OPTIONS,
        )

    @app.route("/media-tools", methods=["GET", "POST"])
    def media_tools():
        processed_image = None
        processed_audio = None
        active_panel = "image"
        image_effects = get_image_effects()
        kmeans_available = is_kmeans_available()
        audio_status = get_audio_status()
        audio_available = bool(audio_status["available"])

        image_params = {
            "image_effect": "neon",
            "rotate_degrees": 45,
            "pixel_size": 8,
            "kmeans_colors": 5,
            "glitch_shift": 16,
        }
        audio_params = {
            "audio_operation": "reverse",
            "speed_factor": 1.25,
            "echo_delay": 180,
            "pitch_steps": 4,
            "fade_duration": 900,
        }

        if request.method == "POST":
            active_panel = request.form.get("panel", "image")

            if active_panel == "image":
                image_params["image_effect"] = str(request.form.get("image_effect", "neon")).strip().lower()
                if image_params["image_effect"] not in image_effects:
                    image_params["image_effect"] = next(iter(image_effects))

                image_params["rotate_degrees"] = _coerce_int(
                    request.form.get("rotate_degrees", "45"), 45, -360, 360
                )
                image_params["pixel_size"] = _coerce_int(
                    request.form.get("pixel_size", "8"), 8, 2, 40
                )
                image_params["kmeans_colors"] = _coerce_int(
                    request.form.get("kmeans_colors", "5"), 5, 2, 10
                )
                image_params["glitch_shift"] = _coerce_int(
                    request.form.get("glitch_shift", "16"), 16, 4, 48
                )

                image_file = request.files.get("image_file")
                image_name = _save_uploaded_file(
                    file_storage=image_file,
                    target_dir=app.config["UPLOAD_FOLDER"],
                    allowed_extensions=IMAGE_EXTENSIONS,
                )
                if not image_name:
                    flash("Please upload a valid image file.", "error")
                else:
                    source_path = app.config["UPLOAD_FOLDER"] / image_name

                    try:
                        if image_params["image_effect"] == "kmeans_palette":
                            if not kmeans_available:
                                raise RuntimeError("K-Means palette extraction is unavailable on this environment.")
                            processed_image = kmeans_color_palette(
                                input_path=source_path,
                                output_dir=app.config["GENERATED_FOLDER"],
                                n_colors=image_params["kmeans_colors"],
                            )
                        else:
                            processed_image = apply_image_filter(
                                input_path=source_path,
                                effect=image_params["image_effect"],
                                output_dir=app.config["GENERATED_FOLDER"],
                                rotate_degrees=image_params["rotate_degrees"],
                                pixel_size=image_params["pixel_size"],
                                glitch_shift=image_params["glitch_shift"],
                            )
                        _cleanup_directory_files(
                            app.config["GENERATED_FOLDER"],
                            keep=app.config["MAX_SAVED_GENERATED_FILES"],
                            allowed_extensions=IMAGE_EXTENSIONS | AUDIO_EXTENSIONS,
                        )
                        flash("Image processed successfully.", "success")
                    except Exception:  # pragma: no cover - defensive path
                        app.logger.exception("Image processing failed.")
                        flash("Image processing failed.", "error")
                    finally:
                        _delete_file_if_exists(source_path)
                        _cleanup_directory_files(
                            app.config["UPLOAD_FOLDER"],
                            keep=app.config["MAX_SAVED_UPLOADS"],
                            allowed_extensions=IMAGE_EXTENSIONS | AUDIO_EXTENSIONS | DATA_EXTENSIONS,
                        )

            elif active_panel == "audio":
                audio_params["audio_operation"] = str(request.form.get("audio_operation", "reverse")).strip().lower()
                if audio_params["audio_operation"] not in AUDIO_OPERATIONS:
                    audio_params["audio_operation"] = "reverse"

                audio_params["speed_factor"] = _coerce_float(
                    request.form.get("speed_factor", "1.25"), 1.25, 0.5, 2.5
                )
                audio_params["echo_delay"] = _coerce_int(
                    request.form.get("echo_delay", "180"), 180, 50, 2000
                )
                audio_params["pitch_steps"] = _coerce_int(
                    request.form.get("pitch_steps", "4"), 4, -12, 12
                )
                audio_params["fade_duration"] = _coerce_int(
                    request.form.get("fade_duration", "900"), 900, 100, 6000
                )

                if not audio_available:
                    flash(f"Audio features are disabled. {audio_status['reason']}", "error")
                else:
                    audio_file = request.files.get("audio_file")
                    audio_name = _save_uploaded_file(
                        file_storage=audio_file,
                        target_dir=app.config["UPLOAD_FOLDER"],
                        allowed_extensions=AUDIO_EXTENSIONS,
                    )
                    if not audio_name:
                        flash("Please upload a valid audio file.", "error")
                    else:
                        audio_path = app.config["UPLOAD_FOLDER"] / audio_name
                        merge_path = None
                        merge_file = request.files.get("merge_file")
                        if merge_file and merge_file.filename:
                            merge_name = _save_uploaded_file(
                                file_storage=merge_file,
                                target_dir=app.config["UPLOAD_FOLDER"],
                                allowed_extensions=AUDIO_EXTENSIONS,
                            )
                            if merge_name:
                                merge_path = app.config["UPLOAD_FOLDER"] / merge_name

                        try:
                            processed_audio = process_audio(
                                input_path=audio_path,
                                output_dir=app.config["GENERATED_FOLDER"],
                                operation=audio_params["audio_operation"],
                                speed=audio_params["speed_factor"],
                                echo_delay=audio_params["echo_delay"],
                                merge_path=merge_path,
                                pitch_steps=audio_params["pitch_steps"],
                                fade_duration=audio_params["fade_duration"],
                            )
                            _cleanup_directory_files(
                                app.config["GENERATED_FOLDER"],
                                keep=app.config["MAX_SAVED_GENERATED_FILES"],
                                allowed_extensions=IMAGE_EXTENSIONS | AUDIO_EXTENSIONS,
                            )
                            flash("Audio processed successfully.", "success")
                        except Exception:  # pragma: no cover - defensive path
                            app.logger.exception("Audio processing failed.")
                            flash("Audio processing failed.", "error")
                        finally:
                            _delete_file_if_exists(audio_path)
                            _delete_file_if_exists(merge_path)
                            _cleanup_directory_files(
                                app.config["UPLOAD_FOLDER"],
                                keep=app.config["MAX_SAVED_UPLOADS"],
                                allowed_extensions=IMAGE_EXTENSIONS | AUDIO_EXTENSIONS | DATA_EXTENSIONS,
                            )

        return render_template(
            "media_tools.html",
            processed_image=processed_image,
            processed_audio=processed_audio,
            audio_available=audio_available,
            audio_status=audio_status,
            active_panel=active_panel,
            image_effects=image_effects,
            audio_operations=AUDIO_OPERATIONS,
            image_params=image_params,
            audio_params=audio_params,
            kmeans_available=kmeans_available,
        )

    @app.route("/upload", methods=["GET", "POST"])
    def upload():
        if request.method == "POST":
            incoming_file = request.files.get("file")
            allowed_extensions = IMAGE_EXTENSIONS | AUDIO_EXTENSIONS | DATA_EXTENSIONS
            saved_name = _save_uploaded_file(
                file_storage=incoming_file,
                target_dir=app.config["UPLOAD_FOLDER"],
                allowed_extensions=allowed_extensions,
            )
            if saved_name:
                _cleanup_directory_files(
                    app.config["UPLOAD_FOLDER"],
                    keep=app.config["MAX_SAVED_UPLOADS"],
                    allowed_extensions=IMAGE_EXTENSIONS | AUDIO_EXTENSIONS | DATA_EXTENSIONS,
                )
                flash(f"File uploaded: {saved_name}", "success")
            else:
                flash("Upload failed. Unsupported or missing file.", "error")
        return redirect(url_for("media_tools"))

    @app.route("/download/<folder>/<path:filename>")
    def download_file(folder: str, filename: str):
        directories = {
            "generated": app.config["GENERATED_FOLDER"],
            "uploads": app.config["UPLOAD_FOLDER"],
        }
        if folder not in directories:
            abort(404)

        safe_name = Path(filename).name
        target = directories[folder] / safe_name
        if not target.exists():
            abort(404)

        return send_from_directory(directories[folder], safe_name, as_attachment=True)

    return app


app = create_app()


if __name__ == "__main__":
    debug_enabled = os.getenv("FLASK_DEBUG", "0").strip().lower() in {"1", "true", "yes", "on"}
    app.run(debug=debug_enabled)
