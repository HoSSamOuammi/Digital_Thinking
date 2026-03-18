from __future__ import annotations

import os
import secrets
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_FOLDER = BASE_DIR / "static" / "uploads"
GENERATED_FOLDER = BASE_DIR / "static" / "generated"
PREVIEW_FOLDER = GENERATED_FOLDER / "previews"
ADMINS_FOLDER = BASE_DIR / "static" / "Admins"

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}
AUDIO_EXTENSIONS = {".wav", ".mp3", ".ogg", ".flac", ".m4a"}
DATA_EXTENSIONS = {".csv"}

GALLERY_IMAGE_PAGE_SIZE = 12
GALLERY_AUDIO_PAGE_SIZE = 8
PREVIEW_CACHE_LIMIT = 24
UPLOAD_CACHE_LIMIT = 40
GENERATED_CACHE_LIMIT = 120

TEAM_MEMBERS = (
    {
        "slug": "aya",
        "name": "Aya EL Amrani",
        "email": "ElAamrani.aya@etu.uae.ac.ma",
        "role": "Interface et expérience utilisateur",
    },
    {
        "slug": "khadija",
        "name": "Khadija Baskar",
        "email": "Baskar.Khadija@etu.uae.ac.ma",
        "role": "Contenus, formulaires et traduction",
    },
    {
        "slug": "hossam",
        "name": "Hossam OUammi",
        "email": "Ouammi.hossam@etu.uae.ac.ma",
        "role": "Intégration Flask et médias",
    },
    {
        "slug": "abdo",
        "name": "Abderrahmane El Garti",
        "email": "ElGarti.abderrahmane@etu.uae.ac.ma",
        "role": "Tests, galerie et documentation",
    },
)


def is_truthy_env(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def resolve_secret_key() -> tuple[str, bool]:
    configured_key = os.getenv("FLASK_SECRET_KEY", "").strip()
    if configured_key:
        return configured_key, False
    return secrets.token_hex(32), True


def debug_is_enabled() -> bool:
    return os.getenv("FLASK_DEBUG", "0").strip().lower() in {"1", "true", "yes", "on"}


def configure_app(app) -> bool:
    secret_key, generated_secret = resolve_secret_key()

    app.config["SECRET_KEY"] = secret_key
    app.config["MAX_CONTENT_LENGTH"] = 30 * 1024 * 1024
    app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
    app.config["GENERATED_FOLDER"] = GENERATED_FOLDER
    app.config["PREVIEW_FOLDER"] = PREVIEW_FOLDER
    app.config["ADMINS_FOLDER"] = ADMINS_FOLDER
    app.config["MAX_SAVED_UPLOADS"] = UPLOAD_CACHE_LIMIT
    app.config["MAX_SAVED_GENERATED_FILES"] = GENERATED_CACHE_LIMIT
    app.config["SESSION_COOKIE_HTTPONLY"] = True
    app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
    app.config["SESSION_COOKIE_SECURE"] = is_truthy_env("FLASK_SESSION_SECURE")

    return generated_secret


def ensure_project_folders(app) -> None:
    app.config["UPLOAD_FOLDER"].mkdir(parents=True, exist_ok=True)
    app.config["GENERATED_FOLDER"].mkdir(parents=True, exist_ok=True)
    app.config["PREVIEW_FOLDER"].mkdir(parents=True, exist_ok=True)
    app.config["ADMINS_FOLDER"].mkdir(parents=True, exist_ok=True)
