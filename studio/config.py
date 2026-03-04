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

