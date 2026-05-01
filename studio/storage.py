from __future__ import annotations

from pathlib import Path
from typing import Optional
from uuid import uuid4

from werkzeug.utils import secure_filename


def save_uploaded_file(file_storage, target_dir: Path, allowed_extensions: set[str]) -> Optional[str]:
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


def delete_file_if_exists(path: Optional[Path]) -> None:
    if not path:
        return
    try:
        path.unlink(missing_ok=True)
    except OSError:
        return


def list_generated_files(directory: Path, image_extensions: set[str], audio_extensions: set[str]) -> tuple[list[str], list[str]]:
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
        if extension in image_extensions:
            images.append(name)
        elif extension in audio_extensions:
            audios.append(name)
    return images, audios


def paginate_items(items: list[str], page: int, per_page: int) -> dict[str, int | bool | list[str]]:
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


def cleanup_preview_files(directory: Path, keep: int) -> None:
    preview_files = [path for path in directory.glob("*.png") if path.is_file()]
    if len(preview_files) <= keep:
        return
    preview_files.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    for path in preview_files[keep:]:
        delete_file_if_exists(path)


def cleanup_directory_files(
    directory: Path,
    *,
    keep: int,
    allowed_extensions: Optional[set[str]] = None,
