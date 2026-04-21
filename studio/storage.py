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
