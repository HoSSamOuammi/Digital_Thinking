from __future__ import annotations

from modules.audio_processing import get_audio_status
from modules.generative_art import create_generative_art
from studio import create_app
from studio.config import debug_is_enabled

app = create_app()


if __name__ == "__main__":
    app.run(debug=debug_is_enabled())
