from __future__ import annotations

from pathlib import Path
from typing import Optional
from uuid import uuid4

try:
    from pydub import AudioSegment
    from pydub.utils import which
except Exception:  # pragma: no cover - optional dependency safeguard
    AudioSegment = None
    which = None

AUDIO_OPERATIONS: dict[str, str] = {
    "reverse": "Reverse the clip from end to start.",
    "speed": "Speed up or slow down the waveform.",
    "echo": "Layer delayed repetitions over the original clip.",
    "merge": "Append a second clip with a short crossfade.",
    "pitch": "Shift the perceived pitch by semitone steps.",
    "fade": "Apply fade in and fade out envelopes.",
}
MAX_AUDIO_DURATION_MS = 5 * 60 * 1000


def get_audio_status() -> dict[str, str | bool]:
    if AudioSegment is None or which is None:
        return {
            "available": False,
            "reason": "PyDub is not installed.",
        }

    converter_name = Path(getattr(AudioSegment, "converter", "ffmpeg")).name or "ffmpeg"
    converter_path = which(converter_name) or which("ffmpeg") or which("avconv")
    if not converter_path:
        return {
            "available": False,
            "reason": "ffmpeg is not available in PATH.",
        }

    return {
        "available": True,
        "reason": "Audio processing is ready.",
        "converter": converter_path,
    }


def is_audio_available() -> bool:
    return bool(get_audio_status()["available"])


def _change_speed(clip, speed: float):
    speed = max(0.5, min(2.5, speed))
    adjusted = clip._spawn(clip.raw_data, overrides={"frame_rate": int(clip.frame_rate * speed)})
    return adjusted.set_frame_rate(clip.frame_rate)


def _add_echo(clip, delay_ms: int = 180):
    delay_ms = max(50, min(2500, int(delay_ms)))
    output = clip

    for repeat, attenuation in ((1, 7), (2, 12), (3, 18)):
        delayed = AudioSegment.silent(duration=delay_ms * repeat) + (clip - attenuation)
        output = output.overlay(delayed)

    return output


def _merge_clips(primary, secondary):
    crossfade_ms = min(220, len(primary) // 8, len(secondary) // 8)
    return primary.append(secondary, crossfade=max(0, crossfade_ms))


def _pitch_shift(clip, semitones: int):
    semitones = max(-12, min(12, int(semitones)))
    ratio = 2 ** (semitones / 12)
    shifted = clip._spawn(clip.raw_data, overrides={"frame_rate": int(clip.frame_rate * ratio)})
    return shifted.set_frame_rate(clip.frame_rate)


def _fade_clip(clip, duration_ms: int):
    duration_ms = max(100, min(6000, int(duration_ms)))
    return clip.fade_in(duration_ms).fade_out(duration_ms)


def process_audio(
    input_path: Path,
    output_dir: Path,
    operation: str = "reverse",
    speed: float = 1.25,
    echo_delay: int = 180,
    merge_path: Optional[Path] = None,
    pitch_steps: int = 4,
    fade_duration: int = 900,
) -> str:
    """Apply an audio transformation and return generated filename."""

    if not is_audio_available():
        reason = str(get_audio_status()["reason"])
        raise RuntimeError(f"Audio processing is unavailable. {reason}")

    output_dir.mkdir(parents=True, exist_ok=True)
    clip = AudioSegment.from_file(input_path)
    if len(clip) > MAX_AUDIO_DURATION_MS:
        raise ValueError("Audio clips longer than 5 minutes are not supported.")
    normalized_operation = operation.lower()

    if normalized_operation == "speed":
        processed = _change_speed(clip, speed)
    elif normalized_operation == "echo":
        processed = _add_echo(clip, echo_delay)
    elif normalized_operation == "reverse":
        processed = clip.reverse()
    elif normalized_operation == "merge":
        if merge_path is None:
            raise ValueError("Merge operation requires a second audio clip.")
        secondary = AudioSegment.from_file(merge_path)
        if len(secondary) > MAX_AUDIO_DURATION_MS:
            raise ValueError("Merge clips longer than 5 minutes are not supported.")
        processed = _merge_clips(clip, secondary)
    elif normalized_operation == "pitch":
        processed = _pitch_shift(clip, pitch_steps)
    elif normalized_operation == "fade":
        processed = _fade_clip(clip, fade_duration)
    else:
        raise ValueError(f"Unsupported audio operation: {operation}")

    filename = f"audio_{normalized_operation}_{uuid4().hex[:12]}.wav"
    output_path = output_dir / filename
    processed.export(output_path, format="wav")
    return filename
