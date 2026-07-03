from __future__ import annotations

from pathlib import Path

from mutagen import File


def audio_duration_seconds(path: str | Path) -> float:
    audio = File(path)
    if audio is None or audio.info is None:
        raise ValueError(f"无法读取音频时长: {path}")
    return float(audio.info.length)


def is_audio_file(path: Path) -> bool:
    return path.suffix.lower() in {".mp3", ".wav", ".flac", ".m4a", ".ogg"}
