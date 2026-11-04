"""Turn an uploaded speech recording into something every browser can play.

Browser MediaRecorder produces WebM/Opus, which Safari can't play and which
often carries no duration metadata even in Chrome. The clinician's review
screen needs the recording to just work, so it's transcoded to MP3 on the way
out (ffmpeg is already in the image for the feature-extraction path).
"""
from __future__ import annotations

import subprocess

PLAYER_MEDIA_TYPE = "audio/mpeg"
PLAYER_FILENAME = "recording.mp3"


def to_playable_mp3(raw: bytes, filename: str) -> bytes:
    """Return the recording as MP3 bytes. Pass through if it already is one."""
    if filename.lower().endswith(".mp3"):
        return raw
    try:
        completed = subprocess.run(
            [
                "ffmpeg", "-loglevel", "error",
                "-i", "pipe:0",
                "-vn",
                "-c:a", "libmp3lame", "-q:a", "5",
                "-f", "mp3", "pipe:1",
            ],
            input=raw,
            capture_output=True,
            timeout=60,
            check=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("Audio conversion is unavailable on this server.") from exc
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as exc:
        raise RuntimeError("The recording could not be converted for playback.") from exc
    return completed.stdout
