"""Adjust WAV playback speed using ffmpeg's ``atempo`` filter.

Pitch is preserved — only the timing changes. The ffmpeg dependency is the
same one the project already uses for the Polly MP3 → WAV conversion, so
this introduces no new external requirement.

The standalone reference is ``Qwen_tts/adjust_speed.py``; this module is
the importable equivalent used by the live TTS pipeline.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def _atempo_chain(speed: float) -> str:
    """Build an ffmpeg ``-filter:a`` chain for ``speed``.

    Modern ffmpeg's atempo accepts 0.5..100.0 in one pass, but chaining keeps
    each step within 0.5..2.0 where the filter sounds best.
    """
    if speed <= 0:
        raise ValueError("speed must be > 0")

    factors = []
    remaining = speed
    while remaining < 0.5:
        factors.append(0.5)
        remaining /= 0.5
    while remaining > 2.0:
        factors.append(2.0)
        remaining /= 2.0
    factors.append(remaining)
    return ",".join(f"atempo={f:.6f}" for f in factors)


def adjust_wav_speed(in_path, out_path, speed: float) -> bool:
    """Re-encode ``in_path`` to ``out_path`` at ``speed``x with preserved pitch.

    Returns True on success. Returns False (and logs) if ffmpeg is missing
    or the conversion fails — the caller is expected to fall back to the
    original WAV so a misconfigured environment doesn't break TTS entirely.
    """
    if speed <= 0:
        raise ValueError("speed must be > 0")
    in_p = Path(in_path)
    out_p = Path(out_path)

    if shutil.which("ffmpeg") is None:
        print("[wav_speed] ffmpeg not on PATH — skipping speed adjustment")
        return False

    cmd = [
        "ffmpeg", "-y",
        "-i", str(in_p),
        "-filter:a", _atempo_chain(speed),
        "-vn",
        str(out_p),
    ]
    try:
        subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"[wav_speed] ffmpeg failed (exit {e.returncode}); leaving original WAV")
        return False
