# coding=utf-8
"""
Speak arbitrary text through the QTrobot using your designed Qwen TTS voice.

Usage:
    # Speak a single sentence:
    python speak_on_robot.py "Hello, this is my custom voice."

    # Speak the built-in test set (varied length / mood / punctuation):
    python speak_on_robot.py
"""

import os
import sys
import time
import wave
import base64
import threading

import dashscope
from dashscope.audio.qwen_tts_realtime import (
    QwenTtsRealtime, QwenTtsRealtimeCallback, AudioFormat,
)
from play_on_robot import play_on_robot

# ======= CONFIG (must match the voice you created) =======
dashscope.api_key = os.getenv("DASHSCOPE_API_KEY") or "sk-7a00d999dd654c1cbd82fb3693c5eadc"
VOICE = "qwen-tts-vd-myvoice-voice-20260509042601825-cfa4"
MODEL = "qwen3-tts-vd-realtime-2026-01-15"
URL   = "wss://dashscope-intl.aliyuncs.com/api-ws/v1/realtime"
SAMPLE_RATE = 24000

# Default test set — varied length, intonation, and content
DEFAULT_TEXTS = [
    "Hi there! It's really nice to meet you.",
    "Did you know that I can speak many different sentences with the same voice?"
]

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "speak_runs")
# =========================================================


class CollectingCallback(QwenTtsRealtimeCallback):
    """Accumulates PCM bytes from the streaming session."""
    def __init__(self):
        self.complete_event = threading.Event()
        self.pcm = bytearray()
        self.error = ""

    def on_open(self): pass

    def on_close(self, code, msg):
        if code not in (1000, None):
            self.error = f"closed code={code} msg={msg}"
            self.complete_event.set()

    def on_event(self, response):
        t = response.get("type", "")
        if t == "response.audio.delta":
            self.pcm.extend(base64.b64decode(response["delta"]))
        elif t == "session.finished":
            self.complete_event.set()

    def wait(self, timeout=60):
        return self.complete_event.wait(timeout)


def write_wav(path, pcm_bytes):
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(bytes(pcm_bytes))


def synthesize(text, wav_path):
    """Synthesize one sentence into wav_path. Returns audio duration in seconds."""
    cb = CollectingCallback()
    tts = QwenTtsRealtime(model=MODEL, callback=cb, url=URL)
    tts.connect()
    tts.update_session(
        voice=VOICE,
        response_format=AudioFormat.PCM_24000HZ_MONO_16BIT,
        mode="server_commit",
    )
    tts.append_text(text)
    tts.finish()
    cb.wait(timeout=60)
    if cb.error:
        raise RuntimeError(cb.error)
    write_wav(wav_path, cb.pcm)
    return len(cb.pcm) / (SAMPLE_RATE * 2)  # bytes / (rate * sample_width)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    if len(sys.argv) > 1:
        texts = [" ".join(sys.argv[1:])]
    else:
        texts = DEFAULT_TEXTS
        print(f"No text given; using {len(texts)} built-in test phrases.\n")

    print(f"Voice : {VOICE}")
    print(f"Model : {MODEL}\n")

    for i, text in enumerate(texts, 1):
        print(f"[{i}/{len(texts)}] {text!r}")
        wav_path = os.path.join(OUT_DIR, f"speak_{i:02d}.wav")
        try:
            duration = synthesize(text, wav_path)
            print(f"  synth ok ({duration:.2f}s) -> {os.path.basename(wav_path)}")
            play_on_robot(wav_path)
            print(f"  played on robot")
            time.sleep(duration + 0.3)  # let playback finish before next
        except Exception as e:
            print(f"  FAILED: {e}")
        print()


if __name__ == "__main__":
    main()
