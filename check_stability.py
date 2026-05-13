# coding=utf-8
"""
Stability test for a designed Qwen TTS voice.

Synthesizes the same text N times in independent sessions, then reports how
much the output varies across runs (duration, loudness, latency). Each run is
saved as a WAV so you can A/B them by ear.

A "stable" voice should produce nearly-identical duration and RMS across runs;
small variation in first-audio-delay is normal (network jitter).
"""

import os
import time
import wave
import array
import base64
import math
import threading
import statistics
from dataclasses import dataclass

import dashscope
from dashscope.audio.qwen_tts_realtime import (
    QwenTtsRealtime, QwenTtsRealtimeCallback, AudioFormat,
)

# ======= CONFIG =======
dashscope.api_key = os.getenv("DASHSCOPE_API_KEY") or "sk-7a00d999dd654c1cbd82fb3693c5eadc"

VOICE = "qwen-tts-vd-myvoice-voice-20260509042141880-20bc"
MODEL = "qwen3-tts-vd-realtime-2026-01-15"
URL   = "wss://dashscope-intl.aliyuncs.com/api-ws/v1/realtime"

TEXT = "Hello! I'm your robot assistant. How can I help you today?"
N_RUNS = 5
SAMPLE_RATE = 24000
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stability_runs")
PLAY_ON_ROBOT = True  # set False to only save WAVs without playing through QTrobot speakers
# ======================

if PLAY_ON_ROBOT:
    from play_on_robot import play_on_robot


@dataclass
class RunResult:
    idx: int
    ok: bool
    duration_s: float
    first_audio_delay_s: float
    rms: float
    peak: int
    byte_count: int
    wav_path: str
    error: str = ""


class CollectingCallback(QwenTtsRealtimeCallback):
    """Accumulates PCM bytes instead of playing them."""

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


def write_wav(path, pcm_bytes, rate=SAMPLE_RATE):
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)  # paInt16
        w.setframerate(rate)
        w.writeframes(bytes(pcm_bytes))


def compute_metrics(pcm_bytes):
    if not pcm_bytes:
        return 0.0, 0.0, 0
    samples = array.array("h")
    samples.frombytes(bytes(pcm_bytes))
    n = len(samples)
    duration = n / SAMPLE_RATE
    sumsq = sum(s * s for s in samples)
    rms = math.sqrt(sumsq / n) if n else 0.0
    peak = max(abs(s) for s in samples)
    return duration, rms, peak


def run_once(idx):
    cb = CollectingCallback()
    tts = QwenTtsRealtime(model=MODEL, callback=cb, url=URL)
    try:
        tts.connect()
        tts.update_session(
            voice=VOICE,
            response_format=AudioFormat.PCM_24000HZ_MONO_16BIT,
            mode="server_commit",
        )
        tts.append_text(TEXT)
        tts.finish()
        cb.wait(timeout=60)
    except Exception as e:
        return RunResult(idx, False, 0, 0, 0, 0, 0, "", error=str(e))

    if cb.error:
        return RunResult(idx, False, 0, 0, 0, 0, 0, "", error=cb.error)

    duration, rms, peak = compute_metrics(cb.pcm)
    wav_path = os.path.join(OUT_DIR, f"run_{idx:02d}.wav")
    write_wav(wav_path, cb.pcm)
    delay = tts.get_first_audio_delay() or 0.0

    return RunResult(idx, True, duration, delay, rms, peak, len(cb.pcm), wav_path)


def coef_var(values):
    if len(values) < 2 or statistics.mean(values) == 0:
        return 0.0
    return statistics.stdev(values) / statistics.mean(values) * 100.0


def summarize(results):
    ok = [r for r in results if r.ok]
    print("\n" + "=" * 78)
    print(f"Stability summary  ({len(ok)}/{len(results)} runs successful)")
    print("=" * 78)
    if len(ok) < 2:
        print("Need >=2 successful runs to compute stability.")
        for r in results:
            if not r.ok:
                print(f"  run {r.idx}: FAILED — {r.error}")
        return

    def stat(name, vals, unit=""):
        cv = coef_var(vals)
        print(f"  {name:<22} mean={statistics.mean(vals):.3f}{unit}  "
              f"stdev={statistics.stdev(vals):.3f}{unit}  "
              f"min={min(vals):.3f}  max={max(vals):.3f}  CV={cv:.2f}%")

    durations = [r.duration_s for r in ok]
    delays    = [r.first_audio_delay_s for r in ok]
    rmses     = [r.rms for r in ok]
    peaks     = [r.peak for r in ok]

    stat("duration",        durations, " s")
    stat("first_audio_delay", delays,  " s")
    stat("rms (loudness)",  rmses)
    stat("peak (sample)",   peaks)

    # Verdict — duration drift is the strongest indicator the model rerolled prosody.
    cv_dur = coef_var(durations)
    cv_rms = coef_var(rmses)
    print()
    if cv_dur < 3 and cv_rms < 5:
        verdict = "STABLE — duration & loudness vary <3% / <5%"
    elif cv_dur < 8 and cv_rms < 15:
        verdict = "ACCEPTABLE — minor prosody drift, acceptable for most uses"
    else:
        verdict = "UNSTABLE — large drift; consider re-designing the voice prompt"
    print(f"  Verdict: {verdict}")
    print(f"  Files saved in: {OUT_DIR}")
    print(f"  A/B compare:    aplay {OUT_DIR}/run_01.wav ; aplay {OUT_DIR}/run_02.wav")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Voice : {VOICE}")
    print(f"Model : {MODEL}")
    print(f"Text  : {TEXT!r}")
    print(f"Runs  : {N_RUNS}\n")

    results = []
    for i in range(1, N_RUNS + 1):
        t0 = time.time()
        print(f"[run {i}/{N_RUNS}] synthesizing ...", end=" ", flush=True)
        r = run_once(i)
        elapsed = time.time() - t0
        results.append(r)
        if r.ok:
            print(f"ok  {r.duration_s:.2f}s audio, "
                  f"first_audio_delay={r.first_audio_delay_s:.2f}s, "
                  f"wall={elapsed:.1f}s -> {os.path.basename(r.wav_path)}")
            if PLAY_ON_ROBOT:
                try:
                    play_on_robot(r.wav_path)
                    time.sleep(r.duration_s + 0.3)  # let playback finish before next run
                except Exception as e:
                    print(f"  [robot playback failed: {e}]")
        else:
            print(f"FAILED — {r.error}")
        time.sleep(0.5)  # tiny gap between sessions

    summarize(results)


if __name__ == "__main__":
    main()
