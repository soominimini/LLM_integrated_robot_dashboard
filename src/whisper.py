import os
import sys
import time
import wave
import queue
import tempfile
from collections import deque

import numpy as np
import rospy
from audio_common_msgs.msg import AudioData
from openai import OpenAI

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("Set OPENAI_API_KEY before running this script.")

client = OpenAI(api_key=api_key)

sample_rate = 16000
channels = 1
topic_name = "/qt_respeaker_app/channel0"

audio_queue = queue.Queue(maxsize=1000)


def _callback_audio_stream(msg):
    try:
        audio_queue.put_nowait(bytes(msg.data))
    except queue.Full:
        pass


silence_threshold = 0.012  # RMS threshold; tune if needed
silence_duration = 1.0     # seconds of silence to stop
max_record_seconds = 20.0  # safety cap
pre_roll_seconds = 0.5     # keep a little audio before speech starts
stream_interval = float(os.getenv("WHISPER_STREAM_INTERVAL", "2.0"))

# Approximate 512-sample frames -> 16000/512 ~= 31.25 frames/sec
pre_roll_frames = max(1, int(pre_roll_seconds * (sample_rate / 512)))

print(f"Listening on {topic_name} (start on speech, stop on silence)...", file=sys.stderr)
rospy.init_node("whisper_stt_capture", anonymous=True)
rospy.Subscriber(topic_name, AudioData, _callback_audio_stream, queue_size=10)

frames = []
pre_roll = deque(maxlen=pre_roll_frames)
speech_started = False
last_voice_time = None
record_start_time = time.time()
last_partial_time = 0.0
latest_partial = ""

while True:
    if (time.time() - record_start_time) > max_record_seconds:
        break
    try:
        chunk = audio_queue.get(timeout=0.5)
    except queue.Empty:
        continue

    pre_roll.append(chunk)
    audio_int16 = np.frombuffer(chunk, dtype=np.int16)
    if audio_int16.size == 0:
        continue
    rms = np.sqrt(np.mean((audio_int16.astype("float32") / 32768.0) ** 2))

    if rms >= silence_threshold:
        if not speech_started:
            speech_started = True
            last_voice_time = time.time()
            frames.extend(pre_roll)
        frames.append(chunk)
        last_voice_time = time.time()
    elif speech_started:
        frames.append(chunk)
        if last_voice_time and (time.time() - last_voice_time) >= silence_duration:
            break

    # Emit partial transcription while recording
    if speech_started and (time.time() - last_partial_time) >= stream_interval:
        last_partial_time = time.time()
        pcm_bytes = b"".join(frames)
        if pcm_bytes:
            try:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp_wav:
                    with wave.open(tmp_wav.name, "wb") as wav_file:
                        wav_file.setnchannels(channels)
                        wav_file.setsampwidth(2)
                        wav_file.setframerate(sample_rate)
                        wav_file.writeframes(pcm_bytes)
                    with open(tmp_wav.name, "rb") as audio_file:
                        transcription = client.audio.transcriptions.create(
                            model="gpt-4o-transcribe",
                            file=audio_file,
                        )
                partial_text = (transcription.text or "").strip()
                if partial_text and partial_text != latest_partial:
                    latest_partial = partial_text
                    print(f"PARTIAL:{partial_text}")
                    sys.stdout.flush()
            except Exception as e:
                print(f"[Whisper] partial error: {e}", file=sys.stderr)

if not frames:
    # Avoid hard failure; let caller decide how to handle empty recognition
    print("No audio captured. Check mic topic or threshold.", file=sys.stderr)
    sys.exit(0)

pcm_bytes = b"".join(frames)
audio_int16 = np.frombuffer(pcm_bytes, dtype=np.int16)

# Basic cleanup and normalization to improve transcription quality.
audio_float32 = audio_int16.astype("float32") / 32768.0
audio_float32 = audio_float32 - np.mean(audio_float32)
peak = np.max(np.abs(audio_float32))
if peak > 0:
    audio_float32 = np.clip(audio_float32 * (0.9 / peak), -1.0, 1.0)
audio_int16 = np.int16(audio_float32 * 32767)

wav_path = "mic_recording.wav"
with wave.open(wav_path, "wb") as wav_file:
    wav_file.setnchannels(channels)
    wav_file.setsampwidth(2)  # 16-bit PCM
    wav_file.setframerate(sample_rate)
    wav_file.writeframes(audio_int16.tobytes())



with open(wav_path, "rb") as audio_file:
    transcription = client.audio.transcriptions.create(
        model="gpt-4o-transcribe",
        file=audio_file,
    )

final_text = (transcription.text or "").strip()
if final_text:
    print(f"FINAL:{final_text}")