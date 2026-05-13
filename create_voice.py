# coding=utf-8
"""
One-time voice designer for Qwen TTS.
Run this ONCE to create a custom voice on your DashScope account.
After it succeeds, qwen.py can reuse the same voice name forever (no extra cost).

Cost: ~$0.20 per successful create call. Edit VOICE_PROMPT and re-run if you
want a different voice — each iteration is a separate $0.20 charge.
"""

import os
import json
import base64
import requests

# ======= CONFIG — edit these, then run =======
API_KEY = os.getenv("DASHSCOPE_API_KEY") or "sk-7a00d999dd654c1cbd82fb3693c5eadc"
ENDPOINT = "https://dashscope-intl.aliyuncs.com/api/v1/services/audio/tts/customization"

# Must match the model used in qwen.py (line 93). Realtime variant.
TARGET_MODEL = "qwen3-tts-vd-realtime-2026-01-15"

# Must match voice="..." in qwen.py (line 101). Default keeps qwen.py unchanged.
PREFERRED_NAME = "myvoice"

# Describe the voice you want. Be specific and multi-dimensional:
# gender + age + pitch + pace + emotion + timbre + purpose.
VOICE_PROMPT = (
    "A cute child's voice, around 8 years old,"
    "with a slightly childish tone,  slow-paced, clear pronunciation, "
    "suitable for animation character voice-overs."
)

# Short sample text used to generate the preview audio.
PREVIEW_TEXT = "Hello! I'm your robot assistant. How can I help you today?"
LANGUAGE = "en"  # must match PREVIEW_TEXT language; supported: zh, en, de, it, pt, es, ja, ko, fr, ru
# =============================================


def post(payload):
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    r = requests.post(ENDPOINT, headers=headers, json=payload, timeout=60)
    try:
        return r.status_code, r.json()
    except Exception:
        return r.status_code, {"raw": r.text}


def query_voice(name):
    return post({
        "model": "qwen-voice-design",
        "input": {"action": "query", "voice": name},
    })


def create_voice():
    return post({
        "model": "qwen-voice-design",
        "input": {
            "action": "create",
            "target_model": TARGET_MODEL,
            "voice_prompt": VOICE_PROMPT,
            "preview_text": PREVIEW_TEXT,
            "preferred_name": PREFERRED_NAME,
            "language": LANGUAGE,
        },
        "parameters": {"sample_rate": 24000, "response_format": "wav"},
    })


def main():
    print(f"[1/2] Checking if voice '{PREFERRED_NAME}' already exists ...")
    status, body = query_voice(PREFERRED_NAME)
    if status == 200 and body.get("output", {}).get("voice"):
        out = body["output"]
        print(f"  Voice already exists. target_model={out.get('target_model')}")
        print(f"  No charge. Use voice=\"{PREFERRED_NAME}\" in qwen.py.")
        return

    print(f"  Not found. Creating new voice (this charges ~$0.20) ...")
    print(f"[2/2] Creating voice '{PREFERRED_NAME}' on {TARGET_MODEL}")
    status, body = create_voice()
    if status != 200:
        print(f"  FAILED. HTTP {status}")
        print(json.dumps(body, indent=2, ensure_ascii=False))
        return

    out = body.get("output", {})
    voice_name = out.get("voice")
    preview = out.get("preview_audio", {})
    audio_b64 = preview.get("data")

    if audio_b64:
        preview_path = os.path.join(os.path.dirname(__file__), f"preview_{PREFERRED_NAME}.wav")
        with open(preview_path, "wb") as f:
            f.write(base64.b64decode(audio_b64))
        print(f"  Preview saved: {preview_path}  (play it: aplay '{preview_path}')")

    print(f"\nDone. Voice name: {voice_name}")
    print(f"Use this in qwen.py:  voice=\"{voice_name}\"")
    print(f"Synthesis model must stay: model=\"{out.get('target_model')}\"")


if __name__ == "__main__":
    main()
