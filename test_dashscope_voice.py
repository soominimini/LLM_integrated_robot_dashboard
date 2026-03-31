#!/usr/bin/env python3
"""
Standalone test: Synthesize speech using DashScope Qwen TTS (Bunny voice)
and play it on the QT robot.

Usage:
    python3 test_dashscope_voice.py
    python3 test_dashscope_voice.py --text "Hello, my name is QT!"
    python3 test_dashscope_voice.py --voice Cherry
    python3 test_dashscope_voice.py --no-robot   # just save the WAV locally
"""

import argparse
import os
import shutil
import subprocess
import requests


# ---------- defaults ----------
DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")
DASHSCOPE_TTS_URL = "https://dashscope-intl.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"

ROBOT_HOST = os.environ.get("ROBOT_HOST", "192.168.100.1")
ROBOT_USER = os.environ.get("ROBOT_USER", "developer")
ROBOT_PASSWORD = os.environ.get("ROBOT_PASSWORD", "")
ROBOT_TMP_AUDIO_DIR = os.environ.get("ROBOT_TMP_AUDIO_DIR", "/tmp/qwen_voices")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULT_TEXT = (
    "Luna is a four-year-old girl who loves picnics. One bright morning, "
    "she wakes up and looks outside. The sun is smiling, and the sky is blue. "
    "Perfect picnic day! Luna says."
)


# ---------- TTS synthesis ----------
def synthesize(text, voice="Bella", api_key=DASHSCOPE_API_KEY):
    """Synthesize speech via DashScope Qwen TTS and return the local WAV path."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    data = {
        "model": "qwen3-tts-flash",
        "input": {
            "text": text,
            "voice": voice,
            "language_type": "English",
        },
    }

    print(f"Synthesizing with voice '{voice}' ...")
    resp = requests.post(DASHSCOPE_TTS_URL, headers=headers, json=data, timeout=90)
    if resp.status_code != 200:
        print(f"API error {resp.status_code}: {resp.text}")
        return None

    result = resp.json()

    # Non-streaming response returns a temporary audio URL
    try:
        audio_url = result["output"]["audio"]["url"]
    except KeyError:
        print(f"Unexpected response format: {result}")
        return None

    # Download the audio file
    print("Downloading audio ...")
    audio_resp = requests.get(audio_url, timeout=60)
    if audio_resp.status_code != 200:
        print(f"Failed to download audio: {audio_resp.status_code}")
        return None

    wav_path = os.path.join(SCRIPT_DIR, f"tts_{voice}.wav")
    with open(wav_path, "wb") as f:
        f.write(audio_resp.content)

    print(f"Audio saved: {wav_path}  ({len(audio_resp.content)} bytes)")
    return wav_path


# ---------- Robot playback ----------
def upload_to_robot(local_path):
    """SCP the file to the robot's tmp audio dir."""
    ssh_opts = ["-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null"]
    filename = os.path.basename(local_path)
    remote = f"{ROBOT_USER}@{ROBOT_HOST}:{ROBOT_TMP_AUDIO_DIR}/{filename}"

    mkdir_cmd = _ssh_cmd(ssh_opts, f"mkdir -p {ROBOT_TMP_AUDIO_DIR}")
    subprocess.run(mkdir_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    scp_cmd = _scp_cmd(ssh_opts, local_path, remote)
    subprocess.run(scp_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"Uploaded to robot: {ROBOT_TMP_AUDIO_DIR}/{filename}")


def play_on_robot(local_path):
    """Play a WAV file on the QT robot via SSH + aplay."""
    ssh_opts = ["-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null"]
    filename = os.path.basename(local_path)
    remote_path = f"{ROBOT_TMP_AUDIO_DIR}/{filename}"
    play_cmd = f"aplay -D plughw:1,0 {remote_path}"

    cmd = _ssh_cmd(ssh_opts, play_cmd, tty=True)
    print("Playing on robot ...")
    subprocess.run(cmd, check=True)
    print("Playback finished.")


def _ssh_cmd(ssh_opts, remote_cmd, tty=False):
    base = ["ssh"] + ssh_opts
    if tty:
        base.append("-t")
    if ROBOT_PASSWORD and shutil.which("sshpass"):
        base = ["sshpass", "-p", ROBOT_PASSWORD] + base
    return base + [f"{ROBOT_USER}@{ROBOT_HOST}", remote_cmd]


def _scp_cmd(ssh_opts, local_file, remote_dest):
    base = ["scp"] + ssh_opts
    if ROBOT_PASSWORD and shutil.which("sshpass"):
        base = ["sshpass", "-p", ROBOT_PASSWORD] + base
    return base + [local_file, remote_dest]


# ---------- main ----------
def main():
    parser = argparse.ArgumentParser(description="Test DashScope Qwen TTS on QT robot")
    parser.add_argument("--text", default=DEFAULT_TEXT,
                        help="Text to speak")
    parser.add_argument("--voice", default="Bunny",
                        help="Voice name (default: Bunny)")
    parser.add_argument("--api-key", default=DASHSCOPE_API_KEY,
                        help="DashScope API key")
    parser.add_argument("--no-robot", action="store_true",
                        help="Skip playing on robot (just save WAV locally)")
    args = parser.parse_args()

    wav_path = synthesize(text=args.text, voice=args.voice, api_key=args.api_key)
    if not wav_path:
        print("Synthesis failed.")
        return 1

    if args.no_robot:
        print("Skipping robot playback (--no-robot).")
        return 0

    try:
        upload_to_robot(wav_path)
        play_on_robot(wav_path)
    except subprocess.CalledProcessError as e:
        print(f"Robot playback failed: {e}")
        print("Tip: set ROBOT_PASSWORD env var, or use --no-robot to just save the file.")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
