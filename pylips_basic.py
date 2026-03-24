import os
import time
import base64
import requests
import subprocess
from pylips.speech import RobotFace


class QwenRobotDesigner:
    def __init__(self, head_ip="192.168.100.1", body_ip="192.168.100.2", user="developer"):
        self.head_ip = head_ip
        self.body_ip = body_ip
        self.user = user
        self.api_key = "sk-7a00d999dd654c1cbd82fb3693c5eadc"
        self.ssh_prefix = f"ssh -o StrictHostKeyChecking=no {self.user}@{self.head_ip}"
        self.face = None
        self.face_started = False

        # Prepare remote directories
        subprocess.run(f"{self.ssh_prefix} 'mkdir -p /tmp/qwen_therapy'", shell=True)
        subprocess.run(f"{self.ssh_prefix} 'mkdir -p /tmp/qwen_voices'", shell=True)

    def set_volume(self, percentage):
        """Adjusts physical speaker volume on Card 1 (Headphones/Internal)"""
        print(f"Calibrating hardware volume to {percentage}%...")
        # Using -c 1 to target the verified hardware path
        volume_cmd = f"{self.ssh_prefix} 'amixer -c 1 sset Headphone {percentage}% unmute'"
        subprocess.run(volume_cmd, shell=True)

    def launch_face(self):
        """Forces Display :0 and Chromium Face (run once)."""
        cmd = (f"{self.ssh_prefix} \"export DISPLAY=:0 && pkill chromium-browser; "
               f"nohup chromium-browser --kiosk --autoplay-policy=no-user-gesture-required "
               f"--disable-gpu --app=http://{self.body_ip}:8000/face > /dev/null 2>&1 &\"")
        subprocess.run(cmd, shell=True)

    def start_face_once(self):
        """Start face display and connect RobotFace once."""
        if self.face_started:
            return
        self.launch_face()
        self.face = RobotFace()
        time.sleep(1)
        self.face_started = True

    def create_and_speak(self, prompt, preview_text, tag="designed_voice"):
        """Calls Qwen to design voice, then plays on Robot Head"""
        if not self.api_key:
            raise RuntimeError("Set QWEN_API_KEY before running this script.")
        url = "https://dashscope-intl.aliyuncs.com/api/v1/services/audio/tts/customization"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

        data = {
            "model": "qwen-voice-design",
            "input": {
                "action": "create",
                "target_model": "qwen3-tts-vd-realtime-2026-01-15",
                "voice_prompt": prompt,
                "preview_text": preview_text,
                "preferred_name":"soomin",
                "language": "en"
            },
            "parameters": {"sample_rate": 24000, "response_format": "wav"}
        }

        print(f"Designing voice for: {preview_text}...")
        response = requests.post(url, headers=headers, json=data, timeout=60)

        if response.status_code == 200:
            result = response.json()
            audio_bytes = base64.b64decode(result["output"]["preview_audio"]["data"])

            # Save locally
            local_file = f"pylips_phrases/{tag}.wav"
            with open(local_file, 'wb') as f:
                f.write(audio_bytes)

            # Sync to Head (QTRP)
            remote_path = f"/tmp/qwen_voices/{tag}.wav"
            subprocess.run(f"scp -q {local_file} {self.user}@{self.head_ip}:{remote_path}", shell=True)

            # Trigger Lips (using preview text)
            if self.face:
                self.face.say(preview_text)

            # Play Sound on Hardware (Card 1, Device 0)
            print("Playing designed voice on head...")
            subprocess.run(f"{self.ssh_prefix} 'aplay -D plughw:1,0 {remote_path}'", shell=True)
        else:
            print(f"API Error: {response.text}")


if __name__ == "__main__":
    bot = QwenRobotDesigner()
    # Start face once (call this at the beginning of your session)
    bot.start_face_once()

    # Custom voice prompt for your therapy context
    my_prompt = "A cute child’s voice, a 7-year-old girl, with a slightly childish tone and a very slow speaking rate, suitable for children audiobook narration."
    my_text = "Hello there! I am so excited to play and learn with you today."

    # Generate audio + play; face stays running
    bot.create_and_speak(my_prompt, my_text)