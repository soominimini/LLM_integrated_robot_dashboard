# coding=utf-8
# Installation instructions for pyaudio:
# APPLE Mac OS X
#   brew install portaudio
#   pip install pyaudio
# Debian/Ubuntu
#   sudo apt-get install python-pyaudio python3-pyaudio
#   or
#   pip install pyaudio
# CentOS
#   sudo yum install -y portaudio portaudio-devel && pip install pyaudio
# Microsoft Windows
#   python -m pip install pyaudio

import pyaudio
import os
import base64
import threading
import time
import dashscope  # DashScope Python SDK version 1.23.9 or later is required
from dashscope.audio.qwen_tts_realtime import QwenTtsRealtime, QwenTtsRealtimeCallback, AudioFormat

# ======= Constant Configuration =======
TEXT_TO_SYNTHESIZE = [
    'Right? I just love this kind of supermarket,',
    'especially during the New Year.',
    'Going to the supermarket',
    'just makes me feel',
    'super, super happy!',
    'I want to buy so many things!'
]


def init_dashscope_api_key():
    """
    Initializes the DashScope SDK API key
    """
    # API keys differ between Singapore and Beijing regions. Get your API key: https://www.alibabacloud.com/help/zh/model-studio/get-api-key
    # If you haven't set an environment variable, replace the line below with: dashscope.api_key = "sk-xxx"
    dashscope.api_key = 'sk-7a00d999dd654c1cbd82fb3693c5eadc'


# ======= Callback Class =======
class MyCallback(QwenTtsRealtimeCallback):
    """
    Custom TTS streaming callback
    """

    def __init__(self):
        self.complete_event = threading.Event()
        self._player = pyaudio.PyAudio()
        self._stream = self._player.open(
            format=pyaudio.paInt16, channels=1, rate=24000, output=True
        )

    def on_open(self) -> None:
        print('[TTS] Connection established')

    def on_close(self, close_status_code, close_msg) -> None:
        self._stream.stop_stream()
        self._stream.close()
        self._player.terminate()
        print(f'[TTS] Connection closed, code={close_status_code}, msg={close_msg}')

    def on_event(self, response: dict) -> None:
        try:
            event_type = response.get('type', '')
            if event_type == 'session.created':
                print(f'[TTS] Session started: {response["session"]["id"]}')
            elif event_type == 'response.audio.delta':
                audio_data = base64.b64decode(response['delta'])
                self._stream.write(audio_data)
            elif event_type == 'response.done':
                print(f'[TTS] Response complete, Response ID: {qwen_tts_realtime.get_last_response_id()}')
            elif event_type == 'session.finished':
                print('[TTS] Session finished')
                self.complete_event.set()
        except Exception as e:
            print(f'[Error] Exception processing callback event: {e}')

    def wait_for_finished(self):
        self.complete_event.wait()


# ======= Main Execution Logic =======
if __name__ == '__main__':
    init_dashscope_api_key()
    print('[System] Initializing Qwen TTS Realtime ...')

    callback = MyCallback()
    qwen_tts_realtime = QwenTtsRealtime(
        # Voice design and speech synthesis must use the same model
        model="qwen3-tts-vd-realtime-2026-01-15",
        callback=callback,
        # URL for Singapore region. For Beijing region, use: wss://dashscope.aliyuncs.com/api-ws/v1/realtime
        url='wss://dashscope-intl.aliyuncs.com/api-ws/v1/realtime'
    )
    qwen_tts_realtime.connect()

    qwen_tts_realtime.update_session(
        voice="qwen-tts-vd-myvoice-voice-20260509040505054-1b8a",  # Custom voice from create_voice.py
        response_format=AudioFormat.PCM_24000HZ_MONO_16BIT,
        mode='server_commit'
    )

    for text_chunk in TEXT_TO_SYNTHESIZE:
        print(f'[Sending text]: {text_chunk}')
        qwen_tts_realtime.append_text(text_chunk)
        time.sleep(0.1)

    qwen_tts_realtime.finish()
    callback.wait_for_finished()

    print(f'[Metric] session_id={qwen_tts_realtime.get_session_id()}, '
          f'first_audio_delay={qwen_tts_realtime.get_first_audio_delay()}s')