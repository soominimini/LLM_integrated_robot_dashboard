#!/usr/bin/env python3.9

# Copyright (c) 2024 LuxAI S.A.
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import os
import sys
import shutil
import subprocess
import shlex
import re
import html
import rospy
from typing import Optional
from qt_robot_interface.srv import behavior_talk_text, behavior_talk_audio, speech_config, setting_setVolume, setting_upload
import math
import threading
import time
import random

# Import kinematic interface for movement
try:
    from kinematics.kinematic_interface import QTrobotKinematicInterface
    KINEMATICS_AVAILABLE = True
except ImportError:
    print("Warning: Kinematic interface not available. Movement features will be disabled.")
    KINEMATICS_AVAILABLE = False

# Optional pylips face sync for lipsync
try:
    from pylips.speech import RobotFace
    PYLIPS_AVAILABLE = True
except Exception:
    PYLIPS_AVAILABLE = False

class TTSHelper:
    """
    Helper class for Text-to-Speech functionality with movement
    """
    
    # Joint limits for safe movement
    JOINT_LIMITS = {
        'head': {
            'HeadYaw': {'min': -90.0, 'max': 90.0},
            'HeadPitch': {'min': -15.0, 'max': 25.0}
        },
        'right_arm': {
            'RightShoulderPitch': {'min': -140.0, 'max': 140.0},
            'RightShoulderRoll': {'min': -75.0, 'max': 7.0},
            'RightElbowRoll': {'min': -90.0, 'max': -7.0}
        },
        'left_arm': {
            'LeftShoulderPitch': {'min': -140.0, 'max': 140.0},
            'LeftShoulderRoll': {'min': -75.0, 'max': 7.0},
            'LeftElbowRoll': {'min': -90.0, 'max': -7.0}
        }
    }
    
    def __init__(self):
        """Initialize TTS services and movement interface"""
        try:
            # Engine selection
            # 'qwen' (default) uses Qwen TTS realtime with a custom voice;
            # 'qt' uses the robot's built-in TTS; 'polly' uses AWS Polly with audio playback
            self.engine = (os.environ.get('TTS_ENGINE') or 'qwen').strip().lower()
            self.aws_voice = os.environ.get('POLLY_VOICE', 'Justin')
            self.polly_rate = os.environ.get('POLLY_RATE')  # e.g., 'slow', 'x-slow', '85%'
            self.polly_volume = os.environ.get('POLLY_VOLUME')  # e.g., 'loud', 'x-loud', '+6dB'
            # Qwen TTS realtime config (must match the voice you created)
            self.qwen_api_key = os.environ.get('DASHSCOPE_API_KEY') or "sk-7a00d999dd654c1cbd82fb3693c5eadc"
            self.qwen_voice = os.environ.get('QWEN_VOICE', 'qwen-tts-vd-myvoice-voice-20260509042601825-cfa4')
            self.qwen_model = os.environ.get('QWEN_MODEL', 'qwen3-tts-vd-realtime-2026-01-15')
            self.qwen_url = os.environ.get('QWEN_URL', 'wss://dashscope-intl.aliyuncs.com/api-ws/v1/realtime')
            self.qwen_sample_rate = int(os.environ.get('QWEN_SAMPLE_RATE', '24000'))
            self.qwen_lipsync = (os.environ.get('QWEN_LIPSYNC', '1').strip().lower() in ('1', 'true', 'yes'))
            self.robot_host = os.environ.get('ROBOT_HOST', '192.168.100.1')
            self.robot_user = os.environ.get('ROBOT_USER', 'developer')
            self.robot_qt_audio_dir = os.environ.get('ROBOT_QT_AUDIO_DIR', '/home/qtrobot/robot/data/audios/')
            self.robot_tmp_audio_dir = os.environ.get('ROBOT_TMP_AUDIO_DIR', '/tmp/qwen_voices')
            self.polly_lipsync = (os.environ.get('POLLY_LIPSYNC', '1').strip().lower() in ('1', 'true', 'yes'))
            self.face = None
            self.user_data_dir = os.environ.get(
                'USER_DATA_DIR',
                os.path.join(os.path.dirname(__file__), 'user_data')
            )
            self.current_user = None

            # Initialize ROS node (needed for some dependencies even in polly mode)
            if not rospy.core.is_initialized():
                rospy.init_node('tts_helper', anonymous=True)

            # Create service proxies for QT engine
            if self.engine == 'qt':
                self.talk_text_service = rospy.ServiceProxy('/qt_robot/behavior/talkText', behavior_talk_text)
                self.talk_audio_service = rospy.ServiceProxy('/qt_robot/behavior/talkAudio', behavior_talk_audio)
                self.speech_config_service = rospy.ServiceProxy('/qt_robot/speech/config', speech_config)
                self.volume_service = rospy.ServiceProxy('/qt_robot/setting/setVolume', setting_setVolume)
            else:
                self.talk_text_service = None
                self.talk_audio_service = None
                self.speech_config_service = None
                self.volume_service = None

            # Qwen engine needs talkAudio (for visemes/lip-sync) and uploadBase64 (to push WAVs to the robot)
            self.upload_service = None
            if self.engine == 'qwen':
                self.talk_audio_service = rospy.ServiceProxy('/qt_robot/behavior/talkAudio', behavior_talk_audio)
                self.upload_service = rospy.ServiceProxy('/qt_robot/setting/uploadBase64', setting_upload)
                self.volume_service = rospy.ServiceProxy('/qt_robot/setting/setVolume', setting_setVolume)

            # Initialize kinematic interface for movement
            self.kinematics = None
            if KINEMATICS_AVAILABLE:
                try:
                    self.kinematics = QTrobotKinematicInterface()
                    print("Kinematic interface initialized successfully")
                except Exception as e:
                    print(f"Warning: Could not initialize kinematic interface: {e}")
                    self.kinematics = None

            if (self.engine == 'polly' and self.polly_lipsync and PYLIPS_AVAILABLE) or \
               (self.engine == 'qwen' and self.qwen_lipsync and PYLIPS_AVAILABLE):
                try:
                    self.face = RobotFace()
                except Exception as e:
                    print(f"Warning: Could not initialize pylips face sync: {e}")
                    self.face = None
            
            # Movement settings (disabled by default; use HumanTracking for motion)
            self.movement_enabled = False
            self.movement_thread = None
            self.stop_movement = False
            # Speaking state
            self._is_speaking = False

            # Set default language and volume
            self.set_language("en-US")
            self.set_volume(50)
            
        except Exception as e:
            print(f"Warning: Could not initialize TTS services: {e}")
            self.talk_text_service = None
            self.talk_audio_service = None
            self.speech_config_service = None
            self.volume_service = None
            self.kinematics = None
    
    def set_language(self, language_code: str) -> bool:
        """
        Set the TTS language

        Args:
            language_code: Language code (e.g., 'en-US', 'fr-FR')

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if self.speech_config_service:
                # Wait for service with a short timeout to avoid hanging
                self.speech_config_service.wait_for_service(timeout=3.0)
                result = self.speech_config_service(language_code, 100, 80)
                return result
            return False
        except rospy.ROSException:
            print(f"Warning: speech config service not available (timeout). Skipping language set.")
            return False
        except Exception as e:
            print(f"Error setting language: {e}")
            return False

    def set_current_user(self, username: Optional[str]):
        """Set the active user for per-user output paths."""
        self.current_user = username.strip().lower() if username else None

    def _polly_output_dir(self) -> str:
        username = self.current_user or os.environ.get('TTS_USERNAME') or 'guest'
        safe = re.sub(r'[^a-zA-Z0-9_\-]', '_', username)
        return os.path.join(self.user_data_dir, safe, 'polly')

    def _qwen_output_dir(self) -> str:
        username = self.current_user or os.environ.get('TTS_USERNAME') or 'guest'
        safe = re.sub(r'[^a-zA-Z0-9_\-]', '_', username)
        return os.path.join(self.user_data_dir, safe, 'qwen')
    
    def set_volume(self, level: int) -> bool:
        """
        Set the robot's speaker volume
        
        Args:
            level: Volume level (0-100)
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if self.volume_service:
                # Convert volume level to robot's internal scale
                robot_volume = int(24 * math.log(max(level, 1)) - 10)
                result = self.volume_service(robot_volume)
                return result
            return False
        except Exception as e:
            print(f"Error setting volume: {e}")
            return False

    def set_hardware_volume(self, percent: int) -> bool:
        """
        Set the head computer's ALSA Headphone mixer level (0-100) via SSH.

        Real-time and engine-agnostic. /qt_robot/setting/setVolume only affects
        the QT TTS engine (talkText), so file-based playback (talkAudio used by
        the qwen/polly engines) needs the hardware mixer instead. Same trick
        pylips_basic.py uses.
        """
        try:
            percent = max(0, min(100, int(percent)))
        except Exception:
            return False
        try:
            ssh_opts = ['-o', 'StrictHostKeyChecking=no', '-o', 'UserKnownHostsFile=/dev/null']
            password = os.environ.get('ROBOT_PASSWORD')
            remote_cmd = f"amixer -c 1 sset Headphone {percent}% unmute"
            if password and shutil.which('sshpass'):
                cmd = ['sshpass', '-p', password, 'ssh'] + ssh_opts + [f"{self.robot_user}@{self.robot_host}", remote_cmd]
            else:
                if password and not shutil.which('sshpass'):
                    print("Hint: install sshpass (e.g., sudo apt-get install -y sshpass) for non-interactive auth.")
                cmd = ['ssh'] + ssh_opts + [f"{self.robot_user}@{self.robot_host}", remote_cmd]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        except subprocess.CalledProcessError as e:
            print(f"Hardware volume change failed: {e}")
            return False

    def set_polly_volume(self, volume_db: str) -> bool:
        """
        Set Polly SSML volume (e.g., "+6dB", "-3dB").
        """
        if not volume_db:
            return False
        volume_db = volume_db.strip()
        if not re.match(r'^[+-]?\d+(\.\d+)?dB$', volume_db):
            return False
        self.polly_volume = volume_db
        return True
    
    def enable_movement(self, enabled: bool = True):
        """
        Enable or disable movement during speech
        
        Args:
            enabled: Whether to enable movement
        """
        self.movement_enabled = enabled
        if not enabled and self.movement_thread and self.movement_thread.is_alive():
            self.stop_movement = True

    
    def _clamp_joint_value(self, joint_name: str, value: float) -> float:
        """
        Clamp a joint value within its safe limits
        
        Args:
            joint_name: Name of the joint
            value: Current value
            
        Returns:
            float: Clamped value within limits
        """
        # Find which part this joint belongs to
        for part, joints in self.JOINT_LIMITS.items():
            if joint_name in joints:
                limits = joints[joint_name]
                return max(limits['min'], min(limits['max'], value))
        
        # If joint not found, return original value
        return value
    
    def _clamp_head_position(self, yaw: float, pitch: float) -> tuple:
        """
        Clamp head position within safe limits
        
        Args:
            yaw: Yaw angle
            pitch: Pitch angle
            
        Returns:
            tuple: (clamped_yaw, clamped_pitch)
        """
        clamped_yaw = self._clamp_joint_value('HeadYaw', yaw)
        clamped_pitch = self._clamp_joint_value('HeadPitch', pitch)
        return clamped_yaw, clamped_pitch
    
    def _clamp_arm_position(self, part: str, positions: list) -> list:
        """
        Clamp arm position within safe limits
        
        Args:
            part: 'right_arm' or 'left_arm'
            positions: List of [shoulder_pitch, shoulder_roll, elbow_roll]
            
        Returns:
            list: Clamped positions
        """
        if part not in ['right_arm', 'left_arm']:
            return positions
        
        joint_names = list(self.JOINT_LIMITS[part].keys())
        clamped_positions = []
        
        for i, position in enumerate(positions):
            if i < len(joint_names):
                clamped_positions.append(self._clamp_joint_value(joint_names[i], position))
            else:
                clamped_positions.append(position)
        
        return clamped_positions
    
    def _gentle_head_movement(self, duration: float):
        """
        Perform a single gentle head movement at the beginning of speech
        
        Args:
            duration: Duration parameter (not used for single movement)
        """
        if not self.kinematics or not self.movement_enabled:
            return
        
        try:
            # Get current head position
            current_pos = self.kinematics.get_head_pos()
            
            # Create a single movement with larger range
            yaw_offset = random.uniform(-8, 8)  # Larger yaw movement
            pitch_offset = random.uniform(-5, 5)  # Larger pitch movement
            
            new_yaw = current_pos[0] + yaw_offset
            new_pitch = current_pos[1] + pitch_offset
            
            # Clamp values within safe limits
            clamped_yaw, clamped_pitch = self._clamp_head_position(new_yaw, new_pitch)
            
            # Move head to new position
            self.kinematics._move_part('head', [clamped_yaw, clamped_pitch], sync=False)
            
            print(f"Head movement: Yaw {current_pos[0]:.1f}° → {clamped_yaw:.1f}° (+{yaw_offset:.1f}°), Pitch {current_pos[1]:.1f}° → {clamped_pitch:.1f}° (+{pitch_offset:.1f}°)")
            
        except Exception as e:
            print(f"Error during head movement: {e}")
    
    def _gentle_arm_movement(self, duration: float):
        """
        Perform a single gentle arm movement at the beginning of speech
        
        Args:
            duration: Duration parameter (not used for single movement)
        """
        if not self.kinematics or not self.movement_enabled:
            return
        
        try:
            # Get current arm positions
            self.kinematics.joints_state_lock.acquire()
            state = self.kinematics.joints_state
            rsp = state.position[state.name.index("RightShoulderPitch")]
            rsr = state.position[state.name.index("RightShoulderRoll")]
            rer = state.position[state.name.index("RightElbowRoll")]
            lsp = state.position[state.name.index("LeftShoulderPitch")]
            lsr = state.position[state.name.index("LeftShoulderRoll")]
            ler = state.position[state.name.index("LeftElbowRoll")]
            self.kinematics.joints_state_lock.release()
            
            # Create single movements with larger range
            right_offset = [random.uniform(-6, 6), random.uniform(-4, 4), random.uniform(-4, 4)]
            left_offset = [random.uniform(-6, 6), random.uniform(-4, 4), random.uniform(-4, 4)]
            
            new_right = [rsp + right_offset[0], rsr + right_offset[1], rer + right_offset[2]]
            new_left = [lsp + left_offset[0], lsr + left_offset[1], ler + left_offset[2]]
            
            # Clamp values within safe limits
            clamped_right = self._clamp_arm_position('right_arm', new_right)
            clamped_left = self._clamp_arm_position('left_arm', new_left)
            
            # Move arms to new positions
            self.kinematics._move_part('right_arm', clamped_right, sync=False)
            self.kinematics._move_part('left_arm', clamped_left, sync=False)
            
            print(f"Arm movement: Right [+{right_offset[0]:.1f}°, +{right_offset[1]:.1f}°, +{right_offset[2]:.1f}°], Left [+{left_offset[0]:.1f}°, +{left_offset[1]:.1f}°, +{left_offset[2]:.1f}°]")
            
        except Exception as e:
            print(f"Error during arm movement: {e}")
    
    def _start_movement_thread(self, duration: float):
        """
        Start single movement at the beginning of speech
        
        Args:
            duration: Estimated duration of speech (not used for single movement)
        """
        if not self.movement_enabled or not self.kinematics:
            return
        
        # Stop any existing movement
        self.stop_movement = True
        if self.movement_thread and self.movement_thread.is_alive():
            self.movement_thread.join(timeout=1.0)
        
        # Start new movement thread for single movement
        self.stop_movement = False
        self.movement_thread = threading.Thread(target=self._gentle_head_movement, args=(duration,))
        self.movement_thread.daemon = True
        self.movement_thread.start()
        
        # Start arm movement in a separate thread
        arm_thread = threading.Thread(target=self._gentle_arm_movement, args=(duration,))
        arm_thread.daemon = True
        arm_thread.start()
    
    def speak(self, text: str) -> bool:
        """
        Make the robot speak the given text with movement
        
        Args:
            text: Text to speak
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self._is_speaking = True
            clean_text = (text or '').strip()
            if not clean_text:
                return False

            # If using AWS Polly path
            print("self.engine : ",self.engine )
            if self.engine == 'polly':
                return self._speak_with_polly(clean_text)

            # If using Qwen TTS realtime path
            if self.engine == 'qwen':
                return self._speak_with_qwen(clean_text)

            # Default: QT built-in TTS
            if self.talk_text_service:
                estimated_duration = len(clean_text) * 0.1
                self._start_movement_thread(estimated_duration)
                result = self.talk_text_service(clean_text)
                self.stop_movement = True
                return result
            return False
        except Exception as e:
            print(f"Error speaking text: {e}")
            self.stop_movement = True
            return False
        finally:
            # Ensure speaking flag resets even on errors
            self._is_speaking = False
    
    def speak_story(self, story_text: str, language: str = "en-US") -> bool:
        """
        Speak a story with proper language setting and movement
        
        Args:
            story_text: The story text to speak
            language: Language code for the story
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Set language for QT engine only; Polly/Qwen voices are selected via env
            if self.engine == 'qt':
                if not self.set_language(language):
                    print(f"Warning: Could not set language to {language}")
            return self.speak(story_text)
            
        except Exception as e:
            print(f"Error speaking story: {e}")
            return False
    
    def is_available(self) -> bool:
        """
        Check if TTS services are available

        Returns:
            bool: True if TTS is available, False otherwise
        """
        if self.engine in ('polly', 'qwen'):
            return True
        return self.talk_text_service is not None

    def is_speaking(self) -> bool:
        """Return True while the robot is currently speaking."""
        return getattr(self, '_is_speaking', False)
    
    def is_movement_available(self) -> bool:
        """
        Check if movement capabilities are available
        
        Returns:
            bool: True if movement is available, False otherwise
        """
        return self.kinematics is not None
    
    def get_joint_limits(self) -> dict:
        """
        Get the current joint limits for safe movement
        
        Returns:
            dict: Joint limits for all parts
        """
        return self.JOINT_LIMITS.copy()
    
    def get_safe_movement_ranges(self) -> dict:
        """
        Get safe movement ranges for gentle motion
        
        Returns:
            dict: Safe movement ranges for each part
        """
        return {
            'head': {
                'yaw_range': (-8, 8),      # Degrees (increased from ±4)
                'pitch_range': (-5, 5),    # Degrees (increased from ±2)
                'center_return_pitch': (-2, 2)  # Degrees (increased from ±1)
            },
            'arms': {
                'shoulder_pitch_range': (-6, 6),  # Degrees (increased from ±3)
                'shoulder_roll_range': (-4, 4),   # Degrees (increased from ±2)
                'elbow_roll_range': (-4, 4)       # Degrees (increased from ±2)
            }
        }
    
    def get_current_head_position(self) -> tuple:
        """
        Get current head position
        
        Returns:
            tuple: (yaw, pitch) in degrees, or (None, None) if not available
        """
        if not self.kinematics:
            return None, None
        
        try:
            return self.kinematics.get_head_pos()
        except Exception as e:
            print(f"Error getting head position: {e}")
            return None, None 

    # Internal helper for Qwen TTS realtime playback
    # Uses /qt_robot/behavior/talkAudio so the robot's built-in visemes drive
    # mouth movement, exactly like talkText does for the QT engine.
    def _speak_with_qwen(self, text: str) -> bool:
        try:
            import wave
            import base64
            import threading
            import dashscope
            from dashscope.audio.qwen_tts_realtime import (
                QwenTtsRealtime, QwenTtsRealtimeCallback, AudioFormat,
            )
        except Exception as e:
            print(f"Qwen TTS unavailable (missing dashscope?): {e}")
            return False

        # Strip any SSML tags — Qwen takes plain text
        plain_text = re.sub(r'<[^>]+>', ' ', text)
        plain_text = ' '.join(plain_text.split()).strip()
        if not plain_text:
            return False

        class _CollectingCallback(QwenTtsRealtimeCallback):
            def __init__(self):
                self.complete_event = threading.Event()
                self.pcm = bytearray()
                self.error = ""

            def on_open(self):
                pass

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

        ts = int(time.time())
        out_dir = self._qwen_output_dir()
        os.makedirs(out_dir, exist_ok=True)
        wav_filename = f"qwen_{ts}.wav"
        wav_path = os.path.join(out_dir, wav_filename)

        try:
            dashscope.api_key = self.qwen_api_key
            cb = _CollectingCallback()
            tts = QwenTtsRealtime(model=self.qwen_model, callback=cb, url=self.qwen_url)
            tts.connect()
            tts.update_session(
                voice=self.qwen_voice,
                response_format=AudioFormat.PCM_24000HZ_MONO_16BIT,
                mode="server_commit",
            )
            tts.append_text(plain_text)
            tts.finish()
            cb.wait(timeout=60)
            if cb.error:
                print(f"Qwen TTS error: {cb.error}")
                return False
            with wave.open(wav_path, "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(self.qwen_sample_rate)
                w.writeframes(bytes(cb.pcm))
            duration = len(cb.pcm) / (self.qwen_sample_rate * 2)
        except Exception as e:
            print(f"Qwen TTS synthesis failed: {e}")
            return False

        # Push WAV to robot's standard audio dir, then trigger talkAudio for QT visemes
        if not self.upload_service or not self.talk_audio_service:
            print("Qwen path: ROS upload/talkAudio services not available")
            return False

        try:
            with open(wav_path, "rb") as f:
                encoded = base64.b64encode(f.read()).decode("ascii")
            remote_path = os.path.join(self.robot_qt_audio_dir, wav_filename)
            self.upload_service.wait_for_service(timeout=5.0)
            up = self.upload_service(data=encoded, filepath=remote_path, permission="644", append=False)
            if not up.status:
                print(f"Qwen path: uploadBase64 returned status=False for {remote_path}")
                return False
        except Exception as e:
            print(f"Qwen path: upload failed: {e}")
            return False

        # Movement during speech
        try:
            self._start_movement_thread(duration)
        except Exception:
            pass

        try:
            # talkAudio takes the basename without extension (same convention as play_on_robot.py)
            name_no_ext = os.path.splitext(wav_filename)[0]
            self.talk_audio_service.wait_for_service(timeout=5.0)
            resp = self.talk_audio_service(name_no_ext, "")
            return bool(getattr(resp, "status", resp))
        except Exception as e:
            print(f"Qwen path: talkAudio failed: {e}")
            return False
        finally:
            self.stop_movement = True

    # Internal helpers for AWS Polly playback
    def _speak_with_polly(self, text: str) -> bool:
        try:
            # Lazy import to avoid hard dependency; add repo root if needed
            try:
                from tts.local_polly_generator import generate_polly_audio
            except Exception:
                repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
                if repo_root not in sys.path:
                    sys.path.insert(0, repo_root)
                from tts.local_polly_generator import generate_polly_audio
        except Exception as e:
            print(f"Polly path unavailable: {e}")
            return False

        # Determine if text is SSML
        is_ssml = ('<' in text and '>' in text)
        # File name (save under user_data/<user>/polly)
        ts = int(time.time())
        polly_dir = self._polly_output_dir()
        os.makedirs(polly_dir, exist_ok=True)
        filename = f"polly_{ts}.mp3"
        output_path = os.path.join(polly_dir, filename)

        # Generate locally via Polly
        # Apply SSML prosody rate if requested
        rate = (self.polly_rate or '').strip()
        volume = (self.polly_volume or '').strip()
        def _maybe_escape_ssml(s: str) -> str:
            return html.escape(s, quote=True) if ('<' not in s and '>' not in s) else s
        if rate or volume:
            try:
                lowered = text.lower()
                if '<speak' in lowered and '</speak>' in lowered:
                    # Insert prosody inside existing <speak> ... </speak>
                    # Find tags conservatively
                    start = lowered.find('<speak')
                    start_close = lowered.find('>', start)
                    end = lowered.rfind('</speak>')
                    if start != -1 and start_close != -1 and end != -1 and end > start_close:
                        inner = text[start_close+1:end]
                        inner = _maybe_escape_ssml(inner)
                        attrs = []
                        if rate:
                            attrs.append(f"rate=\"{rate}\"")
                        if volume:
                            attrs.append(f"volume=\"{volume}\"")
                        attr_str = ' '.join(attrs) if attrs else ''
                        wrapped = f"<speak>\n  <prosody {attr_str}>{inner}</prosody>\n</speak>"
                        text = wrapped
                    else:
                        safe_text = _maybe_escape_ssml(text)
                        attrs = []
                        if rate:
                            attrs.append(f"rate=\"{rate}\"")
                        if volume:
                            attrs.append(f"volume=\"{volume}\"")
                        attr_str = ' '.join(attrs) if attrs else ''
                        text = f"<speak><prosody {attr_str}>{safe_text}</prosody></speak>"
                else:
                    # No SSML: create SSML wrapper
                    safe_text = _maybe_escape_ssml(text)
                    attrs = []
                    if rate:
                        attrs.append(f"rate=\"{rate}\"")
                    if volume:
                        attrs.append(f"volume=\"{volume}\"")
                    attr_str = ' '.join(attrs) if attrs else ''
                    text = f"<speak><prosody {attr_str}>{safe_text}</prosody></speak>"
            except Exception:
                # If anything goes wrong, fall back to original text
                pass

        audio_path = generate_polly_audio(text, self.aws_voice, output_path)
        if not audio_path and is_ssml:
            try:
                # Retry with plain text if SSML was invalid
                plain_text = re.sub(r'<[^>]+>', ' ', text)
                plain_text = ' '.join(plain_text.split()).strip()
                if plain_text:
                    audio_path = generate_polly_audio(
                        plain_text,
                        self.aws_voice,
                        output_path,
                        force_text=True,
                    )
            except Exception:
                pass
        if not audio_path:
            return False

        # Convert MP3 to WAV for reliable playback with aplay
        play_path = audio_path
        play_filename = os.path.basename(audio_path)
        if audio_path.lower().endswith('.mp3'):
            wav_filename = f"polly_{ts}.wav"
            wav_path = os.path.join(os.path.dirname(audio_path), wav_filename)
            try:
                subprocess.run(
                    [
                        'ffmpeg', '-y',
                        '-i', audio_path,
                        '-ar', '24000',
                        '-ac', '1',
                        '-sample_fmt', 's16',
                        wav_path,
                    ],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                play_path = wav_path
                play_filename = wav_filename
            except Exception as e:
                print(f"Warning: MP3->WAV conversion failed, using MP3 playback: {e}")

        # Upload to QTRP host
        if not self._upload_to_robot(play_path):
            return False

        # Trigger lipsync using plain text (strip SSML tags)
        if self.face:
            try:
                plain_text = re.sub(r'<[^>]+>', ' ', text)
                plain_text = ' '.join(plain_text.split()).strip()
                if plain_text:
                    self.face.say(plain_text)
            except Exception as e:
                print(f"Warning: pylips face sync failed: {e}")

        # Play on robot using the same mechanism as pylips_basic.py
        return self._play_on_robot(play_filename)

    def _upload_to_robot(self, local_file: str) -> bool:
        try:
            ssh_opts = ['-o', 'StrictHostKeyChecking=no', '-o', 'UserKnownHostsFile=/dev/null']
            password = os.environ.get('ROBOT_PASSWORD')
            remote_dir = self.robot_tmp_audio_dir
            remote_path = f"{self.robot_user}@{self.robot_host}:{remote_dir}/{os.path.basename(local_file)}"
            mkdir_cmd = ['ssh'] + ssh_opts + [f"{self.robot_user}@{self.robot_host}", f"mkdir -p {remote_dir}"]
            if password and shutil.which('sshpass'):
                mkdir_cmd = ['sshpass', '-p', password, 'ssh'] + ssh_opts + [f"{self.robot_user}@{self.robot_host}", f"mkdir -p {remote_dir}"]
                cmd = ['sshpass', '-p', password, 'scp'] + ssh_opts + [local_file, remote_path]
            else:
                if password and not shutil.which('sshpass'):
                    print("Hint: install sshpass (e.g., sudo apt-get install -y sshpass) for non-interactive auth.")
                cmd = ['scp'] + ssh_opts + [local_file, remote_path]
            subprocess.run(mkdir_cmd, check=True)
            subprocess.run(cmd, check=True)
            return True
        except subprocess.CalledProcessError as e:
            print(f"Upload failed: {e}")
            return False

    def _play_on_robot(self, filename: str) -> bool:
        try:
            ssh_opts = ['-o', 'StrictHostKeyChecking=no', '-o', 'UserKnownHostsFile=/dev/null']
            remote_path = f"{self.robot_tmp_audio_dir}/{filename}"
            if filename.lower().endswith('.mp3'):
                remote_cmd = f"export DISPLAY=:0 && ffplay -nodisp -autoexit {remote_path}"
            else:
                remote_cmd = f"aplay -D plughw:1,0 {remote_path}"
            password = os.environ.get('ROBOT_PASSWORD')
            if password and shutil.which('sshpass'):
                cmd = ['sshpass', '-p', password, 'ssh'] + ssh_opts + ['-t', f"{self.robot_user}@{self.robot_host}", remote_cmd]
            else:
                if password and not shutil.which('sshpass'):
                    print("Hint: install sshpass (e.g., sudo apt-get install -y sshpass) for non-interactive auth.")
                cmd = ['ssh'] + ssh_opts + ['-t', f"{self.robot_user}@{self.robot_host}", remote_cmd]
            subprocess.run(cmd, check=True)
            return True
        except subprocess.CalledProcessError as e:
            print(f"Remote playback failed: {e}")
            return False

    def _copy_to_qtrobot_user(self, filename: str) -> bool:
        try:
            ssh_opts = ['-o', 'StrictHostKeyChecking=no', '-o', 'UserKnownHostsFile=/dev/null']
            # Allow passing sudo password; fallback to ROBOT_PASSWORD
            sudo_pw = os.environ.get('ROBOT_SUDO_PASSWORD') or os.environ.get('ROBOT_PASSWORD')
            if sudo_pw:
                # Use sudo -S to read password from stdin; suppress prompt with -p ''
                remote_cmd = (
                    f"echo {shlex.quote(sudo_pw)} | sudo -S -p '' cp ~/{filename} {self.robot_qt_audio_dir} && "
                    f"echo {shlex.quote(sudo_pw)} | sudo -S -p '' chown qtrobot:qtrobot {os.path.join(self.robot_qt_audio_dir, filename)}"
                )
            else:
                remote_cmd = (
                    f"sudo cp ~/{filename} {self.robot_qt_audio_dir} && "
                    f"sudo chown qtrobot:qtrobot {os.path.join(self.robot_qt_audio_dir, filename)}"
                )

            password = os.environ.get('ROBOT_PASSWORD')
            if password and shutil.which('sshpass'):
                cmd = ['sshpass', '-p', password, 'ssh'] + ssh_opts + ['-t', f"{self.robot_user}@{self.robot_host}", remote_cmd]
            else:
                if password and not shutil.which('sshpass'):
                    print("Hint: install sshpass (e.g., sudo apt-get install -y sshpass) for non-interactive auth.")
                cmd = ['ssh'] + ssh_opts + ['-t', f"{self.robot_user}@{self.robot_host}", remote_cmd]

            subprocess.run(cmd, check=True)
            return True
        except subprocess.CalledProcessError as e:
            print(f"Copy to qtrobot user failed: {e}")
            return False