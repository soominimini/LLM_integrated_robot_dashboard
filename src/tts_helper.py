#!/usr/bin/env python3

# Copyright (c) 2024 LuxAI S.A.
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import os
import sys
import shutil
import subprocess
import shlex
import rospy
from qt_robot_interface.srv import behavior_talk_text, behavior_talk_audio, speech_config, setting_setVolume
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
            # Initialize ROS node if not already done
            if not rospy.core.is_initialized():
                rospy.init_node('tts_helper', anonymous=True)
            
            # Create service proxies
            self.talk_text_service = rospy.ServiceProxy('/qt_robot/behavior/talkText', behavior_talk_text)
            self.talk_audio_service = rospy.ServiceProxy('/qt_robot/behavior/talkAudio', behavior_talk_audio)
            self.speech_config_service = rospy.ServiceProxy('/qt_robot/speech/config', speech_config)
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
            
            # Movement settings (disabled by default; use HumanTracking for motion)
            self.movement_enabled = False
            self.movement_thread = None
            self.stop_movement = False
            # Speaking state
            self._is_speaking = False

            # Engine selection
            # 'qt' (default) uses robot's built-in TTS; 'polly' uses AWS Polly with audio playback
            self.engine = (os.environ.get('TTS_ENGINE') or 'qt').strip().lower()
            self.aws_voice = os.environ.get('POLLY_VOICE', 'Justin')
            self.polly_rate = os.environ.get('POLLY_RATE')  # e.g., 'slow', 'x-slow', '85%'
            self.polly_volume = os.environ.get('POLLY_VOLUME')  # e.g., 'loud', 'x-loud', '+6dB'
            self.robot_host = os.environ.get('ROBOT_HOST', '192.168.100.1')
            self.robot_user = os.environ.get('ROBOT_USER', 'developer')
            self.robot_qt_audio_dir = os.environ.get('ROBOT_QT_AUDIO_DIR', '/home/qtrobot/robot/data/audios/')
            
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
                # Set language with default pitch and speed
                result = self.speech_config_service(language_code, 100, 80)
                return result
            return False
        except Exception as e:
            print(f"Error setting language: {e}")
            return False
    
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
            # Set language for QT engine only; Polly voice is selected via env
            if self.engine != 'polly':
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
        # File name
        ts = int(time.time())
        filename = f"polly_{ts}.mp3"

        # Generate locally via Polly
        # Apply SSML prosody rate if requested
        rate = (self.polly_rate or '').strip()
        volume = (self.polly_volume or '').strip()
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
                        attrs = []
                        if rate:
                            attrs.append(f"rate=\"{rate}\"")
                        if volume:
                            attrs.append(f"volume=\"{volume}\"")
                        attr_str = ' '.join(attrs) if attrs else ''
                        wrapped = f"<speak>\n  <prosody {attr_str}>{inner}</prosody>\n</speak>"
                        text = wrapped
                    else:
                        attrs = []
                        if rate:
                            attrs.append(f"rate=\"{rate}\"")
                        if volume:
                            attrs.append(f"volume=\"{volume}\"")
                        attr_str = ' '.join(attrs) if attrs else ''
                        text = f"<speak><prosody {attr_str}>{text}</prosody></speak>"
                else:
                    # No SSML: create SSML wrapper
                    attrs = []
                    if rate:
                        attrs.append(f"rate=\"{rate}\"")
                    if volume:
                        attrs.append(f"volume=\"{volume}\"")
                    attr_str = ' '.join(attrs) if attrs else ''
                    text = f"<speak><prosody {attr_str}>{text}</prosody></speak>"
            except Exception:
                # If anything goes wrong, fall back to original text
                pass

        audio_path = generate_polly_audio(text, self.aws_voice, filename)
        if not audio_path:
            return False

        # Upload to QTRP host
        if not self._upload_to_robot(audio_path):
            return False

        # Copy into qtrobot audio dir and chown
        if not self._copy_to_qtrobot_user(filename):
            return False

        # Trigger playback on robot via ROS service
        try:
            if self.talk_audio_service:
                # Movement: approximate duration if possible (not precise for MP3). Use 0.08s per char.
                estimated_duration = max(2.0, len(text) * 0.08)
                self._start_movement_thread(estimated_duration)
                res = self.talk_audio_service(filename, "")
                self.stop_movement = True
                return bool(res)
            print("talkAudio service not available")
            return False
        except Exception as e:
            print(f"Error triggering talkAudio: {e}")
            return False

    def _upload_to_robot(self, local_file: str) -> bool:
        try:
            ssh_opts = ['-o', 'StrictHostKeyChecking=no', '-o', 'UserKnownHostsFile=/dev/null']
            password = os.environ.get('ROBOT_PASSWORD')
            if password and shutil.which('sshpass'):
                cmd = ['sshpass', '-p', password, 'scp'] + ssh_opts + [local_file, f"{self.robot_user}@{self.robot_host}:~/"]
            else:
                if password and not shutil.which('sshpass'):
                    print("Hint: install sshpass (e.g., sudo apt-get install -y sshpass) for non-interactive auth.")
                cmd = ['scp'] + ssh_opts + [local_file, f"{self.robot_user}@{self.robot_host}:~/"]
            subprocess.run(cmd, check=True)
            return True
        except subprocess.CalledProcessError as e:
            print(f"Upload failed: {e}")
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