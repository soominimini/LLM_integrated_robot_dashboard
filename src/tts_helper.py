#!/usr/bin/env python3

# Copyright (c) 2024 LuxAI S.A.
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import rospy
from qt_robot_interface.srv import behavior_talk_text, speech_config, setting_setVolume
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
            
            # Movement settings
            self.movement_enabled = True
            self.movement_thread = None
            self.stop_movement = False
            
            # Set default language and volume
            self.set_language("en-US")
            self.set_volume(50)
            
        except Exception as e:
            print(f"Warning: Could not initialize TTS services: {e}")
            self.talk_text_service = None
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
                result = self.speech_config_service(language_code, 100, 100)
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
            if self.talk_text_service and text.strip():
                # Estimate speech duration (rough approximation: 0.1 seconds per character)
                estimated_duration = len(text.strip()) * 0.1
                
                # Start movement thread
                self._start_movement_thread(estimated_duration)
                
                # Speak the text
                result = self.talk_text_service(text.strip())
                
                # Stop movement after speech
                self.stop_movement = True
                
                return result
            return False
        except Exception as e:
            print(f"Error speaking text: {e}")
            self.stop_movement = True
            return False
    
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
            # Set the language first
            if not self.set_language(language):
                print(f"Warning: Could not set language to {language}")
            
            # Speak the story with movement
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