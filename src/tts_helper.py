#!/usr/bin/env python3

# Copyright (c) 2024 LuxAI S.A.
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import rospy
from qt_robot_interface.srv import behavior_talk_text, speech_config, setting_setVolume
import math

class TTSHelper:
    """
    Helper class for Text-to-Speech functionality
    """
    
    def __init__(self):
        """Initialize TTS services"""
        try:
            # Initialize ROS node if not already done
            if not rospy.core.is_initialized():
                rospy.init_node('tts_helper', anonymous=True)
            
            # Create service proxies
            self.talk_text_service = rospy.ServiceProxy('/qt_robot/behavior/talkText', behavior_talk_text)
            self.speech_config_service = rospy.ServiceProxy('/qt_robot/speech/config', speech_config)
            self.volume_service = rospy.ServiceProxy('/qt_robot/setting/setVolume', setting_setVolume)
            
            # Set default language and volume
            self.set_language("en-US")
            self.set_volume(50)
            
        except Exception as e:
            print(f"Warning: Could not initialize TTS services: {e}")
            self.talk_text_service = None
            self.speech_config_service = None
            self.volume_service = None
    
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
    
    def speak(self, text: str) -> bool:
        """
        Make the robot speak the given text
        
        Args:
            text: Text to speak
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if self.talk_text_service and text.strip():
                result = self.talk_text_service(text.strip())
                return result
            return False
        except Exception as e:
            print(f"Error speaking text: {e}")
            return False
    
    def speak_story(self, story_text: str, language: str = "en-US") -> bool:
        """
        Speak a story with proper language setting
        
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
            
            # Speak the story
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