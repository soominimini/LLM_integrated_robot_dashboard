#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test script specifically for head movement improvements
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from tts_helper import TTSHelper
import time

def test_head_movement():
    """Test the improved head movement functionality"""
    print("Testing Improved Head Movement")
    print("=" * 50)
    
    # Initialize TTS helper
    print("Initializing TTS helper...")
    tts = TTSHelper()
    
    # Check if movement is available
    if not tts.is_movement_available():
        print("❌ Movement not available")
        return False
    
    print("✅ Movement available")
    
    # Get current head position
    print("\nCurrent head position:")
    yaw, pitch = tts.get_current_head_position()
    if yaw is not None and pitch is not None:
        print(f"  Yaw: {yaw:.1f}°")
        print(f"  Pitch: {pitch:.1f}°")
    else:
        print("  Head position not available")
    
    # Get joint limits
    joint_limits = tts.get_joint_limits()
    head_limits = joint_limits['head']
    print(f"\nHead joint limits:")
    print(f"  HeadYaw: {head_limits['HeadYaw']['min']}° to {head_limits['HeadYaw']['max']}°")
    print(f"  HeadPitch: {head_limits['HeadPitch']['min']}° to {head_limits['HeadPitch']['max']}°")
    
    # Get safe movement ranges
    safe_ranges = tts.get_safe_movement_ranges()
    head_ranges = safe_ranges['head']
    print(f"\nSafe movement ranges:")
    print(f"  Yaw: {head_ranges['yaw_range'][0]}° to {head_ranges['yaw_range'][1]}°")
    print(f"  Pitch: {head_ranges['pitch_range'][0]}° to {head_ranges['pitch_range'][1]}°")
    print(f"  Center return pitch: {head_ranges['center_return_pitch'][0]}° to {head_ranges['center_return_pitch'][1]}°")
    
    # Enable movement
    print("\nEnabling movement...")
    tts.enable_movement(True)
    
    # Test short speech with movement
    print("\nTesting single head movement at sentence start...")
    test_text = "Testing single head movement with larger range at the beginning of speech."
    
    print(f"Speaking: '{test_text}'")
    print("Watch for a single, larger head movement at the start...")
    
    success = tts.speak(test_text)
    
    if success:
        print("✅ Speech with improved head movement completed")
    else:
        print("❌ Speech failed")
    
    # Check final head position
    print("\nFinal head position:")
    final_yaw, final_pitch = tts.get_current_head_position()
    if final_yaw is not None and final_pitch is not None:
        print(f"  Yaw: {final_yaw:.1f}°")
        print(f"  Pitch: {final_pitch:.1f}°")
        
        # Check if movement occurred
        yaw_diff = abs(final_yaw - yaw)
        pitch_diff = abs(final_pitch - pitch)
        
        if yaw_diff > 1 or pitch_diff > 1:
            print(f"✅ Movement detected: Yaw changed by {yaw_diff:.1f}°, Pitch changed by {pitch_diff:.1f}°")
        else:
            print("⚠️ Little or no movement detected")
    
    # Disable movement
    print("\nDisabling movement...")
    tts.enable_movement(False)
    
    print("\n" + "=" * 50)
    print("Head movement test completed!")
    
    return True

if __name__ == "__main__":
    try:
        test_head_movement()
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"\nTest failed with error: {e}") 