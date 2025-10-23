#!/usr/bin/env python3.9
# -*- coding: utf-8 -*-

"""
Test script for robot movement during speech
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from tts_helper import TTSHelper
import time

def test_movement():
    """Test the movement functionality"""
    print("Testing Robot Movement During Speech")
    print("=" * 50)
    
    # Initialize TTS helper
    print("Initializing TTS helper...")
    tts = TTSHelper()
    
    # Check if TTS is available
    if not tts.is_available():
        print("❌ TTS not available")
        return False
    
    print("✅ TTS available")
    
    # Check if movement is available
    if not tts.is_movement_available():
        print("❌ Movement not available")
        return False
    
    print("✅ Movement available")
    
    # Test joint limits
    print("\nTesting joint limits...")
    joint_limits = tts.get_joint_limits()
    safe_ranges = tts.get_safe_movement_ranges()
    
    print("Joint Limits:")
    for part, joints in joint_limits.items():
        print(f"  {part}:")
        for joint, limits in joints.items():
            print(f"    {joint}: {limits['min']} to {limits['max']} degrees")
    
    print("\nSafe Movement Ranges:")
    for part, ranges in safe_ranges.items():
        print(f"  {part}:")
        for movement, range_vals in ranges.items():
            print(f"    {movement}: {range_vals[0]} to {range_vals[1]} degrees")
    
    print("✅ Joint limits configured")
    
    # Test current head position
    print("\nTesting current head position...")
    yaw, pitch = tts.get_current_head_position()
    if yaw is not None and pitch is not None:
        print(f"Current head position: Yaw={yaw:.1f}°, Pitch={pitch:.1f}°")
        print("✅ Head position monitoring available")
    else:
        print("⚠️ Head position not available")
    
    # Test movement settings
    print("\nTesting movement settings...")
    tts.enable_movement(True)
    print("✅ Movement enabled")
    
    # Test speech with movement
    print("\nTesting speech with single movement at start...")
    test_text = "Hello! I am QTrobot and I can move my arms and head once at the beginning of speech with larger movements."
    
    print(f"Speaking: '{test_text}'")
    print("Watch for a single, larger movement at the start of speech...")
    
    success = tts.speak(test_text)
    
    if success:
        print("✅ Speech with single movement completed successfully")
    else:
        print("❌ Speech failed")
    
    # Test movement disable
    print("\nTesting movement disable...")
    tts.enable_movement(False)
    print("✅ Movement disabled")
    
    # Test speech without movement
    print("\nTesting speech without movement...")
    test_text2 = "Now I am speaking without movement."
    
    print(f"Speaking: '{test_text2}'")
    success2 = tts.speak(test_text2)
    
    if success2:
        print("✅ Speech without movement completed successfully")
    else:
        print("❌ Speech failed")
    
    print("\n" + "=" * 50)
    print("Movement test completed!")
    
    return True

if __name__ == "__main__":
    try:
        test_movement()
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"\nTest failed with error: {e}") 