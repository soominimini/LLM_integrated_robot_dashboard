#!/usr/bin/env python3.9
"""
Complete Polly TTS Workflow
1. Generate Polly audio locally
2. Upload to QTRP
3. Play with QT visemes
"""

import os
import sys
import subprocess
import time
from tts.local_polly_generator import generate_polly_audio, upload_to_robot

# ROS imports
try:
    import rospy
    from qt_robot_interface.srv import *

    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False


def copy_to_qtrobot_user(audio_file, robot_host="192.168.100.1", robot_user="developer"):
    """
    Copy audio file to qtrobot user's audio directory using interactive SSH
    """
    print(f" Copying {audio_file} to qtrobot user: ")

    # Use ssh -t for interactive sudo
    ssh_cmd = [
        'ssh', '-t', f'{robot_user}@{robot_host}',
        f'sudo cp ~/{audio_file} ~/../qtrobot/robot/data/audios/ && sudo chown qtrobot:qtrobot ~/../qtrobot/robot/data/audios/{audio_file}'
    ]

    try:
        result = subprocess.run(ssh_cmd, check=True)
        print(f"✅ File copied to qtrobot user successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to copy file: {e}")
        print(f"💡 Manual step required:")
        print(f"   1. SSH to QTRP: ssh {robot_user}@{robot_host}")
        print(f"   2. Run: sudo cp ~/{audio_file} ~/../qtrobot/robot/data/audios/")
        print(f"   3. Run: sudo chown qtrobot:qtrobot ~/../qtrobot/robot/data/audios/{audio_file}")
        return False


def trigger_audio_playback(audio_file):
    """
    Trigger audio playback using ROS service call like QTrobot example
    """
    print(f"🎵 Triggering audio playback: {audio_file}")

    try:
        if not ROS_AVAILABLE:
            raise ImportError("ROS libraries not available")

        # Initialize ROS node if not already initialized
        try:
            rospy.get_node_uri()
        except rospy.exceptions.ROSException:
            rospy.init_node('pylips_audio_trigger', anonymous=True)

        # Create service proxy for behaviorTalkAudio
        behaviorTalkAudio = rospy.ServiceProxy('/qt_robot/behavior/talkAudio', behavior_talk_audio)

        # Wait for service to be available
        print("⏳ Waiting for service connection...")
        rospy.wait_for_service('/qt_robot/behavior/talkAudio', timeout=5)

        # Call the service - format: behaviorTalkAudio("filename", "filepath")
        print(f"🚀 Calling service: behaviorTalkAudio('{audio_file}', '')")
        response = behaviorTalkAudio(audio_file, "")

        print(f"✅ Audio playback triggered successfully!")
        print(f"🤖 Robot should now play audio and show lip sync!")
        return True

    except ImportError:
        print("❌ ROS libraries not available. Using fallback command method...")
        # Fallback to command line method
        rostopic_cmd = [
            'rostopic', 'pub', '/qt_robot/behavior/talkAudio', 'std_msgs/String',
            f'"data: \'{audio_file}\'"', '--once'
        ]
        try:
            result = subprocess.run(rostopic_cmd, capture_output=True, text=True, check=True)
            print(f"✅ Audio playback triggered successfully!")
            print(f"Output: {result.stdout}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to trigger playback: {e}")
            return False
    except rospy.ServiceException as e:
        print(f"❌ Service call failed: {e}")
        return False
    except rospy.ROSException as e:
        print(f"❌ ROS error: {e}")
        print("💡 Make sure ROS is running: source /opt/ros/noetic/setup.bash")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


def main():
    print("🎯 Complete Polly TTS Workflow")
    print("=" * 40)
    print("1. Generate Polly audio locally")
    print("2. Upload to QTRP")
    print("3. Copy to qtrobot user")
    print("4. Trigger playback with QT visemes")
    print()

    # Configuration
    # text = "<speak> I like <break time='500ms'/> <emphasis level='strong'>lions</emphasis>.</speak>"
    text = "<speak> I like lions.</speak>"
    voice_name = "Ivy"
    audio_file = "polly_audio_new.mp3"

    # Step 1: Generate Polly audio locally
    print("🎤 Step 1: Generating Polly TTS audio...")
    generated_file = generate_polly_audio(text, voice_name, audio_file)

    if not generated_file:
        print("❌ Step 1 failed: Could not generate audio")
        return

    print("✅ Step 1 completed: Audio generated")
    print()

    # Step 2: Upload to QTRP
    print("📤 Step 2: Uploading to QTRP...")
    if not upload_to_robot(generated_file):
        print("❌ Step 2 failed: Could not upload audio")
        return

    print("✅ Step 2 completed: Audio uploaded")
    print()

    # Step 3: Copy to qtrobot user
    print("📋 Step 3: Copying to qtrobot user...")
    if not copy_to_qtrobot_user(generated_file):
        print("❌ Step 3 failed: Could not copy to qtrobot user")
        return

    print("✅ Step 3 completed: Audio copied to qtrobot user")
    print()

    # Step 4: Trigger audio playback
    print("🎵 Step 4: Triggering audio playback...")
    if not trigger_audio_playback(generated_file):
        print("❌ Step 4 failed: Could not trigger playback")
        return

    print("✅ Step 4 completed: Audio playback triggered")
    print()

    print("🎉 Complete workflow finished successfully!")
    print(f"📁 Audio file: {audio_file}")
    print(f"🎭 Visemes: QTrobot's default talking animation")
    print(f"🎤 Voice: Amazon Polly ({voice_name})")
    print()
    print("💡 Note: Only audio files are uploaded to QTRP.")
    print("   No Python scripts needed - QTRP uses built-in ROS interface!")


if __name__ == "__main__":
    main()