#!/usr/bin/env python3.9
"""
Local Polly TTS Generator
Generates Polly audio files locally on QTPC
"""

import os
import sys
import boto3
from botocore.exceptions import ClientError, NoCredentialsError, NoRegionError


def generate_polly_audio(text, voice_name="Justin", output_file="polly_audio.mp3"):
    """
    Generate Polly TTS audio file locally

    Args:
        text (str): Text to convert to speech
        voice_name (str): Polly voice name (default: Joanna)
        output_file (str): Output audio file name
    """
    print(f"🎤 Generating Polly TTS audio...")
    print(f"   Text: '{text}'")
    print(f"   Voice: {voice_name}")
    print(f"   Output: {output_file}")

    try:
        # Initialize Polly client with region fallback
        region = os.environ.get('AWS_REGION') or os.environ.get('AWS_DEFAULT_REGION') or 'us-east-1'
        polly = boto3.client('polly', region_name=region)

        # Check if text contains SSML tags
        is_ssml = '<' in text and '>' in text

        if is_ssml:
            print(f"   📝 Using SSML format for enhanced speech")
            # Generate speech with SSML (standard engine required for SSML)
            response = polly.synthesize_speech(
                Text=text,
                TextType='ssml',  # Specify SSML format
                OutputFormat='mp3',
                VoiceId=voice_name
                # Note: No Engine='neural' for SSML as it's not supported
            )
        else:
            print(f"   📝 Using plain text format")
            # Generate speech with plain text
            response = polly.synthesize_speech(
                Text=text,
                OutputFormat='mp3',
                VoiceId=voice_name,
                Engine='neural'  # Use neural engine for better quality
            )

        # Save audio file
        with open(output_file, 'wb') as file:
            file.write(response['AudioStream'].read())

        print(f"✅ Audio generated successfully: {output_file}")
        return output_file

    except NoCredentialsError:
        print("❌ AWS credentials not found!")
        print("Please set up AWS credentials:")
        print("   export AWS_ACCESS_KEY_ID='your-access-key'")
        print("   export AWS_SECRET_ACCESS_KEY='your-secret-key'")
        print("   export AWS_DEFAULT_REGION='us-east-1'")
        return None

    except NoRegionError:
        print("❌ AWS region not specified!")
        print("Please set a region, e.g.:")
        print("   export AWS_DEFAULT_REGION='us-east-1'")
        print("Or pass AWS_REGION/AWS_DEFAULT_REGION in your environment.")
        return None

    except ClientError as e:
        print(f"❌ AWS Polly error: {e}")
        return None


def upload_to_robot(local_file, robot_host="192.168.100.1", robot_user="developer"):
    """
    Upload audio file to QTRP

    Args:
        local_file (str): Local audio file path
        robot_host (str): QTRP IP address
        robot_user (str): QTRP username
    """
    print(f"📤 Uploading {local_file} to QTRP...")

    import subprocess

    # Upload to developer user's home
    scp_cmd = ['scp', local_file, f'{robot_user}@{robot_host}:~/']

    try:
        result = subprocess.run(scp_cmd, capture_output=True, text=True, check=True)
        print(f"✅ Upload successful!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Upload failed: {e}")
        print(f"Error output: {e.stderr}")
        return False


def main():
    print("🎯 Local Polly TTS Generator for QTrobot")
    print("=" * 50)

    # Configuration
    text = "Hello! I am QTrobot speaking with Amazon Polly voice."
    voice_name = "Joanna"  # You can change this to "Matthew", "Amy", etc.
    output_file = "polly_audio.mp3"

    # Step 1: Generate Polly audio locally
    audio_file = generate_polly_audio(text, voice_name, output_file)

    if not audio_file:
        print("❌ Failed to generate audio")
        return

    # Step 2: Upload to QTRP
    if upload_to_robot(audio_file):
        print(f"\n🎉 Success! Audio file '{audio_file}' uploaded to QTRP")
        print(f"📁 File location on QTRP: ~/{audio_file}")
        print(f"\n🚀 Next steps:")
        print(f"   1. SSH to QTRP: ssh developer@192.168.100.1")
        print(f"   2. Copy to qtrobot: sudo cp ~/{audio_file} /home/qtrobot/data/audios/")
        print(f"   3. Run audio player: python qtrp_audio_player.py {audio_file}")
    else:
        print("❌ Failed to upload audio file")


if __name__ == "__main__":
    main()