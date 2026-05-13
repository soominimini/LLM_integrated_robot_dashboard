# coding=utf-8
"""
Upload a WAV from this PC to the QTrobot's onboard computer, then play it
through the robot's speakers.

The audio service /qt_robot/audio/play runs on the robot's onboard computer
(QTRP), which has its own filesystem and cannot see files on this PC. So we:
  1. base64-encode the WAV locally
  2. push it to QTRP via /qt_robot/setting/uploadBase64 -> /tmp/qwen_tts/<name>
  3. ask QTRP to play that file via /qt_robot/audio/play (filepath form)

Use as a module:
    from play_on_robot import play_on_robot
    play_on_robot("/abs/path/to/file.wav")

Or as a CLI:
    python play_on_robot.py stability_runs/run_01.wav
"""

import os
import sys
import base64
import rospy
from qt_robot_interface.srv import audio_play, setting_upload

ROBOT_AUDIO_DIR = "/home/qtrobot/robot/data/audios"  # standard QTrobot audio dir on QTRP
_NODE_INITED = False
_play_proxy = None
_upload_proxy = None


def _ensure_node():
    global _NODE_INITED, _play_proxy, _upload_proxy
    if not _NODE_INITED:
        rospy.init_node("qwen_tts_player", anonymous=True, disable_signals=True)
        _NODE_INITED = True
    if _play_proxy is None:
        rospy.wait_for_service("/qt_robot/audio/play", timeout=5.0)
        _play_proxy = rospy.ServiceProxy("/qt_robot/audio/play", audio_play)
    if _upload_proxy is None:
        rospy.wait_for_service("/qt_robot/setting/uploadBase64", timeout=5.0)
        _upload_proxy = rospy.ServiceProxy(
            "/qt_robot/setting/uploadBase64", setting_upload
        )


def upload_to_robot(local_path, remote_path=None):
    """Upload a local file to QTRP. Returns the remote path used."""
    abs_path = os.path.abspath(local_path)
    if not os.path.isfile(abs_path):
        raise FileNotFoundError(abs_path)
    if remote_path is None:
        remote_path = os.path.join(ROBOT_AUDIO_DIR, os.path.basename(abs_path))

    _ensure_node()
    with open(abs_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("ascii")

    resp = _upload_proxy(
        data=encoded,
        filepath=remote_path,
        permission="644",
        append=False,
    )
    if not resp.status:
        raise RuntimeError(f"uploadBase64 returned status=False for {remote_path}")
    return remote_path


def play_on_robot(local_path):
    """Upload then play a WAV through the robot. Returns True on success."""
    remote_path = upload_to_robot(local_path)
    # QTrobot audio_play only accepts `filename` (basename) for files in
    # /home/qtrobot/robot/data/audios/; `filepath` is rejected. Strip extension.
    name = os.path.splitext(os.path.basename(remote_path))[0]
    resp = _play_proxy(filename=name, filepath="")
    if not resp.status:
        raise RuntimeError(f"audio/play returned status=False for filename={name}")
    return True


def main():
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <path/to/file.wav>")
        sys.exit(1)
    path = sys.argv[1]
    print(f"Uploading and playing on robot: {path}")
    try:
        play_on_robot(path)
        print("OK")
        sys.exit(0)
    except Exception as e:
        print(f"FAILED: {e}")
        sys.exit(2)


if __name__ == "__main__":
    main()
