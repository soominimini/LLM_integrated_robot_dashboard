import os
import subprocess
from pylips.speech import RobotFace

# ROS imports (optional)
try:
    import rospy
    from qt_robot_interface.srv import behavior_talk_text
    ROS_AVAILABLE = True
except Exception:
    ROS_AVAILABLE = False

# Fallback to ROS-based playback if pylips face server is unavailable.
def _speak_with_robot(text: str, robot_host: str, robot_user: str) -> bool:
    # Try local ROS first
    if ROS_AVAILABLE:
        try:
            try:
                rospy.get_node_uri()
            except rospy.exceptions.ROSException:
                rospy.init_node('pylips_default_test', anonymous=True)
            talk_text = rospy.ServiceProxy('/qt_robot/behavior/talkText', behavior_talk_text)
            talk_text = rospy.ServiceProxy('/qt_robot/behavior/talkText', behavior_talk_text)
            rospy.wait_for_service('/qt_robot/behavior/talkText', timeout=5)
            talk_text(text)
            return True
        except Exception as e:
            print(f"Local ROS talkText failed: {e}")

    # Fallback: SSH to robot and call rosservice directly (interactive password)
    try:
        cmd = [
            "ssh", "-t",
            f"{robot_user}@{robot_host}",
            f"rosservice call /qt_robot/behavior/talkText \"text: '{text}'\""
        ]
        subprocess.run(cmd, check=True)
        return True
    except Exception as e:
        print(f"Remote rosservice call failed: {e}")
        return False


def main():
    robot_host = os.getenv("ROBOT_HOST", "192.168.100.1")
    robot_user = os.getenv("ROBOT_USER", "developer")
    try:
        face = RobotFace()
        # you may need to wait here for a minute or two to let allosaurus download on the first run
        face.say("Hello, welcome to pylips!")
    except Exception as e:
        print(f"pylips server unavailable, falling back to ROS workflow: {e}")
        # ensure envs are set for pylips_test workflow
        os.environ.setdefault("ROBOT_HOST", robot_host)
        os.environ.setdefault("ROBOT_USER", robot_user)
        if not _speak_with_robot("Hello, welcome to pylips!", robot_host, robot_user):
            try:
                from pylips_test import main as fallback_main
                fallback_main()
            except Exception as inner_e:
                print(f"Fallback workflow failed: {inner_e}")


if __name__ == "__main__":
    main()