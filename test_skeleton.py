#!/usr/bin/env python3
import rospy
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from qt_nuitrack_app.msg import Skeletons
import threading

# Global state
latest_skeletons = None
latest_image = None
lock = threading.Lock()
bridge = CvBridge()

# Visualization config
JOINT_COLOR = (0, 255, 0)  # Green
LINE_COLOR = (0, 0, 255)   # Red

# Joint connection map (based on Nuitrack indices)
BONES = [
    (0, 1), (1, 2), (2, 3),      # head-neck-torso
    (1, 5), (5, 6), (6, 7),      # left arm
    (1, 9), (9, 10), (10, 11),   # right arm
    (2, 12), (12, 13), (13, 14), # left leg
    (2, 15), (15, 16), (16, 17)  # right leg
]

def skeletons_callback(msg):
    global latest_skeletons
    with lock:
        latest_skeletons = msg
    rospy.logdebug("Skeleton message received.")

def image_callback(msg):
    global latest_image
    try:
        img = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        with lock:
            latest_image = img
        rospy.loginfo_once("[INFO] Image stream is active.")
    except Exception as e:
        rospy.logerr("CVBridge error: %s", str(e))

def draw_skeletons_on_image(image, skeletons_msg):
    h, w, _ = image.shape

    for skeleton in skeletons_msg.skeletons:
        joints = skeleton.joints
        points = {}

        for joint in joints:
            if joint.confidence > 0.5:
                x_proj, y_proj = joint.projection[:2]
                cx = int(x_proj * w)
                cy = int(y_proj * h)
                points[joint.type] = (cx, cy)
                cv2.circle(image, (cx, cy), 5, JOINT_COLOR, -1)

        for j1, j2 in BONES:
            if j1 in points and j2 in points:
                cv2.line(image, points[j1], points[j2], LINE_COLOR, 2)

    return image

def main():
    rospy.init_node('skeleton_image_overlay', anonymous=True)

    # ✅ Use corrected camera topic
    rospy.Subscriber('/camera/color/image_raw', Image, image_callback)
    rospy.Subscriber('/qt_nuitrack_app/skeletons', Skeletons, skeletons_callback)

    rospy.loginfo("[INFO] Subscribers initialized. Waiting for data...")
    rospy.sleep(1.0)  # Let subscribers initialize

    rate = rospy.Rate(10)  # 10 Hz

    while not rospy.is_shutdown():
        display_image = None
        skeletons_msg = None

        with lock:
            if latest_image is not None:
                display_image = latest_image.copy()
            if latest_skeletons is not None:
                skeletons_msg = latest_skeletons

        if display_image is not None:
            if skeletons_msg is not None:
                display_image = draw_skeletons_on_image(display_image, skeletons_msg)

            try:
                cv2.imshow("Robot View with Skeleton", display_image)
                key = cv2.waitKey(1)
                if key == 27:  # ESC to quit
                    break
            except Exception as e:
                rospy.logerr("OpenCV error: %s", str(e))
        else:
            rospy.loginfo_throttle(5, "[INFO] Waiting for image stream...")

        rate.sleep()

    cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
