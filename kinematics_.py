#!/usr/bin/env python3
import rospy
import numpy as np
from qt_nuitrack_app.msg import Skeletons
from src.kinematics.kinematic_interface import QTrobotKinematicInterface

class SkeletonToRobotXYZMirror:
    def __init__(self):
        rospy.init_node('skeleton_to_robot_xyz_mirror', anonymous=True)
        self.kin = QTrobotKinematicInterface()
        rospy.Subscriber('/qt_nuitrack_app/skeletons', Skeletons, self.callback)
        rospy.loginfo("Started skeleton-to-robot XYZ mirroring node.")

    def callback(self, msg):
        if not msg.skeletons:
            return

        joints = msg.skeletons[0].joints  # First person
        print("human joints: ",joints)

        try:
            # Get 3D coordinates (real[]) from Nuitrack skeleton
            head_pos = np.array(joints[0].real)
            l_hand_pos = np.array(joints[7].real)
            r_hand_pos = np.array(joints[11].real)

            print("head left right: ", head_pos, l_hand_pos, r_hand_pos)

            # Make robot look at human head
            self.kin.look_at_xyz(head_pos)

            # Move robot arms toward human hand positions
            self.kin.reach_left(l_hand_pos)
            self.kin.reach_right(r_hand_pos)

        except Exception as e:
            rospy.logerr(f"Error while mapping skeleton to robot: {e}")

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        mirror = SkeletonToRobotXYZMirror()
        mirror.run()
    except rospy.ROSInterruptException:
        pass
