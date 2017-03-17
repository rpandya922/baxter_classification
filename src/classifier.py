#!/usr/bin/env python
from __future__ import division  
import rospy
import math
import tf
import geometry_msgs.msg
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from hmm import HMM
import scipy
import scipy.io
from sklearn.svm import SVC
from random import shuffle
import sys
import copy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
import baxter_interface.gripper as gripper

TRAIN_PER = 0.7
NUM_CLASSES = 3

def add_norm(x):
    norm = np.linalg.norm(x)
    return np.hstack((x, norm))
def print_accuracy(classifier):
    print "\n" + str(classifier.accuracy())
def get_belief(x):
    return {
        0 : 'handle',
        1 : 'top',
        2 : 'no grasp'
    }[x]

handle = np.load("handle2.npy")
top = np.load("top2.npy")
no_grasp = np.load("no_grasp2.npy")

training = [[0, add_norm(h)] for h in handle[(int)(len(handle) * TRAIN_PER):]]
testing = [[0, add_norm(h)] for h in handle[:(int)(len(handle) * TRAIN_PER)]]
training.extend([[1, add_norm(t)] for t in top[(int)(len(top) * TRAIN_PER):]])
testing.extend([[1, add_norm(t)] for t in top[:(int)(len(top) * TRAIN_PER)]])
training.extend([[2, add_norm(n)] for n in no_grasp[(int)(len(no_grasp) * TRAIN_PER):]])
testing.extend([[2, add_norm(n)] for n in no_grasp[:(int)(len(no_grasp) * TRAIN_PER)]])

testing_data = [np.array(i[1]).reshape(1, -1) for i in testing]
testing_labels = [i[0] for i in testing]

train_X = [i[1] for i in training]
train_y = [i[0] for i in training]
initial = [0.0001, 0.0001, 0.9998]
# arbitrarily defined, to be trained on later
transitions = [[7/8, 0, 1/8],
               [0, 7/8, 1/8],
               [1/8, 1/8, 3/8]]

classifier = HMM(initial, transitions, train_X, train_y, NUM_CLASSES)
print "Finished Training"
for i in range(len(testing_data)):
    classifier.update(testing_data[i], testing_labels[i])
    # print(str(classifier.update(testing_data[i], testing_labels[i])) + " " + str(testing_labels[i]))

print(classifier.accuracy())

print "============ Starting tutorial setup"
moveit_commander.roscpp_initialize(sys.argv)
rospy.init_node('move_group_python_interface_tutorial',
                anonymous=True)
robot = moveit_commander.RobotCommander()
scene = moveit_commander.PlanningSceneInterface()
group = moveit_commander.MoveGroupCommander("right_arm")
display_trajectory_publisher = rospy.Publisher(
                                    '/move_group/display_planned_path',
                                    moveit_msgs.msg.DisplayTrajectory)
print "============ Reference frame: %s" % group.get_planning_frame()
print "============ Reference frame: %s" % group.get_end_effector_link()

pose_target = geometry_msgs.msg.Pose()
pose_handoff = geometry_msgs.msg.Pose()

pose_target.orientation.x = 0.713
pose_target.orientation.y = 0.701
pose_target.orientation.z = -0.013
pose_target.orientation.w = 0.001
pose_target.position.x = 0.665
pose_target.position.y = -0.626
pose_target.position.z = 0.026

pose_handoff.orientation.x = 0.473
pose_handoff.orientation.y = 0.601
pose_handoff.orientation.z = 0.395
pose_handoff.orientation.w = 0.510
pose_handoff.position.x = 0.967
pose_handoff.position.y = -0.582
pose_handoff.position.z = 0.408

g = gripper.Gripper('right')
g.calibrate()

group.set_pose_target(pose_target)
plan1 = group.plan()
group.go(wait=True)

rospy.sleep(2)
g.close()
rospy.sleep(0.5)

group.set_pose_target(pose_handoff)
plan2 = group.plan()
group.go(wait=True)

if __name__ == '__main__':
    rospy.on_shutdown(lambda: print_accuracy(classifier))
    listener = tf.TransformListener()

    rate = rospy.Rate(50.0)
    i = 0
    prev = 2
    num_same = 0
    while not rospy.is_shutdown():
        try:
            (trans,rot) = listener.lookupTransform('/mug', '/hand', rospy.Time(0))
            trans = np.array(trans)
            rot = np.array(rot)
            d = np.hstack((trans, rot))
            belief = classifier.update(add_norm(d), 2)
            b = np.argmax(belief)
            if prev == b:
                num_same += 1
            else:
                num_same = 0
            print get_belief(b)
            if(prev != 2 and num_same > 10):
                print prev
                g.open()
                rospy.sleep(1)
                group.set_pose_target(pose_target)
                plan1 = group.plan()
                group.go(wait=True)
                rospy.signal_shutdown('')
            prev = b
            i += 1

        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            continue
        rate.sleep()




