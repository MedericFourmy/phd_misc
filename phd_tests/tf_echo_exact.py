#!/usr/bin/python

import rospy
import tf

frame1 = 'camera_accel_optical_frame'
frame2 = 'camera_color_optical_frame'

rospy.init_node('tf_a')
listener = tf.TransformListener()

rate = rospy.Rate(10.0)
while not rospy.is_shutdown():
    try:
        print frame1, ' -> ', frame2
        (trans,rot) = listener.lookupTransform(frame1, frame2, rospy.Time(0))
        print 'trans + rot: ', trans, rot
    except (tf.LookupException, tf.ConnectivityException):
        continue

    rate.sleep()