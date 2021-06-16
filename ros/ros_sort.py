#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
Detectron2 training script with a plain training loop.

This script reads a given config file and runs the training or evaluation.
It is an entry point that is able to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as a library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.

Compared to "train_net.py", this script supports fewer default features.
It also includes fewer abstraction, therefore is easier to add custom logic.
"""
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from perception_messages.msg import Detections2DArray
from perception_messages.msg import Detection2D

import time

import logging
import os
import numpy as np

from sort import *

class Sort_Tracking(Node):

    def __init__(self):
        super().__init__('tracking_sort_node')

        # Declare parameters
        self.declare_parameters(
            namespace='',
            parameters=[
                ('topic_in', 'image'),
                ('topic_out', 'detections'),
            ]
        )

        # get parameters
        topic_in = self.get_parameter('topic_in').get_parameter_value().string_value
        topic_out = self.get_parameter('topic_out').get_parameter_value().string_value

        # create subscriber
        self.subscription = self.create_subscription(Detections2DArray, topic_in, self.listener_callback, 5)
        self.subscription  # prevent unused variable warning
        self.get_logger().info('Listening to %s topic' % topic_in)

        # create publisher
        self.publisher_ = self.create_publisher(Detections2DArray, topic_out, 5)

        # initialize variables
        self.last_time = time.time()

        # Detector initialization
        self.model = self._init_model()
        self.get_logger().info('tracking model initialized')

    def listener_callback(self, msg):
        # transform message
        detections = self._msg2detections(msg)

        if len(detections)>0:
            # segment image
            track_bbs_ids = self.model.update(detections)

            # publish results
            if len(track_bbs_ids)>0 and len(track_bbs_ids)==len(detections):
                # iou_matrix = iou_batch(detections, trackers)
                tracked_msg = self._create_bbox_message(msg, track_bbs_ids)
            else:
                tracked_msg = Detections2DArray()
        else:
            tracked_msg = Detections2DArray()

        tracked_msg.header = msg.header
        self.publisher_.publish(tracked_msg)

        # compute true fps
        curr_time = time.time()
        fps = 1 / (curr_time - self.last_time)
        self.last_time = curr_time
        self.get_logger().info('Computing tracking at %.01f fps' % fps)

    def _init_model(self):
        model = Sort() 

        return model

    def _create_bbox_message(self, msg, tracking_ids):
        for obj_indx in range(len(msg.detections)):
        	msg.detections[obj_indx].instance = int(tracking_ids[obj_indx][4])
        return msg

    def _msg2detections(self, msg):
        detections = []
        for obj_indx in range(len(msg.detections)):
        	detection = msg.detections[obj_indx]
        	bbox = np.asarray([detection.center_x - detection.size_x/2., detection.center_y - detection.size_y/2., detection.center_x + detection.size_x/2., detection.center_y + detection.size_y/2.])
        	detections.append(bbox)
        return np.asarray(detections)


def main(args=None):
    rclpy.init(args=args)

    tracker_publisher = Sort_Tracking()

    rclpy.spin(tracker_publisher)

    tracker_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
