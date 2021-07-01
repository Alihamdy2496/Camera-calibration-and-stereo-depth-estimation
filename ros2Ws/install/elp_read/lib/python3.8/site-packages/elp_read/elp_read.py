#!/usr/bin/env python3
import rclpy
import numpy as np
from rclpy.node import Node
import numpy as np
from matplotlib import pyplot as plt
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import time
import sys
sys.path.append('/home/ali/anaconda3/lib/python3.8/site-packages')
import cv2
bridge = CvBridge()

class my_node (Node):
    def __init__(self):
        super().__init__("elp_read")
        self.obj_pub1=self.create_publisher(Image,"elp/row_left",10)
        self.obj_pub2=self.create_publisher(Image,"elp/row_right",10)
        self.create_timer(1/30,self.timer_call)
        self.cap1 = cv2.VideoCapture(4)
        self.cap2 = cv2.VideoCapture(2)

    def timer_call(self):
        msg=Image()
        if (self.cap1.isOpened()):
            ret, frame = self.cap1.read()
            if (ret):
                cv2_img1 = bridge.cv2_to_imgmsg(frame,encoding='passthrough')
                self.obj_pub1.publish(cv2_img1)
        if (self.cap2.isOpened()):
            ret, frame = self.cap2.read()
            if (ret):
                cv2_img1 = bridge.cv2_to_imgmsg(frame,encoding='passthrough')
                self.obj_pub2.publish(cv2_img1) 


def main (args=None):
    rclpy.init(args=args)
    node=my_node()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__=="__main__":
    main()




