import cv2
import rospy
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError 
import torch

bridge = CvBridge()

def image_callback(msg):
    print("Received an image!")
    try:
        cv2_img = bridge.imgmsg_to_cv2(msg, "bgr8")
    except (CvBridgeError):
        print(CvBridgeError)
    else:
        cv2.imshow("Image Display", cv2_img)
        cv2.waitKey(1) 

    
def image_listener():
    rospy.init_node('py_image_listener')
    rospy.Subscriber("/rs_front/color/image", Image, image_callback)
    rospy.spin()
    cv2.destroyWindow("Image Display")

if __name__ == '__main__':
    image_listener()