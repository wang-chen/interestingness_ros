import cv2
import torch
import rospy
import argparse
import interestingness
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import torchvision.transforms as transforms

import PIL
from torchvision.transforms.functional import vflip
from interestingness import show_batch

class InterestNode:
    def __init__(self, topic, transform):
        super(InterestNode, self).__init__()
        rospy.init_node('interestingness_node')
        rospy.Subscriber(topic, Image, self.callback)
        self.transform = transform
        self.bridge = CvBridge()

    def spin(self):
        rospy.spin()

    def callback(self, msg):
        rospy.loginfo("Received image: %d"%(msg.header.seq))
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "rgb8")
            frame = vflip(PIL.Image.fromarray(frame))
            frame = self.transform(frame)
        except CvBridgeError:
            rospy.logerr(CvBridgeError)
        else:
            frame = frame.cuda() if torch.cuda.is_available() else frame
            show_batch(frame)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ROS Interestingness Networks')
    parser.add_argument("--image-topic", type=str, default='/rs_front/color/image', help="image topic to subscribe")
    parser.add_argument("--model-save", type=str, default='../saves/ae.pt.DroneFilming.interest.mse', help="read model")
    parser.add_argument("--crop-size", type=int, default=320, help='crop size')
    parser.add_argument("--num-interest", type=int, default=10, help='loss compute by grid')
    parser.add_argument("--skip-frames", type=int, default=1, help='number of skip frame')
    parser.add_argument("--window-size", type=int, default=1, help='smooth window size >=1')
    parser.add_argument('--save-flag', type=str, default='interests', help='save name flag')
    parser.add_argument("--rr", type=float, default=5, help="reading rate")
    parser.add_argument("--wr", type=float, default=5, help="writing rate")
    args = parser.parse_args(); print(args)

    transform = transforms.Compose([
        transforms.CenterCrop(args.crop_size),
        transforms.Resize((args.crop_size,args.crop_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    node = InterestNode(args.image_topic, transform)
    node.spin()