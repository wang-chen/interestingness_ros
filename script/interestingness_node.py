#!/usr/bin/python3

# Copyright <2019> <Chen Wang [https://chenwang.site], Carnegie Mellon University>

# Redistribution and use in source and binary forms, with or without modification, are 
# permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this list of 
# conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice, this list 
# of conditions and the following disclaimer in the documentation and/or other materials 
# provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its contributors may be 
# used to endorse or promote products derived from this software without specific prior 
# written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY 
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES 
# OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT 
# SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, 
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED 
# TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; 
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN 
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN 
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH 
# DAMAGE.


import os
import cv2
import PIL
import sys
import torch
import rospy
import rospkg
import argparse
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import torchvision.transforms as transforms
from torchvision.transforms.functional import vflip

rospack = rospkg.RosPack()
pack_path = rospack.get_path('interestingness_ros')
sys.path.append(pack_path)
interestingness_path = os.path.join(pack_path,'interestingness')
sys.path.append(interestingness_path)
from interestingness import *


class InterestNode:
    def __init__(self, transform):
        super(InterestNode, self).__init__()
        rospy.init_node('interestingness_node')
        rospy.Subscriber(args.image_topic, Image, self.callback)
        self.transform = transform
        self.bridge = CvBridge()
        # self.net = torch.load(args.model_save)
        # self.net.set_train(False)
        # self.net.memory.set_learning_rate(rr=args.rr, wr=args.wr)

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
    parser.add_argument("--model-save", type=str, default=pack_path+'/saves/ae.pt.SubTF.interest.mse', help="read model")
    parser.add_argument("--crop-size", type=int, default=320, help='crop size')
    parser.add_argument("--num-interest", type=int, default=10, help='loss compute by grid')
    parser.add_argument("--skip-frames", type=int, default=1, help='number of skip frame')
    parser.add_argument("--window-size", type=int, default=1, help='smooth window size >=1')
    parser.add_argument('--save-flag', type=str, default='interests', help='save name flag')
    parser.add_argument("--rr", type=float, default=5, help="reading rate")
    parser.add_argument("--wr", type=float, default=5, help="writing rate")
    args = parser.parse_args(); print(args)

    results_path = os.path.join(pack_path,'results')
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    transform = transforms.Compose([
        transforms.CenterCrop(args.crop_size),
        transforms.Resize((args.crop_size,args.crop_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    interest = Interest(args.num_interest, os.path.join(results_path, '%s.txt'%('test_name')))

    movavg = MovAvg(args.window_size)

    node = InterestNode(transform)

    node.spin()