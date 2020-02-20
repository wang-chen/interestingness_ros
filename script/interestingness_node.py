#!/usr/bin/env python3

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
import numpy as np
from sensor_msgs.msg import Image
from rospy.numpy_msg import numpy_msg
from cv_bridge import CvBridge, CvBridgeError
import torchvision.transforms as transforms

rospack = rospkg.RosPack()
pack_path = rospack.get_path('interestingness_ros')
interestingness_path = os.path.join(pack_path,'interestingness')
sys.path.append(pack_path)
sys.path.append(interestingness_path)

from rosutil import ROSArgparse
from interestingness_ros.msg import InterestInfo
from interestingness.test_interest import MovAvg, show_batch_box, level_height
from interestingness.interestingness import AE, VAE, AutoEncoder, Interestingness
from interestingness.dataset import ImageData, Dronefilm, DroneFilming, SubT, SubTF, PersonalVideo
from interestingness.torchutil import VerticalFlip, count_parameters, show_batch, show_batch_origin, Timer, MovAvg
from interestingness.torchutil import ConvLoss, CosineLoss, CorrelationLoss, Split2d, Merge2d, PearsonLoss, FiveSplit2d


class InterestNode:
    def __init__(self, args, transform):
        super(InterestNode, self).__init__()
        self.config(args)
        self.movavg = MovAvg(self.window_size)
        self.transform, self.bridge = transform, CvBridge()
        net = torch.load(self.model_save)
        net.set_train(False)
        net.memory.set_learning_rate(rr=self.rr, wr=self.wr)
        self.net = net.cuda() if torch.cuda.is_available() else net
        for topic in self.image_topic:
            rospy.Subscriber(topic, Image, self.callback)
        self.frame_pub = rospy.Publisher('interestingness/image', Image, queue_size=10)
        self.info_pub = rospy.Publisher('interestingness/info', numpy_msg(InterestInfo), queue_size=10)

    def config(self, args):
        self.rr, self.wr = args.rr, args.wr
        self.model_save = args.model_save
        self.image_topic = args.image_topic
        self.skip_frames = args.skip_frames
        self.window_size = args.window_size

    def spin(self):
        rospy.spin()

    def callback(self, msg):
        if msg.header.seq % self.skip_frames is not 0:
            return
        rospy.loginfo("Received image %s: %d"%(msg.header.frame_id, msg.header.seq))
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "rgb8")
            frame = PIL.Image.fromarray(frame)
            frame = self.transform(frame).unsqueeze(dim=0)
        except CvBridgeError:
            rospy.logerr(CvBridgeError)
        else:
            frame = frame.cuda() if torch.cuda.is_available() else frame
            _, loss = self.net(frame)
            loss = self.movavg.append(loss)
            frame = 255*show_batch_box(frame, msg.header.seq, loss.item(),show_now=False)
            frame_msg = self.bridge.cv2_to_imgmsg(frame.astype(np.uint8))
            info = InterestInfo()
            info.level = level_height(loss.item())
            info.feature_shape = self.net.states.shape
            info.feature_map = self.net.states.cpu().view(-1).numpy()
            info.reading_weights = self.net.memory.rw.cpu().view(-1).numpy()
            info.header = frame_msg.header = msg.header
            self.frame_pub.publish(frame_msg)
            self.info_pub.publish(info)


if __name__ == '__main__':

    rospy.init_node('interestingness_node')
    parser = ROSArgparse(relative='interestingness_node/')
    parser.add_argument("image-topic", default=['/rs_front/color/image'])
    parser.add_argument("data-root", type=str, default='/data/datasets', help="dataset root folder")
    parser.add_argument("model-save", type=str, default=pack_path+'/saves/ae.pt.SubTF.interest.mse', help="read model")
    parser.add_argument("crop-size", type=int, default=320, help='crop size')
    parser.add_argument("num-interest", type=int, default=10, help='loss compute by grid')
    parser.add_argument("skip-frames", type=int, default=1, help='number of skip frame')
    parser.add_argument("window-size", type=int, default=1, help='smooth window size >=1')
    parser.add_argument('save-flag', type=str, default='interests', help='save name flag')
    parser.add_argument("rr", type=float, default=5, help="reading rate")
    parser.add_argument("wr", type=float, default=5, help="writing rate")
    args = parser.parse_args()

    results_path = os.path.join(pack_path,'results')
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    transform = transforms.Compose([
        VerticalFlip(), # Front camera of SubTF is mounted vertical flipped.
        transforms.CenterCrop(args.crop_size),
        transforms.Resize((args.crop_size, args.crop_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    node = InterestNode(args, transform)

    node.spin()
