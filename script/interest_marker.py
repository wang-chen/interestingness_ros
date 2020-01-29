#!/usr/bin/env python

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

import math
import rospy
import tf2_ros

from rosutil import ROSArgparse
from interestingness_ros.msg import InterestInfo
from visualization_msgs.msg import Marker, MarkerArray


def info_callback(msg):
    if msg.level < args.min_level:
        rospy.loginfo('Skip marker with level {}'.format(msg.level))
        return

    marker = Marker()
    marker.id = msg.header.seq
    marker.header = msg.header
    marker.type = marker.SPHERE
    marker.action = marker.ADD

    marker.color.a = msg.level
    marker.color.r, marker.color.g, marker.color.b = 1, 0, 0
    marker.scale.x, marker.scale.y, marker.scale.z = 5, 5, 5

    marker.pose.orientation.w = 1
    marker.pose.position.z = 1
    marker.lifetime.secs = 999999999
    publisher.publish(marker)
    rospy.loginfo('Published marker with level {}.'.format(msg.level))


if __name__ == '__main__':
    rospy.init_node('interestmarker_node')

    parser = ROSArgparse(relative='interestmarker_node/')
    parser.add_argument("min-level", default=0.1, help="minimum interest level to show")
    args = parser.parse_args()

    rospy.Subscriber('/interestingness/info', InterestInfo, info_callback)

    publisher = rospy.Publisher('interestmarker/marker', Marker, queue_size=10)

    rospy.spin()
