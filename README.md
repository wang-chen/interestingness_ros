# interestingness_ros

ROS wrapper for interestingness package.

If you want plain python package, go to [interestingness](https://github.com/wang-chen/interestingness) instead.

# Instructions:

* Only work with Python 3.

* Dependencies:
  
     ROS, PyTorch 1.4+, Matplotlib.

* For ROS Melodic with Ubuntu 18.04.

  * Following [this instruction](https://medium.com/@beta_b0t/how-to-setup-ros-with-python-3-44a69ca36674) to setup Python 3. Here what I put in the '.bashrc'.

        source /opt/ros/melodic/setup.bash
        source ~/catkin_ws/devel/setup.bash
        source ~/catkin_build_ws/install/setup.bash --extend # from the above medium post
        export ROS_PYTHON_VERSION=3

  * Install husky_gazebo for visualization.

        sudo apt install ros-melodic-husky-gazebo

* For ROS Noetic with Ubuntu 20.04.
  
  * ROS Noetic is native with Python 3.

  * Until June 2020, husky_gazebo doesn't work on ROS Noetic.  You need to comment 'robot.launch' in all launch files.

        <!-- <include file="$(find interestingness_ros)/launch/robot.launch" /> -->

  * We have not provide other visualization methods. But you can still run the program.

* Remember to update the submodule before catkin_make.
 
      cd ~/catkin_ws/src
      git clone https://github.com/wang-chen/interestingness_ros
      cd interestingness_ros
      git submodule init
      git submodule update

* Download pre-trained model [ae.pt.SubTF.n1000.mse](https://github.com/wang-chen/interestingness/releases/download/v1.0/ae.pt.SubTF.n1000.mse) into folder "saves", or change argument in "interestingness.launch".

      <param name="model-save" value="$(find interestingness_ros)/saves/ae.pt.SubTF.n1000.mse" />

* Download the [SubT](https://github.com/wang-chen/SubT) ROS bag files into folder [datalocation].

* Change the argument "datalocation" in [L4](https://github.com/wang-chen/interestingness_ros/blob/16168a074c0866e832ab46bb62c6c8382dc14b6c/launch/subtf_bags.launch#L4) of "subtf_bags.launch" to [datalocation], e.g.,

      <arg name="datalocation" default="/data/datasets"/>
 
* You may change sequence ID and bag playing speed in [L321](https://github.com/wang-chen/interestingness_ros/blob/16168a074c0866e832ab46bb62c6c8382dc14b6c/launch/subtf_bags.launch#L321) of "subtf_bags.launch".

      <node pkg="rosbag" type="play" name="rosbag" args="--clock -r 3 $(arg SubT4)"/>

* The corresponding sequence ID is listed:

    |   Data sequence     | ID   | ROS args|
    |  :-----------:      | :--: | :----: |
    |  0817-ugv0-tunnel0  |  0   | SubT0 |
    |  0817-ugv1-tunnel0  |  1   | SubT1 |
    |  0818-ugv0-tunnel1  |  2   | SubT2 |
    |  0818-ugv1-tunnel1  |  3   | SubT3 |
    |  0820-ugv0-tunnel1  |  4   | SubT4 |
    |  0821-ugv0-tunnel0  |  5   | SubT5 |
    |  0821-ugv1-tunnel0  |  6   | SubT6 |

* Run

      roslaunch interestingness_ros interestingness_subtf.launch
      # You need to wait for a while for first launch.

* You may need to uncomment image transform in [L147](https://github.com/wang-chen/interestingness_ros/blob/f63e152a03d7c5eb1bcdd7ef95aeb026c8b5b234/script/interestingness_node.py#L147) in "interestingness_node.py" for UGV0 sequences in SubTF.

      VerticalFlip(), # Front camera of UGV0 in SubTF is mounted vertical flipped. Uncomment this line when needed.



---
## Citation

    @inproceedings{wang2020visual,
      title={Visual memorability for robotic interestingness via unsupervised online learning},
      author={Wang, Chen and Wang, Wenshan and Qiu, Yuheng and Hu, Yafei and Scherer, Sebastian},
      booktitle={European Conference on Computer Vision (ECCV)},
      year={2020},
      organization={Springer}
    }

* Download [this paper](https://arxiv.org/pdf/2005.08829.pdf).

---
You may watch the following video to catch the idea of this work.

[<img src="https://img.youtube.com/vi/PXIcm17fEko/maxresdefault.jpg" width="100%">](https://youtu.be/PXIcm17fEko)
