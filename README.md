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
 
* You may change sequence ID and bag playing speed in [L187](https://github.com/wang-chen/interestingness_ros/blob/a7b921a4b8f70ef9bdf80e162d528f42bac485f6/launch/subtf_bags.launch#L187) of "subtf_bags.launch".

      <node pkg="rosbag" type="play" name="rosbag" args="--clock -r 3 $(arg SubT4)"/>

* Run

      roslaunch interestingness_ros interestingness_subtf.launch
      # You need to wait for a while for first launch.

* You may need to uncomment image transform in [L321](https://github.com/wang-chen/interestingness_ros/blob/16168a074c0866e832ab46bb62c6c8382dc14b6c/launch/subtf_bags.launch#L321) in "interestingness_node.py" for UGV0 sequences in SubTF.

      VerticalFlip(), # Front camera of UGV0 in SubTF is mounted vertical flipped. Uncomment this line when needed.

---
## Citation

      @article{wang2020visual,
        author = {Wang, Chen and Wang, Wenshan and Qiu, Yuheng and Hu, Yafei and Scherer, Sebastian},
        journal = {arXiv preprint arXiv:2005.08829},
        title = {{Visual Memorability for Robotic Interestingness via Unsupervised Online Learning}},
        year = {2020}
      }

* Download [this paper](https://arxiv.org/pdf/2005.08829.pdf).

---
You may watch the following video to catch the idea of this work.

[<img src="https://img.youtube.com/vi/gBBdYdUrIcw/maxresdefault.jpg" width="100%">](https://youtu.be/gBBdYdUrIcw)
