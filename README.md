# interestingness_ros

ROS wrapper for interestingness package.

If you want plain python package, go to [interestingness](https://github.com/wang-chen/interestingness) instead.

# Instructions:

* Only work with Python 3.

* Dependencies:
  
     PyTorch 1.4+, matplotlib.

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

* Download the pre-trained model named [ae.pt.SubTF.n1000.mse](https://github.com/wang-chen/interestingness/releases/download/v1.0/ae.pt.SubTF.n1000.mse) into folder "saves".

* Download the [SubT](https://github.com/wang-chen/SubT) ROS bag files into folder [datalocation].

* Change the argument "datalocation" in "subtf_bags.launch" to [datalocation], e.g.,

      <arg name="datalocation" default="/data/datasets"/>
* Run

      roslaunch interestingness_ros interestingness_subtf.launch
 
