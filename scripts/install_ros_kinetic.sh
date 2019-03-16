#!/bin/bash
# update & upgrade #
sudo apt-get update
sudo apt-get upgrade

# install ros indigo #
sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
sudo apt-key adv --keyserver hkp://ha.pool.sks-keyservers.net:80 --recv-key 0xB01FA116
sudo apt-get update
sudo apt-get -y install ros-kinetic-desktop-full
sudo rosdep init
rosdep update
echo "source /opt/ros/kinetic/setup.bash" >> ~/.bashrc
source ~/.bashrc
sudo apt-get -y install python-rosinstall

# install catkin #
sudo apt-get -y install ros-kinetic-catkin


# create ros workspace #
mkdir -p ~/catkin_ws/src
cd ~/catkin_ws/src
catkin_init_workspace
cd ~/catkin_ws/
catkin_make
echo "source ~/catkin_ws/devel/setup.bash" >> ~/.bashrc

# update gazebo to latest version #
#sudo sh -c 'echo "deb http://packages.osrfoundation.org/gazebo/ubuntu trusty #main" > /etc/apt/sources.list.d/gazebo-latest.list'
#wget http://packages.osrfoundation.org/gazebo.key -O - | sudo apt-key add -
#sudo apt-get update
#sudo apt-get upgrade

# install gazebo ros pkgs #
sudo sh -c 'echo "deb http://packages.osrfoundation.org/gazebo/ubuntu-stable `lsb_release -cs` main" > /etc/apt/sources.list.d/gazebo-stable.list'
wget http://packages.osrfoundation.org/gazebo.key -O - | sudo apt-key add -
sudo apt-get -y install ros-kinetic-gazebo-ros-pkgs ros-kinetic-gazebo-ros-control

# install ros control #
sudo apt-get -y install ros-kinetic-ros-control ros-kinetic-ros-controllers

# additional setting #
#echo "export LC_NUMERIC=C" >> ~/.bashrc
#echo "killall -q roscore" >> ~/.bashrc
#echo "killall -q rosmaster" >> ~/.bashrc
#echo "killall -q gazebo" >> ~/.bashrc
#echo "killall -q gzserver" >> ~/.bashrc
#echo "killall -q gzclient" >> ~/.bashrc