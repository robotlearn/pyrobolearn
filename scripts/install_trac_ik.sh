#!/bin/sh
# This bash script install the trac_ik library and the official python wrapper
#
# wiki: http://wiki.ros.org/trac_ik
# repo: https://bitbucket.org/traclabs/trac_ik

# install from Debian package
#sudo apt-get install ros-kinetic-trac-ik

# Define few variables
currentdir=$PWD

# Check if user is root/running with sudo
if [ `whoami` != root ]; then
    echo "Please run this script with sudo"
    exit
fi

# check the first argument which must be the source directory
if [ -z $1 ]
then
    echo "Specify the absolute location (or relative to home) where to install the Openpose package."
    exit
fi

# move to that directory
cd; cd $1

# install requirements
sudo apt-get install libccd-dev libccd2 libfcl-0.5-dev libfcl0.5
sudo apt-get install libnlopt-dev libnlopt0  # nlopt
sudo apt-get -y install libeigen3-dev  # eigen
sudo pip install nlopt

# install trac_ik library from sources
git clone https://bitbucket.org/traclabs/trac_ik.git
cd trac_ik/trac_ik_lib/
mkdir build && cd build
cmake -DPYTHON_VERSION=2.7 ..
make -j2
source devel/setup.bash

# install python
cd ../../trac_ik_python/
mkdir build && cd build
cmake -DPYTHON_VERSION=2.7 ..
make -j2
sudo make install
