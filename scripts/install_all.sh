#!/bin/sh
# Install everything: for each package, ask the user if he/she wishes to install it

# check if user is root (i.e. running with sudo)
if [ `whoami` != root ]; then
    echo "Please run this script with sudo"
    exit
fi

# check the first argument which must be the source directory
if [ -z $1 ]
then
    echo "Specify the absolute location (or relative to home) where to install the various packages."
    exit
fi


## Install packages

# install ros-kinetic
read -p 'Do you wish to install the `ros-kinetic` package on this computer [yes/no]?: ' reply
if [ $reply = 'yes' ]; then
    echo "Installing ros-kinetic..."
    $PWD/install_ros_kinetic.sh
else
    echo "Not installing ros-kinetic"
fi

# install rbdl
read -p 'Do you wish to install the `rbdl` package on this computer [yes/no]?: ' reply
if [ $reply = 'yes' ]; then
    echo "Installing rbdl..."
    $PWD/install_rbdl.sh
else
    echo "Not installing rbdl"
fi

# install ipopt
read -p 'Do you wish to install the `ipopt` package on this computer [yes/no]?: ' reply
if [ $reply = 'yes' ]; then
    echo "Installing ipopt..."
    $PWD/install_ipopt.sh
else
    echo "Not installing ipopt"
fi

# install openni2 & nite2
read -p 'Do you wish to install the `openni2`, `libfreenect`, and `nite2` packages on this computer [yes/no]?: ' reply
if [ $reply = 'yes' ]; then
    echo "Installing openni2, libfreenect, and nite2..."
    $PWD/install_openni2_nite2.sh $1
else
    echo "Not installing openni2, libfreenect, and nite2"
fi

# install openpose
read -p 'Do you wish to install the `openpose` package on this computer [yes/no]?: ' reply
if [ $reply = 'yes' ]; then
    echo "Installing openpose..."
    $PWD/install_openpose.sh $1
else
    echo "Not installing openpose"
fi