#!/bin/sh
# This bash script install the openpose and the official python wrapper
# 
# Github repo: https://github.com/CMU-Perceptual-Computing-Lab/openpose
# Reference: https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/installation.md
# FAQ (troubleshooting): https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/faq.md


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

# clone repository
git clone https://github.com/CMU-Perceptual-Computing-Lab/openpose
cd openpose
git pull origin master

# Prerequisites
sudo apt-get install cmake-qt-gui
sudo apt install cmake-curses-gui
read -p 'Do you wish to install the CUDA8 on this computer [yes/no]?: ' reply
if [ $reply = 'yes' ]; then
    echo "Installing CUDA8..."
    sudo ./scripts/ubuntu/install_cuda.sh
else
    echo "Not installing CUDA8"
fi
read -p 'Do you wish to install the cuDNN5.1 on this computer [yes/no]?: ' reply
if [ $reply = 'yes' ]; then
    echo "Installing cuDNN5.1..."
    sudo ./scripts/ubuntu/install_cudnn.sh
else
    echo "Not installing cuDNN5.1"
fi
sudo ./scripts/ubuntu/install_deps.sh
sudo apt-get install libopencv-dev
sudo apt-get install nvidia-modprobe

mkdir build && cd build
echo "Check https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/installation.md#openpose-configuration for more information on how to configure the openpose package"
#cmake-gui
# to build with python 2.7
#export PYTHON_EXECUTABLE=/usr/bin/python2.7
#export PYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython2.7.so
cmake -DBUILD_PYTHON=ON -DPYBIND11_PYTHON_VERSION=2.7 ..
make -j`nproc`
sudo make install
sudo cp python/openpose/pyopenpose.so /usr/local/lib/python2.7/dist-packages/
cd ..
OPENPOSE_PATH_DIR=$PWD

# go to home directory and export environment variables in the bashrc
cd
echo "\n\n# Export OPENPOSE environment variables" >> ".bashrc"
echo "export OPENPOSE_PATH=$OPENPOSE_PATH_DIR" >> ".bashrc"

# Test installation (check github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/quick_start.md)
#cd OPENPOSE_PATH_DIR
#./build/examples/openpose/openpose.bin
# Python examples are in github.com/CMU-Perceptual-Computing-Lab/openpose/tree/master/examples/tutorial_api_python

# return to original directory
cd currentdir


# Troubleshootings
# - If you obtain the following error "nvcc fatal : Unsupported gpu architecture 'compute_<xx>'", just modify the
#   cmake/Cuda.cmake file, from the line "set(Caffe_known_gpu_archs "...")", remove the architectures that your
#   computer does not possess. You can also specify the architecture in the cmake line with the option "CUDA_ARCH".
# - If you obtain an error with the python version, you have to specify the version you want to use to pybind11, by
#   setting the option "-DPYBIND11_PYTHON_VERSION=2.7"
# - If you have the following error "Cuda check failed (30 vs. 0): unknown error" when launching a basic example
#   in openpose as given in "https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/quick_start.md",
#   then the error comes from your Nvidia GPU driver or your CUDA toolkit. Check the solutions given in the following
#   link: https://github.com/NVIDIA/DIGITS/issues/1663
#   Before tempting any solutions, try to reboot your computer (this solution worked for me). Otherwise, this will
#   probably require you to reinstall the Nvidia drivers / CUDA toolkit.
# - If you have an error with Qt such as: qt.qpa.plugin: Could not find the Qt platform plugin "xcb" in "", reinstall
#   the package, or install qtcreator.
