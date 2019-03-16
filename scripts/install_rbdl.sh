#!/bin/sh
# This bash script install the rbdl library and the official python wrapper
# 
# repo + documentation: https://rbdl.bitbucket.io/
# paper: https://pdfs.semanticscholar.org/f129/5068f646ea9379da72ba5183276ca73a038d.pdf

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

# Install requirements
# update & upgrade #
sudo apt-get update
sudo apt-get upgrade

#install cython
sudo apt-get -y install cython
# build-tools
sudo apt-get -y install build-essential
sudo apt-get -y install cmake
# eigen 
sudo apt-get -y install libeigen3-dev 

# download and install rbdl
sudo rm -r rbdl
wget https://bitbucket.org/rbdl/rbdl/get/default.zip
unzip default.zip
rm default.zip
mv rbdl-rbdl-* rbdl
cd rbdl
mkdir build && cd build/
cmake -D CMAKE_BUILD_TYPE=Release -D RBDL_BUILD_ADDON_URDFREADER=ON -D RBDL_BUILD_PYTHON_WRAPPER=ON ../ 
# make error fix: cd addons/urdfreader
# open urdfreader.cc
# remove the line: #include "ros/ros.h"
make -j`nproc`
sudo make install

## to use python wrapper, you need to install Cython and NumPy
cd python
sudo cp rbdl.so /usr/local/lib/python2.7/dist-packages 
sudo cp rbdl.so /usr/local/lib/python2.7/site-packages

# return to original directory
cd currentdir


## to wrap GetBodyId() function
# 
# Inside crbdl.pxd, add:
# unsigned int GetBodyId(const char *body_name)
# inside cdef cppclass Model:
# 
# Inside rbdl-wrapper.pyx, add:
# def GetBodyId (self, char* body_name):
#      return self.thisptr.GetBodyId(body_name)
# inside cdef class Model:
# 
# then do cmake, make, install again
