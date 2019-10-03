#!/bin/bash
# Install the Dart simulator and its Python wrapper (dartpy)
# References
# - Install DART on Ubuntu: https://dartsim.github.io/install_dart_on_ubuntu.html
# - Install dartpy on Ubuntu: https://dartsim.github.io/install_dartpy_on_ubuntu.html

# Define few variables
ORIGIN_DIR=$PWD

# install dart

# Pass as the first argument the location where to install DART
if [[ -z $1 ]]; then
    echo "Specify the location where to install the DART source directory as the first argument."
    exit
fi
cd $1

sudo apt-get remove libdart*

# install dependencies
sudo apt-get install build-essential cmake pkg-config git
sudo apt-get install libeigen3-dev libassimp-dev libccd-dev libfcl-dev libboost-regex-dev libboost-system-dev
sudo apt-get install libnlopt-dev
sudo apt-get install coinor-libipopt-dev
sudo apt-get install libbullet-dev
sudo apt-get install libode-dev
sudo apt-get install liboctomap-dev
sudo apt-get install libflann-dev
sudo apt-get install libtinyxml2-dev liburdfdom-dev
sudo apt-get install libxi-dev libxmu-dev freeglut3-dev libopenscenegraph-dev
sudo apt-get install python3-pip

# build and install DART and dartpy
git clone git://github.com/dartsim/dart.git
cd dart
git checkout tags/v6.9.0
mkdir build
cd build
cmake .. -DDART_BUILD_DARTPY=ON -DCMAKE_INSTALL_PREFIX=/usr/ -DCMAKE_BUILD_TYPE=Release
make -j4
make -j4 dartpy

sudo make install

# return to the original directory
cd $ORIGIN_DIR
