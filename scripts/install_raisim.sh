#!/bin/bash
# Install the raisim simulator (raisimLib and raisimOgre) and its Python wrapper (raisimpy)
# ./install_raisim [<workspace_path> <local_build_path> <python_version> [<path/to/python/bin>]]
# References:
# - install raisimLib: https://github.com/leggedrobotics/raisimLib
# - install raisimOgre: https://github.com/leggedrobotics/raisimOgre
# - install raisimpy: https://github.com/robotlearn/raisimpy

# Define few variables
ORIGIN_DIR=$PWD

# define workspace, local build, and python version

# Pass as the first argument the workspace location
if [[ -z $1 ]]; then
    echo "Specify the WORKSPACE location where to clone/install the various RaiSim repositories."
    exit
fi

# Pass as the second argument the local build location
if [[ -z $2 ]]; then
    echo "Specify the local build location where to install the exported cmake libraries."
    exit
fi

# Pass as the third argument the python version
if [[ -z $3 ]]; then
    echo "Specify the python version."
    exit
fi

WORKSPACE=$1
LOCAL_BUILD=$2
PYTHON_VERSION=$3

#####################
# install raisimLib #
#####################

# install raisimLib dependencies
sudo apt-get install libeigen-dev

# install raisimLib
cd $WORKSPACE/raisimLib
mkdir build && cd build && cmake .. -DCMAKE_INSTALL_PREFIX=$LOCAL_BUILD && make install

######################
# install raisimOgre #
######################

# install cmake > 3.10
export cmake_version=3.14
export cmake_build=5
mkdir ~/temp
cd ~/temp
wget https://cmake.org/files/v$cmake_version/cmake-$cmake_version.$cmake_build.tar.gz
tar -xzvf cmake-$cmake_version.$cmake_build.tar.gz
cd cmake-$cmake_version.$cmake_build/
./bootstrap
make -j4
sudo make install

# install g++
sudo apt-get install -y software-properties-common
sudo add-apt-repository ppa:ubuntu-toolchain-r/test
sudo apt update
sudo apt install g++-8 gcc-8 -y

# install raisimOgre dependencies
sudo apt-get install libgles2-mesa-dev libxt-dev libxaw7-dev libsdl2-dev libzzip-dev libfreeimage-dev libfreetype6-dev libpugixml-dev

# build Ogre from source
cd $WORKSPACE
git clone https://github.com/leggedrobotics/ogre.git
cd ogre
git checkout raisimOgre
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$LOCAL_BUILD -DOGRE_BUILD_COMPONENT_BITES=ON -OGRE_BUILD_COMPONENT_JAVA=OFF -DOGRE_BUILD_DEPENDENCIES=OFF -DOGRE_BUILD_SAMPLES=False
make install -j8

# install raisimOgre
cd $WORKSPACE
cd raisimOgre && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=$LOCAL_BUILD -DCMAKE_INSTALL_PREFIX=$LOCAL_BUILD -DRAISIM_OGRE_EXAMPLES=True
make install -j8


####################
# Install raisimpy #
####################

# install pybind11
cd $WORKSPACE
git clone https://github.com/pybind/pybind11
mkdir build
cd build
if [[ -z $4 ]]; then  # Check if path to python binary is provided
    cmake ..
else
    cmake -DPYTHON_EXECUTABLE=$4 ..
fi
make check -j 4

# install raisimpy
cd $WORKSPACE
git clone https://github.com/robotlearn/raisimpy
cd raisimpy
mkdir build && cd build
cmake -DPYBIND11_PYTHON_VERSION=$PYTHON_VERSION -DCMAKE_PREFIX_PATH=$LOCAL_BUILD -DCMAKE_INSTALL_PREFIX=$LOCAL_BUILD ..
make -j4
make install
export PYTHONPATH=$PYTHONPATH:$LOCAL_BUILD/lib

# write variables to export in the ~/.bashrc file, and source it
cd
echo "" >> .bashrc
echo "# RaiSim" >> .bashrc
echo "export PYTHONPATH=\$PYTHONPATH:${LOCAL_BUILD}/lib" >> .bashrc
# source .bashrc

# return to the original directory
cd $ORIGIN_DIR
