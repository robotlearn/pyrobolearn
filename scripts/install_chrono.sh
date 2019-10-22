#!/bin/bash
# Install the Chrono simulator and its Python wrapper (pychrono)
# References:
# - install chrono:
#   - https://github.com/projectchrono/chrono
#   - http://api.projectchrono.org/development/tutorial_install_chrono.html
# - install pychrono:
#   - http://api.projectchrono.org/development/module_python_installation.html

# Define few variables
ORIGIN_DIR=$PWD

# install chrono

# Pass as the first argument the location where to install chrono and pychrono
if [[ -z $1 ]]; then
    echo "Specify the location where to install the chrono and pychrono source directories as the first argument."
    exit
fi
CHRONO_PATH=$1
cd $CHRONO_PATH

# install dependencies
sudo apt-get install libeigen3-dev cmake git
sudo apt install build-essential x11-common libxxf86vm-dev libglu1-mesa-dev freeglut3 xorg-dev
sudo apt install libirrlicht1.8 libirrlicht-dev libirrlicht-doc

# install SWIG
git clone https://github.com/swig/swig.git
cd swig
sudo apt-get install automake
./autogen.sh
./configure
sudo apt-get install bison flex
make -j4
sudo make install  # install swig in /usr/local/bin/

# install chrono and pychrono
cd $CHRONO_PATH
git clone https://github.com/projectchrono/chrono
cd chrono
mkdir build
cd build
cmake .. -DENABLE_MODULE_IRRLICHT=True -DENABLE_MODULE_PYTHON=True -DSWIG_EXECUTABLE=/usr/local/bin/swig
make -j4
sudo make install
# the pychrono Python library is in chrono/build/bin

# return to the original directory
cd $ORIGIN_DIR
