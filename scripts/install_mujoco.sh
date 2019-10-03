#!/bin/bash
# Install the MuJoCo simulator and the Python wrapper (mujoco_py) provided by OpenAI
# You have to provide the directory where to install the Mujoco and mujoco_py packages and which contains the mjkey.txt
# References:
# - install mujoco: https://www.roboti.us/index.html
# - install mujoco_py: https://github.com/openai/mujoco-py

# Define few variables
ORIGIN_DIR=$PWD

# install mujoco

# Pass as the first argument the location where to install MuJoCo and mujoco_py
if [[ -z $1 ]]; then
    echo "Specify the location where to install the MuJoCo and mujoco_py source directories as the first argument."
    exit
fi
MUJOCO_PATH=$1
cd $MUJOCO_PATH

# check OS
if [[ "$OSTYPE" == "linux-gnu" ]]; then  # LINUX
    PACKAGE="mujoco200_linux"
elif [[ "$OSTYPE" == "darwin"* ]]; then  # Mac OSX
    PACKAGE="mujoco200_macos"
elif [[ "$OSTYPE" == "msys" ]]; then  # Windows
    PACKAGE="mujoco200_win64"
else
    echo "This OS is not supported. Expecting a Linux, Mac OSX, or Windows system."
    exit
fi

# download package and unzip it
wget "https://www.roboti.us/download/${PACKAGE}.zip"
unzip "${PACKAGE}.zip"

# install mujoco_py
sudo apt install libosmesa6-dev libgl1-mesa-glx libglfw3

export MUJOCO_PY_MJKEY_PATH="${MUJOCO_PATH}/mjkey.txt"
export MUJOCO_PY_MJPRO_PATH="${MUJOCO_PATH}/${PACKAGE}"
export MUJOCO_PY_MUJOCO_PATH="${MUJOCO_PATH}/${PACKAGE}"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${MUJOCO_PATH}/${PACKAGE}/bin
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so  # :/usr/lib/x86_64-linux-gnu/libGL.so  # for visualization

# write variables to export in the ~/.bashrc file, and source it
cd
echo "" >> .bashrc
echo "# MuJoCo" >> .bashrc
echo "export MUJOCO_PY_MJKEY_PATH=${MUJOCO_PATH}/mjkey.txt" >> .bashrc
echo "export MUJOCO_PY_MJPRO_PATH=${MUJOCO_PATH}/${PACKAGE}" >> .bashrc
echo "export MUJOCO_PY_MUJOCO_PATH=${MUJOCO_PATH}/${PACKAGE}" >> .bashrc
echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${MUJOCO_PATH}/${PACKAGE}/bin" >> .bashrc
echo "export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so" >> .bashrc
# source .bashrc

git clone https://github.com/openai/mujoco-py
cd mujoco-py
pip install -e .
# sudo python setup.py install

# return to the original directory
cd $ORIGIN_DIR
