#!/bin/bash
# Install the VREP simulator and the Python wrapper (PyRep)
# If you use a virtual environment for Python, activate this last one before launching this script
# References:
# - install VREP: http://www.coppeliarobotics.com/downloads.html
# - install PyRep: https://github.com/stepjam/PyRep

# Define few variables
ORIGIN_DIR=$PWD

# install VREP

# Pass as the first argument the location where to install VREP and PyRep
if [[ -z $1 ]]; then
    echo "Specify the location where to install the VREP and PyRep source directories as the first argument."
    exit
fi
cd $1

# check OS
if [[ "$OSTYPE" == "linux-gnu" ]]; then
    # LINUX
    OS=`cat /etc/os-release | grep NAME= | head -1 | cut -c 7- | sed 's/.$//'`  # e.g. Ubuntu
    VERSION=`cat /etc/os-release | grep VERSION= | head -1 | cut -c 10- | cut -c -5`  # e.g. 16.04

    # Check Ubuntu version
    if [[ $VERSION == "16.04" ]]; then
        PACKAGE="V-REP_PRO_EDU_V3_6_2_Ubuntu16_04"
    elif [[ $VERSION == "18.04" ]]; then
        PACKAGE="V-REP_PRO_EDU_V3_6_2_Ubuntu18_04"
    else
        echo "This Linux distribution ${OS} ${VERSION} is not supported; Ubuntu 16.04 or 18.04 is required."
        exit
    fi

    # download package and unpack it
    wget "http://www.coppeliarobotics.com/files/${PACKAGE}.tar.xz"
    tar -xvf "${PACKAGE}.tar.xz"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    # Mac OSX
    PACKAGE="V-REP_PRO_EDU_V3_6_2_Mac"
    # download package and unpack it
    wget "http://www.coppeliarobotics.com/files/${PACKAGE}.zip"
    unzip "${PACKAGE}.zip"
#elif [[ "$OSTYPE" == "msys" ]]; then
#    # Windows
#    PACKAGE="V-REP_PRO_EDU_V3_6_2_Setup"
#    wget "http://www.coppeliarobotics.com/files/$PACKAGE.exe"
else
    echo "This OS is not supported. Expecting an Ubuntu or Mac OSX system."
    exit
fi

# export variables
export VREP_ROOT="${PWD}/${PACKAGE}"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$VREP_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$VREP_ROOT

# write variables to export in the ~/.bashrc file, and source it
cd
echo "" >> .bashrc
echo "# VREP" >> .bashrc
echo "export VREP_ROOT=${PWD}/${PACKAGE}" >> .bashrc
echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:\$VREP_ROOT" >> .bashrc
echo "export QT_QPA_PLATFORM_PLUGIN_PATH=\$VREP_ROOT" >> .bashrc
# source .bashrc

# install PyRep
cd $1
git clone https://github.com/stepjam/PyRep.git
cd PyRep

# install requirements and setup (if you don't have a virtual environment with Py3, you might need to replace
# pip by pip3, and python by python3.
pip install -r requirements.txt
python setup.py install --user

# return to the original directory
cd $ORIGIN_DIR
