#!/bin/sh
# This bash script install the OpenNI2 and Nite2 packages (tested on Ubuntu 16.04)
#
# To use this script: 
# $ sudo ./install_openni2_nite2.sh <path/to/install/packages>
#
# You might need to source your .bashrc file at the end.
#
# Github repo: 
# - https://github.com/OpenNI/OpenNI
# - https://github.com/occipital/OpenNI2
# - https://github.com/OpenKinect/libfreenect
#
# References: 
# - https://roboticslab-uc3m.gitbook.io/installation-guides/install-openni-nite
# - https://stackoverflow.com/questions/32415089/install-openni2-and-nite2-on-linux
# - https://codeyarns.com/2015/08/04/how-to-install-and-use-openni2/
#
# To test kinect (with libfreenect):
# $ freenect-glview
# To test asus/kinect (with openni, check in the Bin and Tools folders):
# $ NiViewer
# $ UserViewer

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
    echo "Specify the absolute location (or relative to home) where to install the OpenNI2, libfreenect, and NiTE2 packages."
    exit
fi

# move to that directory
cd; cd $1

# install requirement packages
echo "Installing prerequired packages..."
sudo apt-get install git g++ cmake libxi-dev libxmu-dev libusb-1.0-0-dev pkg-config freeglut3-dev build-essential libudev-dev openjdk-8-jdk usbutils

# driver for Asus Xtion Pro Live OpenNI driver (Ubuntu)
sudo apt-get install libopenni-sensor-primesense0

# install OpenNI2
echo "Installing OpenNI2..."
cd; cd $1
git clone https://github.com/occipital/OpenNI2.git  # We used to have a fork off 6857677beee08e264fc5aeecb1adf647a7d616ab with working copy of Xtion Pro Live OpenNI2 driver.
cd OpenNI2
make -j`nproc`
sudo ln -s $PWD/Bin/x64-Release/libOpenNI2.so /usr/local/lib/  # $PWD should be /yourPathTo/OpenNI2
sudo ln -s $PWD/Bin/x64-Release/OpenNI2/ /usr/local/lib/  # $PWD should be /yourPathTo/OpenNI2
sudo ln -s $PWD/Include /usr/local/include/OpenNI2  # $PWD should be /yourPathTo/OpenNI2
sudo ldconfig
#sudo $PWD/Packaging/Linux/install.sh
sudo cp $PWD/Packaging/Linux/primesense-usb.rules /etc/udev/rules.d/557-primesense-usb.rules  # install udev rules for usb devices
# Install prebuilt SDK from online (instead of git)
#wget https://s3.amazonaws.com/com.occipital.openni/OpenNI-Linux-x64-2.2.0.33.tar.bz2
#tar -xjvf OpenNI-Linux-x64-2.2.0.33.tar.bz2 && rm OpenNI-Linux-x64-2.2.0.33.tar.bz2
#cd OpenNI-Linux-x64-2.2
#sudo ./install.sh  # this installs 'primesense-usb.rules' in /etc/udev/rules.d/, and defines environment variables 'OPENNI2_*'
#cd; cd $1
#cp libfreenect/build/lib/OpenNI2-FreenectDriver/libFreenectDriver.so OpenNI-Linux-x64-2.2/Redist/OpenNI2/Drivers/
#cp libfreenect/build/lib/OpenNI2-FreenectDriver/libFreenectDriver.so OpenNI-Linux-x64-2.2/Tools/OpenNI2/Drivers/
#cd OpenNI-Linux-x64-2.2/Tools
#./NiViewer

# Test the installation
#cd Bin/x64-Release/
#./NiViewer

# install libfreenect because it contains a driver to be used with OpenNI2
echo "Installing libfreenect..."
cd; cd $1
git clone https://github.com/OpenKinect/libfreenect
cd libfreenect && mkdir -p build
cd build
cmake .. -DBUILD_OPENNI2_DRIVER=ON
make -j`nproc`
sudo make install
sudo ldconfig
sudo ln -s /usr/local/lib/OpenNI2-FreenectDriver/libFreenectDriver.so /usr/local/lib/OpenNI2/Drivers
cd ..
sudo cp platform/linux/udev/51-kinect.rules /etc/udev/rules.d/  # don't need to run the app as root
cd wrappers/python/
sudo python setup.py install

# Test the installation
#freenect-glview

# Set rules to avoid needing sudo when trying to read data from the sensors connected via USB port
sudo sh -c "echo 'KERNEL == \"ttyUSB0\", MODE = \"0777\"' > 80-persistent-local-usb.rules"

# install NiTE2
echo "Installing NiTE2..."
cd; cd $1
wget https://sourceforge.net/projects/roboticslab/files/External/nite/NiTE-Linux-x64-2.2.tar.bz2
tar xvf NiTE-Linux-x64-2.2.tar.bz2 && rm NiTE-Linux-x64-2.2.tar.bz2
cd NiTE-Linux-x64-2.2/
sudo ./install.sh    # defines environment variables 'NITE2_*'
sudo ln -s $PWD/Redist/libNiTE2.so /usr/local/lib/
sudo ln -s $PWD/Include /usr/local/include/NiTE-Linux-x64-2.2
cd ..
sudo ldconfig
# Test the installation
#cd NiTE-Linux-x64-2.2/Samples/Bin
#./UserViewer

# write variables to export in the ~/.bashrc file, and source it
cd
OUT_FILE=".bashrc"
echo "" >> $OUT_FILE
echo "# Export OPENNI and NITE environment variables" >> $OUT_FILE
echo "export OPENNI2_INCLUDE=$1/OpenNI2/Include" >> $OUT_FILE
if [ `uname -m` = "x86_64" ];
then
    #echo "export OPENNI2_REDIST64=$1/OpenNI-Linux-x64-2.2/Redist" >> $OUT_FILE
    echo "export OPENNI2_REDIST64=$1/OpenNI2/Bin/x64-Release" >> $OUT_FILE
    echo "export NITE2_REDIST64=$1/NiTE-Linux-x64-2.2/Redist" >> $OUT_FILE 
else
    #echo "export OPENNI2_REDIST=$1/OpenNI-Linux-x64-2.2/Redist" >> $OUT_FILE
    echo "export OPENNI2_REDIST=$1/OpenNI2/Bin/x64-Release" >> $OUT_FILE
    echo "export NITE2_REDIST=$1/NiTE-Linux-x64-2.2/Redist" >> $OUT_FILE
fi
# source .bashrc

# return to the original directory
cd $currentdir
