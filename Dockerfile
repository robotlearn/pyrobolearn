FROM ubuntu:18.04
LABEL maintainer="Brian Delhaisse <brian.delhaisse@iit.it> - Daniele Bonatto <dbonatto@ulb.ac.be>"

RUN apt-get update && yes | apt-get upgrade && apt-get -y install wget

### Main Directory

RUN mkdir /pyrobolearn/
WORKDIR /pyrobolearn/

### Main Downloads to avoid rebuilding the docker image every time we change something

RUN wget https://bitbucket.org/rbdl/rbdl/get/default.zip 
RUN wget https://www.coin-or.org/download/source/Ipopt/Ipopt-3.12.12.zip
RUN wget https://sourceforge.net/projects/roboticslab/files/External/nite/NiTE-Linux-x64-2.2.tar.bz2
RUN wget -c "https://developer.nvidia.com/compute/cuda/10.1/Prod/local_installers/cuda-repo-ubuntu1804-10-1-local-10.1.168-418.67_1.0-1_amd64.deb"
RUN wget -c "https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/libcudnn7_7.6.0.64-1+cuda10.1_amd64.deb"
RUN wget -c "https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/libcudnn7-dev_7.6.0.64-1+cuda10.1_amd64.deb"
# ! We never remove those, the image may be a little big


### Main packages

# tzdata is interactive, we don't want this
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get -y install python3.6 python3-pip build-essential clang  tzdata python3-tk cmake gcc gfortran g++ wget libpng-dev libopenblas-dev git python3.6-dev python-numpy lsb-release libeigen3-dev unzip libxi-dev libxmu-dev libusb-1.0.0-dev pkg-config freeglut3-dev libudev-dev openjdk-8-jdk usbutils libopenni-sensor-primesense0 libatlas-base-dev libprotobuf-dev libleveldb-dev libsnappy-dev libhdf5-serial-dev protobuf-compiler libboost-all-dev libgflags-dev libgoogle-glog-dev liblmdb-dev python3-protobuf protobuf-compiler python3-setuptools python3-dev libopencv-dev opencl-headers ocl-icd-opencl-dev libviennacl-dev
# TODO I we should put all those in the requirements.txt instead but we need to be careful with the ln -s command
RUN yes | pip3 install numpy
RUN yes | pip3 install torch torchvision
RUN yes | pip3 install cython
RUN yes | pip3 install gpytorch
# scikit-build needed first for installing the requirements properly
RUN yes | pip3 install scikit-build
RUN yes | pip3 install cmake
RUN ln -s /usr/include/locale.h /usr/include/xlocale.h
RUN yes | pip3 install git+https://github.com/python-control/python-control

# Needed for slycot as the cmake did not found fortranobject.c
# /usr/local/lib/python3.6/dist-packages/numpy/f2py/src/fortranobject.c
RUN ln -sf /usr/local/lib/python3.6/dist-packages/numpy/ /usr/lib/python3/dist-packages/numpy
ADD requirements.txt .
RUN yes | pip3 install -r requirements.txt


### rbdl
# We need to install this before ROS as ROS will trigger all the tests of rbdl and they don't work

RUN mkdir rbdl
WORKDIR /pyrobolearn/rbdl/
RUN mv ../default.zip . && unzip default.zip && mv rbdl-rbdl-* rbdl && cd rbdl && mkdir build && cd build/ && cmake -D CMAKE_BUILD_TYPE=Release -D RBDL_BUILD_ADDON_URDFREADER=ON -D RBDL_BUILD_PYTHON_WRAPPER=ON ../ && make -j`nproc` &&  make install && cp python/rbdl.so /usr/local/lib/python3.6/dist-packages && cp python/rbdl.so /usr/local/lib/python3.6/site-packages

WORKDIR /pyrobolearn/


### ros-melodic

RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
RUN yes | apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654
RUN apt-get update
RUN apt-get -y install ros-melodic-desktop-full
RUN rosdep init
RUN rosdep update
RUN apt-get -y install python-rosinstall ros-melodic-catkin 
RUN sh -c 'echo "deb http://packages.osrfoundation.org/gazebo/ubuntu-stable `lsb_release -cs` main" > /etc/apt/sources.list.d/gazebo-stable.list'
RUN wget http://packages.osrfoundation.org/gazebo.key -O - | apt-key add -
RUN apt-get -y install ros-melodic-gazebo-ros-pkgs ros-melodic-gazebo-ros-control
RUN apt-get -y install ros-melodic-ros-control ros-melodic-ros-controllers

RUN apt-get -y install python-catkin-tools
#RUN echo "source /opt/ros/melodic/setup.bash" >> ~/.bashrc && /bin/bash -c 'source ~/.bashrc'
#RUN /bin/bash -x -c "source /opt/ros/melodic/setup.bash"
ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/lib"
ENV OPENPOSE_PATH="/pyrobolearn/openpose/"
ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/opt/ros/melodic/lib"
ENV PATH="/opt/ros/melodic/bin:${PATH}"
ENV ROS_DISTRO="melodic"
ENV ROS_ETC_DIR="/opt/ros/melodic/etc/ros"
ENV ROS_ROOT="/opt/ros/melodic/share/ros"
ENV ROS_PYTHON_VERSION=2
ENV ROS_VERSION=1
ENV ROS_PACKAGE_PATH="/opt/ros/melodic/share"
ENV ROS_MASTER_URI="http://localhost:11311"
ENV PYTHONPATH="/opt/ros/melodic/lib/python2.7/dist-packages"
ENV CMAKE_PREFIX_PATH="/opt/ros/melodic:${CMAKE_PREFIX_PATH}"
ENV LD_LIBRARY_PATH="/opt/ros/melodic/lib:${LD_LIBRARY_PATH}"
ENV PKG_CONFIG_PATH="/opt/ros/melodic/lib/pkgconfig"

RUN mkdir -p /pyrobolearn/catkin_ws/src && cd /pyrobolearn/catkin_ws/src && catkin_init_workspace && cd /pyrobolearn/catkin_ws/ && catkin_make
#RUN /bin/bash -x -c "source /pyrobolearn/catkin_ws/devel/setup.bash"
#RUN echo "source /pyrobolearn/catkin_ws/devel/setup.bash" >> ~/.bashrc && /bin/bash -c 'source ~/.bashrc'
ENV CMAKE_PREFIX_PATH="/pyrobolearn/catkin_ws/devel:${CMAKE_PREFIX_PATH}"
ENV LD_LIBRARY_PATH="/pyrobolearn/catkin_ws/devel/lib:${LD_LIBRARY_PATH}"
ENV ROS_PACKAGE_PATH="/pyrobolearn/catkin_ws/src:${ROS_PACKAGE_PATH}"
ENV ROSLIST_PACKAGE_DIRECTORIES="/pyrobolearn/catkin_ws/devel/share/common-lisp"


WORKDIR /pyrobolearn/


### lpopt

RUN unzip Ipopt-3.12.12.zip
WORKDIR /pyrobolearn/Ipopt-3.12.12/ThirdParty/Blas
RUN ./get.Blas && mkdir -p build && cd build && ../configure --prefix=/usr/local --disable-shared --with-pic && make && make install
WORKDIR /pyrobolearn/Ipopt-3.12.12/ThirdParty/Lapack
RUN ./get.Lapack && mkdir -p build && cd build && ../configure --prefix=/usr/local --disable-shared --with-pic --with-blas="/usr/local/lib/libcoinblas.a -lgfortran" && make && make install
WORKDIR /pyrobolearn/Ipopt-3.12.12/ThirdParty/ASL
RUN ./get.ASL

WORKDIR /pyrobolearn/Ipopt-3.12.12/ThirdParty/Mumps
RUN ./get.Mumps


WORKDIR /pyrobolearn/Ipopt-3.12.12/
RUN ./configure --prefix=/usr/local/ coin_skip_warn_cxxflags=yes --with-blas="/usr/local/lib/libcoinblas.a -lgfortran" --with-lapack=/usr/local/lib/libcoinlapack.a && make -j`nproc` && make test && make install
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/lib/"
RUN yes | pip3 install ipopt

WORKDIR /pyrobolearn/

### openni2 and nite2

RUN git clone https://github.com/occipital/OpenNI2.git && cd OpenNI2 && make -j`nproc` && ln -s $PWD/Bin/x64-Release/libOpenNI2.so /usr/local/lib && ln -s $PWD/Bin/x64-Release/OpenNI2/ /usr/local/lib/ && ldconfig && cp $PWD/Packaging/Linux/primesense-usb.rules /etc/udev/rules.d/557-primesense-usb.rules

RUN git clone https://github.com/OpenKinect/libfreenect && cd libfreenect && mkdir -p build && cd build && cmake .. -DBUILD_OPENNI2_DRIVER=ON && make -j`nproc` && make install && ldconfig && ln -s /usr/local/lib/OpenNI2-FreenectDriver/libFreenectDriver.so /usr/local/lib/OpenNI2/Drivers && cd .. && cp platform/linux/udev/51-kinect.rules /etc/udev/rules.d/ && cd wrappers/python/ && python setup.py install && echo 'KERNEL == \"ttyUSB0\", MODE = \"0777\"' > 80-persistent-local-usb.rules

RUN tar xvf NiTE-Linux-*.tar.bz2 && cd NiTE-*/ && ./install.sh && ln -s $PWD/Redist/libNiTE2.so /usr/local/lib/ && ln -s $PWD/Include /usr/local/include/NiTE-Linux-x64-2.2 && cd .. && ldconfig

# x64
ENV OPENNI2_INCLUDE="/pyrobolearn/OpenNI2/Include"
ENV OPENNI2_REDIST64="/pyrobolearn/OpenNI2/Bin/x64-Release"
ENV NITE2_REDIST64="/pyrobolearn/NiTE-Linux-x64-2.2/Redist"


WORKDIR /pyrobolearn/

### openpose

RUN git clone https://github.com/CMU-Perceptual-Computing-Lab/openpose
WORKDIR /pyrobolearn/openpose
#RUN apt-get -y install sudo && useradd -m docker && echo "docker:docker" | chpasswd && adduser docker sudo
# BASE  REQUIREMENTS
# Already installed with apt at start
# INSTALL CUDA
RUN mv ../cuda-repo*.deb .
RUN dpkg -i cuda-repo*.deb
RUN apt-key add '/var/cuda-repo-10-1-local-10.1.168-418.67/7fa2af80.pub'
RUN apt-get update && apt-get -y install cuda
ENV PATH="/usr/local/cuda-10.1/bin:${PATH}"

# INSTALL CUDNN
RUN mv ../libcudnn*.deb .
RUN dpkg -i libcudnn7_7*.deb
RUN dpkg -i libcudnn7-dev*.deb

# Openpose installation
RUN mkdir build && cd build && cmake -DBUILD_PYTHON=ON -DPYBIND11_PYTHON_VERSION=3.6 .. && make -j`nproc` && make install

RUN cp build/python/openpose/pyopenpose.cpython-36m-x86_64-linux-gnu.so /usr/local/lib/python3.6/dist-packages/
ENV OPENPOSE_PATH=/pyrobolearn/openpose/


### Finishing

WORKDIR /pyrobolearn/

RUN ldconfig
RUN cat /proc/driver/nvidia/version && nvcc -V
RUN printenv

ADD . .

# TODO maybe transform for alpine linux

#ENTRYPOINT "python3"
