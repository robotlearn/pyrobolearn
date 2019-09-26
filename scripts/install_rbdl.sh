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


## to wrap GetBodyId(), GetBodyName() function
# 
# Inside crbdl.pxd,
# - under `cdef cppclass Model`, add:
#
# unsigned int GetBodyId(const char *body_name)
#
# string GetBodyName(unsigned int body_id)
#
# - under `cdef extern from "<rbdl/Kinematics.h>" namespace "RigidBodyDynamics":`, add:
#
# cdef void UpdateKinematics (Model& model,
#            const VectorNd &q,
#            const VectorNd &qdot,
#            const VectorNd &qddot)
#
# cdef Matrix3d CalcBodyWorldOrientation (Model& model,
#            const VectorNd &q,
#            const unsigned int body_id,
#            bool update_kinematics)
#
# Inside rbdl-wrapper.pyx,
# - under `cdef class Model`, add:
#
# def GetBodyId (self, char* body_name):
#      return self.thisptr.GetBodyId(body_name)
#
# def GetBodyName(self, unsigned int index):
#      return self.thisptr.GetBodyName(index)
#
# - under `kinematics.h` comment block, add:
#
# def UpdateKinematics(
#        Model model,
#        np.ndarray[double, ndim=1, mode="c"] q,
#        np.ndarray[double, ndim=1, mode="c"] qdot,
#        np.ndarray[double, ndim=1, mode="c"] qddot
#):
#    crbdl.UpdateKinematics(
#            model.thisptr[0],
#            NumpyToVectorNd (q),
#            NumpyToVectorNd (qdot),
#            NumpyToVectorNd (qddot)
#    )
#
# def CalcBodyWorldOrientation (Model model,
#        np.ndarray[double, ndim=1, mode="c"] q,
#        unsigned int body_id,
#        update_kinematics=True):
#    return Matrix3dToNumpy (crbdl.CalcBodyWorldOrientation (
#            model.thisptr[0],
#            NumpyToVectorNd (q),
#            body_id,
#            update_kinematics
#            ))
#
# - under `Conversion Numpy <-> Eigen` comment block, add:
#
# cdef np.ndarray Matrix3dToNumpy (crbdl.Matrix3d cM):
#    result = np.ndarray ([3, 3])
#    for i in range (3):
#        for j in range (3):
#            result[i,j] = cM.coeff(i,j)
#    return result
#
# then do cmake, make, install again
