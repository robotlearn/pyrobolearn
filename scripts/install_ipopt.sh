#!/bin/sh
# File taken from CarND-MPC-Project from Udacity but modified to install python wrapper as well
# Source: https://github.com/udacity/CarND-MPC-Project
#
# To use this script:
# $ sudo ./install_openni2_nite2.sh <path/to/install/packages>
#
# You might need to source your .bashrc file at the end.
#
# IPopt installation on MacOSX:
# brew install ipopt --with-openblas

# Define few variables
currentdir=$PWD

# Pass the Ipopt source directory as the first argument
if [ -z $1 ]
then
    echo "Specifiy the location of the Ipopt source directory in the first argument."
    exit
fi
cd $1

# specify ipopt version
version=3.12.12

# install gfortran
sudo apt-get install gfortran

# download ipopt
wget https://www.coin-or.org/download/source/Ipopt/Ipopt-${version}.zip
unzip Ipopt-${version}.zip
rm Ipopt-${version}.zip

cd Ipopt-${version}

prefix=/usr/local
srcdir=$PWD

echo "Building Ipopt from ${srcdir}"
echo "Saving headers and libraries to ${prefix}"

# BLAS
cd $srcdir/ThirdParty/Blas
./get.Blas
mkdir -p build && cd build
../configure --prefix=$prefix --disable-shared --with-pic
make
sudo make install

# Lapack
cd $srcdir/ThirdParty/Lapack
./get.Lapack
mkdir -p build && cd build
../configure --prefix=$prefix --disable-shared --with-pic \
    --with-blas="$prefix/lib/libcoinblas.a -lgfortran"
make
sudo make install

# ASL
cd $srcdir/ThirdParty/ASL
./get.ASL

# MUMPS
cd $srcdir/ThirdParty/Mumps
./get.Mumps

# build everything
cd $srcdir
./configure --prefix=$prefix coin_skip_warn_cxxflags=yes \
    --with-blas="$prefix/lib/libcoinblas.a -lgfortran" \
    --with-lapack=$prefix/lib/libcoinlapack.a
make
make test
sudo make -j1 install

# write variables to export in the ~/.bashrc file, and source it
cd
echo "" >> .bashrc
echo "# Export environment variables for IPopt" >> .bashrc
echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/usr/local/lib/" >> .bashrc
# source .bashrc

# install python package
pip install ipopt

# return to the original directory
cd $currentdir
