#!/bin/sh
# This bash script install the gdal library and the official python wrapper

sudo apt-get install libgdal-dev
export CPLUS_INCLUDE_PATH=/usr/include/gdal
export C_INCLUDE_PATH=/usr/include/gdal
#sudo pip install gdal
sudo apt-get install python-gdal