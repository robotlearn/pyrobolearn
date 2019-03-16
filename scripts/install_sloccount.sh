#!/bin/sh
# This bash installs the 'sloccount' command which counts source lines of code, and provides several statisticsabout this last one.
# Once installed, go to the pyrobolearn package and type:
# $ sloccount *.py */*.py

# define variables
origindir=$PWD
scriptdir=`dirname $0`

# install package
sudo apt install sloccount -y

# compute statistics for pyrobolearn software
cd $scriptdir; cd ..
sloccount *.py */*.py

# return to the original directory
cd $origindir
