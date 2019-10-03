#!/bin/sh
# This bash installs the 'sloccount' and 'cloc' commands which provide statistics such as the number of lines of code,
# and others of a certain software.
# Once installed, go to the pyrobolearn package and type:
# $ sloccount .
# or
# $ cloc .

# define variables
origindir=$PWD
scriptdir=`dirname $0`

# install packages
sudo apt install sloccount -y
sudo apt install cloc -y

# compute statistics for pyrobolearn software
cd $scriptdir; cd ..
echo "\n\nUsing Sloccount:"
sloccount .
echo "\n\nUsing clock:"
cloc .

# return to the original directory
cd $origindir
