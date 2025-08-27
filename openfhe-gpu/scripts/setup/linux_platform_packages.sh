#!/bin/bash
#
# This script can be used to install all of the requistes for the Linux Platform
#

# run update before installing every package

# install cmake
apt-get update
apt-get install -y cmake

# install required packages
apt-get update
apt-get install -y build-essential
apt-get update
apt-get install -y autoconf
apt-get update
apt-get install -y libtool
apt-get update
apt-get install -y libgmp-dev
apt-get update
apt-get install -y libntl-dev

# install all necessary and additional compilers
apt-get update
apt-get install -y g++-9 g++-10 g++-11 g++-12
apt-get update
apt-get install -y clang-12 clang-13 clang-14 clang-15

# for documentation
apt-get update
apt-get install -y doxygen
apt-get update
apt-get install -y graphviz

# python packages
apt-get update
apt-get install -y python3-pip
apt-get update
pip install pybind11[global]
apt-get update
apt-get install -y python3-pytest
# to verify pytest installation: python3 -m pip show pytest