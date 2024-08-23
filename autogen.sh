#!/usr/bin/env bash

BUILD_DIR="build"
LIB_DIR=$HOME/lib

WD=`pwd`
PROJECT_DIR=$(cd `dirname "${BASH_SOURCE[0]}"`; pwd)

cd $PROJECT_DIR

lib/configure_gtest.sh
lib/configure_wavelib.sh

if [ -d $BUILD_DIR ]
then
    rm -rf $BUILD_DIR/*
else
    mkdir $BUILD_DIR
fi

cd $BUILD_DIR

HOST_NAME=`hostname -f`

if [[ $HOST_NAME == sl* ]] 
then
    HOST_TARGET="paramtune"
    HOST_INCLUDE_OPTIONS="-I/opt/soft/boost-${TEMPUS_BOOST_VERSION}/include/ -I/opt/soft/lz4/include -I/home/jarko/lib/ViennaCL-1.7.0/ -I/opt/intel/opencl-sdk/include -I$PROJECT_DIR/lib"
    HOST_LINKER_OPTIONS="-L/opt/soft/boost-${TEMPUS_BOOST_VERSION}/lib/ -L/opt/soft/lz4/lib -L/opt/intel/opencl-sdk/lib64 -L$PROJECT_DIR/lib/wavelib/build/Bin/"
    HOST_SPECIFIC_LIBRARIES="/opt/soft/glibc-2.18/lib/libc.so"

    cmake -DCMAKE_MODULE_PATH="$PROJECT_DIR/host_config/sl/cmake/" -DMODULES=$HOST_TARGET \
        -DCMAKE_CXX_FLAGS="-Wno-ignored-attributes $HOST_INCLUDE_OPTIONS $HOST_LINKER_OPTIONS -DBUILD_WITHOUT_SVR_FIX -DPGSTD=std" \
        -DHOST_SPECIFIC_LIBRARIES="$HOST_SPECIFIC_LIBRARIES" \
        $@ \
        ..
    echo
    echo "Configuring for BAS complete." 
else 
   cmake $@ .. 
fi

if [[ `basename $WD` != "$BUILD_DIR"  ]]
then
    echo "Please cd to the build directory."
fi

CPU_NUMBER=`cat /proc/cpuinfo | grep processor | wc -l`

echo "Please run 'make -j$CPU_NUMBER $HOST_TARGET' to build the project"
