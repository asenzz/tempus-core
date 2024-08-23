#!/usr/bin/env bash

BUILD_DIR="build"

WD=`pwd`
PROJECT_DIR=$(cd `dirname "${BASH_SOURCE[0]}"`; pwd)/wavelib

cd $PROJECT_DIR

if [ -d $BUILD_DIR ]
then
    rm -rf $BUILD_DIR/*
else
    mkdir $BUILD_DIR
fi

cd $BUILD_DIR


cmake ..
make -k -j$((`nproc` - 1))

sudo cp ./Bin/libwavelib.a /usr/local/lib/
sudo mkdir -p /usr/local/include/wavelib/header
sudo cp ../header/wavelib.h /usr/local/include/wavelib/header
