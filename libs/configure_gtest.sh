#!/usr/bin/env bash

BUILD_DIR="build"

WD=`pwd`
PROJECT_DIR=$(cd `dirname "${BASH_SOURCE[0]}"`; pwd)/gtest-1.7.0

cd $PROJECT_DIR

if [ -d $BUILD_DIR ]
then
    rm -rf $BUILD_DIR/*
else
    mkdir $BUILD_DIR
fi

cd $BUILD_DIR


cmake ..
make -j$((`nproc` - 1))
cp *.a ../lib

