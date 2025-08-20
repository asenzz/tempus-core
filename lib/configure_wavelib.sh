#!/usr/bin/env bash
PREFIX=/usr/local
if [ ! -z $1 ]; then
    PREFIX=$1
fi
echo "Prefix is $PREFIX."
BUILD_DIR="build"

WD=`pwd`
PROJECT_DIR=$(cd `dirname "${BASH_SOURCE[0]}"`; pwd)/wavelib

cd $PROJECT_DIR

if [ -d $BUILD_DIR ]; then
    rm -rf $BUILD_DIR/*
else
    mkdir $BUILD_DIR
fi

cd $BUILD_DIR

cmake ..
make -k -j$((`nproc` - 1))

cp ./Bin/libwavelib.a $PREFIX/lib/
mkdir -p $PREFIX/include/wavelib/header
cp ../header/wavelib.h $PREFIX/include/wavelib/header