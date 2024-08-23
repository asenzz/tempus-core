#!/usr/bin/env bash

cd quickfix-v.1.14.4

./bootstrap
./configure CC=/usr/bin/gcc CXX=/usr/bin/g++ CFLAGS="" CXXFLAGS=""

make clean && make -j$((`nproc` - 1)) && sudo make install

