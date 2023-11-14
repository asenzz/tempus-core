#!/usr/bin/env bash

cd gbenchmark
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$((`nproc` - 1))
