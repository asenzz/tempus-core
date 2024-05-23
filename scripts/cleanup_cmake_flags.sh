#!/bin/bash
find . -name flags.make -exec sed -i 's/\(CUDA_FLAGS = .*\) \-fiopenmp \-fopenmp-targets=spir64\(.*\)/\1\2/g' {} \;
