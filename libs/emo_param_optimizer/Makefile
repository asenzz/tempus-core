all:dimenser.out 


#CCOPT=-g -D_GLIBCXX_DEBUG -D_GLIBCXX_DEBUG_PEDANTIC
CCOPT=-O3
dimenser.out:dimenser.cpp trainings.cpp  split_oemd.cu model.cpp cuda_blas_funcs.cu cuda_path.cu firefly.cpp vmdm.cpp
	nvcc  $(CCOPT)    -I/users/eatanassov/mmodel/cuosqpn/osqp-cuda-1.0/include   -std=c++11  -ccbin=mpiicpc -I/usr/local/cuda-11.5/targets/x86_64-linux/include -I/usr/local/cuda-11.5/samples/common/inc -Xcompiler="  -std=c++11 -mt_mpi  -qopenmp $(CCOPT) -xHost -std=c++11 -mkl"   -lnvblas -lcublas -L/usr/local/cuda-11.5/targets/x86_64-linux/lib -lcusparse  -lcudart      -I/users/eatanassov/arma/include  dimenser.cpp vmdm.cpp  model.cpp firefly.cpp split_oemd.cu trainings.cpp  cuda_blas_funcs.cu cuda_path.cu  -L/users/eatanassov/arma/lib64 -lcufft  -lcusolver -lcublasLt  -larmadillo    -o dimenser.out
