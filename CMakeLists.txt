CMAKE_MINIMUM_REQUIRED(VERSION 3.10)
IF(USE_CUDA)
    PROJECT(SVR LANGUAGES CXX CUDA)
    INCLUDE(FindCUDA)
ELSE()
    PROJECT(SVR CXX)
ENDIF()

SET(OpenMP_iomp_LIBRARY "/opt/intel/oneapi/compiler/latest/linux/compiler/lib/intel64_lin/libiomp5.a")
SET(CMAKE_CXX_STANDARD 17)

OPTION(USE_CUDA "Compile for CUDA system." ON)
if (USE_CUDA)
    FIND_PACKAGE(CUDA)
    ADD_DEFINITIONS(-DUSE_CUDA)
    # ADD_DEFINITIONS(-DVIENNACL_WITH_CUDA)
    SET(CUDA_ARCH_LIST Auto CACHE STRING "List of CUDA architectures (e.g. Pascal, Volta, etc) or \
                                        compute capability versions (6.1, 7.0, etc) to generate code for. \
                                        Set to Auto for automatic detection (default)."
        )
    CUDA_SELECT_NVCC_ARCH_FLAGS(CUDA_ARCH_FLAGS ${CUDA_ARCH_LIST})
    SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --default-stream per-thread --extended-lambda --default-stream per-thread --std=c++17 -gencode arch=compute_70,code=sm_70 -forward-unknown-to-host-compiler -lineinfo -liomp5 -fopenmp -Xcompiler '-std=c++17' -Xcompiler '-fopenmp' -Xcompiler '-liomp5' -Xcompiler '-mtune=native' -Xcompiler '-march=native' -Xcompiler '-O3' -Xcompiler '-mavx2' -Xcompiler '-mavx' -Xcompiler '-fPIC'")
    #LIST(APPEND CMAKE_CUDA_FLAGS " --default-stream per-thread -gencode arch=compute_70,code=sm_70 ")
    CUDA_SELECT_NVCC_ARCH_FLAGS(CUDA_ARCH_FLAGS ${CUDA_ARCH_LIST})
    SET(CMAKE_CUDA_ARCHITECTURES 70)
    INCLUDE_DIRECTORIES(include ${CUDA_INCLUDE_DIRS} ${CUDA_TOOLKIT_ROOT_DIR}/samples/common)
endif()

if(NOT WITH_TESTS)
    SET(WITH_TESTS ".*")
endif()
    
if(NOT WITH_TOOLS)
    SET(WITH_TOOLS ".*")
endif()

LIST(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

if($ENV{HOSTNAME} MATCHES "sl[0-9]+$")
    LIST(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/host_config/sl/cmake/")
endif()

FIND_PACKAGE(TBB REQUIRED)
FIND_PACKAGE(OpenMP REQUIRED)
FIND_PACKAGE(PQXX)
FIND_PACKAGE(PQ)
FIND_PACKAGE(Armadillo)
FIND_PACKAGE(MKL)
INCLUDE_DIRECTORIES(BEFORE SYSTEM ${TBB_INCLUDE_DIRS} "/opt/intel/oneapi/tbb/latest/include/")

SET(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})
SET(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR})

MESSAGE(STATUS "BUILD_TYPE: " ${CMAKE_BUILD_TYPE} )
MESSAGE(STATUS "CMAKE_BINARY_DIR " ${CMAKE_BINARY_DIR} )
#include(PCH_GCC4_v2)

# Workaround for QtCreator to show headers
SET(IDE QT_CREATOR)
SET(SVRROOT_DIR SVRRoot)

if(NOT DEFINED MODULES)
    SET(MODULES SVRWeb SVRDaemon SVRBusiness OnlineSVR SVRPersist SVRCommon SVRModel SVRFix)
else(NOT DEFINED MODULES)
    LIST(APPEND MODULES SVRBusiness OnlineSVR SVRPersist SVRCommon SVRModel)
endif(NOT DEFINED MODULES)

if(WITH_TESTS)
    LIST(APPEND MODULES SVRBusiness-tests SVRDaemon-blackbox-tests)
endif()
    
if(WITH_TOOLS)
    LIST(APPEND MODULES zTools)
endif()

SET(LIBS_BUILD_TYPE SHARED)
SET(MKL_USE_interface "lp64")

OPTION(MKL_USE_parallel ON)
IF(CMAKE_BUILD_TYPE MATCHES Rel)
    SET (CXX_OPTIMIZATION_FLAGS "-flto -mtune=native -march=native -m64 -mstackrealign -mmovbe -malign-double -mfpmath=sse -O3")
    SET (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CXX_OPTIMIZATION_FLAGS}")
    SET (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${CXX_OPTIMIZATION_FLAGS}")
    SET (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${CXX_OPTIMIZATION_FLAGS}")
    SET (CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ")
ENDIF(CMAKE_BUILD_TYPE MATCHES Rel)

SET(COMPILER_EXTENSIONS "-DBOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS=1 -mavx2 -mavx -ggdb -Wall -larmadillo -fopenmp -liomp5 -DCL_TARGET_OPENCL_VERSION=220") # -fcilkplus -lcilkrts

option(USE_FAST_MATH "Use fast math" OFF)
IF(USE_FAST_MATH)
    SET(COMPILER_EXTENSIONS "${COMPILER_EXTENSIONS} -ffast-math")
    SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math")
ENDIF()

option(EXPERIMENTAL_FEATURES "Enable experimental features" OFF)
IF(EXPERIMENTAL_FEATURES)
    execute_process(COMMAND python3-config --ldflags --cflags OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE PY_CLDFLAGS)
    string(REGEX REPLACE "[\n ]" "" PY_CLDFLAGS "${PY_CLDFLAGS}")
    set(PY_CLDFLAGS "-fPIC ${PY_CLDFLAGS}")
    SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${PY_CLDFLAGS}")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${PY_CLDFLAGS}")
    SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${PY_CLDFLAGS}")
    SET(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${PY_CLDFLAGS}")
    SET(CMAKE_STATIC_LINKER_FLAGS "${CMAKE_STATIC_LINKER_FLAGS} ${PY_CLDFLAGS}")
ENDIF(EXPERIMENTAL_FEATURES)

SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${COMPILER_EXTENSIONS}")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${COMPILER_EXTENSIONS}")
SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler='${COMPILER_EXTENSIONS}' -DCUDACC")
SET(MKL_INTERFACE_FULL "intel_lp64") # Never set to intel_ilp64, Armadillo crashes bug with matrix indices
SET(CMAKE_PEDANTIC_FLAGS "-pedantic -Werror") #   -fsanitize=address -lduma  -fstack-protector-strong -ldmallocthcxx -fno-sanitize-recover=all  -fsanitize=leak -fstack-protector-strong -fno-omit-frame-pointer -ldmallocthcxx" )
IF(WIN32 OR CYGWIN)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DGTEST_OS_CYGWIN=1")
ENDIF()

find_package(PkgConfig REQUIRED)
pkg_check_modules (JEMALLOC jemalloc)

pkg_search_module(JEMALLOC REQUIRED jemalloc)
include_directories(${JEMALLOC_INCLUDE_DIRS})
#SET(PQLIBRARY_DIR /usr/lib)
#SET(PQXXLIBRARY_DIR /usr/lib/x86_64-linux-gnu)

IF(APPLE)
    SET(CMAKE_MACOSX_RPATH 1)
    SET(Boost_INCLUDE_DIR /usr/local/include)
    SET(BOOST_LIBRARYDIR /usr/local/lib)
    SET(PQLIBRARY_DIR /usr/local/lib)
#    SET(PQXXLIBRARY_DIR /usr/local/lib)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-c++11-narrowing")
ELSE()
    SET(Boost_INCLUDE_DIR /usr/include)
    SET(BOOST_LIBRARYDIR /usr/lib/x86_64-linux-gnu)
    SET(PQLIBRARY_DIR /usr/lib)
#    SET(PQXXLIBRARY_DIR /usr/lib/x86_64-linux-gnu)
ENDIF()


SET(Boost_MIN_VERSION 1.48)

SET(DEPLIBS_DIR /libs)
SET(PQXX_LIBRARIES -lpqxx -lpq -lcuda)

add_definitions(-DENABLE_DEBUGGING -DBOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS=1)

option(ENABLE_OPENCL "Use the Vienna OpenCL backend for BLAS" ON)
IF(ENABLE_OPENCL)
    add_definitions(-DVIENNACL_WITH_OPENCL)
    FIND_PACKAGE(OpenCL REQUIRED)
ENDIF(ENABLE_OPENCL)

IF(EXPERIMENTAL_FEATURES)
    add_definitions(-DEXPERIMENTAL_FEATURES)
    SET(MATLAB_INCLUDE_DIR "/mnt/slowstore/usr/local/MATLAB/R2021b/extern/include/")
    SET(MATLAB_LINK_DIR "/mnt/slowstore/usr/local/MATLAB/R2021b/extern/bin/glnxa64/")
ENDIF(EXPERIMENTAL_FEATURES)

FOREACH(module ${MODULES})
    INCLUDE_DIRECTORIES(${SVRROOT_DIR}/${module}/include)
    ADD_SUBDIRECTORY(${SVRROOT_DIR}/${module})
    set(DIR_TEST_MODULES)
    get_property(DIR_TEST_MODULES DIRECTORY ${CMAKE_SOURCE_DIR}/${SVRROOT_DIR}/${module} PROPERTY TEST_MODULES)
    if(DIR_TEST_MODULES)
        set(TEST_MODULES ${TEST_MODULES} ${DIR_TEST_MODULES})
    endif()
ENDFOREACH(module)

if(WITH_BENCHMARKS)
    message("********************************************************************************")
    message("*  Registering the following benchmarks:")
    ADD_SUBDIRECTORY(${SVRROOT_DIR}/SVRBenchmarks)
    message("********************************************************************************")
endif()


if(WITH_TESTS)
    enable_testing()
    message("********************************************************************************")
    message("*  Registering the following tests:")
    FOREACH(module ${TEST_MODULES})
        if("${module}" MATCHES "${WITH_TESTS}")
            message("* ~~ ${module}")
            add_test(NAME ${module} COMMAND ${module})

            if("${module}" MATCHES "SVRBusiness-tests")
                add_test(NAME ${module}-async COMMAND "${module}" "-d async" )
                message("* ~~ ${module}-async")                        
            endif()
        endif()
    ENDFOREACH()
    message("*  Use make test to run the tests")
    message("********************************************************************************")
endif()