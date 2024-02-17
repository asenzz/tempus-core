CMAKE_MINIMUM_REQUIRED(VERSION 3.10)
IF(USE_CUDA)
    PROJECT(SVR LANGUAGES CXX CUDA)
ELSE()
    PROJECT(SVR CXX)
ENDIF()

SET(POSITION_INDEPENDENT_CODE OFF)
# SET(CMAKE_BUILD_TYPE "Release" CACHE STRING "CMAKE_BUILD_TYPE")
SET(CMAKE_CXX_STANDARD "17" CACHE STRING "CMAKE_CXX_STANDARD")
SET(CMAKE_CUDA_HOST_COMPILER "/usr/bin/g++" CACHE STRING "CMAKE_CUDA_HOST_COMPILER")

SET(CMAKE_PEDANTIC_FLAGS -Werror -pedantic -Werror -Wall -Wno-deprecated)
IF(WIN32 OR CYGWIN)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DGTEST_OS_CYGWIN=1")
ENDIF()

IF (CMAKE_BUILD_TYPE MATCHES "Rel")
    SET(CXX_OPTIMIZATION_FLAGS -fomit-frame-pointer -mtune=native -march=native -m64 -mstackrealign -mmovbe -mavx2 -mavx -malign-double -msse -msse2 -mfpmath=sse -O3) # Find a way to disable -fPIC switch, which is on by default
    SET(CXX_OPTIMIZATION_FLAGS_ICX -fp-model=precise)
    SET(CXX_OPTIMIZATION_FLAGS_GCC "")
    SET(LINKER_OPTIMIZATION_FLAGS -flto=full -fwhole-program) # ICX linking optimization options: -Wl,--copy-dt-needed-entries -fvisibility=hidden "-Dexternally_visible=visibility(default)"
ENDIF()

IF (CMAKE_BUILD_TYPE MATCHES "Debug")
    SET(LSAN_BLACKLIST "lsan-blacklist.txt" CACHE STRING "")
    SET(UBSAN_BLACKLIST "ubsan-blacklist.txt" CACHE STRING "")
    SET(CXX_OPTIMIZATION_FLAGS -fno-omit-frame-pointer --param=max-vartrack-size=1500000 -fsanitize=undefined -fstack-protector-strong -fno-sanitize-recover=all) # Enable as necessary -fsanitize=thread -fsanitize=undefined # Memory profilers -lduma -ldmallocthcxx -ldmallocthcxx"
    SET(CXX_OPTIMIZATION_FLAGS_ICX -fsanitize-blacklist="${LSAN_BLACKLIST}")
    SET(LINKER_OPTIMIZATION_FLAGS -fsanitize=address -static-libasan)
    SET(ENV {LSAN_OPTIONS} "suppressions=${LSAN_BLACKLIST},print_suppressions=1,use_unaligned=1,report_objects=1")
    SET(ENV {UBSAN_OPTIONS} "print_stacktrace=1,log_path=${CMAKE_CURRENT_SOURCE_DIR}/ubsan.log,suppressions=${UBSAN_BLACKLIST},print_suppressions=1,use_unaligned=1,report_objects=1")
ENDIF()

ADD_DEFINITIONS(-DSVR_ENABLE_DEBUGGING -DBOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS=1) # Compatibility

OPTION(USE_FAST_MATH "Use fast math" OFF)
IF(USE_FAST_MATH)
    LIST(APPEND CXX_OPTIMIZATION_FLAGS ${CXX_OPTIMIZATION_FLAGS} -ffast-math)
    LIST(APPEND CXX_OPTIMIZATION_FLAGS_GCC)
    LIST(APPEND CXX_OPTIMIZATION_FLAGS_ICX -fp-model=fast1 -fp-speculation=safe)
    SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math" CACHE STRING "")
ENDIF()

SET(COMPILER_EXTENSIONS -std=c++17 -DBOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS=1 -ggdb -Wall -DCL_TARGET_OPENCL_VERSION=220)
SET(COMPILER_EXTENSIONS_CC -std=c17) # Deprecated -fcilkplus -lcilkrts
SET(COMPILER_EXTENSIONS_ICX -qopenmp-simd -fiopenmp -fopenmp=libiomp5 -fopenmp-targets=spir64_gen)
SET(COMPILER_EXTENSIONS_GCC -fopenmp)

FOREACH (OPT IN LISTS CMAKE_PEDANTIC_FLAGS COMPILER_EXTENSIONS COMPILER_EXTENSIONS_GCC CXX_OPTIMIZATION_FLAGS CXX_OPTIMIZATION_FLAGS_GCC)
   ADD_COMPILE_OPTIONS("$<$<COMPILE_LANGUAGE:CXX>:${OPT}>")
ENDFOREACH()


# ADD_COMPILE_OPTIONS(${CXX_OPTIMIZATION_FLAGS} ${CXX_OPTIMIZATION_FLAGS_ICX} ${COMPILER_EXTENSIONS} ${COMPILER_EXTENSIONS_ICX} ${COMPILER_EXTENSIONS_CC})
ADD_LINK_OPTIONS(${LINKER_OPTIMIZATION_FLAGS} ${CXX_OPTIMIZATION_FLAGS} ${COMPILER_EXTENSIONS} ${COMPILER_EXTENSIONS_GCC} "-Wl,--copy-dt-needed-entries")

OPTION(USE_CUDA "Compile for CUDA system." ON)
if (USE_CUDA)
    ADD_DEFINITIONS(-DUSE_CUDA)
    # ADD_DEFINITIONS(-DVIENNACL_WITH_CUDA) # ViennaCL is DEPRECATED and to be removed
    SET(CUDA_ARCH_LIST Auto CACHE STRING "List of CUDA architectures (e.g. Pascal, Volta, etc) or \
                                        compute capability versions (6.1, 7.0, etc) to generate code for. \
                                        Set to Auto for automatic detection (default)."
    )
    SET(CMAKE_CUDA_ARCHITECTURES "70" CACHE STRING "")
    SET(CUDA_ARCH_LIST "7.0" CACHE STRING "")
    INCLUDE_DIRECTORIES(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

    SET(CMAKE_CUDA_COMPILER "nvcc" CACHE STRING "CMAKE_CUDA_COMPILER")
    SET(CMAKE_CUDA_HOST_COMPILER "g++-11" CACHE STRING "CMAKE_CUDA_HOST_COMPILER")
    SET(CUDA_HOST_COMPILER "g++-11" CACHE STRING "CUDA_HOST_COMPILER")
    SET(COMPILER_EXTENSIONS_CUDA -DCUDACC -fopenmp --extended-lambda --default-stream per-thread --std=c++17 -forward-unknown-to-host-compiler -lineinfo)
    FOREACH(OPT IN LISTS COMPILER_EXTENSIONS COMPILER_EXTENSIONS_GCC CXX_OPTIMIZATION_FLAGS CXX_OPTIMIZATION_FLAGS_GCC)
        ADD_COMPILE_OPTIONS("$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${OPT}>")
    ENDFOREACH()
    FOREACH(OPT IN LISTS COMPILER_EXTENSIONS_CUDA)
        ADD_COMPILE_OPTIONS("$<$<COMPILE_LANGUAGE:CUDA>:${OPT}>")
    ENDFOREACH()
    IF (NOT CUDA_SDK_ROOT_DIR MATCHES "/")
        SET(CUDA_SDK_ROOT_DIR "${CUDA_TOOLKIT_ROOT_DIR}" CACHE STRING "CUDA_SDK_ROOT_DIR" FORCE)
    ENDIF()
ENDIF()

OPTION(ENABLE_OPENCL "Use the Vienna OpenCL backend for BLAS" ON)
IF(ENABLE_OPENCL)
    ADD_DEFINITIONS(-DVIENNACL_WITH_OPENCL)
    FIND_PACKAGE(OpenCL REQUIRED)
    INCLUDE_DIRECTORIES("/usr/local/cuda/include") # or /opt/intel/oneapi/compiler/latest/include/sycl/
ENDIF(ENABLE_OPENCL)

INCLUDE_DIRECTORIES("/opt/intel/oneapi/tbb/latest/include" "/usr/include/hdf5/serial")


SET(MKL_INTERFACE_FULL "intel_lp64" CACHE STRING "MKL_INTERFACE_FULL" FORCE)
SET(MKL_LINK "static" CACHE STRING "MKL_LINK" FORCE)
SET(OpenMP_iomp_LIBRARY "/opt/intel/oneapi/compiler/latest/linux/compiler/lib/intel64_lin/libiomp5.a" CACHE STRING "OpenMP_iomp_LIBRARY")
SET(OpenMP_CXX_LIB_NAMES "iomp;pthread" CACHE STRING "OpenMP_CXX_LIB_NAMES")

OPTION(EXPERIMENTAL_FEATURES "Enable experimental features" OFF)
IF(EXPERIMENTAL_FEATURES)
    ADD_DEFINITIONS(-DEXPERIMENTAL_FEATURES)
    SET(MATLAB_INCLUDE_DIR "/mnt/slowstore/usr/local/MATLAB/R2021b/extern/include/")
    SET(MATLAB_LINK_DIR "/mnt/slowstore/usr/local/MATLAB/R2021b/extern/bin/glnxa64/")
    EXECUTE_PROCESS(COMMAND python3-config --ldflags --cflags OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE PY_CLDFLAGS)
    STRING(REGEX REPLACE "[\n ]" "" PY_CLDFLAGS "${PY_CLDFLAGS}")
    # SET(PY_CLDFLAGS "-fPIC ${PY_CLDFLAGS}")
    SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${PY_CLDFLAGS}")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${PY_CLDFLAGS}")
    SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${PY_CLDFLAGS}")
    SET(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${PY_CLDFLAGS}")
    SET(CMAKE_STATIC_LINKER_FLAGS "${CMAKE_STATIC_LINKER_FLAGS} ${PY_CLDFLAGS}")
ENDIF(EXPERIMENTAL_FEATURES)



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

# SET(NVCC_PREPEND_FLAGS "-allow-unsupported-compiler -ccbin /usr/bin/c++") # Unsafe avoid!
FIND_PACKAGE(TBB REQUIRED)
FIND_PACKAGE(OpenMP REQUIRED)
FIND_PACKAGE(PQXX)
FIND_PACKAGE(PQ)
FIND_PACKAGE(Armadillo)
FIND_PACKAGE(MKL)

SET(ENABLE_OMP_OFFLOAD ON CACHE BOOL "ENABLE_OMP_OFFLOAD" FORCE)
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
SET(PQXX_LIBRARIES -lpqxx -lpq)

FOREACH(module ${MODULES})
    INCLUDE_DIRECTORIES(${SVRROOT_DIR}/${module}/include)
    ADD_SUBDIRECTORY(${SVRROOT_DIR}/${module})
    SET(DIR_TEST_MODULES)
    GET_PROPERTY(DIR_TEST_MODULES DIRECTORY ${CMAKE_SOURCE_DIR}/${SVRROOT_DIR}/${module} PROPERTY TEST_MODULES)
    if(DIR_TEST_MODULES)
        SET(TEST_MODULES ${TEST_MODULES} ${DIR_TEST_MODULES})
    endif()
ENDFOREACH(module)

if(WITH_BENCHMARKS)
    message("********************************************************************************")
    message("*  Registering the following benchmarks:")
    ADD_SUBDIRECTORY(${SVRROOT_DIR}/SVRBenchmarks)
    message("********************************************************************************")
endif()


if(WITH_TESTS)
    ENABLE_TESTING()
    MESSAGE("********************************************************************************")
    MESSAGE("*  Registering the following tests:")
    FOREACH(MODULE ${TEST_MODULES})
        if("${MODULE}" MATCHES "${WITH_TESTS}")
            MESSAGE("* ~~ ${MODULE}")
            ADD_TEST(NAME ${MODULE} COMMAND ${MODULE})

            if("${MODULE}" MATCHES "SVRBusiness-tests")
                ADD_TEST(NAME ${MODULE}-async COMMAND "${MODULE}" "-d async" )
                MESSAGE("* ~~ ${MODULE}-async")
            endif()
        endif()
    ENDFOREACH()
    MESSAGE("*  Use make test to run the tests")
    MESSAGE("********************************************************************************")
endif()
