CMAKE_MINIMUM_REQUIRED(VERSION 3.10)
OPTION(USE_INTEL_COMPILER "Use Intel's ICX compiler" OFF)
FIND_PROGRAM(PROGRAM_ICPX icpx)
IF(PROGRAM_ICPX)
    SET(USE_INTEL_COMPILER ON CACHE BOOL "USE_INTEL_COMPILER")
    SET(CMAKE_CXX_COMPILER ${PROGRAM_ICPX} CACHE STRING "CMAKE_CXX_COMPILER")
    FIND_PROGRAM(PROGRAM_ICX icx REQUIRED)
    SET(CMAKE_C_COMPILER ${PROGRAM_ICX} CACHE STRING "CMAKE_C_COMPILER")
    CMAKE_PATH(GET PROGRAM_ICX PARENT_PATH ICX_DIR)
    SET(CMAKE_LINKER "${ICX_DIR}/compiler/ld.lld" CACHE STRING "CMAKE_LINKER")
    LIST(APPEND LINKER_OPTIMIZATION_FLAGS_ICX -fuse-ld=lld)
ENDIF()

FIND_PROGRAM(PROGRAM_CCACHE ccache)
IF(PROGRAM_CCACHE)
    SET(CMAKE_CXX_COMPILER_LAUNCHER ${PROGRAM_CCACHE})
    SET(CMAKE_C_COMPILER_LAUNCHER ${PROGRAM_CCACHE})
    SET(CMAKE_CUDA_COMPILER_LAUNCHER ${PROGRAM_CCACHE})
    SET(CMAKE_OPENCL_COMPILER_LAUNCHER ${PROGRAM_CCACHE})
ENDIF()
SET(POSITION_INDEPENDENT_CODE OFF)
IF(USE_CUDA)
    PROJECT(SVR LANGUAGES CXX CUDA)
ELSE()
    PROJECT(SVR CXX)
ENDIF()

# SET(CMAKE_BUILD_TYPE "Release" CACHE STRING "CMAKE_BUILD_TYPE")
SET(CMAKE_CXX_STANDARD "17" CACHE STRING "CMAKE_CXX_STANDARD")
SET(CMAKE_PEDANTIC_FLAGS -pedantic -Wall -Werror -debug extended -O3)
#IF(NOT USE_INTEL_COMPILER)
#    LIST(APPEND -Werror)
#ENDIF()

IF(WIN32 OR CYGWIN)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DGTEST_OS_CYGWIN=1")
ENDIF()

IF (CMAKE_BUILD_TYPE MATCHES "Rel")
    # SET(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON) # ICXP segfaults
    SET(CXX_OPTIMIZATION_FLAGS -fomit-frame-pointer -mtune=native -march=native -m64 -mstackrealign -mmovbe -mavx2 -mavx -malign-double -msse4.2 -mfpmath=sse -O3) # TODO Find a way to disable -fPIC switch, which is on by default
    SET(CXX_OPTIMIZATION_FLAGS_ICX -qopt-mem-layout-trans=4 -funroll-loops -Wno-overriding-option -fp-model=precise -fp-speculation=safe -pc80 -fimf-use-svml=true -qopt-prefetch -mbranches-within-32B-boundaries)
    SET(CXX_OPTIMIZATION_FLAGS_GCC "")
    LIST(APPEND LINKER_OPTIMIZATION_FLAGS_GCC -Wl,--icf=safe -flto=full -fwhole-program)
    LIST(APPEND LINKER_OPTIMIZATION_FLAGS_ICX -Wl,-z,muldefs -fdiagnostics-show-option -qopenmp-simd -fiopenmp -fopenmp=libiomp5 -flto=full -Wl,--icf=safe -fopenmp-targets=spir64_x86_64 --offload-arch=native -Xopenmp-target-backend=spir64_x86_64 "-device=cpu -march=sse4.2")
ENDIF()

IF (CMAKE_BUILD_TYPE MATCHES "Debug")
    SET(LSAN_BLACKLIST "lsan-blacklist.txt" CACHE STRING "")
    SET(UBSAN_BLACKLIST "ubsan-blacklist.txt" CACHE STRING "")
    SET(CXX_OPTIMIZATION_FLAGS -fno-omit-frame-pointer --param=max-vartrack-size=1500000 -fsanitize=undefined -fstack-protector-strong -fno-sanitize-recover=all) # Enable as necessary -fsanitize=thread -fsanitize=undefined # Memory profilers -lduma -ldmallocthcxx -ldmallocthcxx"
    SET(CXX_OPTIMIZATION_FLAGS_ICX -ggdb -fsanitize-blacklist="${LSAN_BLACKLIST}" -fsanitize=address -fno-common -fp-model=strict -pc80 -fimf-precision=high -fimf-use-svml=false)
    LIST(APPEND LINKER_OPTIMIZATION_FLAGS  -fsanitize=address -static-libasan)
    LIST(APPEND LINKER_OPTIMIZATION_FLAGS_GCC)
    LIST(APPEND LINKER_OPTIMIZATION_FLAGS_ICX)
    SET(ENV {ASAN_OPTIONS} "log_path=/tmp/asan.log:check_initialization_order=true:detect_invalid_pointer_pairs=1")
    SET(ENV {LSAN_OPTIONS} "suppressions=${LSAN_BLACKLIST}:print_suppressions=1:use_unaligned=1:report_objects=1:log_path=/tmp/lsan.log")
    SET(ENV {UBSAN_OPTIONS} "print_stacktrace=1:log_path=/tmp/ubsan.log:suppressions=${UBSAN_BLACKLIST}:print_suppressions=1:use_unaligned=1:report_objects=1")
ENDIF()

FIND_PROGRAM(PROGRAM_MOLD mold)
IF(PROGRAM_MOLD)
    LIST(APPEND LINKER_OPTIMIZATION_FLAGS_GCC -fuse-ld=mold)
ENDIF()

ADD_DEFINITIONS(-DARMA_USE_FFTW3 -DARMA_ALLOW_FAKE_GCC -DSVR_ENABLE_DEBUGGING -DBOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS=1) # Compatibility

OPTION(USE_FAST_MATH "Use fast math" OFF)
IF(USE_FAST_MATH)
    LIST(APPEND CXX_OPTIMIZATION_FLAGS ${CXX_OPTIMIZATION_FLAGS} -ffast-math)
    LIST(APPEND CXX_OPTIMIZATION_FLAGS_GCC)
    LIST(APPEND CXX_OPTIMIZATION_FLAGS_ICX -fp-model=fast1 -fp-speculation=safe -fimf-use-svml=true -pc64 -fimf-precision=low)
    SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math" CACHE STRING "")
ENDIF()

SET(COMPILER_EXTENSIONS -std=c++17)
SET(COMPILER_EXTENSIONS_CC -std=c17) # Deprecated -fcilkplus -lcilkrts
SET(COMPILER_EXTENSIONS_ICX -Wno-pass-failed -Wno-unsupported-floating-point-opt -fdiagnostics-show-option -qopenmp-simd -fiopenmp -fopenmp=libiomp5 -fopenmp-targets=spir64_x86_64) # ,nvptx64-nvidia-cuda
SET(COMPILER_EXTENSIONS_GCC -fopenmp -ggdb)

IF (USE_INTEL_COMPILER)
    FOREACH (OPT IN LISTS COMPILER_EXTENSIONS COMPILER_EXTENSIONS_ICX CMAKE_PEDANTIC_FLAGS CXX_OPTIMIZATION_FLAGS CXX_OPTIMIZATION_FLAGS_ICX)
        ADD_COMPILE_OPTIONS("$<$<COMPILE_LANGUAGE:CXX>:${OPT}>")
    ENDFOREACH()
    FOREACH (OPT IN LISTS CMAKE_PEDANTIC_FLAGS COMPILER_EXTENSIONS_CC COMPILER_EXTENSIONS_ICX CXX_OPTIMIZATION_FLAGS CXX_OPTIMIZATION_FLAGS_ICX)
        ADD_COMPILE_OPTIONS("$<$<COMPILE_LANGUAGE:C>:${OPT}>")
    ENDFOREACH()
    ADD_LINK_OPTIONS(${LINKER_OPTIMIZATION_FLAGS} ${LINKER_OPTIMIZATION_FLAGS_ICX}) # ${CXX_OPTIMIZATION_FLAGS} ${COMPILER_EXTENSIONS} ${COMPILER_EXTENSIONS_ICX})
    OPTION(PROFILE_OPTIMIZE "Generate profiling code for Intel HWPGO" OFF)
ELSE()
    FOREACH (OPT IN LISTS CMAKE_PEDANTIC_FLAGS COMPILER_EXTENSIONS COMPILER_EXTENSIONS_GCC CXX_OPTIMIZATION_FLAGS CXX_OPTIMIZATION_FLAGS_GCC)
        ADD_COMPILE_OPTIONS("$<$<COMPILE_LANGUAGE:CXX>:${OPT}>")
    ENDFOREACH()
    FOREACH (OPT IN LISTS CMAKE_PEDANTIC_FLAGS COMPILER_EXTENSIONS_CC COMPILER_EXTENSIONS_GCC CXX_OPTIMIZATION_FLAGS CXX_OPTIMIZATION_FLAGS_GCC)
        ADD_COMPILE_OPTIONS("$<$<COMPILE_LANGUAGE:C>:${OPT}>")
    ENDFOREACH()
    ADD_LINK_OPTIONS(${LINKER_OPTIMIZATION_FLAGS_GCC} ${CXX_OPTIMIZATION_FLAGS} ${COMPILER_EXTENSIONS} ${COMPILER_EXTENSIONS_GCC})
ENDIF()


# ADD_COMPILE_OPTIONS(${CXX_OPTIMIZATION_FLAGS} ${CXX_OPTIMIZATION_FLAGS_ICX} ${COMPILER_EXTENSIONS} ${COMPILER_EXTENSIONS_ICX} ${COMPILER_EXTENSIONS_CC})

SET(CMAKE_CUDA_HOST_COMPILER "g++-12" CACHE STRING "CUDA host compiler")
OPTION(USE_CUDA "Compile for CUDA system." ON)
if (USE_CUDA)
    FIND_PACKAGE(CUDAToolkit)
    LINK_DIRECTORIES(${CUDAToolkit_LIBRARY_DIR})
    ADD_DEFINITIONS(-DUSE_CUDA)
    # ADD_DEFINITIONS(-DVIENNACL_WITH_CUDA) # ViennaCL is DEPRECATED and to be removed
    SET(CUDA_ARCH_LIST Auto CACHE STRING "List of CUDA architectures (e.g. Pascal, Volta, etc) or \
                                        compute capability versions (6.1, 7.0, etc) to generate code for. \
                                        Set to Auto for automatic detection (default)."
    )
    SET(CMAKE_CUDA_ARCHITECTURES "70" CACHE STRING "")
    SET(CUDA_ARCH_LIST "Volta" CACHE STRING "")
    INCLUDE_DIRECTORIES(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${CUDAToolkit_INCLUDE_DIRS})

    SET(CMAKE_CUDA_COMPILER "nvcc" CACHE STRING "CMAKE_CUDA_COMPILER")
    SET(CUDA_HOST_COMPILER ${CMAKE_CUDA_HOST_COMPILER} CACHE STRING "CUDA_HOST_COMPILER")
    SET(COMPILER_EXTENSIONS_CUDA -DCUDACC -allow-unsupported-compiler --extended-lambda --default-stream per-thread --std=c++17 -forward-unknown-to-host-compiler -lineinfo)
    FOREACH(OPT IN LISTS COMPILER_EXTENSIONS_CUDA)
        # SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${OPT}")
    #    SET(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} ${COMPILER_EXTENSIONS_CUDA}")
    #    SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${COMPILER_EXTENSIONS_CUDA}")
        ADD_COMPILE_OPTIONS("$<$<COMPILE_LANGUAGE:CUDA>:${OPT}>")
    ENDFOREACH()
    FOREACH(OPT IN LISTS COMPILER_EXTENSIONS COMPILER_EXTENSIONS_GCC CXX_OPTIMIZATION_FLAGS CXX_OPTIMIZATION_FLAGS_GCC)
        # SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=${OPT}")
        ADD_COMPILE_OPTIONS("$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${OPT}>")
    ENDFOREACH()
    IF (NOT ${CUDA_SDK_ROOT_DIR} MATCHES "/")
        SET(CUDA_SDK_ROOT_DIR ${CUDA_TOOLKIT_ROOT_DIR})
    ENDIF()
ENDIF()

OPTION(ENABLE_OPENCL "Use the Vienna OpenCL backend for BLAS" ON)
IF(ENABLE_OPENCL)
    ADD_DEFINITIONS(-DVIENNACL_WITH_OPENCL -DCL_TARGET_OPENCL_VERSION=220 -DCL_HPP_TARGET_OPENCL_VERSION=220)
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

FIND_PACKAGE(TBB REQUIRED)
FIND_PACKAGE(Armadillo REQUIRED)
FIND_PACKAGE(MKL REQUIRED)
FIND_PACKAGE(PostgreSQL REQUIRED)
FIND_PACKAGE(LibPQXX REQUIRED)

SET(ENABLE_OMP_OFFLOAD ON CACHE BOOL "ENABLE_OMP_OFFLOAD" FORCE)
SET(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})
SET(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR})

MESSAGE(STATUS "BUILD_TYPE: " ${CMAKE_BUILD_TYPE} )
MESSAGE(STATUS "CMAKE_BINARY_DIR " ${CMAKE_BINARY_DIR} )

# Workaround for QtCreator to show headers
SET(IDE QT_CREATOR)
SET(SVRROOT_DIR SVRRoot)

IF(NOT DEFINED MODULES)
    SET(MODULES SVRWeb SVRDaemon SVRBusiness OnlineSVR SVRPersist SVRCommon SVRModel SVRFix)
ELSE()
    LIST(APPEND MODULES SVRBusiness OnlineSVR SVRPersist SVRCommon SVRModel)
ENDIF()

IF(WITH_TESTS)
    LIST(APPEND MODULES SVRBusiness-tests SVRDaemon-blackbox-tests)
ENDIF()
    
IF(WITH_TOOLS)
    LIST(APPEND MODULES zTools)
ENDIF()
SET(LIBS_BUILD_TYPE SHARED)

FIND_PACKAGE(PkgConfig REQUIRED)
PKG_CHECK_MODULES (JEMALLOC jemalloc)

PKG_SEARCH_MODULE(JEMALLOC REQUIRED jemalloc)
INCLUDE_DIRECTORIES(${JEMALLOC_INCLUDE_DIRS})

IF(APPLE)
    SET(CMAKE_MACOSX_RPATH 1)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-c++11-narrowing")
ENDIF()


SET(Boost_MIN_VERSION 1.74)

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
        IF("${MODULE}" MATCHES "${WITH_TESTS}")
            MESSAGE("* ~~ ${MODULE}")
            ADD_TEST(NAME ${MODULE} COMMAND ${MODULE})

            IF("${MODULE}" MATCHES "SVRBusiness-tests")
                ADD_TEST(NAME ${MODULE}-async COMMAND "${MODULE}" "-d async" )
                MESSAGE("* ~~ ${MODULE}-async")
            ENDIF()
        ENDIF()
    ENDFOREACH()
    MESSAGE("*  Use make test to run the tests")
    MESSAGE("********************************************************************************")
ENDIF()

LINK_DIRECTORIES(/lib /libexec /usr/lib /usr/lib/libexec /usr/local/lib /usr/local/libexec)