CMAKE_MINIMUM_REQUIRED(VERSION 3.10)
IF(USE_CUDA)
	PROJECT(OnlineSVR LANGUAGES CXX CUDA)
	INCLUDE(FindCUDA)
ELSE()
	PROJECT(OnlineSVR CXX)
ENDIF()

#Define project dependencies
SET(DEPENDENCIES SVRCommon SVRModel)

FOREACH(DEP ${DEPENDENCIES})
    INCLUDE_DIRECTORIES(../${DEP}/include)
ENDFOREACH(DEP)

FILE(GLOB SOURCE_FILES "src/*.cpp" "include/*.h" "include/*.hpp" "include/*.tcc")
IF(IDE STREQUAL "QT_CREATOR")
    FILE(GLOB SOURCE_FILES "src/*.cpp" "include/*.h" "include/*.hpp" "include/*.tcc")
ENDIF(IDE STREQUAL "QT_CREATOR")

IF(USE_CUDA)
    FILE(GLOB SOURCE_FILES "src/*.cpp" "include/*.h" "include/*.hpp" "include/*.tcc" "src/*.cu" "src/*.cuh")
ELSE()
    FILE(GLOB SOURCE_FILES "src/*.cpp" "include/*.h" "include/*.hpp" "include/*.tcc")
ENDIF()

FIND_PACKAGE(Boost ${Boost_MIN_VERSION} COMPONENTS thread system REQUIRED)
FIND_PACKAGE(MKL REQUIRED)


IF(EXPERIMENTAL_FEATURES)
    #FIND_PACKAGE(OSQP REQUIRED)
    #FIND_PACKAGE(Eigen3 REQUIRED)
    LINK_DIRECTORIES(${MATLAB_LINK_DIR} "/usr/lib/google/tensorflow")
    execute_process(COMMAND python3-config --includes OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE PYINCLUDES)
    string(REGEX REPLACE "[\n ]" "" PYINCLUDES "${PYINCLUDES}")
    FIND_PACKAGE(Python3 REQUIRED)
    INCLUDE_DIRECTORIES("${MATLAB_INCLUDE_DIR}" "${PYINCLUDES}" "/usr/include/python3.6m") # "/usr/local/include/eigen3" "/usr/local/include/google/protobuf"
    #find_package(Torch REQUIRED)
ENDIF(EXPERIMENTAL_FEATURES)

INCLUDE_DIRECTORIES(BEFORE SYSTEM ${MKL_INCLUDE_DIRS} "/opt/intel/oneapi/mkl/latest/include/")
INCLUDE_DIRECTORIES(BEFORE SYSTEM ${TBB_INCLUDE_DIRS} "/opt/intel/oneapi/tbb/latest/include/")

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -liomp5") # -Wno-error")

IF (USE_CUDA)
    ADD_LIBRARY(${PROJECT_NAME} ${LIBS_BUILD_TYPE} ${SOURCE_FILES} ${CUDA_SOURCE_FILES})
ELSE()
    ADD_LIBRARY(${PROJECT_NAME} ${LIBS_BUILD_TYPE} ${SOURCE_FILES})
ENDIF()

IF(EXPERIMENTAL_FEATURES)
    TARGET_LINK_LIBRARIES(${PROJECT_NAME} ${DEPENDENCIES} armadillo pthread fftw3 ${OMP_LIBRARY} ${MKL_LIBRARIES} clblast MatlabDataArray MatlabEngine TBB::tbb python3.6m ${JEMALLOC_LIBRARIES}) # osqp::osqpstatic
ELSE(EXPERIMETNAL_FEATURES)
    TARGET_LINK_LIBRARIES(${PROJECT_NAME} ${DEPENDENCIES} ${JEMALLOC_LIBRARIES} armadillo ${OMP_LIBRARY} clblast TBB::tbb)
ENDIF(EXPERIMENTAL_FEATURES)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_PEDANTIC_FLAGS}")

IF(EXPERIMENTAL_FEATURES)
    SET(TEST_DEPENDENCIES ${PROJECT_NAME} SVRBusiness SVRCommon SVRModel wavelib fftw3 python3.6m) # tensorflow_all "${TORCH_LIBRARIES}"
ELSE(EXPERIMENTAL_FEATURES)
    SET(TEST_DEPENDENCIES ${PROJECT_NAME} SVRBusiness SVRCommon SVRModel wavelib fftw3)
ENDIF()

include ("../../cmake/UnitTester.cmake")

if (USE_CUDA)
    SET_TARGET_PROPERTIES(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    TARGET_LINK_LIBRARIES(${PROJECT_NAME} ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES} ${CUDA_cusolver_LIBRARY} ${CUDA_cusparse_LIBRARY} ${CUDA_cufft_LIBRARY})
    #TARGET_COMPILE_OPTIONS(${PROJECT_NAME} PUBLIC -Wno-error)
#    TARGET_COMPILE_OPTIONS(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
#            -gencode arch=compute_70,code=sm_70 -forward-unknown-to-host-compiler -lineinfo -liomp5 -fopenmp -Xcompiler=--std=c++17 -Xcompiler=-fopenmp -Xcompiler=-liomp5 -Xcompiler=-mtune=native -Xcompiler=-march=native >)
endif()

target_compile_definitions(${PROJECT_NAME} PUBLIC ARMA_ALLOW_FAKE_GCC)
