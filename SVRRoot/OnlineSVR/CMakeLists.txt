CMAKE_MINIMUM_REQUIRED(VERSION 3.10)
IF(USE_CUDA)
	PROJECT(OnlineSVR LANGUAGES CUDA CXX)
    LINK_LIBRARIES(CUDA::cudart CUDA::cublas CUDA::cufftw CUDA::cusolver CUDA::nppc CUDA::npps) # CUDA::cuda_driver
ELSE()
	PROJECT(OnlineSVR CXX)
ENDIF()

SET(DEPENDENCIES SVRCommon SVRModel)
INCLUDE(../../cmake/IncludeDependencies.cmake)
INCLUDE(../../cmake/SourceList.cmake)

FIND_PACKAGE(prima REQUIRED)
# FIND_PACKAGE(Bandicoot REQUIRED)
FIND_PACKAGE(Boost ${Boost_MIN_VERSION} COMPONENTS thread system REQUIRED CONFIG)
LINK_LIBRARIES(${DEPENDENCIES} magma MKL::MKL TBB::tbb ${JEMALLOC_LIBRARIES} prima::primac Ginkgo::ginkgo) # bandicoot

IF (EXPERIMENTAL_FEATURES)
    # FIND_PACKAGE(OSQP REQUIRED)
    # FIND_PACKAGE(Eigen3 REQUIRED)
    LINK_DIRECTORIES(${MATLAB_LINK_DIR} "/usr/lib/google/tensorflow")
    EXECUTE_PROCESS(COMMAND python3-config --includes OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE PYINCLUDES)
    STRING(REGEX REPLACE "[\n ]" "" PYINCLUDES "${PYINCLUDES}")
    FIND_PACKAGE(Python3 REQUIRED)
    INCLUDE_DIRECTORIES("${MATLAB_INCLUDE_DIR}" "${PYINCLUDES}" "/usr/include/python3.6m") # "/usr/local/include/eigen3" "/usr/local/include/google/protobuf"
    #FIND_PACKAGE(Torch REQUIRED)
    LINK_LIBRARIES(fftw3 MatlabDataArray MatlabEngine python3.6m) # osqp::osqpstatic clblast tensorflow_all ${TORCH_LIBRARIES}
    APPEND(TEST_DEPENDENCIES ${PROJECT_NAME} python3.6m) # tensorflow_all ${TORCH_LIBRARIES}
ENDIF()

IF(WITH_TESTS AND PROJECT_NAME MATCHES "${WITH_TESTS}")
    MESSAGE("~~ Building ${PROJECT_NAME}-test")
    SET(TEST_DEPENDENCIES ${PROJECT_NAME} SVRCommon SVRBusiness SVRPersist SVRModel wavelib fftw3)

    SET_PROPERTY(DIRECTORY PROPERTY TEST_MODULES "${PROJECT_NAME}")

    SET(GTEST_ROOT "/usr")

    FIND_PACKAGE(GTest REQUIRED)
    FIND_PACKAGE(Boost ${Boost_MIN_VERSION} COMPONENTS program_options REQUIRED CONFIG)
    INCLUDE_DIRECTORIES(${GTEST_ROOT}/include)

    LINK_DIRECTORIES(${GTEST_ROOT}/lib)

    FILE(GLOB TEST_SOURCE_FILES "test/*.cpp")

    ADD_EXECUTABLE(${PROJECT_NAME}-test ${TEST_SOURCE_FILES} ${COMMON_SOURCE_FILES})
    TARGET_LINK_LIBRARIES(${PROJECT_NAME}-test ${DEPENDENCIES} ${TEST_DEPENDENCIES} gtest gtest_main pthread ${Boost_LIBRARIES} sycl ${ONEAPI_ROOT}/compiler/latest/lib/libomptarget.sycl.wrap.so) # absl::hash absl::flags_reflection absl::raw_hash_set
ENDIF()

ADD_LIBRARY(${PROJECT_NAME} ${SOURCE_FILES})
