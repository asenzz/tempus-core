IF(USE_CUDA)
    PROJECT(OnlineSVR LANGUAGES CUDA CXX)
    LINK_LIBRARIES(CUDA::cudart CUDA::cublas CUDA::cufftw CUDA::cusolver CUDA::nppc CUDA::npps) # CUDA::cuda_driver
ELSE()
    PROJECT(OnlineSVR CXX)
ENDIF()

SET(DEPENDENCIES SVRPersist SVRCommon SVRModel)
INCLUDE(../../cmake/IncludeDependencies.cmake)
INCLUDE(../../cmake/SourceList.cmake)

FIND_PACKAGE(prima REQUIRED)
# FIND_PACKAGE(Bandicoot REQUIRED)
FIND_PACKAGE(Boost ${Boost_MIN_VERSION} COMPONENTS thread system iostreams REQUIRED CONFIG)
LINK_LIBRARIES(${DEPENDENCIES} magma MKL::MKL TBB::tbb ${JEMALLOC_LIBRARIES} ${Boost_LIBRARIES} prima::primac) # bandicoot

IF(USE_GINKGO)
    LINK_LIBRARIES(Ginkgo::ginkgo)
ENDIF()

IF (EXPERIMENTAL_FEATURES)
    # FIND_PACKAGE(OSQP REQUIRED)
    # FIND_PACKAGE(Eigen3 REQUIRED)
    LINK_DIRECTORIES(${MATLAB_LINK_DIR} "/usr/lib/google/tensorflow")
    EXECUTE_PROCESS(COMMAND python3-config --includes OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE PYINCLUDES)
    STRING(REGEX REPLACE "[\n ]" "" PYINCLUDES "${PYINCLUDES}")
    FIND_PACKAGE(Python3 REQUIRED)
    INCLUDE_DIRECTORIES("${MATLAB_INCLUDE_DIR}" "${PYINCLUDES}" "/usr/include/python3.6m") # "/usr/local/include/eigen3" "/usr/local/include/google/protobuf"
    LINK_LIBRARIES(fftw3 MatlabDataArray MatlabEngine python3.6m) # osqp::osqpstatic clblast tensorflow_all
    APPEND(TEST_DEPENDENCIES ${PROJECT_NAME} python3.6m) # tensorflow_all
ENDIF()

IF(WITH_TESTS AND PROJECT_NAME MATCHES "${WITH_TESTS}")
    MESSAGE("~~ Building ${PROJECT_NAME}-test")
    SET(TEST_DEPENDENCIES ${PROJECT_NAME} SVRCommon SVRBusiness SVRPersist SVRModel wavelib fftw3)
    SET_PROPERTY(DIRECTORY PROPERTY TEST_MODULES "${PROJECT_NAME}")

    FIND_PACKAGE(GTest REQUIRED )
    FIND_PACKAGE(Boost ${Boost_MIN_VERSION} COMPONENTS program_options REQUIRED CONFIG)
    INCLUDE_DIRECTORIES(${GTEST_ROOT}/include)

    FILE(GLOB TEST_SOURCE_FILES "test/*.cpp")

    ADD_EXECUTABLE(${PROJECT_NAME}-test ${TEST_SOURCE_FILES} ${COMMON_SOURCE_FILES})
    TARGET_LINK_LIBRARIES(${PROJECT_NAME}-test ${DEPENDENCIES} hdbscan hdbscan_utils collections ${TEST_DEPENDENCIES} gtest gtest_main pthread ${Boost_LIBRARIES}) # absl::hash absl::flags_reflection absl::raw_hash_set # ${ONEAPI_ROOT}/compiler/latest/lib/libomptarget.sycl.wrap.so # sycl
ENDIF()

ADD_LIBRARY(${PROJECT_NAME} ${SOURCE_FILES})
