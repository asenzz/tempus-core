CMAKE_MINIMUM_REQUIRED(VERSION 3.10)
INCLUDE("../../cmake/Util.cmake")

PROJECT(SVRCommon)

INCLUDE("../../cmake/SourceList.cmake")

FIND_PACKAGE(Boost ${Boost_MIN_VERSION} COMPONENTS regex date_time serialization REQUIRED)
FIND_PACKAGE(OpenCL REQUIRED)
FIND_PACKAGE(TBB REQUIRED)

ADD_LIBRARY(${PROJECT_NAME} ${LIBS_BUILD_TYPE} ${SOURCE_FILES})

TARGET_LINK_LIBRARIES(${PROJECT_NAME} ${JEMALLOC_LIBRARIES} ${Boost_LIBRARIES} boost_serialization boost_thread boost_system boost_log lz4 crypto ssl TBB::tbb fftw3 IPP::ippcore IPP::ipps IPP::ippi IPP::ippcv jemalloc MKL::MKL) # IPP freeze on first call because linked in two or more shared objects
IF (ENABLE_OPENCL)
    INCLUDE_DIRECTORIES(BEFORE SYSTEM ${OpenCL_INCLUDE_DIRS})
    TARGET_LINK_LIBRARIES(${PROJECT_NAME} ${OpenCL_LIBRARIES})
ENDIF (ENABLE_OPENCL)

SET(TEST_DEPENDENCIES "SVRCommon")

INCLUDE("../../cmake/UnitTester.cmake")