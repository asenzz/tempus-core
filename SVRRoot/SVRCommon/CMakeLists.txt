INCLUDE("../../cmake/Util.cmake")

PROJECT(SVRCommon)

INCLUDE("../../cmake/SourceList.cmake")

GLOB_SOURCES("${CMAKE_CURRENT_SOURCE_DIR}/include/util")
GLOB_SOURCES("${CMAKE_CURRENT_SOURCE_DIR}/include/common")

FIND_PACKAGE(Boost ${Boost_MIN_VERSION} COMPONENTS regex date_time serialization REQUIRED CONFIG)
IF(ENABLE_OPENCL)
    FIND_PACKAGE(OpenCL REQUIRED)
ENDIF()
FIND_PACKAGE(TBB REQUIRED)

ADD_LIBRARY(${PROJECT_NAME} ${SOURCE_FILES})

TARGET_LINK_LIBRARIES(${PROJECT_NAME} ${JEMALLOC_LIBRARIES} ${Boost_LIBRARIES} boost_serialization boost_thread boost_system boost_log lz4 crypto ssl TBB::tbb fftw3 jemalloc MKL::MKL) # IPP freeze on first call because linked in two or more shared objects
IF (USE_IPP)
    TARGET_LINK_LIBRARIES(${PROJECT_NAME} ${IPP_LIBRARIES})
ENDIF()
IF (ENABLE_OPENCL)
    INCLUDE_DIRECTORIES(BEFORE SYSTEM ${OpenCL_INCLUDE_DIRS})
    TARGET_LINK_LIBRARIES(${PROJECT_NAME} ${OpenCL_LIBRARIES})
ENDIF (ENABLE_OPENCL)
SET(TEST_DEPENDENCIES SVRCommon SVRBusiness)

INCLUDE("../../cmake/UnitTester.cmake")
