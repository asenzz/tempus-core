CMAKE_MINIMUM_REQUIRED(VERSION 3.10)
PROJECT(SVRCommon)

FILE(GLOB_RECURSE SOURCE_FILES "src/*.cpp" "src/*.hpp" "include/*.hpp" "include/*.tcc")

FIND_PACKAGE(Boost ${Boost_MIN_VERSION} COMPONENTS regex date_time REQUIRED)
FIND_PACKAGE(OpenCL REQUIRED)
FIND_PACKAGE(TBB REQUIRED)

INCLUDE_DIRECTORIES(BEFORE SYSTEM include ${Boost_INCLUDE_DIR} "/opt/intel/oneapi/tbb/latest/include")

ADD_LIBRARY(${PROJECT_NAME} ${LIBS_BUILD_TYPE} ${SOURCE_FILES})
TARGET_LINK_LIBRARIES(${PROJECT_NAME} ${JEMALLOC_LIBRARIES} ${Boost_LIBRARIES} boost_serialization armadillo lz4 crypto ssl boost_thread boost_system boost_log TBB::tbb)
TARGET_INCLUDE_DIRECTORIES(${PROJECT_NAME} PUBLIC "/opt/intel/oneapi/tbb/latest/include/tbb")
IF(ENABLE_OPENCL)
        INCLUDE_DIRECTORIES(BEFORE SYSTEM ${OpenCL_INCLUDE_DIRS})
        TARGET_LINK_LIBRARIES(${PROJECT_NAME} ${OpenCL_LIBRARIES})
ENDIF(ENABLE_OPENCL)

SET(TEST_DEPENDENCIES "SVRCommon")
        
include ("../../cmake/UnitTester.cmake")
