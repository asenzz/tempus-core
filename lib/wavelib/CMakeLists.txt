set(PROJECT_NAME wavelib)
project(${PROJECT_NAME} C)


# src root path
set(WAVELIB_SRC_ROOT ${PROJECT_SOURCE_DIR} CACHE PATH "Wavelib source root")
# binary output by default
set(COMMON_BIN_PATH ${CMAKE_BINARY_DIR}/Bin)
set(LIBRARY_OUTPUT_PATH ${COMMON_BIN_PATH}/${CMAKE_BUILD_TYPE})
set(EXECUTABLE_OUTPUT_PATH ${COMMON_BIN_PATH}/${CMAKE_BUILD_TYPE})
# set where to find additional cmake modules if any
set(CMAKE_MODULE_PATH ${WAVELIB_SRC_ROOT}/cmake ${CMAKE_MODULE_PATH})


set(WAVELIB_VERSION "1.0.0" CACHE STRING "Wavelib version" FORCE)
message(">>> Building Wavelib version: ${WAVELIB_VERSION}")
message(">>> EXECUTABLE_OUTPUT_PATH = ${EXECUTABLE_OUTPUT_PATH}")

option(BUILD_UT "Enable Unit test" ON)

# cleanup prefix lib for Unix-like OSes
set(CMAKE_SHARED_MODULE_PREFIX)

# install target to this folder by default
set(WAVELIB_BINARY_DIR ${WAVELIB_SRC_ROOT}/bin)
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "${WAVELIB_BINARY_DIR}" CACHE PATH "default install path" FORCE)
endif()

# make include globaly visible
set(PROJECT_WIDE_INCLUDE ${WAVELIB_SRC_ROOT}/include)
include_directories(${PROJECT_WIDE_INCLUDE})


IF(CMAKE_BUILD_TYPE MATCHES Rel)
    SET (CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} -mtune=native -march=native -O3")
ENDIF(CMAKE_BUILD_TYPE MATCHES Rel)

SET(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} -ggdb -std=gnu11 -Wall -mfpmath=sse -msse4.2 -malign-double -msseregparm -mstackrealign -mmovbe -m64 -Wall -Wextra")

include_directories(${COMMON_BIN_PATH})

if(BUILD_UT)
    include(CTest)
    enable_testing()
    add_subdirectory(unitTests)
endif()

add_subdirectory(src)

install(DIRECTORY ${WAVELIB_SRC_ROOT}/include/ DESTINATION include FILES_MATCHING PATTERN "*.h")

install(DIRECTORY ${WAVELIB_SRC_ROOT}/header/ DESTINATION /usr/local/include/wavelib/header)
install(FILES ${WAVELIB_SRC_ROOT}/Bin/libwavelib.a DESTINATION /usr/local/lib)

