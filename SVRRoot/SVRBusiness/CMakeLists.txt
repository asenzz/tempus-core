CMAKE_MINIMUM_REQUIRED(VERSION 3.12)

INCLUDE("../../cmake/Util.cmake")

IF(USE_CUDA)
        PROJECT(SVRBusiness LANGUAGES CXX CUDA)
ELSE()
        PROJECT(SVRBusiness LANGUAGES CXX)
ENDIF()

SET(BUILD_MAIN OFF)

#Define project dependencies
SET(DEPENDENCIES SVRPersist SVRCommon SVRFix-shared SVRModel OnlineSVR)

INCLUDE(../../cmake/IncludeDependencies.cmake)
INCLUDE_DIRECTORIES(../SVRFix/include)
INCLUDE("../../cmake/SourceList.cmake")

INCLUDE_DIRECTORIES(include)
LINK_DIRECTORIES(${PQXX_LIBRARIES_DIRECTORY})
IF(EXPERIMENTAL_FEATURES)
    LINK_DIRECTORIES(${MATLAB_LINK_DIR})
ENDIF(EXPERIMENTAL_FEATURES)

IF(BUILD_MAIN)
    ADD_EXECUTABLE(${PROJECT_NAME} "src/main.cpp")
ELSE()
    LIST(REMOVE_ITEM SOURCE_FILES "main.cpp")
    ADD_LIBRARY(${PROJECT_NAME} ${SOURCE_FILES})
ENDIF()


TARGET_LINK_LIBRARIES(${PROJECT_NAME} ${JEMALLOC_LIBRARIES} ${DEPENDENCIES} crypto rt wavelib)

LIST (FIND MODULES "SVRFix" idx)
if (${idx} GREATER -1)
    TARGET_LINK_LIBRARIES(${PROJECT_NAME} SVRFix-shared)
endif()