PROJECT(SVRBenchmarks)

#Define project dependencies
SET(DEPENDENCIES SVRModel OnlineSVR)

INCLUDE(../../cmake/IncludeDependencies.cmake)

if(WITH_TESTS AND PROJECT_NAME MATCHES "${WITH_TESTS}")
    message("~~ Building ${PROJECT_NAME}")

    
    SET(GBENCH_ROOT gbenchmark)

    INCLUDE_DIRECTORIES(${GBENCH_ROOT}/include)
    link_directories(${GBENCH_ROOT}/build/src)

    INCLUDE("../../cmake/SourceList.cmake")
    ADD_EXECUTABLE(${PROJECT_NAME} ${SOURCE_FILES} ${COMMON_SOURCE_FILES})
    TARGET_LINK_LIBRARIES(${PROJECT_NAME} ${JEMALLOC_LIBRARIES} ${DEPENDENCIES} benchmark)
endif()

