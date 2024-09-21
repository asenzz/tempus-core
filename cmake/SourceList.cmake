MESSAGE("CMAKE_CURRENT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}")
LINK_DIRECTORIES(/lib /libexec /usr/lib /usr/lib/libexec /usr/local/lib /usr/local/libexec)

MACRO(GLOB_SOURCES DIR)
    FILE(GLOB SOURCE_FILES_
            "${DIR}/*.c"
            "${DIR}/*.cpp"
            "${DIR}/include/*.h"
            "${DIR}/include/*.hpp"
            "${DIR}/include/*.tpp"
    )
    IF(USE_CUDA)
        FILE(GLOB SOURCE_FILES_CUDA "${DIR}/*.cu" "${DIR}/*.cuh")
        LIST(APPEND SOURCE_FILES_ ${SOURCE_FILES_CUDA})
        IF(USE_LTO)
            LIST(APPEND COMMON_SOURCE_FILES ${COMMON_SOURCE_FILES_CUDA})
        ENDIF()
    ENDIF()
    LIST(APPEND SOURCE_FILES ${SOURCE_FILES_})
ENDMACRO()

GLOB_SOURCES("${CMAKE_CURRENT_SOURCE_DIR}/src")
GLOB_SOURCES("${CMAKE_CURRENT_SOURCE_DIR}/src/*DAO")
GLOB_SOURCES("${CMAKE_CURRENT_SOURCE_DIR}/include")
