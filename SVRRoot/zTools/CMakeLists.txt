CMAKE_MINIMUM_REQUIRED(VERSION 3.10)
PROJECT(Tools)
LINK_DIRECTORIES(/lib /libexec /usr/lib /usr/lib/libexec /usr/local/lib /usr/local/libexec)
#Define project dependencies
SET(DEPENDENCIES SVRBusiness SVRPersist SVRCommon SVRModel OnlineSVR)

INCLUDE(../../cmake/IncludeDependencies.cmake)

FIND_PACKAGE(Boost ${Boost_MIN_VERSION} COMPONENTS program_options REQUIRED)

MACRO(SUBDIRLIST result curdir)
  FILE(GLOB children RELATIVE "${curdir}" "${curdir}/*")
  SET(dirlist "")
  FOREACH(child ${children})
    IF(IS_DIRECTORY ${curdir}/${child})
        LIST(APPEND dirlist ${child})
    ENDIF()
  ENDFOREACH()
  SET(${result} ${dirlist})
ENDMACRO()
          
SUBDIRLIST(SUBDIRS ${CMAKE_CURRENT_SOURCE_DIR})
  
FOREACH(subdir ${SUBDIRS})
    message("********************************************************************************")
    message("*  Building the following tools:")
    if("${subdir}" MATCHES "${WITH_TOOLS}")
        add_subdirectory(${subdir})
        message("* ~~ ${subdir}")
    endif()
    message("********************************************************************************")
ENDFOREACH()
