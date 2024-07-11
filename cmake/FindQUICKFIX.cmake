# - Find QUICKFIX
# Find the QUICKFIX includes and client library
# This module defines
#  QUICKFIX_INCLUDE_DIRS, where to find quickfix/Session.h
#  QUICKFIX_LIBRARIES, the libraries needed to use QUICKFIX.
#  QUICKFIX_FOUND, If false, do not try to use QUICKFIX.

if (QUICKFIX_INCLUDE_DIRS AND QUICKFIX_LIBRARIES)
    set(QUICKFIX_FOUND TRUE)

else (QUICKFIX_INCLUDE_DIRS AND QUICKFIX_LIBRARIES)

    find_path(QUICKFIX_INCLUDE_DIRS quickfix/Session.h
            /usr/include/
            /usr/local/include/
            /usr/include/quickfix/
            /usr/local/include/quickfix/
            /opt/quickfix/include/
            $ENV{ProgramFiles}/quickfix/*/include
            $ENV{SystemDrive}/quickfix/*/include
            $ENV{QUICKFIX_DIR}/include
    )

    if (WIN32)
        find_library(QUICKFIX_LIBRARIES NAMES quickfix
                PATHS
                $ENV{ProgramFiles}/quickfix/*/lib
                $ENV{SystemDrive}/quickfix/*/lib
                $ENV{QUICKFIX_DIR}/lib
        )
    else (WIN32)
        find_library(QUICKFIX_LIBRARIES NAMES quickfix
                PATHS
                /usr/lib
                /usr/lib/quickfix
                /usr/local/lib
                /usr/local/lib/quickfix
                /opt/quickfix/lib
        )
    endif (WIN32)

    if (QUICKFIX_INCLUDE_DIRS AND QUICKFIX_LIBRARIES)
        set(QUICKFIX_FOUND TRUE)
        message(STATUS "Found QUICKFIX: ${QUICKFIX_INCLUDE_DIRS}, ${QUICKFIX_LIBRARIES}")
    else (QUICKFIX_INCLUDE_DIRS AND QUICKFIX_LIBRARIES)
        set(QUICKFIX_FOUND FALSE)
        if (QUICKFIX_FIND_REQUIRED)
            message(FATAL_ERROR "QUICKFIX not found.Check Environment Variable QUICKFIX_DIR")
        else (QUICKFIX_FIND_REQUIRED)
            message(STATUS "QUICKFIX not found.")
        endif (QUICKFIX_FIND_REQUIRED)
    endif (QUICKFIX_INCLUDE_DIRS AND QUICKFIX_LIBRARIES)

    mark_as_advanced(QUICKFIX_INCLUDE_DIRS QUICKFIX_LIBRARIES)

endif (QUICKFIX_INCLUDE_DIRS AND QUICKFIX_LIBRARIES)
