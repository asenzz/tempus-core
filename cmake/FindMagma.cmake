#  find_package(Magma)
#  if(Magma_FOUND)
#    target_link_libraries(TARGET ${MAGMA_LIBRARIES})
#  endif()

set(BUILD_SHARED_LIBS FALSE)

if(NOT BUILD_SHARED_LIBS)
  set(MAGMA_LIB "libclmagma.a")
else()
  set(MAGMA_LIB "libclmagma.so")
endif()

find_path(MAGMA_INCLUDE_DIR NAMES magma.h HINTS /usr/include)

find_library(MAGMA_LIBRARY
             NAMES ${MAGMA_LIB}
             PATHS /usr/lib
                   /usr/local/lib/
             NO_DEFAULT_PATH)

set(MAGMA_INCLUDE_DIRS ${MAGMA_INCLUDE_DIR})
set(MAGMA_LIBRARIES ${MAGMA_LIBRARY})

# Handle the QUIETLY and REQUIRED arguments and set MKL_FOUND to TRUE if
# all listed variables are TRUE.
INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(Magma DEFAULT_MSG MAGMA_LIBRARIES MAGMA_INCLUDE_DIRS)

MARK_AS_ADVANCED(MAGMA_INCLUDE_DIRS MAGMA_LIBRARIES MAGMA_LIBRARY)
