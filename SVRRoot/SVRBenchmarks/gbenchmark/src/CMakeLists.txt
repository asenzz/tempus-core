# Allow the source files to find headers in src/
include_directories(${PROJECT_SOURCE_DIR}/src)

if (DEFINED BENCHMARK_CXX_LINKER_FLAGS)
  list(APPEND CMAKE_SHARED_LINKER_FLAGS ${BENCHMARK_CXX_LINKER_FLAGS})
  list(APPEND CMAKE_MODULE_LINKER_FLAGS ${BENCHMARK_CXX_LINKER_FLAGS})
endif()

file(GLOB
  SOURCE_FILES
    *.cc
    ${PROJECT_SOURCE_DIR}/include/benchmark/*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/*.h)

add_library(benchmark ${SOURCE_FILES})
set_target_properties(benchmark PROPERTIES
  OUTPUT_NAME "benchmark"
  VERSION ${GENERIC_LIB_VERSION}
  SOVERSION ${GENERIC_LIB_SOVERSION}
)

# Link threads.
target_link_libraries(benchmark ${JEMALLOC_LIBRARIES} ${BENCHMARK_CXX_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})
find_library(LIBRT rt)
if(LIBRT)
  target_link_libraries(benchmark ${LIBRT})
endif()

# We need extra libraries on Windows
if(${CMAKE_SYSTEM_NAME} MATCHES "Windows")
  target_link_libraries(benchmark Shlwapi)
endif()

set(include_install_dir "include")
set(lib_install_dir "lib/")
set(bin_install_dir "bin/")
set(config_install_dir "lib/cmake/${PROJECT_NAME}")

set(generated_dir "${CMAKE_CURRENT_BINARY_DIR}/generated")

set(version_config "${generated_dir}/${PROJECT_NAME}ConfigVersion.cmake")
set(project_config "${generated_dir}/${PROJECT_NAME}Config.cmake")
set(targets_export_name "${PROJECT_NAME}Targets")

set(namespace "${PROJECT_NAME}::")

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${version_config}" VERSION ${GIT_VERSION} COMPATIBILITY SameMajorVersion
)

configure_file("${PROJECT_SOURCE_DIR}/cmake/Config.cmake.in" "${project_config}" @ONLY)

# Install target (will install the library to specified CMAKE_INSTALL_PREFIX variable)
install(
  TARGETS benchmark
  EXPORT ${targets_export_name}
  ARCHIVE DESTINATION ${lib_install_dir}
  LIBRARY DESTINATION ${lib_install_dir}
  RUNTIME DESTINATION ${bin_install_dir}
  INCLUDES DESTINATION ${include_install_dir})

install(
  DIRECTORY "${PROJECT_SOURCE_DIR}/include/benchmark"
  DESTINATION ${include_install_dir}
  FILES_MATCHING PATTERN "*.*h")

install(
    FILES "${project_config}" "${version_config}"
    DESTINATION "${config_install_dir}")

install(
    EXPORT "${targets_export_name}"
    NAMESPACE "${namespace}"
    DESTINATION "${config_install_dir}")
