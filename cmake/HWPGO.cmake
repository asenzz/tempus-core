include_guard(GLOBAL)

set(HWPGO Off CACHE BOOL "Perform HWPGO feedback using a nested CMake build")

# Explicitly etting to "NEW" does not seem to affect the DEFERred
# hwpgo_finalize invocation as expected, but setting a default does.
set(CMAKE_POLICY_DEFAULT_CMP0112 NEW)

add_compile_options(-fprofile-sample-generate)
add_link_options(-fprofile-sample-generate)

if (HWPGO)
    find_program(PERF NAMES amplxe-perf perf)
    if (PERF)
        set(PROFILER "${PERF}")
    else (PERF)
        find_program(SEP NAMES sep)
        set(PROFILER "${SEP}")
    endif (PERF)

    if (NOT PERF AND NOT SEP)
        message(FATAL_ERROR "Didn't find sep or perf.")
    endif (NOT PERF AND NOT SEP)

    message(STATUS "Profiler is: ${PROFILER}")

    add_compile_options(-fprofile-sample-use=${CMAKE_BINARY_DIR}/base.freq.prof)
    add_compile_options(-mllvm -unpredictable-hints-file=${CMAKE_BINARY_DIR}/base.misp.prof)

    set(HWPGO_FREQ_EVENT "BR_INST_RETIRED.NEAR_TAKEN" CACHE STRING "PMU event to use for execution frequency profiling")
    set(HWPGO_MISP_EVENT "BR_MISP_RETIRED.ALL_BRANCHES" CACHE STRING "PMU event to use for branch mispredict profiling")
    set(HWPGO_SAMPLE_PERIOD 1000003 CACHE STRING "PMU event sampling period")

    if (SEP)
        execute_process(COMMAND "${SEP}" "-el"
                COMMAND_ERROR_IS_FATAL ANY
                OUTPUT_VARIABLE SEP_ECLIST)
        if ("${SEP_ECLIST}" MATCHES "${HWPGO_FREQ_EVENT}_PS")
            string(APPEND HWPGO_FREQ_EVENT "_PS")
        endif()
        if ("${SEP_ECLIST}" MATCHES "${HWPGO_MISP_EVENT}_PS")
            string(APPEND HWPGO_MISP_EVENT "_PS")
        endif()
    endif (SEP)

    if (PERF)
        string(APPEND HWPGO_FREQ_EVENT ":uppp")
        string(APPEND HWPGO_MISP_EVENT ":upp")
    elseif(SEP)
        string(APPEND HWPGO_FREQ_EVENT ":pdir")
    endif (PERF)

    message(STATUS "Using execution frequency PMU event: ${HWPGO_FREQ_EVENT}")
    message(STATUS "Using branch mispredict PMU event: ${HWPGO_MISP_EVENT}")

    set(NOLOGO_OPT "")
    set(CLANGOPT_PREFIX "")
    if (MSVC)
        set(NOLOGO_OPT "/nologo")
        set(CLANGOPT_PREFIX "/clang:")
    endif (MSVC)
    execute_process(COMMAND ${CMAKE_C_COMPILER} "${NOLOGO_OPT}" "${CLANGOPT_PREFIX}--print-prog-name=llvm-profgen"
            COMMAND_ERROR_IS_FATAL ANY
            OUTPUT_STRIP_TRAILING_WHITESPACE
            OUTPUT_VARIABLE PROFGEN)
    execute_process(COMMAND ${CMAKE_C_COMPILER} "${NOLOGO_OPT}" "${CLANGOPT_PREFIX}--print-prog-name=llvm-profdata"
            COMMAND_ERROR_IS_FATAL ANY
            OUTPUT_STRIP_TRAILING_WHITESPACE
            OUTPUT_VARIABLE PROFDATA)
endif (HWPGO)

function(hwpgo_executable binary)
    if (NOT HWPGO)
        return()
    endif (NOT HWPGO)

    cmake_parse_arguments(ARG
            ""
            "IDENTIFIER"
            "ARGUMENTS"
            ${ARGN})

    if (ARG_IDENTIFIER)
        set(IDENT "${ARG_IDENTIFIER}")
    else (ARG_IDENTIFIER)
        set(IDENT "train")
    endif (ARG_IDENTIFIER)

    # We avoid using TARGET_FILE here because this would automatically create a
    # dependency on ${binary}, which is not what we want.
    set(BASE_BIN "${CMAKE_BINARY_DIR}/base_build/$<PATH:RELATIVE_PATH,$<TARGET_FILE_DIR:${binary}>/$<TARGET_FILE_NAME:${binary}>,${CMAKE_BINARY_DIR}>")
    set(TRAIN_ID "${binary}-${IDENT}")
    set(PMU_PROFILE "${TRAIN_ID}.perf.data")
    set(FREQ_PROFILE "${TRAIN_ID}.freq.prof")
    set(MISP_PROFILE "${TRAIN_ID}.misp.prof")

    if (SEP)
        set(PROFILER_ARGS "-start;-out;${TRAIN_ID}.tb7;-ec;${HWPGO_FREQ_EVENT}:SA=${HWPGO_SAMPLE_PERIOD}:PRECISE=YES,${HWPGO_MISP_EVENT}:SA=${HWPGO_SAMPLE_PERIOD}:PRECISE=YES;-lbr;no_filter:usr;-perf-script;event,ip,brstack;-app;\"${BASE_BIN}")
    else (SEP)
        set(PROFILER_ARGS "record;-o;${PMU_PROFILE};-b;-e;${HWPGO_FREQ_EVENT},${HWPGO_MISP_EVENT};-c;${HWPGO_SAMPLE_PERIOD};--;${BASE_BIN}")
    endif (SEP)

    if (ARG_ARGUMENTS)
        list(APPEND PROFILER_ARGS "${ARG_ARGUMENTS}")
    endif (ARG_ARGUMENTS)

    if (SEP)
        string(APPEND PROFILER_ARGS "\"")
    endif (SEP)

    set(PROFGEN_FREQ_ARGS "--output=${FREQ_PROFILE};--binary=${BASE_BIN};--sample-period=${HWPGO_SAMPLE_PERIOD};--perf-event=${HWPGO_FREQ_EVENT}")
    set(PROFGEN_MISP_ARGS "--output=${MISP_PROFILE};--binary=${BASE_BIN};--sample-period=${HWPGO_SAMPLE_PERIOD};--perf-event=${HWPGO_MISP_EVENT};--leading-ip-only")

    if (SEP)
        list(APPEND PROFGEN_FREQ_ARGS "--perfscript=${PMU_PROFILE}.script")
        list(APPEND PROFGEN_MISP_ARGS "--perfscript=${PMU_PROFILE}.script")
    else (SEP)
        list(APPEND PROFGEN_FREQ_ARGS "--perfdata=${PMU_PROFILE}")
        list(APPEND PROFGEN_MISP_ARGS "--perfdata=${PMU_PROFILE}")
    endif (SEP)

    add_custom_target(${TRAIN_ID}.profile
            DEPENDS ${PMU_PROFILE} ${FREQ_PROFILE} ${MISP_PROFILE}
            COMMENT "Profiling ${TRAIN_ID}"
    )

    add_custom_command(OUTPUT ${PMU_PROFILE}
            COMMAND ${PROFILER} ARGS ${PROFILER_ARGS}
            DEPENDS base
    )
    add_custom_command(OUTPUT ${FREQ_PROFILE}
            COMMAND ${PROFGEN} ARGS ${PROFGEN_FREQ_ARGS}
            DEPENDS base ${PMU_PROFILE}
    )
    # Note: the dependency on FREQ_PROFILE here is a not a true dependency; it's
    # specified so that we don't get two concurrent llvm-profgen runs clobbering
    # one another's temporary files.
    add_custom_command(OUTPUT ${MISP_PROFILE}
            COMMAND ${PROFGEN} ARGS ${PROFGEN_MISP_ARGS}
            DEPENDS base ${PMU_PROFILE} ${FREQ_PROFILE}
    )

    list(APPEND FREQ_PROFILES ${FREQ_PROFILE})
    list(APPEND MISP_PROFILES ${MISP_PROFILE})
    list(APPEND PROFILE_TARGETS ${TRAIN_ID}.profile)

    set(FREQ_PROFILES ${FREQ_PROFILES} PARENT_SCOPE)
    set(MISP_PROFILES ${MISP_PROFILES} PARENT_SCOPE)
    set(PROFILE_TARGETS ${PROFILE_TARGETS} PARENT_SCOPE)
endfunction()

function(hwpgo_finalize)
    if (NOT HWPGO)
        return()
    endif (NOT HWPGO)

    if (NOT PROFILE_TARGETS)
        message(FATAL_ERROR "HWPGO requested, but no binaries were specified to profile")
    endif(NOT PROFILE_TARGETS)

    macro(get_subdirectory_targets list dir)
        get_property(parent_targets DIRECTORY ${dir} PROPERTY BUILDSYSTEM_TARGETS)
        list(APPEND ${list} ${parent_targets})

        get_property(children DIRECTORY ${dir} PROPERTY SUBDIRECTORIES)
        foreach(child ${children})
            get_subdirectory_targets(${list} ${child})
        endforeach()
    endmacro()

    set(all_targets "")
    get_subdirectory_targets(all_targets ${CMAKE_CURRENT_SOURCE_DIR})
    list(REMOVE_ITEM all_targets ${PROFILE_TARGETS})
    foreach(target ${all_targets})
        add_dependencies(${target} profiles)
    endforeach()

    add_custom_target(profiles DEPENDS base.freq.prof base.misp.prof)
    add_custom_command(OUTPUT base.freq.prof
            COMMAND ${PROFDATA} ARGS merge --sample --output base.freq.prof ${FREQ_PROFILES}
            DEPENDS ${PROFILE_TARGETS})
    add_custom_command(OUTPUT base.misp.prof
            COMMAND ${PROFDATA} ARGS merge --sample --output base.misp.prof ${MISP_PROFILES}
            DEPENDS ${PROFILE_TARGETS})

    set(PASSTHROUGH_ARGS)
    get_cmake_property(CACHE_VARS CACHE_VARIABLES)
    foreach(CACHE_VAR ${CACHE_VARS})
        if (NOT "${CACHE_VAR}" MATCHES "^HWPGO")
            get_property(CACHE_VAR_TYPE CACHE ${CACHE_VAR} PROPERTY TYPE)
            if (NOT "${CACHE_VAR_TYPE}" MATCHES INTERNAL AND
                    NOT "${CACHE_VAR_TYPE}" MATCHES STATIC)
                list(APPEND PASSTHROUGH_ARGS "-D${CACHE_VAR}:${CACHE_VAR_TYPE}=${${CACHE_VAR}}")
            endif ()
        endif ()
    endforeach()

    ExternalProject_Add(base
            PREFIX ${CMAKE_BINARY_DIR}/base_prefix
            SOURCE_DIR ${CMAKE_SOURCE_DIR}
            BINARY_DIR ${CMAKE_BINARY_DIR}/base_build
            INSTALL_COMMAND ""
            CMAKE_CACHE_ARGS ${PASSTHROUGH_ARGS}
    )
endfunction(hwpgo_finalize)

if (HWPGO)
    cmake_language(DEFER DIRECTORY ${CMAKE_SOURCE_DIR} CALL hwpgo_finalize())
endif (HWPGO)