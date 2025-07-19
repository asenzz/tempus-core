PROJECT(connector-tester)

INCLUDE(../../../cmake/SourceList.cmake)

IF(EXPERIMENTAL_FEATURES)
    LINK_DIRECTORIES(${MATLAB_LINK_DIR})
ENDIF(EXPERIMENTAL_FEATURES)

ADD_EXECUTABLE(${PROJECT_NAME} 
    main.cpp 
    tui.cpp tui.hpp
    controller.cpp controller.hpp
    command.cpp command.hpp
    command_text.cpp command_text.hpp
    ${COMMON_SOURCE_FILES}
)

TARGET_LINK_LIBRARIES(${PROJECT_NAME} ${DEPENDENCIES} ${Boost_LIBRARIES} ${JEMALLOC_LIBRARIES})
