CMAKE_MINIMUM_REQUIRED(VERSION 2.8.12)
PROJECT(connector-tester)
    
link_directories(${GTEST_ROOT}/lib)
IF(EXPERIMENTAL_FEATURES)
    LINK_DIRECTORIES(${MATLAB_LINK_DIR})
ENDIF(EXPERIMENTAL_FEATURES)

ADD_EXECUTABLE(${PROJECT_NAME} 
    main.cpp 
    tui.cpp tui.hpp
    controller.cpp controller.hpp
    command.cpp command.hpp
    command_text.cpp command_text.hpp
)

TARGET_LINK_LIBRARIES(${PROJECT_NAME} ${DEPENDENCIES} ${Boost_LIBRARIES} ${JEMALLOC_LIBRARIES})
