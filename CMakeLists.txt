cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

#project("system_security" VERSION 0.1.0)

find_package(OpenCV REQUIRED)


add_executable(
    system_security main.cpp
)

target_link_libraries(system_security  ${OpenCV_LIBS}
)