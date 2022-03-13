cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
project(system_security)

FILE(COPY cars.xml haarcascade_fullbody.xml DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/)
find_package(OpenCV REQUIRED)

add_executable(
    system_security main.cpp
)

target_link_libraries(system_security  ${OpenCV_LIBS}
)