cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
PROJECT(system_security)


find_package(OpenCV REQUIRED)

FILE(COPY yolov3-tiny.cfg yolov3-tiny.weights coco.names DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/)

add_executable(
    system_security main.cpp
)

target_link_libraries(system_security  ${OpenCV_LIBS}
)