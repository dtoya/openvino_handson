# CMakeList.txt for OpenVINO sample
cmake_minimum_required(VERSION 3.10)

project(hello_classification)

set(TARGET_NAME ${PROJECT_NAME})
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fPIE")

find_package(OpenVINO REQUIRED)
find_package(OpenCV REQUIRED)
add_definitions(-DUSE_OPENCV)

include_directories(${OpenCV_INCLUDE_DIRS})
link_directories()

add_executable(${TARGET_NAME} main.cpp)
#set_target_properties(${TARGET_NAME} PROPERTIES "CMAKE_CXX_FLAGS" "${CMAKE_CXX_FLAGS}") 
target_link_libraries(${TARGET_NAME} PRIVATE openvino::runtime ${OpenCV_LIBS})
