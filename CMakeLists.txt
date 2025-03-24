cmake_minimum_required(VERSION 3.10)
project(event_processing)

set(CMAKE_CXX_STANDARD 14)

# Ensure Python interpreter is found
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(include)

add_library(event_processing MODULE src/bindings.cpp src/surface.cpp src/eros.cpp)
target_include_directories(event_processing PRIVATE ${Python3_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
target_link_libraries(event_processing PRIVATE pybind11::module ${OpenCV_LIBS})

set_target_properties(event_processing PROPERTIES PREFIX "" SUFFIX ".so")
