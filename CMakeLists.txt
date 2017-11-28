CMAKE_MINIMUM_REQUIRED(VERSION 2.8 FATAL_ERROR)

set(CMAKE_CXX_FLAGS "-g -Wall -std=c++11")

add_subdirectory(msgs)
add_subdirectory(plugins)
