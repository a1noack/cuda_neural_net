cmake_minimum_required(VERSION 3.15)
project(nn)

set(CMAKE_CXX_STANDARD 14)

include_directories(.)
include_directories(utils)

add_executable(nn
        utils/dataset.cpp
        utils/dataset.hpp
        layer.cpp
        layer.hpp
        #main.cpp
        main3.cpp
        nn.cpp
        nn.hpp)
