cmake_minimum_required(VERSION 3.16)
project(lfm_host CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

add_subdirectory(../lfm lfm)

add_executable(lfm_host main.cpp)
target_link_libraries(lfm_host PRIVATE lfm)
target_compile_options(lfm_host PRIVATE -Wall -Wextra)
