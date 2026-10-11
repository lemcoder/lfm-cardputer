cmake_minimum_required(VERSION 3.25)
project(lfm_host CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()
# Keep assert() on in every host build: CMake's Release/RelWithDebInfo add
# -DNDEBUG, which compiles asserts away (and hides typos inside them).
foreach(cfg RELEASE RELWITHDEBINFO MINSIZEREL)
    string(REPLACE "-DNDEBUG" "" CMAKE_CXX_FLAGS_${cfg} "${CMAKE_CXX_FLAGS_${cfg}}")
endforeach()

add_subdirectory(../lfm lfm)

add_executable(lfm_host main.cpp)
target_link_libraries(lfm_host PRIVATE lfm)
target_compile_options(lfm_host PRIVATE -Wall -Wextra)

# Tests: GoogleTest, fetched at configure time (host only, never in firmware).
include(FetchContent)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/releases/download/v1.17.0/googletest-1.17.0.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    SYSTEM  # treat gtest headers as system headers: no warnings from them
)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

add_executable(lfm_tests tests/test_math.cpp)
target_link_libraries(lfm_tests PRIVATE lfm GTest::gtest_main)
target_compile_options(lfm_tests PRIVATE -Wall -Wextra)

enable_testing()
include(GoogleTest)
gtest_discover_tests(lfm_tests)
