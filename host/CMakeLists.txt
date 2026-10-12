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

# Third-party, fetched at configure time (host only, never in firmware).
include(FetchContent)
FetchContent_Declare(
    json
    URL https://github.com/nlohmann/json/releases/download/v3.12.0/json.tar.xz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    SYSTEM
)
FetchContent_MakeAvailable(json)

# Host-only I/O: safetensors reader, golden-file loader.
file(GLOB LFM_IO_SRCS CONFIGURE_DEPENDS src/*.cpp)
add_library(lfm_io STATIC ${LFM_IO_SRCS})
target_include_directories(lfm_io PUBLIC include)
target_link_libraries(lfm_io PUBLIC lfm PRIVATE nlohmann_json::nlohmann_json)
target_compile_options(lfm_io PRIVATE -Wall -Wextra)

add_executable(lfm_host main.cpp)
target_link_libraries(lfm_host PRIVATE lfm_io)  # lfm_io brings lfm
target_compile_options(lfm_host PRIVATE -Wall -Wextra)

# Tests: GoogleTest.
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/releases/download/v1.17.0/googletest-1.17.0.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    SYSTEM  # treat gtest headers as system headers: no warnings from them
)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

file(GLOB LFM_TEST_SRCS CONFIGURE_DEPENDS tests/*.cpp)
add_executable(lfm_tests ${LFM_TEST_SRCS})
target_link_libraries(lfm_tests PRIVATE lfm_io nlohmann_json::nlohmann_json GTest::gtest_main)
# Absolute repo root, so tests can find golden/ wherever they're run from.
target_compile_definitions(lfm_tests PRIVATE LFM_REPO_ROOT="${CMAKE_CURRENT_SOURCE_DIR}/..")
target_compile_options(lfm_tests PRIVATE -Wall -Wextra)

enable_testing()
include(GoogleTest)
gtest_discover_tests(lfm_tests)

# Quantization quality: perplexity / agreement of .lfm variants vs a reference.
add_executable(lfm_eval eval.cpp)
target_link_libraries(lfm_eval PRIVATE lfm)
target_compile_options(lfm_eval PRIVATE -Wall -Wextra)
