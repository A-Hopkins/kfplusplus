cmake_minimum_required(VERSION 3.5.0)
project(kfplusplus VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_library(kfplusplus INTERFACE)
target_include_directories(kfplusplus INTERFACE include)

add_subdirectory(extern/pybind11)
# Only build bindings if requested
option(BUILD_PYTHON_BINDINGS "Build Python bindings with pybind11" ON)

if(BUILD_PYTHON_BINDINGS)
  pybind11_add_module(kfplusplus_python bindings/bindings.cpp)
  target_include_directories(kfplusplus_python PRIVATE include)
  # If you have dependencies, link them here
endif()

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

option(BUILD_TESTING "Build and run unit tests" ON)
option(BUILD_EXAMPLES "Build example programs" ON)

if(BUILD_TESTING)
  enable_testing()
  add_subdirectory(test)
endif()

if(BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()

# Collect source files with glob only for custom targets
file(GLOB_RECURSE ALL_SOURCE_FILES
  "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/include/*.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/examples/*.cpp")

# Custom target for clang format utility
if(BUILD_TESTING)
  add_custom_target(
    clang_format
    COMMAND clang-format -i ${ALL_SOURCE_FILES}
  )
endif()