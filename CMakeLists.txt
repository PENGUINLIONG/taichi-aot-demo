cmake_minimum_required(VERSION 3.13)

project(TaichiAotDemo)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Default build type to release.
if (NOT CMAKE_BUILD_TYPE)
    message(WARNING "No build type selected, default to Release")
    set(CMAKE_BUILD_TYPE "Release" CACHE PATH "Build Type" FORCE)
endif()

# Find C-API library to link with.
set(TAICHI_C_API_INSTALL_DIR $ENV{TAICHI_C_API_INSTALL_DIR})
if (NOT EXISTS ${TAICHI_C_API_INSTALL_DIR})
    message(FATAL_ERROR "Environment variable TAICHI_C_API_INSTALL_DIR is not specified")
endif()
find_library(taichi_c_api taichi_c_api HINTS ${TAICHI_C_API_INSTALL_DIR}/lib REQUIRED NO_CMAKE_FIND_ROOT_PATH)
link_libraries(taichi_c_api)

# Include directories for all targets.
include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/external/include
    ${TAICHI_C_API_INSTALL_DIR}/include
)

# Example permutations.
add_executable(ImplicitFem_Desktop
    ${PROJECT_SOURCE_DIR}/src/tiaot/platforms/desktop.cpp
    ${PROJECT_SOURCE_DIR}/src/tiaot/demos/implicit_fem.cpp)
