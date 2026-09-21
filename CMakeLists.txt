cmake_minimum_required(VERSION 3.18)
project(cis5650_boids LANGUAGES CUDA CXX)

set(FLOCKING_BLOCK_SIZE 128 CACHE STRING "CUDA threads per block for flocking kernels")
set(FLOCKING_VISUALIZE 1 CACHE STRING "Render OpenGL boids (1) or run simulation without draw calls (0)")

# Helpful for IDE tooling (IntelliSense/clangd). Some generators may ignore it,
# but it is safe to enable.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Enable C++17 for host code
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Set a default build type if none was specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    SET(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
    # Set the possible values of build type for cmake-gui
    SET_PROPERTY(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

find_package(OpenGL REQUIRED)

# CUDA headers are used from host (C++) sources too (e.g. main.cpp). Some IDE
# tooling doesn't automatically pick up CUDA's include path on Windows unless we
# explicitly attach it to the target.
find_package(CUDAToolkit QUIET)

if(UNIX)
    include_directories("${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
    find_package(glfw3 REQUIRED)
    find_package(GLEW REQUIRED)
    set(GL_LIBRARIES glfw ${GLEW_LIBRARIES} ${OPENGL_LIBRARIES})
else(UNIX)
    set(EXTERNAL "${CMAKE_SOURCE_DIR}/external")

    set(GLFW_ROOT_DIR ${EXTERNAL})
    set(GLFW_USE_STATIC_LIBS ON)
    find_package(GLFW REQUIRED)

    set(GLEW_ROOT_DIR ${EXTERNAL})
    set(GLEW_USE_STATIC_LIBS ON)
    find_package(GLEW REQUIRED)

    add_definitions(${GLEW_DEFINITIONS})
    include_directories(${GLEW_INCLUDE_DIR} ${GLFW_INCLUDE_DIR})
    set(GL_LIBRARIES ${GLEW_LIBRARY} ${GLFW_LIBRARY} ${OPENGL_LIBRARY})
endif(UNIX)

set(GLM_ROOT_DIR "${CMAKE_SOURCE_DIR}/external")
find_package(GLM REQUIRED)
include_directories(${GLM_INCLUDE_DIRS})

find_package(CCCL REQUIRED)

set(headers
    src/cudaMat4.hpp
    src/glslUtility.hpp
    src/kernel.h
    src/main.hpp
    src/utilityCore.hpp
    )

set(sources
    src/glslUtility.cpp
    src/kernel.cu
    src/main.cpp
    src/utilityCore.cpp
    )

list(SORT headers)
list(SORT sources)

source_group(Headers FILES ${headers})
source_group(Sources FILES ${sources})

add_executable(${CMAKE_PROJECT_NAME} ${sources} ${headers})

# Headless CUDA-event benchmark. It reuses the simulation kernels but does not
# create a GLFW/OpenGL context, so benchmark timings exclude rendering.
add_executable(cis5650_benchmark
    benchmarks/benchmark_main.cpp
    src/kernel.cu
    src/utilityCore.cpp
    src/utilityCore.hpp
    src/cudaMat4.hpp
)

# Make CUDA headers available to all translation units for both build + IDE.
if(CUDAToolkit_FOUND)
    target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
elseif(DEFINED CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES)
    target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
elseif(DEFINED ENV{CUDA_PATH})
    target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE "$ENV{CUDA_PATH}/include")
endif()
target_link_libraries(${CMAKE_PROJECT_NAME} ${GL_LIBRARIES} CCCL::Thrust)
target_link_libraries(cis5650_benchmark PRIVATE CCCL::Thrust)
target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE
    FLOCKING_BLOCK_SIZE=${FLOCKING_BLOCK_SIZE}
    FLOCKING_VISUALIZE=${FLOCKING_VISUALIZE}
)
target_compile_definitions(cis5650_benchmark PRIVATE
    FLOCKING_BLOCK_SIZE=${FLOCKING_BLOCK_SIZE}
)
if(CMAKE_VERSION VERSION_LESS "3.23.0")
    set_target_properties(${CMAKE_PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES OFF)
elseif(CMAKE_VERSION VERSION_LESS "3.24.0")
    set_target_properties(${CMAKE_PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES all-major)
else()
    set_target_properties(${CMAKE_PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES native)
endif()
if(CMAKE_VERSION VERSION_LESS "3.23.0")
    set_target_properties(cis5650_benchmark PROPERTIES CUDA_ARCHITECTURES OFF)
elseif(CMAKE_VERSION VERSION_LESS "3.24.0")
    set_target_properties(cis5650_benchmark PROPERTIES CUDA_ARCHITECTURES all-major)
else()
    set_target_properties(cis5650_benchmark PROPERTIES CUDA_ARCHITECTURES native)
endif()

if(CUDAToolkit_FOUND)
    target_include_directories(cis5650_benchmark PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
elseif(DEFINED CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES)
    target_include_directories(cis5650_benchmark PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
elseif(DEFINED ENV{CUDA_PATH})
    target_include_directories(cis5650_benchmark PRIVATE "$ENV{CUDA_PATH}/include")
endif()
target_include_directories(cis5650_benchmark PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${GLM_INCLUDE_DIRS}
)
target_compile_options(${CMAKE_PROJECT_NAME} PRIVATE "$<$<AND:$<CONFIG:Debug,RelWithDebInfo>,$<COMPILE_LANGUAGE:CUDA>>:-G>")
set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ${CMAKE_PROJECT_NAME})

add_custom_command(
    TARGET ${CMAKE_PROJECT_NAME}
    PRE_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/shaders
        ${CMAKE_BINARY_DIR}/shaders
    )
