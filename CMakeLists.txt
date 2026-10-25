cmake_minimum_required(VERSION 3.18)

# JetPack's nvcc is not on the PATH by default
if(NOT DEFINED CMAKE_CUDA_COMPILER AND EXISTS /usr/local/cuda/bin/nvcc)
  set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
endif()
# GPU architecture from the SoC: tegra234 = Orin (87), tegra194 = Xavier (72), tegra210 = Nano (53)
# 'compatible' holds several entries separated by \0 (board, module, SoC); file(READ) stops at
# the first \0 and would miss the SoC, while file(STRINGS) splits at the nulls and reads them all.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(_soc "")
  if(EXISTS /proc/device-tree/compatible)
    file(STRINGS /proc/device-tree/compatible _soc)
  endif()
  if(_soc MATCHES "tegra234")
    set(CMAKE_CUDA_ARCHITECTURES 87)
  elseif(_soc MATCHES "tegra194")
    set(CMAKE_CUDA_ARCHITECTURES 72)
  elseif(_soc MATCHES "tegra210")
    set(CMAKE_CUDA_ARCHITECTURES 53)
  else()
    message(FATAL_ERROR "SoC not recognized in /proc/device-tree/compatible (${_soc}). "
      "Set the GPU architecture: cmake -DCMAKE_CUDA_ARCHITECTURES=<sm> ...")
  endif()
endif()

project(edgebench LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)  # nvcc from CUDA 10.2 (JetPack 4) supports up to C++14 only
message(STATUS "GPU sm_${CMAKE_CUDA_ARCHITECTURES}, CUDA ${CMAKE_CUDA_COMPILER_VERSION}")
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

find_package(CUDAToolkit REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)
find_library(NVINFER_LIB nvinfer HINTS /usr/lib/aarch64-linux-gnu REQUIRED)
find_path(NVINFER_INC NvInfer.h HINTS /usr/include/aarch64-linux-gnu REQUIRED)

option(WITH_NVJPEG "JPEG decoding on the GPU with nvJPEG (CUDA)" ON)

add_executable(bench src/bench.cpp src/preproc.cu)
target_include_directories(bench PRIVATE src ${NVINFER_INC} ${OpenCV_INCLUDE_DIRS})
target_link_libraries(bench PRIVATE ${NVINFER_LIB} CUDA::cudart ${OpenCV_LIBS} pthread)
# gcc < 9 (Ubuntu 18.04) needs libstdc++fs for std::(experimental::)filesystem
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9)
  target_link_libraries(bench PRIVATE stdc++fs)
endif()

if(WITH_NVJPEG AND TARGET CUDA::nvjpeg)
  target_compile_definitions(bench PRIVATE WITH_NVJPEG)
  target_link_libraries(bench PRIVATE CUDA::nvjpeg)
  message(STATUS "nvJPEG: yes (${CUDAToolkit_LIBRARY_DIR})")
else()
  message(WARNING "nvJPEG not found: --decode gpu will be unavailable")
endif()

add_executable(probe src/probe.cpp)
