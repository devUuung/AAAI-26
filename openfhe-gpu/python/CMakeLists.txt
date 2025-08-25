# CMakeLists.txt for Python pybind11 module

cmake_minimum_required(VERSION 3.18)

# pybind11 찾기 (시스템 또는 pip 설치 경로)
find_package(pybind11 QUIET)
if(NOT pybind11_FOUND)
    # pip 설치된 pybind11 헤더 경로를 시도
    execute_process(
        COMMAND python3 -c "import pybind11, sys; sys.stdout.write(pybind11.get_include())"
        OUTPUT_VARIABLE PYBIND11_INCLUDE
        OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(PYBIND11_INCLUDE)
        message(STATUS "Using pybind11 headers at ${PYBIND11_INCLUDE}")
        include_directories(${PYBIND11_INCLUDE})
    else()
        message(FATAL_ERROR "pybind11 not found. Install via: pip install pybind11[global]")
    endif()
endif()

# 파이썬 모듈 소스
set(PYBIND_SRC ${CMAKE_CURRENT_SOURCE_DIR}/openfhe_gpu_pybind.cpp)

# 필요한 include 경로 (OpenFHE 헤더, GPU 유틸 등)
include_directories(
    ${CMAKE_SOURCE_DIR}/src/core/include
    ${CMAKE_SOURCE_DIR}/src/pke/include
    ${CMAKE_SOURCE_DIR}/src/binfhe/include
    ${CMAKE_SOURCE_DIR}/src/core/lib
    ${CMAKE_SOURCE_DIR}/src/core/lib/gpu
)

# 파이썬 확장 모듈 이름: openfhe_gpu
if(pybind11_FOUND)
    pybind11_add_module(openfhe_gpu MODULE ${PYBIND_SRC})
else()
    add_library(openfhe_gpu MODULE ${PYBIND_SRC})
    target_compile_definitions(openfhe_gpu PRIVATE PYBIND11_COMPILER_TYPE=\"CMake\")
    target_compile_features(openfhe_gpu PUBLIC cxx_std_17)
    set_target_properties(openfhe_gpu PROPERTIES PREFIX "" SUFFIX ".so")
endif()

# CUDA/Thrust 및 내부 라이브러리 링크
target_link_directories(openfhe_gpu PUBLIC /usr/local/cuda/lib64)
target_link_directories(openfhe_gpu PUBLIC /usr/local/cuda-12.3/lib64)

target_link_libraries(openfhe_gpu
  PUBLIC
    OPENFHEcore
    OPENFHEpke
    OPENFHEbinfhe
    Thrust
    CUDA::cudart
    rmm
  PRIVATE
    CUDA::nvToolsExt
)

target_compile_features(openfhe_gpu PUBLIC cxx_std_17)

# 빌드 산출물 위치를 빌드 트리 루트의 python 하위로 두기(편의)
set_target_properties(openfhe_gpu PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/python
  ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/python
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/python
)

# 설치 규칙(옵션)
install(TARGETS openfhe_gpu DESTINATION lib)


