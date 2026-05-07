cmake_minimum_required(VERSION 3.17)
project(BGALPipelineTest LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT DEFINED BGAL_SRC_DIR)
  message(FATAL_ERROR "BGAL_SRC_DIR is not set")
endif()

if(NOT DEFINED VCGLIB_DIR)
  message(FATAL_ERROR "VCGLIB_DIR is not set")
endif()

find_package(Eigen3 REQUIRED)
find_package(CGAL REQUIRED)
find_package(Boost REQUIRED)
find_package(OpenMP REQUIRED)
find_package(libigl CONFIG QUIET)
if(NOT TARGET igl::igl_core)
  add_library(igl_core_stub INTERFACE)
  add_library(igl::igl_core ALIAS igl_core_stub)
endif()

# 让下游 BGAL 子模块里使用的 ${PROJECT_SOURCE_DIR}/include 指到当前 workspace/include。
# 这个 include 目录由 Python 脚本预先从实际工程复制/同步过来。
if(NOT EXISTS "${PROJECT_SOURCE_DIR}/include")
  message(FATAL_ERROR "workspace/include does not exist")
endif()

add_subdirectory("${BGAL_SRC_DIR}/src" "${CMAKE_BINARY_DIR}/bgal_src_build")

add_executable(export_processed_mesh tools/export_processed_mesh.cpp)
target_link_libraries(export_processed_mesh PRIVATE Model)
target_include_directories(export_processed_mesh PRIVATE "${PROJECT_SOURCE_DIR}/include")

add_executable(poisson_sample_xyz tools/poisson_sample_xyz.cpp)
target_include_directories(poisson_sample_xyz PRIVATE
  "${PROJECT_SOURCE_DIR}/include"
  "${VCGLIB_DIR}"
  "${VCGLIB_DIR}/eigenlib"
)
if(OpenMP_CXX_FOUND)
  target_link_libraries(poisson_sample_xyz PRIVATE OpenMP::OpenMP_CXX)
endif()

add_executable(run_cwf_then_quadcover tools/run_cwf_then_quadcover.cpp)
target_link_libraries(run_cwf_then_quadcover PRIVATE QuadCoverLike CVTLike Model)
target_include_directories(run_cwf_then_quadcover PRIVATE "${PROJECT_SOURCE_DIR}/include")

if(TORCH_CXX_FLAGS)
  separate_arguments(TORCH_CXX_FLAGS_LIST NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
  target_compile_options(run_cwf_then_quadcover PRIVATE ${TORCH_CXX_FLAGS_LIST})
endif()
