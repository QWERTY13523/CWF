cmake_minimum_required(VERSION 3.17)
project(BGAL VERSION 1.0 LANGUAGES CXX)

# ---------- 编译选项 ----------
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release RelWithDebInfo MinSizeRel)
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(VERSION ${PROJECT_VERSION})
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# 统一输出目录（四种配置）
foreach(conf IN ITEMS Debug Release RelWithDebInfo MinSizeRel)
  string(TOUPPER ${conf} CONFU)
  set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${CONFU} ${PROJECT_SOURCE_DIR}/lib)
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${CONFU} ${PROJECT_SOURCE_DIR}/lib)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${CONFU} ${PROJECT_SOURCE_DIR}/bin)
endforeach()

# 安装前缀
set(CMAKE_INSTALL_PREFIX "/usr/local" CACHE PATH "BGAL install path")

# Windows 兼容小项
if(MSVC)
  add_compile_options(/bigobj)
endif()

# 公共头
include_directories(${PROJECT_SOURCE_DIR}/include)

# ---------- 依赖 ----------
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
message(STATUS "Eigen3 ${EIGEN3_VERSION_STRING} FOUND")

# Boost：如需具体组件可改为 `find_package(Boost REQUIRED COMPONENTS filesystem ...)`
find_package(Boost REQUIRED)
message(STATUS "BOOST FOUND: ${Boost_INCLUDE_DIRS}")
include_directories(${Boost_INCLUDE_DIRS})

find_package(CGAL REQUIRED COMPONENTS Core)
message(STATUS "CGAL FOUND: ${CGAL_VERSION}")

find_package(OpenMP) # 可选
if(OpenMP_CXX_FOUND)
  message(STATUS "OpenMP FOUND")
endif()

# 关键：libigl 通过 vcpkg 的 CONFIG 包
# vcpkg 会提供 imported target：igl::core（以及 igl::opengl 等可选组件）
find_package(libigl CONFIG REQUIRED)
message(STATUS "libigl FOUND via: ${libigl_DIR}")

# ---------- 依赖聚合目标（子库复用） ----------
# INTERFACE 目标不能带源码，这里只做“依赖打包”
add_library(bgal_deps INTERFACE
        include/BGAL/MinEnclosingBall/MinEnclosingBall.h
        src/MinEnclosingBall/MinEnclosingBall.cpp)
target_include_directories(bgal_deps INTERFACE
  ${PROJECT_SOURCE_DIR}/include
)
target_link_libraries(bgal_deps INTERFACE
  Eigen3::Eigen
  CGAL::CGAL
  igl::core        # 修正：原先的 igl::igl_core 是错误的
)
if(OpenMP_CXX_FOUND)
  target_link_libraries(bgal_deps INTERFACE OpenMP::OpenMP_CXX)
endif()
# 若 Boost 需要链接具体库组件，请在此追加 ${Boost_LIBRARIES} 或 Boost::xxx

# ---------- 子目录 ----------
add_subdirectory(include)
add_subdirectory(src)
add_subdirectory(MAIN)

# ---------- 安装与导出 ----------
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# 需要安装导出的模块名（与你各子目录 add_library 的目标名一致）
set(BGAL_MODULES
  Algorithm BaseShape Draw Geodesic Integral Model Optimization
  PointCloudProcessing PQP Reconstruction Tessellation2D Tessellation3D CVTLike Sphere MinEnclosingBall
)

# 1) 安装实际库目标
install(TARGETS ${BGAL_MODULES}
  EXPORT ${PROJECT_NAME}Targets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# ⚠️ 不要把第三方 imported target（例如 igl::core）纳入 install/EXPORT
# 移除了你原先的 `install(TARGETS igl_core ...)`

# 3) 安装头文件树
install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        FILES_MATCHING PATTERN "*.h")

# 4) 导出 targets
install(EXPORT ${PROJECT_NAME}Targets
  FILE ${PROJECT_NAME}Targets.cmake
  NAMESPACE ${PROJECT_NAME}::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
)

# 5) 生成并安装 BGALConfig.cmake（用于下游 find_package(BGAL)）
configure_package_config_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/BGALConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/BGALConfig.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
)
install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/BGALConfig.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
)

# 安装时的 rpath（Linux 常用；可按需删除）
if(UNIX AND NOT APPLE)
  set(CMAKE_INSTALL_RPATH_USE_LINK_PATH ON)
endif()
