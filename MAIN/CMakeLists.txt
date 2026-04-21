cmake_minimum_required(VERSION 3.17)

add_executable(MAIN main.cpp)
target_link_libraries(MAIN Algorithm BaseShape Draw Geodesic Integral Model Optimization PointCloudProcessing PQP Reconstruction Tessellation2D Tessellation3D CVTLike igl::igl_core)
target_include_directories(MAIN PUBLIC
	$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
	$<INSTALL_INTERFACE:include>)


add_executable(cwf cwf.cpp)
target_link_libraries(cwf Algorithm BaseShape Draw Geodesic Integral Model Optimization PointCloudProcessing PQP Reconstruction Tessellation2D Tessellation3D CVTLike igl::igl_core)
target_include_directories(cwf PUBLIC
	$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
	$<INSTALL_INTERFACE:include>)


add_executable(quadcover_main quadcover_main.cpp)
target_link_libraries(quadcover_main Algorithm BaseShape Draw Geodesic Integral Model Optimization PointCloudProcessing PQP Reconstruction Tessellation2D Tessellation3D CVTLike QuadCoverLike Sphere igl::igl_core)
target_include_directories(quadcover_main PUBLIC
	$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
	$<INSTALL_INTERFACE:include>)
if(OpenMP_CXX_FOUND)
  target_link_libraries(quadcover_main OpenMP::OpenMP_CXX)
endif()
target_compile_options(quadcover_main PRIVATE
  $<$<AND:$<NOT:$<CXX_COMPILER_ID:MSVC>>,$<CONFIG:Release,RelWithDebInfo,MinSizeRel>>:-O3>
  $<$<AND:$<CXX_COMPILER_ID:MSVC>,$<CONFIG:Release,RelWithDebInfo,MinSizeRel>>:/O2>)

add_executable(quadcover_batch quadcover_batch.cpp)
target_include_directories(quadcover_batch PUBLIC
	$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
	$<INSTALL_INTERFACE:include>)
target_compile_options(quadcover_batch PRIVATE
  $<$<AND:$<NOT:$<CXX_COMPILER_ID:MSVC>>,$<CONFIG:Release,RelWithDebInfo,MinSizeRel>>:-O3>
  $<$<AND:$<CXX_COMPILER_ID:MSVC>,$<CONFIG:Release,RelWithDebInfo,MinSizeRel>>:/O2>)
