cmake_minimum_required(VERSION 3.17)

set(BGAL_QuadCoverLike_SRC
    QuadCover.cpp
)

find_package(CGAL REQUIRED)
find_package(Torch REQUIRED CONFIG)

foreach(_torch_target IN ITEMS caffe2::mkl c10 torch_cpu torch_cpu_library torch torch_library)
  if(TARGET ${_torch_target})
    get_target_property(_torch_target_includes ${_torch_target} INTERFACE_INCLUDE_DIRECTORIES)
    if(_torch_target_includes)
      list(FILTER _torch_target_includes EXCLUDE REGEX "NOTFOUND")
      set_property(TARGET ${_torch_target} PROPERTY INTERFACE_INCLUDE_DIRECTORIES "${_torch_target_includes}")
    endif()
  endif()
endforeach()

if(TARGET caffe2::mkl)
  set_property(TARGET caffe2::mkl PROPERTY INTERFACE_LINK_LIBRARIES "")
endif()

if(TARGET torch_cpu)
  get_target_property(_torch_cpu_link_libs torch_cpu INTERFACE_LINK_LIBRARIES)
  if(_torch_cpu_link_libs)
    list(FILTER _torch_cpu_link_libs EXCLUDE REGEX "^(FALSE|caffe2::mkl)$")
    set_property(TARGET torch_cpu PROPERTY INTERFACE_LINK_LIBRARIES "${_torch_cpu_link_libs}")
  endif()
endif()

add_library(QuadCoverLike STATIC ${BGAL_QuadCoverLike_SRC})

target_compile_definitions(QuadCoverLike PRIVATE
    GLOG_USE_GLOG_EXPORT
    GLOG_USE_GFLAGS)

target_include_directories(QuadCoverLike BEFORE PRIVATE
    ${TORCH_INCLUDE_DIRS})

target_link_libraries(QuadCoverLike
    Algorithm
    BaseShape
    Model
    Tessellation2D
    Tessellation3D
    Sphere
    Optimization
    CGAL::CGAL
    ${TORCH_LIBRARIES}
    ${Boost_LIBRARIES}
)
if(OpenMP_CXX_FOUND)
  target_link_libraries(QuadCoverLike OpenMP::OpenMP_CXX)
endif()
set_target_properties(QuadCoverLike PROPERTIES VERSION ${PROJECT_VERSION})
set_target_properties(QuadCoverLike PROPERTIES CLEAN_DIRECT_OUTPUT 1)

if(TORCH_CXX_FLAGS)
  separate_arguments(TORCH_CXX_FLAGS_LIST NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
  target_compile_options(QuadCoverLike PRIVATE ${TORCH_CXX_FLAGS_LIST})
endif()
target_compile_options(QuadCoverLike PRIVATE
  $<$<AND:$<NOT:$<CXX_COMPILER_ID:MSVC>>,$<CONFIG:Release,RelWithDebInfo,MinSizeRel>>:-O3>
  $<$<AND:$<CXX_COMPILER_ID:MSVC>,$<CONFIG:Release,RelWithDebInfo,MinSizeRel>>:/O2>)

target_include_directories(QuadCoverLike PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
