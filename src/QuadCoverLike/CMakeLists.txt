cmake_minimum_required(VERSION 3.17)

set(BGAL_QuadCoverLike_SRC
    QuadCover.cpp
)

find_package(CGAL REQUIRED)
add_library(QuadCoverLike STATIC ${BGAL_QuadCoverLike_SRC})
target_link_libraries(QuadCoverLike
    Algorithm
    BaseShape
    Model
    Tessellation2D
    Tessellation3D
    Sphere
    Optimization
    CGAL::CGAL
    ${Boost_LIBRARIES})
set_target_properties(QuadCoverLike PROPERTIES VERSION ${PROJECT_VERSION})
set_target_properties(QuadCoverLike PROPERTIES CLEAN_DIRECT_OUTPUT 1)

target_include_directories(QuadCoverLike PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
