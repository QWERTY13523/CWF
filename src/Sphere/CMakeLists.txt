cmake_minimum_required(VERSION 3.17)

set(BGAL_Sphere_SRC
        Sphere.cpp
)
# Get static lib
add_library(Sphere STATIC ${BGAL_Sphere_SRC})
target_link_libraries(Sphere Algorithm BaseShape Model ${Boost_LIBRARIES})
set_target_properties(Sphere PROPERTIES VERSION ${VERSION})
set_target_properties(Sphere PROPERTIES CLEAN_DIRECT_OUTPUT 1)


target_include_directories(Sphere PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>)
