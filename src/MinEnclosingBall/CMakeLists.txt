cmake_minimum_required(VERSION 3.17)

set(BGAL_MinEnclosingBall_SRC
        MinEnclosingBall.cpp
)

add_library(MinEnclosingBall STATIC ${BGAL_MinEnclosingBall_SRC})
target_link_libraries(MinEnclosingBall Algorithm BaseShape Model ${Boost_LIBRARIES})
set_target_properties(MinEnclosingBall PROPERTIES VERSION ${VERSION})
set_target_properties(MinEnclosingBall PROPERTIES CLEAN_DIRECT_OUTPUT 1)

target_include_directories(MinEnclosingBall PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>)