project(DML)
cmake_minimum_required(VERSION 2.8)
add_definitions(-std=c++0x -O3)
include_directories(/usr/local/include)
link_directories(/usr/local/lib)


set(COMMON_SRC_LIST itml.cpp utility.cpp)


add_executable(demo ${COMMON_SRC_LIST} demo.cpp)
add_executable(gen_cross_index gen_cross_index.cpp ${COMMON_SRC_LIST})
add_executable(gen_constraints ${COMMON_SRC_LIST} gen_constraints.cpp)


target_link_libraries(demo boost_system boost_thread)
target_link_libraries(gen_constraints boost_system boost_thread)
