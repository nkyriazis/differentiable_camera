cmake_minimum_required(VERSION 3.13.4)
project(differentiable_camera)

option(USE_BOOST_STATIC "Use static boost libraries (or dynamic)" ON)

set(Boost_USE_STATIC_LIBS ${USE_BOOST_STATIC})
find_package(Boost REQUIRED unit_test_framework filesystem)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

add_subdirectory(camera)
add_subdirectory(tests)

add_executable(main main.cpp Eigen.natvis)
target_link_libraries(main camera)

# copy clang
file(COPY borrowed/clang_format/_clang-format DESTINATION ${CMAKE_CURRENT_SOURCE_DIR})