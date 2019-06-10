add_library(camera INTERFACE)
target_include_directories(camera INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/.. ${Boost_INCLUDE_DIR})
target_link_libraries(camera INTERFACE Eigen3::Eigen)