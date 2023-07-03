cmake_minimum_required(VERSION 3.0.2)
project(length_penalty_metrics_tests)

add_compile_options(-pthread -std=c++17 -funroll-loops -Wall -Ofast)
set(CMAKE_BUILD_TYPE Release)
#set(CMAKE_BUILD_TYPE Debug)

find_package(catkin REQUIRED COMPONENTS
  length_penalty_metrics ssm15066_estimators nn_ssm object_loader_msgs roscpp
)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES length_penalty_metrics_tests
  CATKIN_DEPENDS length_penalty_metrics ssm15066_estimators object_loader_msgs roscpp nn_ssm
  DEPENDS
)

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)

add_executable( test_length_penalty_metrics src/test_length_penalty_metrics.cpp)
add_dependencies( test_length_penalty_metrics ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries( test_length_penalty_metrics
${catkin_LIBRARIES}
)

add_executable( test_length_penalty_metrics_speed src/test_length_penalty_metrics_speed.cpp)
add_dependencies( test_length_penalty_metrics_speed ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries( test_length_penalty_metrics_speed
${catkin_LIBRARIES}
)

add_executable( test_length_penalty_metrics_parallel src/test_length_penalty_metrics_parallel.cpp)
add_dependencies( test_length_penalty_metrics_parallel ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries( test_length_penalty_metrics_parallel
${catkin_LIBRARIES}
)

add_executable( test_rosdyn_clone src/test_rosdyn_clone.cpp)
add_dependencies( test_rosdyn_clone ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries( test_rosdyn_clone
${catkin_LIBRARIES}
)

add_executable( test_nn_performance src/test_nn_performance.cpp)
add_dependencies( test_nn_performance ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries( test_nn_performance
${catkin_LIBRARIES}
)
