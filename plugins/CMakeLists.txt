find_package(Boost 1.40 COMPONENTS system REQUIRED)
find_package(Armadillo REQUIRED)
find_package(PkgConfig)
pkg_check_modules(GAZEBO gazebo)
pkg_check_modules(SDF sdformat)
include_directories(
  ${PROJECT_SOURCE_DIR}
  ${GAZEBO_INCLUDE_DIRS}
  ${SDF_INCLUDE_DIRS}
  ${ARMADILLO_INCLUDE_DIRS}
  ${CMAKE_BINARY_DIR}/msgs)

link_directories(
  ${GAZEBO_LIBRARY_DIRS}
  ${ARMDILLO_LIBRARY_DIRS}
  ${CMAKE_CURRENT_BINARY_DIR}/msgs)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${GAZEBO_CXX_FLAGS} ${ARMADILLO_CXX_FLAGS} -pthread")

add_library(arm_gz SHARED arm_gz.cc)
target_link_libraries(arm_gz
  arm_msgs
  ${GAZEBO_LIBRARIES}
  ${SDF_LIBRARIES}
  ${Boost_LIBRARIES}
  ${ARMADILLO_LIBRARIES})
add_dependencies(arm_gz arm_msgs)

add_library(arm_plugin SHARED arm_plugin.cc)
target_link_libraries(arm_plugin
  arm_msgs
  ${GAZEBO_LIBRARIES}
  ${SDF_LIBRARIES}
  ${ARMADILLO_LIBRARIES})
add_dependencies(arm_plugin arm_msgs)

add_library(ball_gz SHARED ball_gz.cc)
target_link_libraries(ball_gz
  ${GAZEBO_LIBRARIES}
  ${SDF_LIBRARIES}
  ${Boost_LIBRARIES}
  ${ballADILLO_LIBRARIES})

add_library(ball_plugin SHARED ball_plugin.cc)
target_link_libraries(ball_plugin
  ${GAZEBO_LIBRARIES}
  ${SDF_LIBRARIES}
  ${ballADILLO_LIBRARIES})
