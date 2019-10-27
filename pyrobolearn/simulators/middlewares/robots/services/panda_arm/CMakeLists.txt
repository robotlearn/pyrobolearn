cmake_minimum_required(VERSION 2.8.3)
project(move_to_pose_iit)

find_package(catkin REQUIRED COMPONENTS rospy roscpp std_msgs genmsg message_generation)

add_service_files(
  DIRECTORY srv
  FILES MoveJoints.srv
)

generate_messages(
  DEPENDENCIES std_msgs
)

catkin_package()

include_directories(include ${catkin_INCLUDE_DIRS})
