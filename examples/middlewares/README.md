## Middleware examples

In this folder, you will have simple examples on how to use the ROS middleware with PyRoboLearn. Once this middleware 
is instantiated, it is passed to the simulator that uses it to publish or subscribe to topics/services to get the 
various joint states, sensor values, etc. The ROS middleware can then later be used to launch ROS nodes and other 
ROS features. The ROS middleware can also be useful to get the data from a real robotic platform or to send some 
joint trajectories to it.

Here are the few examples that you can find in this folder:
1. `bullet_ros_control_gazebo.py`: After running the corresponding roslaunch file (see file documentation), you will 
be able to teleoperate the manipulator (rrbot, kuka, or franka emika panda) in Gazebo by moving the same robot in 
PyBullet. The robots that are instantiated in Gazebo use position control (by using `ros_control`). A simple video 
demonstrating the results can be found here: https://www.youtube.com/watch?v=OPh-NCfKKK8
2. `bullet_ros_rqt.py`: this will launch RQT along PRL.
3. `bullet_ros_publisher.py`: example where we use ROS to publish the joint position values that were returned by 
the Bullet simulator on the corresponding ROS topic.
4. `bullet_ros_subscriber.py`: example where we use ROS to get the joint values from the ROS topics and change them 
in the simulator. This works with the `bullet_ros_publisher.py` code presented above. By moving the robot with your 
mouse in the publisher version, you will see the robot in this subscriber version moving accordingly. This can be 
useful if you have access to the real platform as well.
