Robot examples
==============

More than 60 robots (of various types) are available through ``pyrobolearn``.

Here are the few examples that you can find in this folder:

1. ``load_robot.py <robot_name>``: load the given robot in the world.
2. ``visualize_robot.py <robot_name>``: test different visualization tools that can be used on the robot to show its
   joint axis, bounding boxes, and others.
3. ``robot_with_sliders.py <robot_name>``: load the given robot in the world and allow you to manipulate the robot's
   joints with sliders.
4. ``distribute_epucks.py``: distribute several e-pucks in the world and make them move forward.
5. ``quadcopter_controller.py``: move a quadcopter in the air using an Xbox or Playstation game controller.
6. ``robots/<robot>.py``: load the given robot in the simulator by directly instantiating it. Some of these files do
   more than just loading the robot.
7. ``attach_gripper_to_manipulator``: attach the specified gripper / hand to the Kuka manipulator robot. In this file,
   you can check the various grippers you can use.
8. ``attach_manipulator_to_quadruped``: attach the Kuka Youbot manipulator to the HyQ2Max quadruped robot.

Notes: to turn the camera in the simulator, keep pressing the ``ctrl`` key and the left button on the mouse, and
move the mouse.


What to check next?
~~~~~~~~~~~~~~~~~~~

Check the ``pyrobolearn/examples/interfaces`` or ``pyrobolearn/examples/kinematics`` folder.
