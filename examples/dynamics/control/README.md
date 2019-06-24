### Force control

In a nutshell, you have different control modes:

* motion control: specify the desired task (or joint) positions / velocities
* force control: specify the desired task (or joint) forces
	* indirect force control:
		* impedance control
		* admittance control
	* direct force control:
		* hybrid force/position control
		* parallel force/position control

Here are the few examples that you can find in this folder:
1. `no_forces.py`: this example loads a RRBot robot and disable the motors. It does not apply any joint torques.
2. `gravity_compensation.py`: compute the necessary joint torques to compensate for gravity.
3. `attractor_point.py`: compute the necessary joint torques (using impedance control) such that the end-effector 
is attracted by a 3D Cartesian point.

For these 3 above examples, try to move the robot's end effector with your mouse and see what happens.
