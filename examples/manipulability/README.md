## Manipulability Ellipsoids

We provide examples on how to use manipulability ellipsoids.

Here are a short description of the various examples the user can try:
1. `2d_manipulability.py`: draw the 2D velocity and force manipulability ellipsoids on the end-effector of a 
3-link planar manipulator.
2. `com_manipulability_tracking.py`: track the velocity manipulability ellipsoid of the center of mass of a robot 
with a fixed base.
3. `com_manipulability_tracking_with_balance.py`: track the velocity manipulability ellipsoid of the center of mass 
of a floating-base robot while keeping its balance.
4. `com_dynamic_manipulability_tracking_with_balance.py`: track the dynamic manipulability ellipsoid of the center 
of mass of a floating-base robot while keeping its balance.
5. `right_arm_manipulability_tracking_with_balance.py`: track a desired manipulability for its right arm, while reaching a desired end-effector position and keeping balance

References:
- [1] "Robotics: Modelling, Planning and Control", Siciliano et al., 2010
- [2] "Springer Handbook of Robotics", Siciliano et al., 2008
- [3] "Geometry-aware Tracking of Manipulability Ellipsoids", Jaquier et al., R:SS, 2018
