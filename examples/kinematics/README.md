## Robot kinematics

We provide examples on how to perform forward and inverse kinematics.

Here are the forward kinematics (FK) examples that the user can try:
1. `fk.py`: simple forward kinematics example where we directly sent desired joint positions to the Kuka 
manipulator.

Here are the inverse kinematics (IK) examples that the user can try:
1. `ik.py`: simple inverse kinematics example where the Kuka manipulator has to reach a certain target position 
in the world. In this example, the user can also choose the damped-least-squares IK solver.
2. `ik_libraries.py`: comparison between different IK libraries including `pybullet`, `PyKDL`, `trac_ik`, and 
`rbdl` using the Kuka manipulator.
3. `moving_sphere.py`: damped-least-squares IK with the Kuka manipulator where the goal is to follow a sphere 
that moves in a circular manner.


References:
- [1] "Robotics: Modelling, Planning and Control", Siciliano et al., 2010
- [2] "Springer Handbook of Robotics", Siciliano et al., 2008
