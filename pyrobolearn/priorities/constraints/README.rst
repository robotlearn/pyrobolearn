Constraints
===========

In this folder, we define the most common inequality and equality optimization constraints used in robotics for 
priority tasks. Several of them were provided in [1].

Constraints include joint limits, joint velocity limits, collision avoidance, and others. They are separated into 
4 folders (velocity, acceleration, torque, and cartesian force); one for each optimization variable vector that is 
being optimized. Note that different type of tasks (and thus constraints) can be combined together; for instance, 
we can combine acceleration tasks with force tasks. This will create an optimization variable vector 
:math:`x = [\ddot{q}^\top, F^\top]^\top` which can then be used with the joint space dynamic equation 
:math:`\tau = H \ddot{q} + C(q,\dot{q})\dot{q} + g(q) - J^\top F` to get the equivalent joint torques to be applied 
on the robot. 

References:

1. "OpenSoT: A whole-body control library for the compliant humanoid robot COMAN" (`code <https://opensot.wixsite.com/opensot>`_, `slides <https://docs.google.com/presentation/d/1kwJsAnVi_3ADtqFSTP8wq3JOGLcvDV_ypcEEjPHnCEA>`_, `tutorial video <https://www.youtube.com/watch?v=yFon-ZDdSyg>`_, `old code <https://github.com/songcheng/OpenSoT>`_, LGPLv2), Rocchi et al., 2015
2. "Robot Control for Dummies: Insights and Examples using OpenSoT", Hoffman et al., 2017

