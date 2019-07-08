Model Interfaces
================

In this folder, we provide the model interfaces that are used and shared by the various tasks and constraints.

A model interface is an abstraction layer that provides a common interface and remove the direct coupling between the
``Robot`` class and the classes (tasks, constraints, solvers) defined in ``pyrobolearn/priorities``. Additionally, it
also serves as a container to different quantities (joint positions, velocities, torques, etc), which avoids the need
to recompute them for each task / constraint that used them.

The model interface API has been heavily inspired (mostly translated from C++ to Python) by the interface provided in
[1]_ and has been improved.

References:

.. [1] https://github.com/ADVRHumanoids/ModelInterfaceRBDL
