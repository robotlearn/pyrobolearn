Priority Tasks
==============

THIS IS UNDER CONSTRUCTION

In this folder, you will find the code for priority "tasks". The "tasks" defined here are different from the tasks 
defined in the ``pyrobolearn/tasks`` folder which defines robot learning tasks. The tasks defined here can be more seen
as "constraints"; for instance, the constraint for the robot to maintain its balance (i.e. have its center of mass 
above the support polygon), the constraint for the robot to have a certain pose, the constraint for the robot's 
end-effectors to track a certain trajectory, the constraint for the robot to not have collisions between its links, 
the constraint for the robot to respect the equation of motions, etc. In the robotics community field, these are known 
as "tasks" and concepts around them such as the stack of tasks [2]_ have been defined. In *pyrobolearn*, we keep a
more abstract definition of a task, which is also probably more related to what people have in mind when talking 
about a robot performing a certain task.

Most of these "tasks" are represented as a constrained optimization problem, where the "task" consists to minimize 
a certain objective function while respecting certain equality and inequality constraints. This is the reason why 
they are not called "constraints" to avoid the confusion with the (inequality and equality) constraints defined in 
the optimization problem. Most of the time, they are formulated as quadratic programming (QP) optimization problem [1]_.
Priority tasks are divided between kinematic and dynamic tasks, where the former only takes into account position and 
velocity information, while the latter also include dynamic information (forces and torques applied on the various 
bodies). The variables that are thus optimized by the optimization problem depends on the type of problem (kinematic 
or dynamic) we are dealing with. In the case of a kinematic task, the variables are often the joint (or end-effector) 
positions and/or velocities, while in the dynamic case, the variables are the joint accelerations and the (reaction) 
forces applied on the robot. 

Priorities can be divided into two categories:

- **Soft** priorities: each objective function is weighted by an importance weight where higher weights mean that we
  give more importance to the corresponding objective function. For instance, we might have a humanoid robot with two
  arms where each arm has to follow a specific trajectory and where we give the same importance to both "tasks". Soft
  priorities use task augmentation.
- **Hard** priorities: the most important constrained optimization problem is first solved, and then the next most
  important one is solved with an additional (optimization) constraint that the solution has to be in the solution
  space of the previous one. For instance, it is more important for a humanoid robot to maintain its balance than to
  follow perfectly a trajectory with its end-effector. This way of putting "tasks" on top of each other is known as
  the stack of tasks in the robotics community [2]_. Hard priorities exploit the null-space of higher priority tasks.

Soft and hard priorities can be mixed together as done in the following C++ framework [3]_.

The code presented here (and the architecture) is mostly inspired by [3]_, [4]_ (the papers and slides). Compared to
this framework, we write it in Python using popular optimization libraries (that are usually written in C/C++ and 
provide Python wrappers), we decouple it from other frameworks/middlewares (such as superbuild, XBotControl,
ROS/YARP, etc.), integrate it with PRL, and make it free and open-source.

In what follows, to avoid confusion with the vocabulary used in the robotics community, we will keep the notions of 
"tasks", (optimization) "constraints", and "solvers". The solvers can be found in the ``pyrobolearn/optimizers`` folder.


References
----------

.. [1] Quadratic Programming `(Wikipedia) <https://en.wikipedia.org/wiki/Quadratic_programming>`_

.. [2] "A Versatile Generalized Inverted Kinematics Implementation for Collaborative Working Humanoid Robots: The Stack of
  Tasks" (`code <https://stack-of-tasks.github.io/>`_), Mansard et al., 2009

.. [3] "OpenSoT: A whole-body control library for the compliant humanoid robot COMAN" (
  `code <https://opensot.wixsite.com/opensot>`_,
  `slides <https://docs.google.com/presentation/d/1kwJsAnVi_3ADtqFSTP8wq3JOGLcvDV_ypcEEjPHnCEA>`_,
  `tutorial video <https://www.youtube.com/watch?v=yFon-ZDdSyg>`_, `old code <https://github.com/songcheng/OpenSoT>`_,
  LGPLv2), Rocchi et al., 2015

.. [4] "Robot Control for Dummies: Insights and Examples using OpenSoT", Hoffman et al., 2017


TODO
~~~~

- [ ] create independent ``pyopensot`` package once it is over.
