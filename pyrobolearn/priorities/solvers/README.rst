Solvers
=======

In this folder, we provide the various task solvers used to solve a task or stack of tasks.

These include:

- ``TaskSolver``: the abstract task solver from which all the other task solvers inherit from.
- ``QPTaskSolver``: the task solver that uses quadratic programming (QP) to solve the task / stack of tasks.
- ``NLPTaskSolver``: the task solver that uses nonlinear programming (NLP) to solve the task / stack of tasks.

References:

1. "OpenSoT: A whole-body control library for the compliant humanoid robot COMAN" (`code <https://opensot.wixsite.com/opensot>`_, `slides <https://docs.google.com/presentation/d/1kwJsAnVi_3ADtqFSTP8wq3JOGLcvDV_ypcEEjPHnCEA>`_, `tutorial video <https://www.youtube.com/watch?v=yFon-ZDdSyg>`_, `old code <https://github.com/songcheng/OpenSoT>`_, LGPLv2), Rocchi et al., 2015
2. "Robot Control for Dummies: Insights and Examples using OpenSoT", Hoffman et al., 2017
