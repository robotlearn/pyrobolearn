Plotting Tools
==============

Plotting tools are extremely important in research. For this purpose, we provide plotting tools that allows to
plot in real-time different quantities that can be observed in the simulator, like the joint states, the position and
orientation of each body in the world by plotting their reference frame, the position and orientation of links of a
specific body, the resulting trajectories in the 3D Cartesian space, etc.

Warnings: Currently, you have to close the figure before closing the simulator. If you close the simulator first,
you might still have the figure process running.

- ``JointRealTimePlot``: plot in real-time the joint positions (in blue), velocities (in green), accelerations (in red),
  and/or torques (in purple).
- ``LinkFrameRealTimePlot``: plot in real-time the frames of the specified links.

To test these classes, you can run the corresponding python file and move the manipulator with the mouse and check the
real-time plots.

References:

- `matplotlib <https://matplotlib.org/>`_: The standard Python plotting library.
- `Seaborn <https://seaborn.pydata.org/>`_: Seaborn is a Python data visualization library built on top of matplotlib.
