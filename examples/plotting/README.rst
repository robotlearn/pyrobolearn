Plotting examples
=================

In this folder, you will find examples where we use the plotting tools provided in PRL.

Warnings: Currently, you have to close the figure before closing the simulator. If you close the simulator first,
you might still have the process responsible to draw the figure running.

- ``joints.py``: plot in real-time the joint positions (in blue), velocities (in green), accelerations (in red),
  and/or torques (in purple).
- ``link_frames.py``: plot in real-time the frames of the specified links.

To test these classes, you can run the corresponding python file, and move the manipulator with the mouse and check the
real-time plots.
