## Imitation learning task

In this folder, you can run the `trajectory_reproduction_kuka_dmp.py` file. This will create a basic world and load the kuka robot in the world. You can then record trajectories using the mouse and keyboard. Pressing `ctrl+r` will start/stop the recording of the joint trajectories, and `shift+r` will stop the recording phase and start the training phase.
In this example, we train a dynamic movement primitive (DMP) for each joint on the recorded data. Once trained, it will plot the joint values predicted by the trained DMP, and it will try to reproduce the demonstrated trajectories.

The user can try to select the joints it wants to move/record, or increase/decrease the number of basis functions that are used for each DMP.
