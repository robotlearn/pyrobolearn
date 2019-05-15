## Simulators

This folder contains the APIs to the various simulators. Currently, the main simulator being supported is PyBullet. 
Work is under progress for other simulators.

```python
import pyrobolearn as prl

sim = prl.simulators.Bullet()
sim1 = prl.simulators.Dart()
```

#### What to check next?

Check the `worlds` folder and the `robots` folder.

#### TODOs

- [x] implement Bullet interface
- [ ] implement Dart interface
- [ ] implement ROS_RBDL interface
- [ ] implement Gazebo_ROS interface
- [ ] implement Mujoco interface
- [ ] implement `simulator_randomizer` (similar to `physics_randomizer`)
