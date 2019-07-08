## Simulators

This folder contains the APIs to the various simulators. Currently, the main simulator being supported is PyBullet. 
Work is under progress for other simulators.

```python
import pyrobolearn as prl

sim = prl.simulators.Bullet()
sim1 = prl.simulators.BulletROS()
sim2 = prl.simulators.Dart()
```

#### What to check next?

Check the `worlds` folder and the `robots` folder.

#### TODOs

- [x] implement Bullet interface
- [ ] implement BulletROS interface (ongoing)
- [ ] implement Mujoco interface
- [ ] implement Isaac interface
- [ ] implement Dart interface
- [ ] implement RBDL_ROS interface
- [ ] implement GazeboROS interface
- [ ] implement V-REP (PyRep) interface
- [ ] implement `simulator_randomizer` (similar to `physics_randomizer`)
