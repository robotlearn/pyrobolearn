## Simulators

This folder contains the APIs to the various simulators. Currently, the main simulator being supported is PyBullet. 
Work is under progress for other simulators.

```python
import pyrobolearn as prl

sim = prl.simulators.Bullet()
sim1 = prl.simulators.Mujoco()
sim2 = prl.simulators.Dart()
```

#### What to check next?

Check the `worlds` folder and the `robots` folder.

#### TODOs

- [x] implement Bullet interface
- [ ] implement Mujoco interface (ongoing)
- [ ] implement Raisim interface (ongoing)
- [ ] implement Dart interface (ongoing)
- [ ] implement ROS middleware (ongoing - currently worked with real franka emika panda robot)
- [ ] implement RBDL_ROS interface
- [ ] implement GazeboROS interface
- [ ] implement V-REP (PyRep) interface
- [ ] implement Isaac interface
- [ ] implement Chrono interface
- [ ] implement OpenSim interface (useful for biomedical models)
- [ ] implement `simulator_randomizer` (similar to `physics_randomizer`)
