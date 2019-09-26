## Worlds

In this folder, we provide the `World` class, and the `BasicWorld` class (which loads the floor and sets the gravity).
With this world, you can load floors, visual and collision objects, robots, and others. The world is with the robot (+ actuators/sensors) and the mouse keyboard interface, the only parts in the framework that can interact with the simulator directly. You can load a robot through the world.

```python
import pyrobolearn as prl

sim = prl.simulators.BulletSim()
world = prl.worlds.BasicWorld(sim)

coman = world.load_robot('Coman', position=<position1>)
wam = world.load_robot(prl.robots.WAM, position=<position2>)

# the following is not advised
littledog = prl.robots.LittleDog(sim, init_pos=<position3>)
world.load_robot(littledog)  # such that the world knows about the robot is present.
```

#### What to check/look next?

Check the `robots` folder.
