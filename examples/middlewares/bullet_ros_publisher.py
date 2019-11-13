#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Example on how to use the Bullet-ROS simulator (the publisher version) in PRL.

The publisher version publish the various data on the corresponding topics every time the joints are set. If the
:attr:`teleoperate` is set to True, it will also publish every time we call a `get_*()` method.

Before running this file, run in the terminal:
```bash
$ roscore               # note that roscore is launched automatically by this file if has not already been launched
$ rostopic list         # to show the list of published topics
```

The last command should print the following:
```bash
/rosout
/rosout_agg
```

Now, run this file and check the published topics again:
```bash
$ rostopic list
```

This time, you should get as well:
```bash
/rosout
/rosout_agg
/rrbot/joint_states
```

You can print the output of the topic with:
```bash
$ rostopic echo /rrbot/joint_states
```

Try to move the robot with the mouse, and see that the published joint position values changed as well.

You can run this code in parallel with `bullet_ros_subscriber.py`, which implements the subscriber version, that is,
it listens to the topics and set them in the bullet simulator. So by moving the robot in this simulator, you should
see that it also moves in the other simulator.

Note: this code also works with other robots.
"""

from itertools import count
import pyrobolearn as prl


# create middleware and simulator (roscore will automatically be launched if it has not already been launched)
ros = prl.middlewares.ROS(publish=True, teleoperate=True)
sim = prl.simulators.Bullet(middleware=ros)

# load world
world = prl.worlds.BasicWorld(sim)

# load robot
robot = world.load_robot('rrbot')

# run simulation
for t in count():
    # get the joint positions from the Bullet simulator (because :attr:`teleoperate` has been set to True,
    # it will publish these read positions on the corresponding topic)
    q = robot.get_joint_positions()

    # perform a step in the simulator (and sleep for `sim.dt`)
    world.step(sim.dt)
































