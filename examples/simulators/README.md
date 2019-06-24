## Simulator examples

In this folder, you will have simple examples on how to use different simulators by loading a simple world (with a 
floor and gravity enabled), and a simple robot.

The various files can be called by `python <simulator>.py`. Note that they are mostly identical, and only the line 

```python
sim = Simulator(args)  # Bullet, BulletROS, ROS, etc
```

need to be changed. The basic idea is that your code should work without depending on which simulator you use. Note 
that currently, the `Bullet` simulator is the only fully functional API, while the others are partially implemented 
or still need to be implemented.

Here are the few examples that you can find in this folder:
1. `bullet.py`: simple example where we use the `Bullet` simulator, load a basic world (with a floor and gravity 
enabled) and the RRBot robot in it.
2. `bullet_ros_publisher.py`: example where we use the `BulletROS(publish=True)` simulator which publishes the 
joint position values that were returned by the Bullet simulator on the corresponding ROS topic.
3. `bullet_ros_subscriber.py`: example where we use the `BulletROS(subscribe=True)` simulator which gets the joint 
values from the ROS topics and change them in the simulator. This works with the `bullet_ros_publisher.py` code 
presented above. By moving the robot with your mouse in the publisher version, you will see the robot in this 
subscriber version moves in accordance with. This can be useful if you have access to the real platform as well.

Later, a `ROS`/`ROS_RBDL` "simulator" (without passing by a real simulator like `Bullet`) will allow you to make 
your code works on a real platform using ROS without changing any other lines of code. This is one of the big 
TODOs but is not currently my priority.
