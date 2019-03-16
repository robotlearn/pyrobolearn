# This file creates a basic world, load each robot that can be found in the PRL framework

from pyrobolearn.simulators import BulletSim
from pyrobolearn.worlds import BasicWorld
from pyrobolearn.robots import implemented_robots

robot_not_working = set(['icub'])

print("All the robots (total number of robots = {}): {}".format(len(implemented_robots), implemented_robots))

# create simulator
sim = BulletSim()

# create basic world with floor and gravity
world = BasicWorld(sim)

# create one robot at a time
for i, robot_name in enumerate(implemented_robots):
    if robot_name not in robot_not_working:
        # instantiate the given robot
        robot = world.loadRobot(robot_name)

        # print info about the robot
        print("Robot n{}: {}".format(i+1, robot))
        # robot.printRobotInfo()

        # run for few moments in the world
        for t in range(250):
            # run one step and sleep a bit
            world.step(sleep_dt=1./240)

        # remove the robot from the world
        world.removeObject(robot)
