## Robots

This folder contains the various robots that can be used in the PRL framework. Currently, they can be loaded in the [PyBullet](https://pybullet.org/wordpress/) simulator. The corresponding URDFs have been downloaded from various repositories and updated/corrected if necessary (adding inertial tags, correcting inertia values, updating collision meshes, etc). All the robots inherit from the main `Robot` class, and can be loaded in the framework using:

```python
import pyrobolearn as prl

sim = prl.simulators.Bullet()
robot = prl.robots.<RobotClass>(sim)
```

or

```python
import pyrobolearn as prl

sim = prl.simulators.Bullet()
world = prl.worlds.BasicWorld(sim)
robot = world.loadRobot(<Robot_name_or_robot_class>)
```

The folder contains different kind of robots including manipulators, legged robots, wheeled robots, and others. It includes:
- [Aibo](https://github.com/dkotfis/aibo_ros)
- [Allegrohand](https://github.com/simlabrobotics/allegro_hand_ros)
- [Ant](https://github.com/bulletphysics/bullet3/tree/master/examples/pybullet/gym/pybullet_data/mjcf)
- Atlas: [1](https://github.com/openai/roboschool), [2](https://github.com/erwincoumans/pybullet_robots)
- [Ballbot](https://github.com/CesMak/bb)
- [Baxter](https://github.com/RethinkRobotics/baxter_common)
- BB8: [1](http://www.theconstructsim.com/bb-8-gazebo-model/), [2](https://github.com/eborghi10/BB-8-ROS)
- [Blackbird](https://hackaday.io/project/160882-blackbird-bipedal-robot)
- [Cartpole](https://github.com/bulletphysics/bullet3/blob/master/data/cartpole.urdf) but modified to be able to have multiple links specified at runtime
- Cassie: [1](https://github.com/UMich-BipedLab/Cassie_Model), [2](https://github.com/agilityrobotics/cassie-gazebo-sim), [3](https://github.com/erwincoumans/pybullet_robots)
- [Centauro](https://github.com/ADVRHumanoids/centauro-simulator)
- [Cogimon](https://github.com/ADVRHumanoids/iit-cogimon-ros-pkg)
- [Coman](https://github.com/ADVRHumanoids/iit-coman-ros-pkg)
- [Crab](https://github.com/tuuzdu/crab_project)
- [Cubli](https://github.com/xinsongyan/cubli)
- [Darwin](https://github.com/HumaRobotics/darwin_description)
- [e.Do](https://github.com/Comau/eDO_description)
- [E-puck](https://github.com/gctronic/epuck_driver_cpp)
- [F10 racecar](https://github.com/erwincoumans/pybullet_robots/tree/master/data/f10_racecar)
- [Fetch](https://github.com/fetchrobotics/fetch_ros)
- [Franka Emika](https://github.com/frankaemika/franka_ros)
- [Half Cheetah](https://github.com/bulletphysics/bullet3/tree/master/examples/pybullet/gym/pybullet_data/mjcf)
- [Hopper](https://github.com/bulletphysics/bullet3/tree/master/examples/pybullet/gym/pybullet_data/mjcf)
- [Hubo](https://github.com/robEllenberg/hubo-urdf)
- [Humanoid](https://github.com/bulletphysics/bullet3/tree/master/examples/pybullet/gym/pybullet_data/mjcf)
- [Husky](https://github.com/husky/husky)
- [HyQ](https://github.com/iit-DLSLab/hyq-description)
- [HyQ2Max](https://github.com/iit-DLSLab/hyq2max-description)
- ICub: [1](https://github.com/robotology-playground/icub-models), [2](https://github.com/robotology-playground/icub-model-generator). There are currently few problems with this robot.
- [Jaco](https://github.com/JenniferBuehler/jaco-arm-pkgs)
- KR5: [1](https://github.com/a-price/KR5sixxR650WP_description), [2](https://github.com/ros-industrial/kuka_experimental)
- Kuka IIWA: [1](https://github.com/IFL-CAMP/iiwa_stack), [2](https://github.com/bulletphysics/bullet3/tree/master/data/kuka_iiwa)
- Kuka LWR: [1](https://github.com/CentroEPiaggio/kuka-lwr), [2](https://github.com/bulletphysics/bullet3/tree/master/data/kuka_lwr)
- [Laikago](https://github.com/erwincoumans/pybullet_robots)
- [Little Dog](https://github.com/RobotLocomotion/LittleDog)
- [Manipulator2D](https://github.com/domingoesteban/robolearn_robots_ros)
- [Minitaur](https://github.com/bulletphysics/bullet3/tree/master/examples/pybullet/gym/pybullet_data/quadruped)
- [Lincoln MKZ car](https://bitbucket.org/DataspeedInc/dbw_mkz_ros)
- [Morphex](https://gist.github.com/lanius/cb8b5e0ede9ff3b2b2c1bc68b95066fb)
- Nao: [1](https://github.com/ros-naoqi/nao_robot), and [2](https://github.com/ros-naoqi/nao_meshes)
- OpenDog: [1](https://github.com/XRobots/openDog), and [2](https://github.com/wiccopruebas/opendog_project)
- [Pepper](https://github.com/ros-naoqi/pepper_robot)
- [Phantom X](https://github.com/HumaRobotics/phantomx_description)
- [Pleurobot](https://github.com/KM-RoBoTa/pleurobot_ros_pkg)
- [PR2](https://github.com/pr2/pr2_common)
- [Quadcopter](https://github.com/wilselby/ROS_quadrotor_simulator)
- [Rhex](https://github.com/grafoteka/rhex)
- [RRbot](https://github.com/ros-simulation/gazebo_ros_demos)
- Sawyer: [1](https://github.com/RethinkRobotics/sawyer_robot), [2](https://github.com/erwincoumans/pybullet_robots)
- [SEA hexapod](https://github.com/alexansari101/snake_ws)
- [SEA snake]( https://github.com/alexansari101/snake_ws)
- [Shadow hand](https://github.com/shadow-robot/sr_common)
- [Soft hand](https://github.com/CentroEPiaggio/pisa-iit-soft-hand)
- [Swimmer](https://github.com/bulletphysics/bullet3/tree/master/examples/pybullet/gym/pybullet_data/mjcf)
- [Valkyrie](https://github.com/openhumanoids/val_description)
- [Walker 2D](https://github.com/bulletphysics/bullet3/tree/master/examples/pybullet/gym/pybullet_data/mjcf)
- [Walk-man](https://github.com/ADVRHumanoids/iit-walkman-ros-pkg)
- [Wam](https://github.com/jhu-lcsr/barrett_model)
- [Youbot](https://github.com/youbot): this includes the youbot base without any arms, one kuka arm, 2 kuka arms, and the kuka arm without the wheeled base.


Here is a list of robots that I plan to add at one point (some of them require to simulate some fluid dynamics, as done in the `quadcopter` class) but can interest already some people:
- [ ] [ANYmal](https://www.anymal-research.org/): I am currently not sure if I can release the URDF of this robot, and thus I will not do it until it is officially released by ETH.
- [ ] [ROS robots](https://robots.ros.org/): I plan to provide soon the robots listed on this website (I am currently cleaning the URDFs of some of them)
- [ ] [Universal robots](https://github.com/ros-industrial/universal_robot)
- [ ] [rotors-simulator](https://github.com/ethz-asl/rotors_simulator)
- [ ] [uuv-simulator](https://github.com/uuvsimulator/uuv_simulator)
- [ ] [usv-simulator](https://github.com/OUXT-Polaris/ros_ship_packages)

Note that currently, we load directly the URDF from the xml/urdf file using the simulator, but later we might first parse it beforehand and allow the user to add, remove, or change few links at runtime.

TODO:
- [ ] correct still few URDFs (some of them have inertia values that are too high!)
- [ ] finish to implement few methods
- [ ] move all the urdfs outside the pyrobolearn framework: put it in another repo or in a dropbox/google drive.
