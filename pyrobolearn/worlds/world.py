#!/usr/bin/env python
"""Define the `World` class which allows to specify what constitutes the world (i.e. what elements are in the world).

Dependencies:
- `pyrobolearn.simulators`
"""

import os
import copy
import inspect
import collections
import time
import numpy as np
import cv2


from pyrobolearn.simulators import Simulator
from pyrobolearn.worlds.world_camera import WorldCamera
# from pyrobolearn.utils import has_method, has_variable
from pyrobolearn.robots import Body, Robot, robot_names_to_classes
from pyrobolearn.utils.transformation import get_quaternion_from_rpy
# TODO: to install the `gdal` library, run the script `pyrobolearn/scripts/install_gdal.sh`, by default do not
#  import it
from pyrobolearn.worlds.utils.heightmaps.diamond_square import diamond_square_heightmap, diamond_square_heightmap_2
from pyrobolearn.worlds.utils.heightmaps.rbf import rbf_heightmap
from pyrobolearn.worlds.utils.heightmaps.equation import equation_heightmap
from pyrobolearn.worlds.utils.obj_generator import create_obj_from_heightmap


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class World(object):
    r"""World class.

    The world contains all the objects that constitutes the world. This includes immovable bodies such as the
    terrain/floor, walls, and so on, as well as movable bodies such as the various robots, etc.
    Properties of the world are also defined here, such as gravity, friction, and other dynamical properties.

    The world is responsible to load the different objects part of the world and keeping a map of bodies;
    based on where the agent(s) is(are), the bodies will be removed or added from/to the simulator allowing it
    to run faster. (TODO)

    It is independent of the simulator and environment used in RL, and is often provided as inputs to some
    `rewards/costs` and to the `environment`.
    A world can only be defined in a simulator. Because of this, robots (which normally should be part of the
    world) can be independent from this last one. This allows to use robots in reality where the world is already
    defined (i.e. the real world).

    For an excellent overview of available 3D models/scenes, check references [1, 2].

    References:
        - [1] "3D Machine Learning": https://github.com/timzhang642/3D-Machine-Learning
        - [2] Open3D: http://www.open3d.org/
    """

    def __init__(self, simulator, gravity=(0., 0., -9.81)):
        """
        Initialize the world.

        Args:
            simulator (Simulator): simulator instance.
            gravity (tuple/list of 3 float, np.array[3]): gravity vector.
        """
        # set simulator
        self.simulator = simulator
        self.gravity = gravity

        # set world camera
        self.camera = WorldCamera(self.simulator)

        # keep track of the all the unique ids present in the world
        # the following dictionary contains {id1: [(method_name, kwargs), [parent_ids], [child_ids]], id2: Body}
        # ids like id1 include visual shapes, collision shapes, textures, bodies that were created here
        self.ids = collections.OrderedDict()
        self.bodies = {}  # this contains {id: Body}

        self.map = None
        self.floor_id = -1

        # create world state
        self.world_state = None

        # configure debug visualizer
        self.sim.configure_debug_visualizer(self.sim.COV_ENABLE_GUI, 0)

        # interfaces and bridges
        # self.interfaces = set([])
        # self.bridges = []

        # keep in memory all the constraints that are created using `attach` and `detach` methods
        self.constraints = {}

    ##############
    # Properties #
    ##############

    @property
    def simulator(self):
        """Return the simulator instance."""
        return self.sim

    @simulator.setter
    def simulator(self, simulator):
        """Set the simulator instance."""
        if not isinstance(simulator, Simulator):
            raise TypeError("Expecting the given simulator to be an instance of `Simulator`, instead got: "
                            "{}".format(type(simulator)))
        self.sim = simulator

    @property
    def gravity(self):
        """Return the gravity vector."""
        return self.simulator.gravity

    @gravity.setter
    def gravity(self, gravity):
        """Set the gravity vector in the world.

        Args:
            gravity (np.array[3]): 3d gravity vector.
        """
        self.simulator.gravity = gravity

    @property
    def lateral_friction(self):
        """Return the floor lateral friction coefficient."""
        if self.floor_id > 0:
            return self.sim.get_dynamics_info(self.floor_id, link_id=-1)[1]

    @lateral_friction.setter
    def lateral_friction(self, coefficient):
        """Set the floor lateral friction coefficient."""
        if self.floor_id > 0:
            self.sim.change_dynamics(body_id=self.floor_id, link_id=-1, lateral_friction=coefficient)

    @property
    def rolling_friction(self):
        """Return the floor rolling friction coefficient."""
        if self.floor_id > 0:
            return self.sim.get_dynamics_info(self.floor_id, -1)[6]

    @rolling_friction.setter
    def rolling_friction(self, coefficient):
        """Set the floor rolling friction coefficient."""
        if self.floor_id > 0:
            self.sim.change_dynamics(body_id=self.floor_id, link_id=-1, rolling_friction=coefficient)

    @property
    def spinning_friction(self):
        """Return the floor spinning friction coefficient."""
        if self.floor_id > 0:
            return self.sim.get_dynamics_info(self.floor_id, -1)[7]

    @spinning_friction.setter
    def spinning_friction(self, coefficient):
        """Set the spinning friction coefficient."""
        if self.floor_id > 0:
            self.sim.change_dynamics(body_id=self.floor_id, link_id=-1, spinning_friction=coefficient)

    @property
    def restitution(self):
        """Return the floor restitution (bounciness) coefficient."""
        if self.floor_id > 0:
            return self.sim.get_dynamics_info(self.floor_id, -1)[5]

    @restitution.setter
    def restitution(self, coefficient):
        """Set the floor restitution (bounciness) coefficient."""
        if self.floor_id > 0:
            self.sim.change_dynamics(body_id=self.floor_id, link_id=-1, restitution=coefficient)

    @property
    def contact_damping(self):
        """Return the floor contact damping."""
        if self.floor_id > 0:
            return self.sim.get_dynamics_info(self.floor_id, -1)[8]

    @contact_damping.setter
    def contact_damping(self, value):
        """Set the floor contact damping value."""
        if self.floor_id > 0:
            self.sim.change_dynamics(body_id=self.floor_id, link_id=-1, contact_damping=value)

    @property
    def contact_stiffness(self):
        """Return the floor contact stiffness."""
        if self.floor_id > 0:
            return self.sim.get_dynamics_info(self.floor_id, -1)[9]

    @contact_stiffness.setter
    def contact_stiffness(self, value):
        """Set the floor contact stiffness value."""
        if self.floor_id > 0:
            self.sim.change_dynamics(body_id=self.floor_id, link_id=-1, contact_stiffness=value)

    @property
    def floor_dynamics(self):
        """Return the floor dynamical parameters (friction, restitution, etc).

        Returns:
            float: lateral friction coefficient
            float: rolling friction coefficient
            float: spinning friction coefficient
            float: restitution coefficient
            float: contact damping value
            float: contact stiffness value
        """
        if self.floor_id > 0:
            info = self.sim.get_dynamics_info(self.floor_id, -1)
            return info[1], info[6], info[7], info[5], info[8], info[9]

    @floor_dynamics.setter
    def floor_dynamics(self, dynamics):
        """
        Set the floor dynamics.

        Args:
            dynamics (dict): dictionary of coefficients.
        """
        if self.floor_id > 0:
            self.sim.change_dynamics(body_id=self.floor_id, link_id=-1, **dynamics)

    #############
    # Operators #
    #############

    def __str__(self):
        """Return a string describing the world."""
        return self.__class__.__name__

    def __contains__(self, item):
        """
        Check if the given item is in the world.

        Args:
            item (int, Body): if it is an integer, it will check if the given body id is in the world.

        Returns:
            bool: True if the world contains the given item
        """
        if isinstance(item, Body):
            item = item.id
        return item in self.bodies

    def __copy__(self):  # TODO: add the bodies in the copy
        """Return a shallow copy of the world. This can be overridden in the child class."""
        return self.__class__(simulator=self.simulator, gravity=self.gravity)

    def __deepcopy__(self, memo={}):  # TODO: add the bodies in the copy
        """Return a deep copy of the world. This can be overridden in the child class.

        Args:
            memo (dict): memo dictionary of objects already copied during the current copying pass
        """
        if self in memo:
            return memo[self]

        # copy world
        simulator = copy.deepcopy(self.simulator, memo)
        gravity = copy.deepcopy(self.gravity)
        world = self.__class__(simulator=simulator, gravity=gravity)

        # load bodies in world
        for id_, items in self.ids.iteritems():
            if not isinstance(items, list):
                items = [items]
            for item in items:
                if isinstance(item, tuple):  # (method_name, arguments)
                    method = getattr(world, item[0])
                    method(**item[1])
                else:
                    copy.deepcopy(item, memo)

        memo[self] = world
        return world

    ###########
    # Methods #
    ###########

    @staticmethod
    def __get_method_and_parameters(frame):
        """
        Return the method name and the parameters with their values.

        Args:
            frame (types.FrameType): frame of the method

        Returns:
            str: method name.
            dict: parameters with their values.
        """
        args, _, _, values = inspect.getargvalues(frame)
        method_name = frame.f_code.co_name
        parameters = {arg: values[arg] for arg in args[1:]}
        return method_name, parameters

    # def set_bridges(self, bridges):  # TODO: remove this
    #     """
    #     This append the given bridges to various interfaces to the list of bridges.
    #
    #     See `pyrobolearn.tools.interface` and `pyrobolearn.tools.bridge` for more information.
    #
    #     Args:
    #         bridges (list, Bridge): list of bridges
    #     """
    #     if isinstance(bridges, collections.Iterable):
    #         for bridge in bridges:
    #             # if not isinstance(bridge, Bridge):
    #             #    raise TypeError("Expecting a list of bridges (must be an instance of Bridge)")
    #             if not has_method(bridge, 'step') and not has_variable(bridge, 'interface'):
    #                 raise TypeError("Expecting bridge to have a `step` method and an `interface` variable")
    #             if not has_method(bridge.interface, 'step'):
    #                 raise TypeError("Expecting the bridge.interface to have a `step` method")
    #             self.bridges.append(bridge)
    #             self.interfaces.add(bridge.interface)
    #     # elif isinstance(bridges, Bridge):
    #     elif has_method(bridges, 'step') and has_variable(bridges, 'interface') and \
    #             has_method(bridges.interface, 'step'):
    #         self.bridges.append(bridges)
    #         self.interfaces.add(bridges.interface)
    #     else:
    #         raise TypeError("Expecting a bridge (instance of Bridge) or a list of instances of Bridge")

    def save(self, filename=None):
        """
        Save the world in the given filename, or in the RAM.

        Args:
            filename (str, None): path to file to save the state of the world. If None, it will save it in the main
                memory (RAM).

        Returns:
            str or int: filename, or unique state id.
        """
        self.world_state = self.sim.save(filename)
        return self.world_state

    def reset(self, world_state=None):
        """
        Reset the world. Put back each object where they were when added to the world.

        Args:
            world_state (int, str, None): world state id (int), or path to file (str). If None, it will restore the
                last saved world state.
        """
        if world_state is None:
            # save to current instance if not already saved
            if self.world_state is None:
                self.world_state = self.save()

            # reset world to a previous instance
            if isinstance(self.world_state, (int, str)):
                # load world from the disk / memory
                self.sim.load(self.world_state)
                # reset the robot
                # self.reset_robots()
            else:
                # reset simulation: remove all objects from the world and reset the world to initial conditions
                self.sim.reset()
        else:
            if isinstance(world_state, (int, float)):
                # load world state from memory / disk
                self.sim.load(world_state)
            else:
                raise TypeError("Expecting the world state to be an int (id) or a string (path to file), instead got "
                                "{}".format(type(world_state)))

    def reset_simulator(self):
        """
        Reset the simulator; remove the world from the simulator.
        """
        self.sim.reset()

    def step(self, sleep_dt=None):
        """
        Perform one step for the interfaces and bridges, and one step in the world/simulator.
        """
        # for interface in self.interfaces:
        #     interface.step()
        # for bridge in self.bridges:
        #     bridge.step()

        # call the step method for each body
        for body in self.bodies.values():
            body.step()

        # call simulation step
        self.sim.step()

        # sleep
        if sleep_dt is not None:
            time.sleep(sleep_dt)

    def follow(self, body, distance=None, yaw=None, pitch=None):
        """
        Follow the given body with the world camera at the specified distance, yaw and pitch angles.

        Args:
            body (Body, int, long): body (or body id) to follow with the world camera.
            distance (float, None): distance (in meter) from the camera and the body position. If None, it will take
                the current distance.
            yaw (float, None): camera yaw angle (in radians) left/right. If None, it will take the current yaw angle.
            pitch (float, None): camera pitch angle (in radians) up/down. If None, it will take the current pitch angle.
        """
        if isinstance(body, Body):
            body = body.id
        self.camera.follow(body_id=body, distance=distance, yaw=yaw, pitch=pitch)

    def load_robot(self, robot, position=None, orientation=None, fixed_base=None, *args, **kwargs):
        """
        Load the robot into the world. If the robot parameter is a known robot name or the path to the urdf file,
        it will create a `Robot` instance and return it. If the robot is already an instance of `Robot` it will
        just add it to the list of bodies present in the world.

        Args:
            robot (Robot, str, class): the robot instance or name. For the list of possible robot names, import
                `implemented_robots` from `pyrobolearn.robots` module.
            position (float[3], None): position of the robot. If None, it will take the default position.
            orientation (float[4], None): orientation of the robot. If None, it will be the default orientation.
            fixed_base (bool, None): if True, it will fix the robot's base. If None, it will be the default option.

        Return:
            Robot: instance of the Robot class
        """
        # TODO check the height of the terrain where we wish to load the robot
        def check_height(position):
            """check that the position of the robot is in accordance with the height of the terrain for that
            position."""
            return position

        if isinstance(robot, Robot):  # the robot is already loaded, then add it to the list
            pass

        elif isinstance(robot, str):  # robot's name
            robot = robot.lower()

            if robot in robot_names_to_classes:
                robot_class = robot_names_to_classes[robot]
                robot = robot_class(self.sim, position=position, orientation=orientation, fixed_base=fixed_base,
                                    *args, **kwargs)

            elif robot[-4:] == 'urdf':  # robot is the path to the urdf
                robot = Robot(self.sim, urdf=robot, position=position, orientation=orientation, fixed_base=fixed_base,
                              *args, **kwargs)

            else:
                raise ValueError("The given string does not correspond to any robots or urdfs: {}".format(robot))

        elif inspect.isclass(robot):  # robot class
            robot = robot(self.sim, position=position, orientation=orientation, fixed_base=fixed_base, *args, **kwargs)

        else:  # unknown type
            raise TypeError('Unknown type for robot: {}. It must be a string or '
                            'an instance of Robot'.format(type(robot)))

        self.bodies[robot.id] = robot
        # self.ids[robot.id] = [robot]
        self.ids[robot.id] = [self.__get_method_and_parameters(frame=inspect.currentframe())]
        return robot

    def is_body_id(self, body_id):
        """
        Check if the given id is a body id.

        Args:
            body_id (int): the possible body id

        Returns:
            bool: True if the id is a body id, False otherwise
        """
        if body_id in self.bodies:
            return True  # isinstance(self.bodies[body_id], Body)
        return False

    def is_robot_id(self, body_id):
        """
        Check if the given id is a robot id.

        Args:
            body_id (int): the possible robot id

        Returns:
            bool: True if the id is a robot id, False otherwise
        """
        if body_id in self.bodies:
            body = self.bodies[body_id]
            return isinstance(body, Robot)
        return False

    def get_body(self, body_id):
        """
        Return the instance (Body, Robot) associated to the given body id.

        Args:
            body_id (int): unique body id.

        Raises:
            KeyError: if the given body id is not in the world.

        Returns:
            Body, Robot, int: Body/Robot instance, or unique id.
        """
        return self.bodies[body_id]

    def wrap(self, body_id, wrapper=Body, *args, **kwargs):
        """
        Wrap the given body_id with the provided wrapper. This will replace the

        Args:
            body_id (int): unique body id.
            wrapper (class, Body): wrapper class. By default, it will wrap the provided body_id with the `Body` class.
                The wrapper (its constructor) must at least accepts two parameters: the simulator and the body_id.
            args (tuple, list): list of arguments that are given to the wrapper class.
            kwargs (dict): dictionary of arguments that are given to the wrapper class.

        Returns:
            type(wrapper), Body: instance of the wrapper (by default, it is `Body`)
        """
        if body_id not in self.bodies:
            raise TypeError("Expecting the 'body_id' to be in `self.bodies` (i.e. to have been loaded with one of the "
                            "methods provided in `World`)")
        body = wrapper(self.sim, body_id, *args, **kwargs)
        self.bodies[body_id] = body
        self.ids[body_id].append(body)
        return body

    def reset_robots(self):
        """
        Reset the base and joint states of each robot
        """
        for body_id, body in self.bodies.items():
            if isinstance(body, Robot):
                # reset base
                self.sim.reset_base_pose(body_id, body.init_position, body.init_orientation)
                self.sim.reset_base_velocity(body_id, linear_velocity=[0, 0, 0], angular_velocity=[0, 0, 0])

                # reset joint positions
                positions = body.init_joint_positions
                velocities = np.zeros(len(positions))
                for joint_id, position, velocity in zip(body.joints, positions, velocities):
                    self.sim.reset_joint_state(body_id, joint_id, position, velocity)

    def load_urdf(self, filename, position, orientation=(0, 0, 0, 1), fixed_base=False, scale=1., return_body=False):
        """
        Load the URDF specified by the given path. This will return the body described in the URDF.

        Args:
            filename (str): path to the URDF file
            position (float[3]): position of the object described in the URDF
            orientation (float[4]): orientation represented as a quaternion [x,y,z,w]
            fixed_base (bool): if the base of the object should be fixed or not
            scale (float): scale factor for the object
            name (str, None): name of the object. If None, it will extract it from the URDF.
            return_body (bool): if True, it will return an instance of the `Body`, otherwise, it will return the
                unique id.

        Returns:
            int, Body: unique id, or Body
        """
        body = self.sim.load_urdf(filename, position, orientation, use_fixed_base=fixed_base, scale=scale)
        self.bodies[body] = body
        self.ids[body] = [self.__get_method_and_parameters(frame=inspect.currentframe())]
        if return_body:
            return self.wrap(body_id=body, wrapper=Body)
        return body

    def load_sdf(self, filename, scaling=1., return_body=False):
        """
        Load the given SDF file; this will thus load all the bodies described in a SDF file.

        Args:
            filename (str): path to the SDF file
            scaling (float): scale factor for the object
            return_body (bool): if True, it will return an instance of the `Body`, otherwise, it will return the
                unique id.

        Returns:
            list of int, list of Body: list of unique ids or bodies
        """
        bodies = self.sim.load_sdf(filename, scaling=scaling)
        self.ids[tuple(bodies)] = [self.__get_method_and_parameters(frame=inspect.currentframe())]
        bodies_ = []
        for body in bodies:
            self.bodies[body] = body
            if return_body:
                bodies_.append(self.wrap(body_id=body, wrapper=Body))
            else:
                bodies_.append(body)
        return bodies_

    def load_mjcf(self, filename, scaling=1., return_body=False):
        """
        Load the given MJCF file; this will thus load all the object described in a MJCF file.

        Args:
            filename (str): path to the MJCF file
            scaling (float): scale factor for the object
            return_body (bool): if True, it will return an instance of the `Body`, otherwise, it will return the
                unique id.

        Returns:
            list of int, list of Body: list of bodies
        """
        bodies = self.sim.load_mjcf(filename, scaling=scaling)
        self.ids[tuple(bodies)] = [self.__get_method_and_parameters(frame=inspect.currentframe())]
        bodies_ = []
        for body in bodies:
            self.bodies[body] = body
            if return_body:
                bodies_.append(self.wrap(body_id=body, wrapper=Body))
            else:
                bodies_.append(body)
        return bodies_

    def create_body(self, position, visual_shape_id, collision_shape_id=-1, mass=0., orientation=(0., 0., 0., 1.),
                    return_body=False, *args, **kwargs):
        """Create a body in the simulator.

        Args:
            position (np.array[3]): Cartesian world position of the base
            visual_shape_id (int): unique id from createVisualShape or -1. You can reuse the visual shape (instancing)
            collision_shape_id (int): unique id from createCollisionShape or -1. You can re-use the collision shape
                for multiple multibodies (instancing)
            mass (float): mass of the base, in kg (if using SI units)
            orientation (np.array[4]): Orientation of base as quaternion [x,y,z,w]
            return_body (bool): if True, it will return an instance of the `Body`, otherwise, it will return the
                unique id.

        Returns:
            int: non-negative unique id or -1 for failure.
        """
        body = self.sim.create_body(visual_shape_id=visual_shape_id, collision_shape_id=collision_shape_id, mass=mass,
                                    position=position, orientation=orientation, *args, **kwargs)
        self.bodies[body] = body
        self.ids[body] = [self.__get_method_and_parameters(frame=inspect.currentframe())]
        if return_body:
            return self.wrap(body_id=body, wrapper=Body)
        return body

    def get_available_sdfs(self, fullpath=False):
        """Return the list of available SDFs from the `pybullet_data.getDataPath()` method.

        Args:
            fullpath (bool): If True, it will return the full path to the SDFs. If False, it will just return the
                name of the SDF files (without the extension).
        """
        return self.sim.get_available_sdfs(fullpath=fullpath)

    def get_available_urdfs(self, fullpath=False):
        """Return the list of available URDFs from the `pybullet_data.getDataPath()` method.

        Args:
            fullpath (bool): If True, it will return the full path to the URDFs. If False, it will just return the
                name of the URDF files (without the extension).
        """
        return self.sim.get_available_urdfs(fullpath=fullpath)

    def get_available_mjcfs(self, fullpath=False):
        """Return the list of available MJCFs (=XMLs) from the `pybullet_data.getDataPath()` method.

        Args:
            fullpath (bool): If True, it will return the full path to the MJCFs/XMLs. If False, it will just return
                the name of the MJCF/XML files (without the extension).
        """
        return self.sim.get_available_mjcfs(fullpath=fullpath)

    def get_available_objs(self, fullpath=False):
        """Return the list of available OBJs from the `pybullet_data.getDataPath()` method.

        Args:
            fullpath (bool): If True, it will return the full path to the OBJs. If False, it will just return the
                name of the OBJ files (without the extension).
        """
        return self.sim.get_available_objs(fullpath=fullpath)

    def move_object(self, body_id, position=None, orientation=None):
        """
        Move the given object at the specified position and orientation.

        Args:
            body_id (int): body id
            position (float[3]): new position of the object. If None, it will keep the old position.
            orientation (float[4]): new orientation of the object. If None, it will keep the old orientation.
        """
        if isinstance(body_id, Body):
            body_id = body_id.id
        if position is None:
            position = self.sim.get_base_pose(body_id)[0]
        if orientation is None:
            orientation = self.sim.get_base_pose(body_id)[1]
        self.sim.reset_base_pose(body_id, position, orientation)

    def apply_force(self, body_id, link_id=-1, force=(0., 0., 0.), position=None, frame=2):
        """
        Apply the given force on the specified object or link of the object.

        Warnings:
            - after each simulation step, the external forces are cleared to 0.
            - this does not work when using `sim.setRealTimeSimulation(1)`.

        Args:
            body_id (int): body id to apply the force on
            link_id (int): link id to apply the force, if -1 it will apply the force on the base
            force (np.array[3]): Cartesian forces to be applied on the body
            position (np.array[3]): position on the link where the force is applied. If None, it is the center of mass
                of the object (or the link if specified)
            frame (int): allows to specify the coordinate system of force/position. sim.LINK_FRAME (=1) for local
                link frame, and sim.WORLD_FRAME (=2) for world frame. By default, it is the world frame.
        """
        if isinstance(body_id, Body):
            body_id = body_id.id
        self.sim.apply_external_force(body_id, link_id, force, position, frame)

    def get_body_color(self, body_id):
        """
        Return the RGBA color of the given body.

        Args:
            body_id (int): body id

        Returns:
            float[4]: RGBA color
        """
        return self.sim.get_visual_shape_data(body_id)[-1]

    def change_body_color(self, body_id, color, link_id=-1):
        """
        Change the color of the given body.

        Args:
            body_id (int, Body): body (id)
            color (float[4]): RGBA color where each channel is between 0 and 1.
            link_id (int): link id
        """
        self.sim.change_visual_shape(body_id, link_id, rgba_color=color)

    def get_body_position(self, body_id):
        """
        Return the position of the given body.

        Args:
            body_id (int): body id

        Returns:
            np.array[3]: position of the body (expressed in the world Cartesian frame)
        """
        return self.sim.get_base_pose(body_id)[0]

    def get_body_orientation(self, body_id):
        """
        Return the orientation of the given body.

        Args:
            body_id (int): body id

        Returns:
            np.array[4]: orientation of the body (expressed as a quaternion [x,y,z,w]).
        """
        return self.sim.get_base_pose(body_id)[1]

    def get_body_velocity(self, body_id):
        """
        Return the linear and angular velocities of the given body.

        Args:
            body_id (int): body id

        Returns:
            np.array[6]: linear and angular velocities of the body (in the world frame)
        """
        lin_vel, ang_vel = self.sim.get_base_velocity(body_id)
        return np.concatenate((lin_vel, ang_vel))

    def get_body_linear_velocity(self, body_id):
        """
        Return the linear velocity of the given body.

        Args:
            body_id (int): body id

        Returns:
            np.array[3]: linear velocity of the body
        """
        return self.sim.get_base_velocity(body_id)[0]

    def get_body_angular_velocity(self, body_id):
        """
        Return the angular velocity of the given body.

        Args:
            body_id (int): body id

        Returns:
            np.array[3]: angular velocity of the body
        """
        return self.sim.get_base_velocity(body_id)[1]

    def hide_body(self, body_id):
        """
        Hide (visually) the given body; by making it transparent.

        Args:
            body_id (int): body id
        """
        color = self.get_body_color(body_id)
        color[-1] = 0.
        self.change_body_color(body_id, color=color)

    def show_body(self, body_id):
        """
        Show (visually) a hidden body; by making it opaque.

        Args:
            body_id (int): body id
        """
        color = self.get_body_color(body_id)
        color[-1] = 1.
        self.change_body_color(body_id, color=color)

    def remove(self, body):
        """
        Remove the body specified by its unique id from the world/simulator.

        Args:
            body (int, Body): unique id of the body in the simulator.

        Returns:
            bool: True if succeeded, False if not. This method does not raise any errors.
        """
        if isinstance(body, Body):
            body = body.id
        if body in self.bodies:
            self.bodies.pop(body)
            if body in self.ids:
                self.ids.pop(body)
        else:
            return False

        self.sim.remove_body(body)
        return True

    def get_body_dimensions(self, body_id):
        """
        Return the body dimensions of the given body.

        Args:
            body_id (int): body id

        Returns:
            float[3]: dimensions of the body
        """
        return self.sim.get_visual_shape_data(body_id)[0][3]

    def change_body_scale(self, body_id, scale=(1., 1., 1.)):
        """
        Change the scale of the given body; it changes the scale for the visual and collision shapes.

        Args:
            body_id (int): body id
            scale (float[3]): scaling factors in each direction
        """
        # TODO: currently not possible in PyBullet
        pass

    def get_body_aabb(self, body_id, link_id=-1):
        """
        Return the axis-aligned bounding box (AABB) in world space of the given body.

        Args:
            body_id (int): body id
            link_id (int): optional link id

        Returns:
            np.array[3]: coordinates in world space of the min corner of the AABB
            np.array[3]: coordinates in world space of the max corner of the AABB
        """
        aabb_min, aabb_max = self.sim.get_aabb(body_id, link_id)
        return aabb_min, aabb_max

    def get_body_ids_in_aabb(self, aabb_min, aabb_max):
        """
        Get the list of body ids that have AABB overlap with a given AABB.

        Args:
            aabb_min (float[3]): coordinates of the min corner of the bounding box
            aabb_max (float[3]): coordinates of the max corner of the bounding box

        Returns:
            int[N]: list of body ids
        """
        overlapping_bodies = self.sim.get_overlapping_objects(aabb_min, aabb_max)
        if overlapping_bodies is None:
            return []
        return overlapping_bodies

    def is_there_an_object(self, aabb_min, aabb_max, except_floor=True):
        """
        Return True if there is an object in the bounding box defined by aabb_min and aabb_max.

        Args:
            aabb_min (float[3]): minimum coordinates of the bounding box
            aabb_max (float[3]): maximum coordinates of the bounding box
            except_floor (bool): if the floor should be counted as an object

        Returns:
            bool: True if there is an object in the specified bounding box
        """
        bodies = self.sim.get_overlapping_objects(aabb_min, aabb_max)
        if len(bodies) > 2:
            return True
        if len(bodies) == 0:
            return False
        idx = bodies[0]
        if idx == self.floor_id and except_floor:
            return False
        return True

    def get_closest_bodies(self, body, radius=1, link_id=-1, body2=None, link2_id=-1):  # Not possible for now
        """
        Get the closest bodies from the specified body (or link) within the specified radius.

        Args:
            body (Body): body.
            radius (float): radius around the body in which we check the closest bodies.
            link_id (int): link id. Only report contact points that involve link index of body A.
            body2 (int): only report contact points that involve body B. Important: you need to have a valid body A
                if you provide body B
            link2_id (int): only report contact points that involve link index of body B

        Returns:
            list:
                int: contact flag (reserved)
                int: body unique id of body A
                int: body unique id of body B
                int: link index of body A, -1 for base
                int: link index of body B, -1 for base
                np.array[3]: contact position on A, in Cartesian world coordinates
                np.array[3]: contact position on B, in Cartesian world coordinates
                np.array[3]: contact normal on B, pointing towards A
                float: contact distance, positive for separation, negative for penetration
                float: normal force applied during the last `step`. Always equal to 0.
                float: lateral friction force in the first lateral friction direction (see next returned value)
                np.array[3]: first lateral friction direction
                float: lateral friction force in the second lateral friction direction (see next returned value)
                np.array[3]: second lateral friction direction
        """
        if isinstance(body, Body):
            body = body.id
        if isinstance(body2, Body):
            body2 = body2.id

        if body2 is not None:
            return self.sim.get_closest_points(body1=body, body2=body2, distance=radius,
                                               link1_id=link_id, link2_id=link2_id)

        raise NotImplementedError("Currently, the second body has to be provided...")

    def get_contact_bodies(self):
        pass

    def create_constraint(self, parent_body, parent_link_id=-1, child_body=-1, child_link_id=-1,
                          joint_type=Simulator.JOINT_FIXED, joint_axis=(0., 0., 0.), parent_frame_position=(0., 0., 0.),
                          child_frame_position=(0., 0., 0.), parent_frame_orientation=(0., 0., 0., 1.),
                          child_frame_orientation=(0., 0., 0., 1.)):
        """
        Create a constraint between two links belonging to the same body or two different bodies. You can also create
        a constraint between a body/link and a world frame.

        Args:
            parent_body (int, Body): parent body (or its unique id)
            parent_link_id (int): parent link index (or -1 for the base)
            child_body (int, Body): child body (or its unique id), or -1 for no body (specify a non-dynamic child
                frame in world coordinates)
            child_link_id (int): child link index, or -1 for the base
            joint_type (int): joint type: JOINT_PRISMATIC (=1), JOINT_FIXED (=4), JOINT_POINT2POINT (=5),
                JOINT_GEAR (=6). If the JOINT_FIXED is set, the child body's link will not move with respect to the
                parent body's link. If the JOINT_PRISMATIC is set, the child body's link will only be able to move
                along the given joint axis with respect to the parent body's link. If the JOINT_POINT2POINT is set
                (which should really be called spherical), the child body's link will be able to rotate along the 3
                axis while maintaining the given position relative to the parent body's link.
            joint_axis (np.array[3]): joint axis, in child link frame
            parent_frame_position (np.array[3]): position of the joint frame relative to parent CoM frame.
            child_frame_position (np.array[3]): position of the joint frame relative to a given child CoM frame (or
                world origin if no child specified)
            parent_frame_orientation (np.array[4]): the orientation of the joint frame relative to parent CoM
                coordinate frame (expressed as a quaternion [x,y,z,w])
            child_frame_orientation (np.array[4]): the orientation of the joint frame relative to the child CoM
                coordinate frame, or world origin frame if no child specified (expressed as a quaternion [x,y,z,w])

        Returns:
            int: constraint unique id.
        """
        if isinstance(parent_body, Body):
            parent_body = parent_body.id
        if isinstance(child_body, Body):
            child_body = child_body.id
        return self.sim.create_constraint(parent_body_id=parent_body, parent_link_id=parent_link_id,
                                          child_body_id=child_body, child_link_id=child_link_id,
                                          joint_type=joint_type, joint_axis=joint_axis,
                                          parent_frame_position=parent_frame_position,
                                          child_frame_position=child_frame_position,
                                          parent_frame_orientation=parent_frame_orientation,
                                          child_frame_orientation=child_frame_orientation)

    def remove_constraint(self, constraint_id):
        """
        Remove the specified constraint.

        Args:
            constraint_id (int): constraint unique id.
        """
        self.sim.remove_constraint(constraint_id)

    def attach(self, body1, body2, link1=-1, link2=-1, joint_axis=(0., 0., 0.),
               parent_frame_position=(0., 0., 0.), child_frame_position=(0., 0., 0.),
               parent_frame_orientation=None, child_frame_orientation=(0., 0., 0., 1.)):
        """
        Attach two bodies (links) together at the specified contact point.

        To detach them, call ``world.detach(body1, body2, link1, link2)``.

        Note that this method, in addition to call ``create_constraint`` (with a fixed joint), it checks for collisions
        between body1 and body2, and if there are, check how much body2 penetrates body1 and recomputes the parent
        frame position such that there are no more collisions. Additionally, attaching an object can change the state
        of body1, as such we reset the state as it was before calling this method.

        Args:
            body1 (int, Body): body unique id, or a Body instance.
            body2 (int, Body): body unique id, or a Body instance.
            link1 (int, None): link id. By default, it will be the base (=-1).
            link2 (int, None): link id. By default, it will be the base (=-1).
            joint_axis (np.array[3]): joint axis, in child link frame
            parent_frame_position (np.array[3]): position of the joint frame relative to parent CoM frame.
            child_frame_position (np.array[3]): position of the joint frame relative to a given child CoM frame (or
                world origin if no child specified)
            parent_frame_orientation (np.array[4]): the orientation of the joint frame relative to parent CoM
                coordinate frame (expressed as a quaternion [x,y,z,w])
            child_frame_orientation (np.array[4]): the orientation of the joint frame relative to the child CoM
                coordinate frame, or world origin frame if no child specified (expressed as a quaternion [x,y,z,w])

        Returns:
            bool: True if it was successful.
        """
        # check arguments
        if isinstance(body1, Body):
            body1 = body1.id
        if isinstance(body2, Body):
            body2 = body2.id

        if parent_frame_orientation is None:
            parent_frame_orientation = self.sim.get_base_orientation(body2)

        # save the state of body1
        pose = self.sim.get_base_pose(body1)
        velocity = self.sim.get_base_velocity(body1)
        joint_ids = [joint_id for joint_id in range(self.sim.num_joints(body1))
                     if self.sim.get_joint_info(body1, joint_id)[2] != self.sim.JOINT_FIXED]
        joint_states = self.sim.get_joint_states(body1, joint_ids=joint_ids)

        # create constraint
        constraint_id = self.create_constraint(parent_body=body1, parent_link_id=link1, child_body=body2,
                                               child_link_id=link2, joint_axis=joint_axis,
                                               parent_frame_position=parent_frame_position,
                                               child_frame_position=child_frame_position,
                                               parent_frame_orientation=parent_frame_orientation,
                                               child_frame_orientation=child_frame_orientation)

        # advance for few steps (in simulation)
        for i in range(10):
            self.step()

        # check collisions
        collisions = self.sim.get_contact_points(body1, body2, link1_id=link1, link2_id=link2)

        # if collisions, remove the constraint and recompute the parent frame position such that there are no more
        # collisions. This is achieved by checking the penetrating distance and moving along the contact normal
        # direction (pointing from body1 to body2) by minus that amount.
        if len(collisions) > 0:
            # TODO: it seems that with PyBullet this does nothing, they check for collisions when creating the
            #  constraint
            # get worst penetrating distance and associated contact normal direction (from body2 to body1)
            worst_distance, worst_normal_direction = 0, None
            for collision in collisions:
                distance, normal_direction = collision[7:9]
                if distance < worst_distance:
                    worst_distance, worst_normal_direction = distance, normal_direction

            # compute new parent frame position (the 0.002 is a safety margin)
            parent_frame_position = np.asarray(parent_frame_position)
            parent_frame_position += (-worst_distance + 0.002) * -worst_normal_direction

            # remove the previous constraint
            self.remove_constraint(constraint_id)

            # recreate the constraint with the correct parent frame position
            constraint_id = self.create_constraint(parent_body=body1, parent_link_id=link1, child_body=body2,
                                                   child_link_id=link2, joint_axis=joint_axis,
                                                   parent_frame_position=parent_frame_position,
                                                   child_frame_position=child_frame_position,
                                                   parent_frame_orientation=parent_frame_orientation,
                                                   child_frame_orientation=child_frame_orientation)

        # remember the constraint such that we can call later ``detach`` without providing too much information.
        self.constraints[(body1, body2)] = {(link1, link2): constraint_id}

        # restore state
        self.sim.reset_base_pose(body1, position=pose[0], orientation=pose[1])
        self.sim.reset_base_velocity(body1, linear_velocity=velocity[0], angular_velocity=velocity[1])
        for joint_id, joint_state in zip(joint_ids, joint_states):
            self.sim.reset_joint_state(body1, joint_id, position=joint_state[0], velocity=joint_state[1])

        # return constraint id
        return constraint_id > 0

    def detach(self, body1, body2, link1=None, link2=None):
        """
        Detach two bodies that were previously attached.

        Args:
            body1 (int, Body): body unique id, or a Body instance.
            body2 (int, Body): body unique id, or a Body instance.
            link1 (int, None): link id. By default, it will be the base (=-1). If None, all the links of the first
                body that were attached to the second body will be detached.
            link2 (int, None): link id. By default, it will be the base (=-1). If None, all the links of the second
                body that were attached to the first body will be detached.

        Returns:
            bool: True if it was successful.
        """
        if (body1, body2) in self.constraints:
            if link1 is None or link2 is None:
                for (link_id1, link_id2), constraint_id in self.constraints[(body1, body2)].items():
                    if link1 is None:
                        if link2 is None:  # link1 and link2 are both None
                            self.sim.remove_constraint(constraint_id)
                        else:  # link1 is None and link2 is not
                            if link2 == link_id2:
                                self.sim.remove_constraint(constraint_id)
                    else:
                        if link2 is None:  # link1 is not None and link2 is None
                            if link1 == link_id1:
                                self.sim.remove_constraint(constraint_id)
                        else:  # link1 and link2 are not None
                            if link1 == link_id1 and link2 == link_id2:
                                self.sim.remove_constraint(constraint_id)
            else:
                self.sim.remove_constraint(self.constraints[(body1, body2, link1, link2)])
            return True
        return False

    def are_attached(self, body1, body2, link1=None, link2=None):
        """
        Return True if the given links/bodies are attached.

        Args:
            body1 (int, Body): body unique id, or a Body instance.
            body2 (int, Body): body unique id, or a Body instance.
            link1 (int, None): link id. By default, it will be the base (=-1). If None, all the links of the first
                body that were attached to the second body will be detached.
            link2 (int, None): link id. By default, it will be the base (=-1). If None, all the links of the second
                body that were attached to the first body will be detached.

        Returns:
            bool: True if the bodies/links are attached.
        """
        if (body1, body2) in self.constraints:
            if link1 is None and link2 is None:
                return True
            else:
                for link_1, link_2 in self.constraints[(body1, body2)].keys():
                    if link1 == link_1 and link2 == link_2:
                        return True
        return False

    def load_floor(self, scaling=1.):
        """
        Load a basic floor in the world.

        Args:
            scaling (float): scaling for the floor.

        Returns:
            int: unique id of the floor in the world
        """
        # self.floor_id = self.sim.load_urdf('plane100.urdf', use_fixed_base=True, scale=scaling)
        self.floor_id = self.sim.load_urdf('plane.urdf', position=[0., 0., 0.], use_fixed_base=True, scale=scaling)
        # distance = self.camera.distance
        # self.camera.reset(distance=scaling * distance)
        return self.floor_id

    def load_terrain(self, heightmap, position=(0., 0., 0.), orientation=(.707, 0, 0, .707), scaling=1.,
                     replace_floor=True, remove_obj=False, texture=None):
        """
        Load the given terrain/heightmap into the world.

        Args:
            heightmap (str, np.array[heigth, width]): path to the urdf, sdf, xml, or obj file of the terrain. It can
                also be the path to a heightmap in tif, jpg, or png format. Alternatively, it can represents
                the heightmap as a 2D numpy array where the values represent the height in meters.
            position (float[3]): position of the terrain. By default, origin of the world.
            orientation (tuple of 4 float): orientation of the terrain (expressed as quaternion [x,y,z,w]).
            scaling (float, tuple of 3 float): scaling factor of the terrain.
            replace_floor (bool): if True, it will replace the existing floor. Be careful, that it can cause
                problems with collision.
            remove_obj (bool): if True, it will remove the obj file.
            texture (str, None): texture to apply.

        Returns:
            int: unique id of the terrain.

        Modules to create mesh files (.obj):
        - `mayavi`: https://docs.enthought.com/mayavi/mayavi/
        - `openmesh`: https://www.openmesh.org/media/Documentations/OpenMesh-6.2-Documentation/a00036.html
        - `bpy`: Blender python API - https://docs.blender.org/api/current/
        """
        # if there is already a floor, remove it if specified
        if self.floor_id > -1:
            if replace_floor:
                self.sim.remove_body(self.floor_id)

        # if heightmap is a 2D array
        if isinstance(heightmap, np.ndarray):
            self.generate_terrain(heightmap, filename='heightmap.obj')
            heightmap = 'heightmap.obj'

        filename = heightmap

        # if heightmap is a string, it is the path to the 3d terrain or image
        if isinstance(filename, str):

            if filename[-3:] == 'obj':  # obj (mesh)
                if not isinstance(scaling, (list, tuple)):
                    scaling = [scaling] * 3
                self.floor_id = self.load_mesh(filename, position, orientation, mass=0., scale=scaling, flags=1)

            elif filename[-3:] == 'sdf':  # SDF
                self.floor_id = self.load_sdf(filename=filename, scaling=scaling)

            elif filename[-3:] == 'xml':  # MJCF
                self.floor_id = self.load_mjcf(filename=filename, scaling=scaling)

            elif filename[-4:] == 'urdf':  # URDF
                self.floor_id = self.sim.load_urdf(filename, position, use_fixed_base=True, scale=scaling)

            else:  # heightmap (.tif, .jpg, .png, etc)
                # if extension is jpg or png
                if filename[-3:] == 'png' or filename[-3:] == 'jpg':
                    heightmap = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
                else:  # use gdal to open the heightmap
                    heightmap = self.generate_heightmap(algo=6, filename=filename)

                self.generate_terrain(heightmap, filename=filename)

                # load the obj
                if not isinstance(scaling, (list, tuple)):
                    scaling = [scaling] * 3
                self.floor_id = self.load_mesh(filename, position, orientation, mass=0., scale=scaling, flags=1)
        else:
            raise TypeError("Expecting the given 'heightmap' to be a string or a numpy array, instead got: "
                            "{}".format(type(heightmap)))

        if filename[-3:] == 'obj' and remove_obj:
            # remove mesh from memory
            os.remove(filename + '.obj')  # remove mesh from memory
            # os.remove(filename + '.mtl')

        # apply the given texture if provided
        if isinstance(texture, str):
            texture = self.sim.load_texture(texture)
            self.sim.change_visual_shape(object_id=self.floor_id, link_id=-1, texture_id=texture)

        # return the floor id
        return self.floor_id

    def load_heightmap(self, filename):
        """
        Load a heightmap from an image.

        Args:
            filename (str): filename containing the heightmap in the png, jpg, tif, bmp format

        Returns:
            np.array[H,W]: heightmap (height, width)

        Modules to create mesh files (.obj):
        - `mayavi`: https://docs.enthought.com/mayavi/mayavi/
        - `openmesh`: https://www.openmesh.org/media/Documentations/OpenMesh-6.2-Documentation/a00036.html
        - `bpy`: Blender python API - https://docs.blender.org/api/current/
        """
        if not isinstance(filename, str):
            raise TypeError("Expecting the given 'filename' to be a string (i.e. the path to the heightmap image), "
                            "instead got: {}".format(type(filename)))

        # load heightmap
        if filename[-3:] == 'png' or filename[-3:] == 'jpg':
            heightmap = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        else:  # use gdal to open the heightmap
            heightmap = self.generate_heightmap(algo=6, filename=filename)

        return heightmap

    # aliases
    load_dem = load_heightmap

    @staticmethod
    def generate_heightmap(algo=2, filename=None, width=256, height=256, n=8, min_height=0, max_height=255, noise=0,
                           noise_factor=1, init_values=None, x=None, y=None, z=None, function='multiquadric',
                           dtype=np.int):
        """
        Generate a heightmap (png) using the specified algorithm. We provide 4 algorithms to generate this last one:
        1. by generating it randomly (not advised)
        2. by using the diamond-square algorithm
        3. by using gaussian process regression
        4. by interpolating the given initial points using RBF functions.
        5. from a given 3D equation
        6. by using the Geospatial Data Abstraction Library (GDAL), which allows to open Digital Elevation Models
        (DEM) or Geographic Information System (GIS). It can open a .tiff, .geotiff, ascii grid, or image
        (jpg, png,...) file. Warning: the GDAL option requires the gdal library to be installed.

        By default, the diamond-square algorithm is used. Two good other algorithms are the GDAL and the RBF
        approaches.

        Args:
            algo (int): specifies which algorithm to use to generate the heightmap.
                1. randomly
                2. diamond-square algorithm
                3. diamond-square algorithm (version 2)
                4. RBF interpolation
                5. 3d equation
                6. geospatial
            filename (str, None): if algo=6, path to a DEM, GIS, or image file (e.g. png) to open.
            n (int): used to create a square array of width and height of 2**n + 1. It also specifies the number of
                diamond and square steps.
            min_height (int,float): lower bound; each value in the heightmap will be higher than or equal to this bound
            max_height (int,float): upper bound; each value in the heightmap will be lower than or equal to this bound
            noise (float): magnitude of the noise added to the computed height.
            noise_factor (float): after each step, the jitter is divided by the given factor.
            init_values (np.array[M,3]): list of `M` 3D points which corresponds to initial values that are used to fit
                the gaussian process.
            x (np.array[N], np.array[N,O]): If 1d array, it will compute the meshgrid. Otherwise, the resulting 2D
                array from the meshgrid is expected. This is used to predict the heightmap at the given points.
            y (np.array[0], np.array[N,O]): If 1d array, it will compute the meshgrid. Otherwise, the resulting 2D
                array from the meshgrid is expected. This is used to predict the heightmap at the given points.
            z (callable): it must be a function that accepts two arguments `x` and `y` which will be the arrays from
                the meshgrid.
            function (str, callable): "The radial basis function, based on the radius, r, given by the norm
                (default is Euclidean distance);
                    'multiquadric': sqrt((r/self.epsilon)**2 + 1)
                    'inverse': 1.0/sqrt((r/self.epsilon)**2 + 1)
                    'gaussian': exp(-(r/self.epsilon)**2)
                    'linear': r
                    'cubic': r**3
                    'quintic': r**5
                    'thin_plate': r**2 * log(r)
                If callable, then it must take 2 arguments (self, r). The epsilon parameter will be available as
                self.epsilon. Other keyword arguments passed in will be available as well."
            dtype (np.int, np.float): type of the returned array for the heightmap

        Returns:
            np.array[H,W]: heightmap (height, width)
        """
        if algo == 1:  # random
            heightmap = np.random.randint(min_height, max_height)
        elif algo == 2:  # diamond-square
            heightmap = diamond_square_heightmap(n=n, min_height=min_height, max_height=max_height, noise=noise,
                                                 noise_factor=noise_factor)
        elif algo == 3:  # diamond-square (version 2)
            heightmap = diamond_square_heightmap_2(n=n, min_height=min_height, max_height=max_height, noise=noise,
                                                   noise_factor=noise_factor)
        elif algo == 4:  # RBF interpolation
            heightmap = rbf_heightmap(init_values=init_values, x=x, y=y, function=function, min_height=min_height,
                                      max_height=max_height, dtype=dtype)
        elif algo == 5:  # 3D equation
            heightmap = equation_heightmap(x=x, y=y, z=z, min_height=min_height, max_height=max_height, dtype=dtype)
        elif algo == 6:  # geospatial
            from pyrobolearn.worlds.utils.heightmaps.geospatial import gdal_heightmap
            heightmap = gdal_heightmap(filename=filename)
        else:
            raise NotImplementedError("The algo should be between 1 and 6 (see documentation).")
        return heightmap

    @staticmethod
    def generate_terrain(heightmap, filename, scale=600, smooth=True, verbose=True):
        """
        Generate the terrain (obj) file; that is, create the OBJ file from the heightmap.

        Args:
            heightmap (np.array[W,H]): 2D heightmap.
            filename (str): filename.
            scale (float): scaling factor.
            smooth (bool): if the normals should be smooth.
            verbose (bool): if True, it will output information about the creation of the terrain.

        Returns:
            str: content of the OBJ file.

        References:
            [1] Wavefront .obj file (Wikipedia): https://en.wikipedia.org/wiki/Wavefront_.obj_file
        """
        obj = create_obj_from_heightmap(heightmap=heightmap, scale=scale, smooth=smooth, verbose=verbose,
                                        filename=filename)
        return obj

    def load_stadium(self, scaling=1.):
        """
        Load a stadium as the floor.

        Args:
            scaling (float): scaling for the stadium.

        Returns:
            int: unique id of the floor/stadium in the world
        """
        if self.floor_id > -1:
            self.sim.remove_body(self.floor_id)
        self.floor_id = self.sim.load_urdf('stadium.urdf', use_fixed_base=True, scale=scaling)
        return self.floor_id

    def load_japanese_monastery(self, scaling=1.):
        """
        Load a japanese monastery.

        Args:
            scaling (float): scaling for the japanese monastery

        Returns:
            int: unique id of the monastery
        """
        # replace the floor if there is already one present
        if self.floor_id > -1:
            self.sim.remove_body(self.floor_id)
        self.floor_id = self.sim.load_urdf('samurai.urdf', use_fixed_base=True, scale=scaling)
        return self.floor_id

    def load_bot_lab(self, scaling=2.):
        """
        Load the robot laboratory.

        Args:
            scaling (float): scaling for the robot laboratory.

        Returns:
            int: unique id of the robot lab.
        """
        return self.load_sdf('sdf/botlab/botlab.sdf', scaling=scaling)

    def load_stairs(self):
        pass

    def create_city(self):
        pass

    def load_table(self, position, orientation=None, scaling=1.):
        """
        Load a table in the world.

        Args:
            position (float[3]): position of the table
            orientation (float[4]): orientation of the table (quaternion [x,y,z,w])
            scaling (float): scaling for the table

        Returns:
            int: unique id of the table
        """
        table = self.load_urdf('table/table.urdf', position=position, orientation=orientation, scale=scaling)
        return table

    def load_kiva_shelf(self, scaling=1.):
        """
        Load a Kiva shelf.

        Args:
            scaling (float): scaling of the shelf

        Returns:
            int: unique id of the shelf
        """
        shelf = self.sim.load_sdf('kiva_shelf/model.sdf', scale=scaling)[0]
        self.bodies[shelf] = shelf
        self.ids[shelf] = [self.__get_method_and_parameters(frame=inspect.currentframe())]
        return shelf

    def load_visual_sphere(self, position, radius=0.5, color=None, return_body=False):
        """
        Load a visual sphere in the world (only available in the simulator).

        Args:
            position (float[3]): position of the sphere in Cartesian world space (in meters)
            radius (float): radius of the sphere (in meters)
            color (int[4], None): color of the sphere for red, green, blue, and alpha, each in range [0,1]
            return_body (bool): if True, it will return an instance of the `Body`, otherwise, it will return the
                unique id.

        Returns:
            int, Body: unique id of the visual sphere in the world, or the sphere body
        """
        visual_shape = self.sim.create_visual_shape(self.sim.GEOM_SPHERE, radius=radius, rgba_color=color)
        sphere = self.sim.create_body(visual_shape_id=visual_shape, mass=0., position=position)
        self.bodies[sphere] = sphere
        self.ids[sphere] = [self.__get_method_and_parameters(frame=inspect.currentframe())]
        if return_body:
            return self.wrap(body_id=sphere, wrapper=Body, name='sphere'+str(sphere))
        return sphere

    def load_sphere(self, position, mass=1., radius=0.5, color=None, return_body=False):
        """
        Load a sphere in the world (only available in the simulator).

        Args:
            position (float[3]): position of the sphere in Cartesian world space (in meters)
            mass (float): mass of the sphere (in kg). If mass = 0, the sphere won't move even if there is a collision.
            radius (float): radius of the sphere (in meters).
            color (int[4], None): color of the sphere for red, green, blue, and alpha, each in range [0,1]
            return_body (bool): if True, it will return an instance of the `Body`, otherwise, it will return the
                unique id.

        Returns:
            int, Body: unique id of the sphere in the world, or the sphere body
        """
        collision_shape = self.sim.create_collision_shape(self.sim.GEOM_SPHERE, radius=radius)
        visual_shape = self.sim.create_visual_shape(self.sim.GEOM_SPHERE, radius=radius, rgba_color=color)

        sphere = self.sim.create_body(mass=mass, collision_shape_id=collision_shape, visual_shape_id=visual_shape,
                                      position=position)

        self.bodies[sphere] = sphere
        self.ids[sphere] = [self.__get_method_and_parameters(frame=inspect.currentframe())]
        if return_body:
            return self.wrap(body_id=sphere, wrapper=Body, name='sphere'+str(sphere))
        return sphere

    def load_visual_box(self, position, orientation=(0, 0, 0, 1), dimensions=(1., 1., 1.), color=None,
                        return_body=False):
        """
        Load a visual box in the world (only available in the simulator).

        Args:
            position (float[3]): position of the box in the Cartesian world space (in meters)
            orientation (float[4]): orientation of the box using quaternion [x,y,z,w].
            dimensions (float[3]): dimensions of the box
            color (int[4], None): color of the box for red, green, blue, and alpha, each in range [0,1]
            return_body (bool): if True, it will return an instance of the `Body`, otherwise, it will return the
                unique id.

        Returns:
            int, Body: unique id of the box in the world, or the box body
        """
        dimensions = np.asarray(dimensions)
        visual_shape = self.sim.create_visual_shape(self.sim.GEOM_BOX, half_extents=dimensions / 2., rgba_color=color)

        box = self.sim.create_body(mass=0., visual_shape_id=visual_shape, position=position, orientation=orientation)

        self.bodies[box] = box
        self.ids[box] = [self.__get_method_and_parameters(frame=inspect.currentframe())]
        if return_body:
            return self.wrap(body_id=box, wrapper=Body, name='box'+str(box))
        return box

    def load_box(self, position, orientation=(0, 0, 0, 1), mass=1., dimensions=(1., 1., 1.), color=None,
                 return_body=False):
        """
        Load a box in the world (only available in the simulator).

        Args:
            position (float[3]): position of the box in the Cartesian world space (in meters)
            orientation (float[4]): orientation of the box using quaternion [x,y,z,w].
            mass (float): mass of the box (in kg). If mass = 0, the box won't move even if there is a collision.
            dimensions (float[3]): dimensions of the box (in meter)
            color (int[4], None): color of the box for red, green, blue, and alpha, each in range [0,1]
            return_body (bool): if True, it will return an instance of the `Body`, otherwise, it will return the
                unique id.

        Returns:
            int, Body: unique id of the box in the world, or the box body
        """
        dimensions = np.asarray(dimensions)
        collision_shape = self.sim.create_collision_shape(self.sim.GEOM_BOX, half_extents=dimensions / 2.)
        visual_shape = self.sim.create_visual_shape(self.sim.GEOM_BOX, half_extents=dimensions / 2., rgba_color=color)

        box = self.sim.create_body(mass=mass, collision_shape_id=collision_shape, visual_shape_id=visual_shape,
                                   position=position, orientation=orientation)

        self.bodies[box] = box
        self.ids[box] = [self.__get_method_and_parameters(frame=inspect.currentframe())]
        if return_body:
            return self.wrap(body_id=box, wrapper=Body, name='box'+str(box))
        return box

    def load_visual_cylinder(self, position, orientation=(0, 0, 0, 1), radius=0.5, height=1., color=None,
                             return_body=False):
        """
        Load a visual cylinder in the world (only available in the simulator).

        Args:
            position (float[3]): position of the cylinder in the Cartesian world space (in meters)
            orientation (float[4]): orientation of the cylinder using quaternion [x,y,z,w].
            radius (float): radius of the cylinder (in meters)
            height (float): height of the cylinder (in meters)
            color (int[4], None): color of the cylinder for red, green, blue, and alpha, each in range [0,1]
            return_body (bool): if True, it will return an instance of the `Body`, otherwise, it will return the
                unique id.

        Returns:
            int, Body: unique id of the cylinder in the world, or the cylinder body
        """
        visual_shape = self.sim.create_visual_shape(self.sim.GEOM_CYLINDER, radius=radius, length=height,
                                                    rgba_color=color)
        
        cylinder = self.sim.create_body(mass=0., visual_shape_id=visual_shape, position=position,
                                        orientation=orientation)
        self.bodies[cylinder] = cylinder
        self.ids[cylinder] = [self.__get_method_and_parameters(frame=inspect.currentframe())]
        if return_body:
            return self.wrap(body_id=cylinder, wrapper=Body, name='cylinder'+str(cylinder))
        return cylinder

    def load_cylinder(self, position, orientation=(0, 0, 0, 1), mass=1., radius=0.5, height=1., color=None,
                      return_body=False):
        """
        Load a cylinder in the world (only available in the simulator).

        Args:
            position (float[3]): position of the cylinder in the Cartesian world space (in meters)
            orientation (float[4]): orientation of the cylinder using quaternion [x,y,z,w].
            mass (float): mass of the cylinder (in kg). If mass = 0, it won't move even if there is a collision.
            radius (float): radius of the cylinder (in meters)
            height (float): height of the cylinder (in meters)
            color (int[4], None): color of the cylinder for red, green, blue, and alpha, each in range [0,1]
            return_body (bool): if True, it will return an instance of the `Body`, otherwise, it will return the
                unique id.

        Returns:
            int, Body: unique id of the cylinder in the world, or the cylinder body
        """
        collision_shape = self.sim.create_collision_shape(self.sim.GEOM_CYLINDER, radius=radius, height=height)
        visual_shape = self.sim.create_visual_shape(self.sim.GEOM_CYLINDER, radius=radius, length=height,
                                                    rgba_color=color)
        
        cylinder = self.sim.create_body(mass=mass, collision_shape_id=collision_shape, visual_shape_id=visual_shape,
                                        position=position, orientation=orientation)

        self.bodies[cylinder] = cylinder
        self.ids[cylinder] = [self.__get_method_and_parameters(frame=inspect.currentframe())]
        if return_body:
            return self.wrap(body_id=cylinder, wrapper=Body, name='cylinder'+str(cylinder))
        return cylinder

    def load_visual_capsule(self, position, orientation=(0, 0, 0, 1), radius=0.5, height=1., color=None,
                            return_body=False):
        """
        Load a visual capsule in the world (only available in the simulator).

        Args:
            position (float[3]): position of the capsule in the Cartesian world space (in meters)
            orientation (float[4]): orientation of the capsule using quaternion [x,y,z,w].
            radius (float): radius of the capsule (in meters)
            height (float): height of the capsule (in meters)
            color (int[4], None): color of the capsule for red, green, blue, and alpha, each in range [0,1]
            return_body (bool): if True, it will return an instance of the `Body`, otherwise, it will return the
                unique id.

        Returns:
            int, Body: unique id of the capsule in the world, or the capsule body
        """
        visual_shape = self.sim.create_visual_shape(self.sim.GEOM_CAPSULE, radius=radius, length=height/2.,
                                                    rgba_color=color)
        capsule = self.sim.create_body(mass=0., visual_shape_id=visual_shape, position=position,
                                       orientation=orientation)

        self.bodies[capsule] = capsule
        self.ids[capsule] = [self.__get_method_and_parameters(frame=inspect.currentframe())]
        if return_body:
            return self.wrap(body_id=capsule, wrapper=Body, name='capsule'+str(capsule))
        return capsule

    def load_capsule(self, position, orientation=(0, 0, 0, 1), mass=1., radius=0.5, height=1., color=None,
                     return_body=False):
        """
        Load a capsule in the world (only available in the simulator).

        Args:
            position (float[3]): position of the capsule in the Cartesian world space (in meters)
            orientation (float[4]): orientation of the capsule using quaternion [x,y,z,w].
            mass (float): mass of the capsule (in kg). If mass = 0, it won't move even if there is a collision.
            radius (float): radius of the capsule (in meters)
            height (float): height of the capsule (in meters)
            color (int[4], None): color of the capsule for red, green, blue, and alpha, each in range [0,1]
            return_body (bool): if True, it will return an instance of the `Body`, otherwise, it will return the
                unique id.

        Returns:
            int, Body: unique id of the capsule in the world, or the caspule body
        """
        collision_shape = self.sim.create_collision_shape(self.sim.GEOM_CAPSULE, radius=radius, height=height / 2.)
        visual_shape = self.sim.create_visual_shape(self.sim.GEOM_CAPSULE, radius=radius, length=height / 2.,
                                                    rgba_color=color)
        
        capsule = self.sim.create_body(mass=mass, collision_shape_id=collision_shape, visual_shape_id=visual_shape,
                                       position=position, orientation=orientation)

        self.bodies[capsule] = capsule
        self.ids[capsule] = [self.__get_method_and_parameters(frame=inspect.currentframe())]
        if return_body:
            return self.wrap(body_id=capsule, wrapper=Body, name='capsule'+str(capsule))
        return capsule

    def load_visual_mesh(self, filename, position, orientation=(0, 0, 0, 1), scale=(1., 1., 1.), color=None,
                         return_body=False):
        """
        Load a visual mesh in the world (only available in the simulator).

        Args:
            filename (str): path to file for the mesh. Currently, only Wavefront .obj. It will create convex hulls
                for each object (marked as 'o') in the .obj file.
            position (float[3]): position of the mesh in the Cartesian world space (in meters)
            orientation (float[4]): orientation of the mesh using quaternion [x,y,z,w].
            scale (float[3]): scale the mesh in the (x,y,z) directions
            color (int[4], None): color of the mesh for red, green, blue, and alpha, each in range [0,1]
            return_body (bool): if True, it will return an instance of the `Body`, otherwise, it will return the
                unique id.

        Returns:
            int, Body: unique id of the mesh in the world, or the mesh body
        """
        mesh = self.sim.load_mesh(filename, position, orientation, mass=0., scale=scale, color=color,
                                  with_collision=False)
        self.bodies[mesh] = mesh
        self.ids[mesh] = [self.__get_method_and_parameters(frame=inspect.currentframe())]
        if return_body:
            return self.wrap(body_id=mesh, wrapper=Body, name='mesh'+str(mesh))
        return mesh

    def load_mesh(self, filename, position, orientation=(0, 0, 0, 1), mass=1., scale=(1., 1., 1.), color=None,
                  flags=None, return_body=False):
        """
        Load a mesh in the world (only available in the simulator).

        Args:
            filename (str): path to file for the mesh. Currently, only Wavefront .obj. It will create convex hulls
                for each object (marked as 'o') in the .obj file.
            position (float[3]): position of the mesh in the Cartesian world space (in meters)
            orientation (float[4]): orientation of the mesh using quaternion [x,y,z,w].
            mass (float): mass of the mesh (in kg). If mass = 0, it won't move even if there is a collision.
            scale (float[3]): scale the mesh in the (x,y,z) directions
            color (int[4], None): color of the mesh for red, green, blue, and alpha, each in range [0,1]
            flags (int, None): if flag = `sim.GEOM_FORCE_CONCAVE_TRIMESH` (=1), this will create a concave static
                triangle mesh. This should not be used with dynamic/moving bodies, only for static (mass=0) terrain.
            return_body (bool): if True, it will return an instance of the `Body`, otherwise, it will return the
            unique id.

        Returns:
            int, Body: unique id of the mesh in the world, or the mesh body
        """
        mesh = self.sim.load_mesh(filename, position, orientation, mass, scale, color, with_collision=True,
                                  flags=flags)
        self.bodies[mesh] = mesh
        self.ids[mesh] = [self.__get_method_and_parameters(frame=inspect.currentframe())]
        if return_body:
            return self.wrap(body_id=mesh, wrapper=Body, name='mesh'+str(mesh))
        return mesh

    # The following commented code does not work currently because URDF_GEOM_PLANE is not set in Bullet
    # Note that a plane can be seen as a thin box.
    # def load_visual_plane(self, position, orientation, normal=(0.,0.,1.), color=(1,1,1,1)):
    #     """
    #     Load a visual plane in the world (only available in the simulator).
    #
    #     Args:
    #         position (float[3]): position of the plane in the Cartesian world space (in meters)
    #         orientation (float[4]): orientation of the plane using quaternion [x,y,z,w].
    #         normal (float[3]): normal to the plane
    #         color (int[4]): color of the plane
    #
    #     Returns:
    #         int: unique id of the plane in the world
    #     """
    #     visual_shape = self.sim.create_visual_shape(self.sim.GEOM_PLANE, planeNormal=normal, rgba_color=color)
    #     
    #     plane = self.sim.create_body(mass=0.,
    #                                     visual_shape_id=visual_shape,
    #                                     position=position,
    #                                     orientation=orientation)
    #     self.bodies[plane] = plane
    #     self.ids[plane] = [self.__get_method_and_parameters(frame=inspect.currentframe())]
    #     return plane
    #
    # def load_plane(self, position, orientation, mass=1., normal=(0.,0.,1.), color=(1,1,1,1)):
    #     """
    #     Load a plane in the world (only available in the simulator).
    #
    #     Args:
    #         position (float[3]): position of the plane in the Cartesian world space (in meters)
    #         orientation (float[4]): orientation of the plane using quaternion [x,y,z,w].
    #         mass (float): mass of the plane (in kg). If mass = 0, it won't move even if there is a collision.
    #         normal (float[3]): normal to the plane
    #         color (int[4]): color of the plane
    #
    #     Returns:
    #         int: unique id of the plane in the world
    #     """
    #     collision_shape = self.sim.create_collision_shape(self.sim.GEOM_PLANE, planeNormal=normal)
    #     visual_shape = self.sim.create_visual_shape(self.sim.GEOM_PLANE, planeNormal=normal, rgba_color=color)
    #     
    #     plane = self.sim.create_body(mass=mass,
    #                                      collision_shape_id=collision_shape,
    #                                      visual_shape_id=visual_shape,
    #                                      position=position,
    #                                      orientation=orientation)
    #     self.bodies[plane] = plane
    #     self.ids[plane] = [self.__get_method_and_parameters(frame=inspect.currentframe())]
    #     return plane

    # Temporary because the code above doesn't work
    def load_plane(self, position=(0., 0., 0.), orientation=(0., 0., 0., 1.), scale=1., return_body=False):
        """
        Load a plane in the world (only available in the simulator)

        Args:
            position (float[3]): position of the plane
            orientation (float[4]): orientation of the plane
            scale (float): scale factor of the plane
            return_body (bool): if True, it will return an instance of the `Body`, otherwise, it will return the
                unique id.

        Returns:
            int, Body: unique id of the plane, or body instance
        """
        plane = self.sim.load_urdf('plane.urdf', position, orientation, use_fixed_base=True, scale=scale)
        self.bodies[plane] = plane
        self.ids[plane] = [self.__get_method_and_parameters(frame=inspect.currentframe())]
        if return_body:
            return self.wrap(body_id=plane, wrapper=Body, name='plane'+str(plane))
        return plane

    def load_visual_ellipsoid(self, position, orientation=(0, 0, 0, 1), scale=(1., 1., 1.), color=None,
                              return_body=False):
        """
        Load a visual ellipsoid (using a mesh) in the world (only available in the simulator).

        Args:
            position (float[3]): position in the Cartesian world space (in meters)
            orientation (float[4]): orientation using quaternion [x,y,z,w].
            scale (float[3]): scale in the (x,y,z) directions
            color (int[4], None): color of the ellipsoid for red, green, blue, and alpha, each in range [0,1]
            return_body (bool): if True, it will return an instance of the `Body`, otherwise, it will return the
                unique id.

        Returns:
            int, body: unique id of the ellipsoid in the world, or body instance
        """
        filename = os.path.dirname(__file__) + '/meshes/ellipsoid.obj'
        mesh = self.load_visual_mesh(filename, position, orientation, scale=scale, color=color)
        if return_body:
            return self.wrap(body_id=mesh, wrapper=Body, name='ellipsoid'+str(mesh))
        return mesh

    def load_ellipsoid(self, position, orientation=(0, 0, 0, 1), mass=1., scale=(1., 1., 1.), color=None,
                       return_body=False):
        """
        Load a ellipsoid (using a mesh) in the world (only available in the simulator).

        Args:
            position (float[3]): position in the Cartesian world space (in meters)
            orientation (float[4]): orientation using quaternion [x,y,z,w].
            mass (float): mass [kg]
            scale (float[3]): scale in the (x,y,z) directions
            color (int[4], None): color of the ellipsoid for red, green, blue, and alpha, each in range [0,1]
            return_body (bool): if True, it will return an instance of the `Body`, otherwise, it will return the
                unique id.

        Returns:
            int, Body: unique id of the ellipsoid in the world, or body instance
        """
        filename = os.path.dirname(__file__) + '/meshes/ellipsoid.obj'
        mesh = self.load_mesh(filename, position, orientation, mass=mass, scale=scale, color=color)
        if return_body:
            return self.wrap(body_id=mesh, wrapper=Body, name='ellipsoid'+str(mesh))
        return mesh

    def load_visual_right_triangular_prism(self, position, orientation=(0, 0, 0, 1), scale=(1., 1., 1.),
                                           color=None, return_body=False):
        """
        Load a visual right triangular prism (using a mesh) in the world (only available in the simulator).

        Args:
            position (float[3]): position in the Cartesian world space (in meters)
            orientation (float[4]): orientation using quaternion [x,y,z,w].
            scale (float[3]): scale in the (x,y,z) directions
            color (int[4], None): color of the prism for red, green, blue, and alpha, each in range [0,1]
            return_body (bool): if True, it will return an instance of the `Body`, otherwise, it will return the
                unique id.

        Returns:
            int, Body: unique id of the triangular prism in the world, or body instance
        """
        filename = os.path.dirname(__file__) + '/meshes/right_triangular_prism.obj'
        mesh = self.load_visual_mesh(filename, position, orientation, scale=scale, color=color)
        if return_body:
            return self.wrap(body_id=mesh, wrapper=Body, name='right_triangular_prism'+str(mesh))
        return mesh

    def load_right_triangular_prism(self, position, orientation=(0, 0, 0, 1), mass=1., scale=(1., 1., 1.),
                                    color=None, return_body=False):
        """
        Load a right triangular prism (using a mesh) in the world (only available in the simulator).

        Args:
            position (float[3]): position in the Cartesian world space (in meters)
            orientation (float[4]): orientation using quaternion [x,y,z,w].
            mass (float): mass [kg]
            scale (float[3]): scale in the (x,y,z) directions
            color (int[4], None): color of the prism for red, green, blue, and alpha, each in range [0,1]
            return_body (bool): if True, it will return an instance of the `Body`, otherwise, it will return the
                unique id.

        Returns:
            int, Body: unique id of the triangular prism in the world, or body instance
        """
        filename = os.path.dirname(__file__) + '/meshes/right_triangular_prism.obj'
        mesh = self.load_mesh(filename, position, orientation, mass=mass, scale=scale, color=color)
        if return_body:
            return self.wrap(body_id=mesh, wrapper=Body, name='right_triangular_prism'+str(mesh))
        return mesh

    def load_visual_cone(self, position, orientation=(0, 0, 0, 1), scale=(1., 1., 1.), color=None, return_body=False):
        """
        Load a visual cone (using a mesh) in the world (only available in the simulator).

        Args:
            position (float[3]): position in the Cartesian world space (in meters)
            orientation (float[4]): orientation using quaternion [x,y,z,w].
            scale (float[3]): scale in the (x,y,z) directions
            color (int[4], None): color of the cone for red, green, blue, and alpha, each in range [0,1]
            return_body (bool): if True, it will return an instance of the `Body`, otherwise, it will return the
                unique id.

        Returns:
            int, Body: unique id of the cone in the world, or body instance
        """
        filename = os.path.dirname(__file__) + '/meshes/cone.obj'
        mesh = self.load_visual_mesh(filename, position, orientation, scale=scale, color=color)
        if return_body:
            return self.wrap(body_id=mesh, wrapper=Body, name='cone'+str(mesh))
        return mesh

    def load_cone(self, position, orientation=(0, 0, 0, 1), mass=1., scale=(1., 1., 1.), color=None, return_body=False):
        """
        Load a visual cone (using a mesh) in the world (only available in the simulator).

        Args:
            position (float[3]): position in the Cartesian world space (in meters)
            orientation (float[4]): orientation using quaternion [x,y,z,w].
            mass (float): mass [kg]
            scale (float[3]): scale in the (x,y,z) directions
            color (int[4], None): color of the cone for red, green, blue, and alpha, each in range [0,1]
            return_body (bool): if True, it will return an instance of the `Body`, otherwise, it will return the
                unique id.

        Returns:
            int, Body: unique id of the cone in the world, or body instance
        """
        filename = os.path.dirname(__file__) + '/meshes/cone.obj'
        mesh = self.load_mesh(filename, position, orientation, mass=mass, scale=scale, color=color)
        if return_body:
            return self.wrap(body_id=mesh, wrapper=Body, name='cone'+str(mesh))
        return mesh

    def load_visual_arrow(self, position, orientation=(0, 0, 0, 1), scale=(1., 1., 1.), color=None, return_body=False):
        pass

    def load_arrow(self, position, orientation=(0, 0, 0, 1), mass=1., scale=(1., 1., 1.), color=None,
                   return_body=False):
        pass

    # TODO: check collisions when distributing the various bodies (need to know the dimensions)
    def distribute(self, body, size=2, position_range=(-1, 1), rpy_range=(0, 0), return_body=False, *args,
                   **kwargs):
        r"""
        Spawn several bodies in the specified range.

        Args:
            body (int, Body, callable): if int, it is assumed to be the unique id of a body that has been loaded in
                the world. If callable, it is assumed to be a method of this class (such as `load_box`, `load_sphere`,
                etc) that will load a body in the world. This method will be called multiple times to load the various
                bodies in the world.
            size (int): the total number of bodies to distribute. This takes into account if the given :attr:`body`
                has already been spawned once (which is the case if type(body) is an int or an instance of Body).
            position_range (tuple of float, tuple of np.array): range of the uniform distribution interval for the
                position of each body. The first element is the lower boundary, and the second one the higher boundary
                of the interval.
            rpy_range (tuple of float, tuple of np.array): range of the uniform distribution interval for the
                orientation (expressed as roll-pitch-yaw angles) of each body. The first element is the lower boundary,
                and the second one the higher boundary of the interval.
            return_body (bool): if True, it will return an instance of the `Body`, otherwise, it will return the
                unique id.
            *args: list of arguments to be given to :attr:`body` if this last one is callable.
            **kwargs: dictionary of arguments to be given to :attr:`body` if this last one is callable.

        Returns:
            list of int, list of Body: list of unique ids for each body, or list of bodies
        """
        # check size
        if size < 1:
            raise ValueError("Expecting the given `size` to be an integer bigger than 0, instead got: {}".format(size))

        # create positions (using uniform distribution)
        low, high = position_range
        if isinstance(low, (float, int)) and isinstance(high, (float, int)):
            positions = np.random.uniform(low=low, high=high, size=(size,))
        else:
            if isinstance(low, collections.Iterable):
                positions = np.random.uniform(low=low, high=high, size=(size, len(low)))
            else:
                positions = np.random.uniform(low=low, high=high, size=(size, len(high)))

        # create orientations (using uniform distribution)
        low, high = rpy_range
        rpys = np.random.uniform(low=low, high=high, size=(size, 3))

        # check given body argument
        bodies = []
        if self.is_body_id(body):  # unique id
            body = self.wrap(body, wrapper=Body)
        elif isinstance(body, Body):  # Body
            pass
        elif callable(body) and hasattr(self, body.__name__) and 'position' in inspect.getargspec(body).args:
            body = body(position=positions[0], orientation=get_quaternion_from_rpy(rpys[0]), *args, **kwargs)
            if not isinstance(body, Body):
                body = self.wrap(body, wrapper=Body)
            positions, rpys = positions[1:], rpys[1:]
        else:
            raise TypeError("Expecting the given `body` to be a unique id (int), an instance of Body, or a method of "
                            "`World`, instead got: {} (type={})".format(body, type(body)))

        # add first given body
        if return_body:
            bodies.append(body)
        else:
            bodies.append(body.id)

        # get method and previous arguments
        method_name, kwargs = self.ids[body.id][0]
        method = getattr(self, method_name)
        kwargs = dict(kwargs)

        if isinstance(body, Robot):  # if robot, get the class
            if 'robot' in kwargs:
                kwargs['robot'] = kwargs['robot'].__class__

        # distribute the various other bodies  # TODO: check for collisions
        for position, rpy in zip(positions, rpys):

            # update new position and create body
            kwargs['position'] = position
            kwargs['orientation'] = get_quaternion_from_rpy(rpy)
            body = method(**kwargs)

            if return_body:
                if not isinstance(body, Body):
                    body = self.wrap(body, wrapper=Body)
                bodies.append(body)
            else:
                if isinstance(body, Body):
                    body = body.id
                bodies.append(body)

        return bodies

    def get_dynamics_info(self, body_id, link_id=-1):
        """
        Return the dynamics information about bodies that are in the world.

        Args:
            body_id (int): object unique id.
            link_id (int): link index (or -1 for the base).

        Returns:
            float: mass in kg
            float: lateral friction coefficient
            np.array[3]: local inertia diagonal
            np.array[3]: position of inertial frame in local coordinates of joint frame
            np.array[4]: orientation of inertial frame in local coordinates of joint frame
            float: restitution coefficient (if 0, the object does not bounce)
            float: rolling friction coefficient orthogonal to contact normal
            float: spinning friction coefficient around contact normal
            float: -1 if not available, damping of contact constraints
            float: -1 if not available, stiffness contact constraints
        """
        return self.sim.get_dynamics_info(body_id, link_id)

    def print_dynamics_info(self, body_id, link_id=-1):
        """
        Print the dynamics information related to the given body id and link id.

        Args:
            body_id (int): object unique id.
            link_id (int): link index (or -1 for the base).
        """
        info = self.sim.get_dynamics_info(body_id, link_id)
        print("Mass: {}".format(info[0]))
        print("Lateral friction coefficient: {}".format(info[1]))
        print("Local inertia diagonal: {}".format(info[2]))
        print("Local inertial position: {}".format(info[3]))
        print("Local inertial orientation (quat=[x,y,z,w]): {}".format(info[4]))
        print("Restitution coefficient (bounciness): {}".format(info[5]))
        print("Rolling friction coefficient: {}".format(info[6]))
        print("Spinning friction coefficient: {}".format(info[7]))
        print("Contact damping coefficient (-1 if not available): {}".format(info[8]))
        print("Contact stiffness coefficient (-1 if not available): {}".format(info[9]))

    def change_dynamics(self, body_id=None, lateral_friction=1., spinning_friction=0., rolling_friction=0.,
                        restitution=0., linear_damping=0.04, angular_damping=0.04, contact_stiffness=-1,
                        contact_damping=-1, **kwargs):
        """
        Change the dynamics of a body. If no body is specified, it will be change the world floor dynamics.

        Args:
            body_id (int, None): unique body id. If None, it will be the world floor.
            lateral_friction (float): lateral (linear) contact friction
            spinning_friction (float): torsional friction around the contact normal
            rolling_friction (float): torsional friction orthogonal to contact normal
            restitution (float): bounciness of contact. Keep it a bit less than 1.
            linear_damping (float): linear damping of the link.
            angular_damping (float): angular damping of the link.
            contact_stiffness (float): stiffness of the contact constraints, used together with `contact_damping`
            contact_damping (float): damping of the contact constraints for this body/link. Used together with
                `contact_stiffness`. This overrides the value if it was specified in the URDF file in the contact
                section.
        """
        if body_id is None:
            body_id = self.floor_id
        elif isinstance(body_id, Body):
            body_id = body_id.id
        self.sim.change_dynamics(body_id=body_id, link_id=-1, lateral_friction=lateral_friction,
                                 spinning_friction=spinning_friction, rolling_friction=rolling_friction,
                                 restitution=restitution, linear_damping=linear_damping,
                                 angular_damping=angular_damping, contact_stiffness=contact_stiffness,
                                 contact_damping=contact_damping)

    def apply_texture(self, texture, body_id, link_id=-1):
        """
        Apply the texture to the given object.

        Args:
            texture (str): path to the texture.
            body_id (int): unique body id.
            link_id (int): link id. If -1, it will be the base.
        """
        if isinstance(body_id, Body):
            body_id = body_id.id
        texture = self.sim.load_texture(texture)
        self.sim.change_visual_shape(object_id=body_id, link_id=link_id, texture_id=texture)


class BasicWorld(World):
    r"""Basic World class.

    It creates a basic world with a floor and set the gravity.
    """

    def __init__(self, simulator, floor_path=None, gravity=(0., 0., -9.81), scaling=1., lateral_friction=1.,
                 spinning_friction=0., rolling_friction=0., contact_stiffness=-1, contact_damping=-1):
        super(BasicWorld, self).__init__(simulator, gravity=gravity)

        if floor_path is None:
            self.load_floor(scaling=scaling)
            self.change_dynamics(lateral_friction=lateral_friction, spinning_friction=spinning_friction,
                                 rolling_friction=rolling_friction, linear_damping=0.04, angular_damping=0.04,
                                 contact_stiffness=contact_stiffness, contact_damping=contact_damping)
            # self.simulator.setDefaultContactERP(0.9)
            self.simulator.set_physics_properties(erp=0.9)
        else:
            self.load_terrain(floor_path, replace_floor=True)

        self.print_dynamics_info(self.floor_id)


class RobotPartyWorld(BasicWorld):
    r"""Robot Party World.

    It creates a basic world (see `BasicWorld`) with each available robot loaded into the world.
    """

    def __init__(self, simulator):
        super(RobotPartyWorld, self).__init__(simulator)


class DRCWorld(World):
    r"""Darpa Robotics Challenge World

    This recreates the Darpa Robotics Challenge world.
    """

    def __init__(self, simulator):
        super(DRCWorld, self).__init__(simulator)


# Tests
if __name__ == '__main__':
    import numpy as np
    from itertools import count
    from pyrobolearn.simulators import Bullet

    # create simulator
    sim = Bullet()

    # create world
    world = BasicWorld(sim)
    # world = World(sim)
    # world.load_bot_lab()

    # load meshes
    # world.load_mesh('utils/terrains/terrain_map.obj',
    #                 position=[0, 0, -2],
    #                 orientation=[.707, 0, 0, .707],
    #                 mass=0.,
    #                 scale=(.1, .1, .1),
    #                 # color=[1, 0, 0, 1],
    #                 flags=1)
    # world.load_mesh('cube.obj', position=[0, 0, 2], scale=(.1, .1, .1), flags=0)
    # world.load_mesh('bedroom.obj', [0, 0, 0], mass=0., color=[0.4, 0.4, 0.4, 1], flags=1) #, scale=(0.01, 0.01, 0.01))
    # world.load_mesh('mtsthelens.obj', [0, 0, -8], mass=0., color=[0.2, 0.5, 0.2, 1], flags=1, scale=(0.01,0.01,0.01))
    # world.load_mesh('meshes/terrain.obj', [0,0,0], mass=0., color=[1,1,1,1], flags=1)

    # # load robots
    # world.load_robot('Cogimon', position=[0,0,1.])
    # world.load_robot('coman', use_fixed_base=False)

    # load basic shapes
    sphere = world.load_visual_sphere([1., 0, 1.], color=(1, 0, 0, 0.5))
    sphere = world.wrap(sphere, name='sphere')
    # world.load_visual_box([-1,0,1], dimensions=[1.,1.,1.], color=[0,0,1,0.5])
    # world.load_cylinder([0, -1, 1], color=[1, 0, 0, 1])
    #  world.load_capsule([0, 1, 1],  color=[1, 0, 0, 1])
    #  world.load_mesh(filename='duck.obj', [1, 0, 2], [0.707, 0, 0, 0.707], mass=0.1, scale=[0.1,0.1,0.1],
    #                 color=[1, 0, 0, 1])

    # world.load_ellipsoid([0,0,2], mass=0, scale=[2.,1.,1.], color=(0,0,1,1))
    world.load_visual_cone([0, 0, 0.1*0.5], orientation=(0, 1, 0, 0), scale=(0.1, 0.1, 0.1), color=(0.5, 0, 0, 0.5))
    # world.load_right_triangular_prism([-1, -1, 2])
    # floor = world.load_mesh(filename='box', [1, 0, 2], mass=0, color=None)

    # floor = world.load_floor()
    # floor = world.load_mesh([1,0,0], [0,0,0,1], filename='grass.obj', mass=0, color=(1,1,1,1))
    # texture = sim.load_texture('grass.png')
    # sim.change_visual_shape(floor, -1, texture_id=texture)

    # grass = sim.load_urdf('grass.urdf')
    # texture = sim.load_texture('grass.png')
    # sim.change_visual_shape(grass, -1, texture_id=texture)

    # vs = world.load_visual_sphere([0,0,2], radius=0.1, color=(0,0,1,1))

    # world.load_plane([1.,0.,1.], [0,0,0,1], color=(1,0,0,1))

    T = 1000
    w = 2.*np.pi / T
    red = True
    # loop
    for t in count():
        # p = world.get_object_position(sphere)
        # p -= 0.001 * np.array([1.,0,0])
        p = np.array([np.cos(w*t), np.sin(w*t), 1.])

        if t % T == 0:
            if red:
                world.change_body_color(sphere.id, (1, 0, 0, 0.5))
            else:
                world.change_body_color(sphere.id, (0, 0, 1, 0.5))
            red = not red
        # world.move_object(sphere, p)
        sphere.position = p
        world.apply_force(body_id=sphere.id, force=(0, 0, 100))
        world.step(sleep_dt=1./240)
