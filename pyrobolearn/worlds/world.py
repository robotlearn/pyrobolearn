#!/usr/bin/env python
"""Define the the `World` class which allows to specify what constitutes the world (i.e. what elements are in
 the world).

Dependencies:
- `pyrobolearn.simulators`
"""

import collections
import multiprocessing
import os

import cv2
import time

from pyrobolearn.utils.converter import QuaternionListConverter
# from pyrobolearn.utils.heightmap_generator import *  # TODO: problem with gdal installation
from pyrobolearn.utils import hasMethod, hasVariable, isClass

from pyrobolearn.robots import Robot, robot_names_to_classes
# from pyrobolearn.tools.bridges.bridge import Bridge

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class WorldCamera(object):
    r"""World camera.

    Camera that looks at the world (only available in the simulator).

    The following operations carried out (in the given order) by OpenGL in order to display images seen by the
    camera are:
    * M: Model space --> World space. This transforms the coordinates of each model described in their own local
     frame :math:`[x_{l}, y_{l}, z_{l}, 1]` to world coordinates :math:`[x_{w}, y_{w}, z_{w}, 1]`.
    * V: World space --> View space. This transforms world coordinates :math:`[x_{w}, y_{w}, z_{w}, 1]` into eye
     coordinates :math:`[x_{e}, y_{e}, z_{e}, 1]`. That is, it rotates and translates the world such that it is in
     front of the camera.
    * P: View space --> Projection space. Transforms the eye coordinates into clip coordinates using an orthographic
     or perspective projection. The new coordinates are given by :math:`[x_{c}, y_{c}, z_{c}, w_{c}]`. This is not
     normalized, i.e. w_{c} is not equal to 1. See next operation.
    * norm: Screen space --> NDC space. This normalizes the previous clipped coordinates into
     Normalized Device Coordinates (NDC) where each coordinate is normalized and is between -1 and 1. That is,
     we now have :math:`[x_{n}, y_{n}, z_{n}, 1] = [x_c/w_c, y_c/w_c, z_c, w_c/w_c]`
    * Vp: NDC space --> Screen space. Finally, this maps the previous normalized clip coordinates to pixel
     coordinates :math:`[x_{s}, y_{s}, z_{s}, 1]` where :math:`x_s` (:math:`y_s`) is between 0 and the width
     (height) of the screen respectively, and :math:`z_s` represents the depth which is between 0 and 1.

    References:
        [1] http://www.codinglabs.net/article_world_view_projection_matrix.aspx
        [2] https://learnopengl.com/Getting-started/Coordinate-Systems
        [3] http://www.thecodecrate.com/opengl-es/opengl-transformation-matrices/
        [4] http://learnwebgl.brown37.net/08_projections/projections_perspective.html
    """

    def __init__(self, simulator):
        self.sim = simulator

    def __repr__(self):
        return self.__class__.__name__

    @property
    def info(self):
        """
        Return all the information about the camera.
        """
        return self.getDebugVisualizerCamera(convert=False)

    # alias
    def getDebugVisualizerCamera(self, convert=True):
        """
        Return all the information provided by the camera.

        Args:
            convert (bool): if True, it will convert the lists into numpy vectors and matrices

        Returns:
            width (int): width of the camera image in pixels
            height (int): height of the camera image in pixels
            viewMatrix (float[16], float[4x4]): view matrix of the camera
            projectionMatrix (float[16], float[4x4]): projection matrix of the camera
            cameraUp (float[3]): up axis of the camera, in Cartesian world space coordinates
            cameraForward (float[3]): forward axis of the camera, in Cartesian world space coordinates
            horizontal (float[3]): TBD. This is a horizontal vector that can be used to generate rays (for mouse
                picking or creating a simple ray tracer for example)
            vertical (float[3]): TBD.This is a vertical vector that can be used to generate rays(for mouse picking
                or creating a simple ray tracer for example).
            yaw (float): yaw angle of the camera, in Cartesian local space coordinates
            pitch (float): pitch angle of the camera, in Cartesian local space coordinates
            dist (float): distance between the camera and the camera target
            target (float[3]): target of the camera, in Cartesian world space coordinates

        """
        if convert:
            width, height, viewMatrix, projectionMatrix, cameraUp,\
                cameraForward, horizontal, vertical, yaw, pitch, dist, target = self.sim.getDebugVisualizerCamera()

            # convert
            viewMatrix = np.array(viewMatrix).reshape(4, 4).T
            projectionMatrix = np.array(projectionMatrix).reshape(4, 4).T
            cameraUp = np.array(cameraUp)
            cameraForward = np.array(cameraForward)
            horizontal = np.array(horizontal)
            vertical = np.array(vertical)
            target = np.array(target)
            yaw, pitch = np.deg2rad(yaw), np.deg2rad(pitch)

            # return
            return width, height, viewMatrix, projectionMatrix, cameraForward, cameraForward, horizontal, vertical,\
                    yaw, pitch, dist, target
        return self.sim.getDebugVisualizerCamera()

    @property
    def width(self):
        """
        Return the width of the pictures (in pixel)
        """
        return self.sim.getDebugVisualizerCamera()[0]

    @property
    def height(self):
        """
        Return the height of the pictures (in pixel)
        """
        return self.sim.getDebugVisualizerCamera()[1]

    @property
    def V(self):
        """
        Return the view matrix, which maps from the world to the view space.
        """
        viewMatrix = self.sim.getDebugVisualizerCamera()[2]
        return np.array(viewMatrix).reshape(4, 4).T

    # alias
    viewMatrix = V

    @property
    def Vinv(self):
        """
        Return the inverse of the view matrix
        """
        return np.linalg.inv(self.V)

    @property
    def P(self):
        """
        Return the projection matrix, which maps from the view to the projected/clipped space.
        """
        projectionMatrix = self.sim.getDebugVisualizerCamera()[3]
        return np.array(projectionMatrix).reshape(4, 4).T

    # alias
    projectionMatrix = P

    @property
    def Pinv(self):
        """
        Return the inverse of the projection matrix
        """
        return np.linalg.inv(self.P)

    @property
    def Vp(self):
        """
        Return the viewport matrix, which maps from the normalized clip coordinates to pixel coordinates.
        """
        width, height = self.sim.getDebugVisualizerCamera()[:2]
        return np.array([[width / 2, 0, 0, width / 2],
                         [0, height / 2, 0, height / 2],
                         [0, 0, 0.5, 0.5],
                         [0, 0, 0, 1]])

    viewportMatrix = Vp

    @property
    def Vp_inv(self):
        """
        Return the inverse of the viewport matrix.
        """
        return np.linalg.inv(self.Vp)

    def getMatrices(self, inverse=False):
        """
        Return the view, projection, and viewport matrices.
        """
        width, height, V, P = self.sim.getDebugVisualizerCamera()[:4]
        V = np.array(V).reshape(4, 4).T
        P = np.array(P).reshape(4, 4).T
        Vp = np.array([[width / 2, 0, 0, width / 2],
                         [0, height / 2, 0, height / 2],
                         [0, 0, 0.5, 0.5],
                         [0, 0, 0, 1]])
        if inverse:
            Vinv = np.linalg.inv(V)
            Pinv = np.linalg.inv(P)
            Vpinv = np.linalg.inv(Vp)
            return V, P, Vp, Vinv, Pinv, Vpinv
        return V, P, Vp

    @property
    def upVector(self):
        """
        Return the up axis of the camera in the Cartesian world space coordinates
        """
        return np.array(self.sim.getDebugVisualizerCamera()[4])

    @property
    def forwardVector(self):
        """
        Return the forward axis of the camera in the Cartesian world space coordinates.
        """
        return np.array(self.sim.getDebugVisualizerCamera()[5])

    def getVectors(self):
        """
        Return the forward, up, and lateral vectors of the camera.
        """
        upVector, forwardVector = self.sim.getDebugVisualizerCamera()[4:6]
        upVector, forwardVector = np.array(upVector), np.array(forwardVector)
        lateralVector = np.cross(forwardVector, upVector)
        return forwardVector, upVector, lateralVector

    @property
    def yaw(self):
        """
        Return the yaw angle of the camera in radian
        """
        return np.deg2rad(self.sim.getDebugVisualizerCamera()[8])

    @property
    def pitch(self):
        """
        Return the pitch angle of the camera.
        """
        return np.deg2rad(self.sim.getDebugVisualizerCamera()[9])

    @property
    def dist(self):
        """
        Return the distance between the camera and the camera target.
        """
        return self.sim.getDebugVisualizerCamera()[10]

    @property
    def targetPosition(self):
        """
        Return the target of the camera in the Cartesian world space coordinates.
        """
        return np.array(self.sim.getDebugVisualizerCamera()[11])

    @targetPosition.setter
    def targetPosition(self, pos):
        yaw, pitch, dist = self.sim.getDebugVisualizerCamera()[-4:-1]
        self.sim.resetDebugVisualizerCamera(dist, yaw, pitch, pos)

    @property
    def position(self):
        """
        Return the current position of the camera in the Cartesian world space coordinates.
        """
        Vinv = np.linalg.inv(self.V) # compute inverse of the view matrix
        position = Vinv[:3,3] # the last column is the current position of the camera
        return position

    @position.setter
    def position(self, pos):
        self.sim.resetDebugVisualizerCamera(dist, yaw, pitch, targetPos)

    @property
    def orientation(self):
        # based on forwardVector and upVector
        pass

    @orientation.setter
    def orientation(self, orientation):
        pass

    def setYawPitch(self, yaw, pitch, radian=True):
        if radian:
            yaw, pitch = np.rad2deg(yaw), np.rad2deg(pitch)
        dist, targetPos = self.sim.getDebugVisualizerCamera()[-2:]
        self.sim.resetDebugVisualizerCamera(dist, yaw, pitch, targetPos)

    def addYawPitch(self, dyaw, dpitch, radian=True):
        yaw, pitch, dist, targetPos = self.sim.getDebugVisualizerCamera()[-4:]
        if radian:
            dyaw, dpitch = np.rad2deg(dyaw), np.rad2deg(dpitch)
        yaw += dyaw
        pitch += dpitch
        self.sim.resetDebugVisualizerCamera(dist, yaw, pitch, targetPos)

    def getRGBImage(self):
        """
        Return the captured RGB image.
        """
        return self.getRGBAImage()[:, :, :3]

    def getRGBAImage(self):
        """
        Return the captured RGBA image. 'A' stands for alpha channel (for opacity/transparency)
        """
        width, height, viewMatrix, projectionMatrix = self.sim.getDebugVisualizerCamera()[:4]
        img = np.array(self.sim.getCameraImage(width, height, viewMatrix, projectionMatrix)[2])
        img = img.reshape(width, height, 4)  # RGBA
        return img

    def getDepthImage(self):
        """
        Return the depth image.
        """
        width, height, viewMatrix, projectionMatrix = self.sim.getDebugVisualizerCamera()[:4]
        img = np.array(self.sim.getCameraImage(width, height, viewMatrix, projectionMatrix)[3])
        img = img.reshape(width, height)
        return img

    def getRGBADImage(self, concatenate=True):
        """
        Return the RGBA and depth images.
        """
        width, height, viewMatrix, projectionMatrix = self.sim.getDebugVisualizerCamera()[:4]
        rgba, depth = self.sim.getCameraImage(width, height, viewMatrix, projectionMatrix)[2:4]
        rgba = np.array(rgba).reshape(width, height, 4)
        depth = np.array(depth).reshape(width, height)
        if concatenate:
            return np.dstack((rgba, depth))
        return (rgba, depth)

    def screenToWorld(self, x_screen, Vp_inv=None, P_inv=None, V_inv=None):
        """
        Return the corresponding coordinates in the Cartesian world space from the coordinates of a point
        on the screen.

        Args:
            x_screen (float[4]): augmented vector coordinates of a point on the screen
            Vp_inv (float[4,4])): inverse of viewport matrix
            P_inv (float[4,4]): inverse of projection matrix
            V_inv (float[4,4]): inverse of view matrix

        Returns:
            float[4]: augmented vector coordinates of the corresponding point in the world
        """
        if Vp_inv is None: Vp_inv = self.Vp_inv
        if P_inv is None: P_inv = self.Pinv
        if V_inv is None: V_inv = self.Vinv

        x_ndc = Vp_inv.dot(x_screen)
        x_ndc[1] = -x_ndc[1]  # invert y-axis
        x_ndc[2] = -x_ndc[2]  # invert z-axis
        x_eye = P_inv.dot(x_ndc)
        x_eye = x_eye / x_eye[3]  # normalize
        x_world = V_inv.dot(x_eye)
        return x_world

    def worldToScreen(self, x_world, V=None, P=None, Vp=None):
        """
        Return the corresponding screen coordinates from a 3D point in the world.

        Args:
            x_world (float[4]): augmented vector coordinates of a point in the Cartesian world space
            V (float[4,4], None): view matrix
            P (float[4,4], None): projection matrix
            Vp (float[4,4], None): viewport matrix

        Returns:
            float[4]: augmented vector coordinates of the corresponding point on the screen
        """
        if V is None: V = self.V
        if P is None: P = self.P
        if Vp is None: Vp = self.Vp

        x_eye = V.dot(x_world)
        x_clip = P.dot(x_eye)
        x_ndc = x_clip / x_clip[3]  # normalize
        x_ndc[1] = -x_ndc[1]  # invert y-axis (as y pointing upward in projection but should point downward in screen)
        x_ndc[2] = -x_ndc[2]  # invert z-axis (to get right-handed coord. system, -1=close and 1=far)
        x_screen = Vp.dot(x_ndc)    # for depth between 0(=close) and 1(=far)
        return x_screen


class World(object):
    r"""World class.

    The world contains all the objects that constitutes the world. This includes immovable objects such as the
    terrain/floor, walls, and so on, as well as movable objects such as the various robots, etc.
    Properties of the world are also defined here, such as gravity, friction, and other dynamical properties.

    The world is responsible to load the different objects part of the world and keeping a map of objects;
    based on where the agent(s) is(are), the objects will be removed or added from/to the simulator allowing it
    to run faster.

    It is independent of the simulator and environment used in RL, and is often provided as inputs to some
    `rewards/costs` and to the `environment`.
    A world can only be defined in a simulator. Because of this, robots (which normally should be part of the
    world) can be independent from this last one. This allows to use robots in reality where the world is already
    defined (i.e. the real world).

    For an excellent overview of available 3D models/scenes, check references [1, 2].

    References:
        [1] "3D Machine Learning": https://github.com/timzhang642/3D-Machine-Learning
        [2] Open3D: http://www.open3d.org/
    """

    def __init__(self, simulator, set_gravity=True):
        self.sim = simulator
        # self.sim.setAdditionalSearchPath(pybullet_data.getDataPath())

        self.robots = {}
        self.robot_init_states = {}
        self.movable_objects = {} # set()
        self.immovable_objects = {} # set()
        self.visual_objects = {} # set()

        self.visual_shapes = {}
        self.map = None
        self.floor_id = -1

        self.quaternion_converter = QuaternionListConverter(convention=1)

        # By default, set the gravity
        if set_gravity:
            self.setGravity()

        self.worldState = None

        # configure debug visualizer
        # self.sim.configureDebugVisualizer(self.sim.COV_ENABLE_GUI, 0)

        # interfaces and bridges
        self.interfaces = set([])
        self.bridges = []

    ##############
    # Properties #
    ##############

    @property
    def simulator(self):
        return self.sim

    @property
    def mainCamera(self):
        return self.getMainCamera()

    ########################
    # Operator Overloading #
    ########################

    def __repr__(self):
        return self.__class__.__name__

    def __contains__(self, item):
        """
        Check if the given item is in the world.

        Args:
            item (int, Object, Robot): if it is an integer, it will check if the given object id is in the world.

        Returns:
            bool: True if the world contains the given item
        """
        if not isinstance(item, int):
            item = item.id

        return (item in self.robots) or (item in self.movable_objects) or (item in self.immovable_objects) \
                or (item in self.visual_objects)

    ###########
    # Methods #
    ###########

    def setBridges(self, bridges):
        """
        This append the given bridges to various interfaces to the list of bridges.

        See `pyrobolearn.tools.interface` and `pyrobolearn.tools.bridge` for more information.

        Args:
            bridges (list, Bridge): list of bridges
        """
        if isinstance(bridges, collections.Iterable):
            for bridge in bridges:
                # if not isinstance(bridge, Bridge):
                #    raise TypeError("Expecting a list of bridges (must be an instance of Bridge)")
                if not hasMethod(bridge, 'step') and not hasVariable(bridge, 'interface'):
                    raise TypeError("Expecting bridge to have a `step` method and an `interface` variable")
                if not hasMethod(bridge.interface, 'step'):
                    raise TypeError("Expecting the bridge.interface to have a `step` method")
                self.bridges.append(bridge)
                self.interfaces.add(bridge.interface)
        # elif isinstance(bridges, Bridge):
        elif hasMethod(bridges, 'step') and hasVariable(bridges, 'interface') and hasMethod(bridges.interface, 'step'):
            self.bridges.append(bridges)
            self.interfaces.add(bridges.interface)
        else:
            raise TypeError("Expecting a bridge (instance of Bridge) or a list of instances of Bridge")

    def setRealTimeSimulation(self, enable=True):
        """
        Simulate in real-time. This is wrapper around the original `setRealTimeSimulation` method.
        From the user-guide (https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA):
        "By default, the physics server will not step the simulation, unless you explicitly send a 'stepSimulation'
        command. This way you can maintain control determinism of the simulation. It is possible to run the
        simulation in real-time by letting the physics server automatically step the simulation according to
        its real-time-clock (RTC) using the setRealTimeSimulation command. If you enable the real-time simulation,
        you don't need to call 'stepSimulation'.

        Note that setRealTimeSimulation has no effect in DIRECT mode: in DIRECT mode the physics server and client
        happen in the same thread and you trigger every command. In GUI mode and in Virtual Reality mode, and TCP/UDP
        mode, the physics server runs in a separate thread from the client (PyBullet), and setRealTimeSimulation
        allows the physicsserver  thread to add additional calls to stepSimulation."

        Args:
            enable (bool): If True, it enables the real-time simulation. If False, it disables it.
        """
        self.sim.setRealTimeSimulation(enable)

    def save(self, filename=None):
        """
        Save the world in the given filename. If the filename is None, it will save it in memory (RAM).
        """
        # save approximate world state on the disk
        # self.sim.saveWorld(filename)

        if filename is None:
            # save world state in memory (RAM)
            self.worldState = self.sim.saveState()
        else:
            # save world state on the disk
            self.sim.saveBullet(filename)
            self.worldState = filename
        return self.worldState

    def reset(self):
        """
        Reset the world. Put back each object where they were when added to the world.
        """
        # save to current instance if not already saved
        if self.worldState is None:
            self.worldState = self.save()

        # reset world to a previous instance
        if isinstance(self.worldState, int):
            # load world state from memory
            self.sim.restoreState(self.worldState)
            # reset the robot
            # for robot in self.robots.values():
            #     # robot.resetBasePositionAndOrientation()
            #     # robot.resetBaseVelocity()
            #     robot.resetJointStates()
            #     robot.setJointInitPositions()
        elif isinstance(self.worldState, str):
            # load world state from the disk
            self.sim.restoreState(fileName=self.worldState)
        else:
            # reset simulation: remove all objects from the world and reset the world to initial conditions
            self.sim.resetSimulation()

    def resetSimulator(self):
        """
        Reset the simulator.
        """
        self.sim.resetSimulation()

    def step(self, sleep_dt=None):
        """
        Perform one step for the interfaces and bridges, and one step in the world/simulator.
        """
        for interface in self.interfaces:
            interface.step()
        for bridge in self.bridges:
            bridge.step()
        self.sim.stepSimulation()
        if sleep_dt is not None:
            time.sleep(sleep_dt)

    def setGravity(self, xyz=(0.,0.,-9.81)):
        """
        Set the given gravity.

        Args:
            xyz (float[3]): gravity (acceleration) along the 3 axis.
        """
        x,y,z = xyz
        self.sim.setGravity(x, y, z)

    def getMainCamera(self):
        """
        Return the main camera of the simulator.
        """
        return WorldCamera(self.sim)

    def loadRobot(self, robot, position=None, orientation=None, useFixedBase=None):
        """
        Load the robot into the world. If the robot parameter is a known robot name or the path to the urdf file,
        it will create a `Robot` instance and return it. If the robot is already an instance of `Robot` it will
        just add it to the list of objects present in the world.

        Args:
            robot (Robot, str, class): the robot instance or name. For the list of possible robot names, import
                `implemented_robots` from `pyrobolearn.robots` module.
            position (float[3], None): position of the robot. If None, it will take the default position.
            orientation (float[4], None): orientation of the robot. If None, it will be the default orientation.
            useFixedBase (bool, None): if True, it will fix the robot's base. If None, it will be the default option.

        Return:
            Robot: instance of the Robot class
        """
        if isinstance(robot, Robot):  # the robot is already loaded, then add it to the list
            pass

        elif isinstance(robot, str):  # robot's name
            robot = robot.lower()

            if robot in robot_names_to_classes:
                robot_class = robot_names_to_classes[robot]
                robot = robot_class(self.sim, init_pos=position, init_orient=orientation, useFixedBase=useFixedBase)

            else:  # robot is the path to the urdf
                robot = Robot(self.sim, urdf_path=robot, init_pos=position, init_orient=orientation,
                              useFixedBase=useFixedBase)

        elif isClass(robot):  # robot class
            robot = robot(self.sim, init_pos=position, init_orient=orientation, useFixedBase=useFixedBase)

        else: # unknown type
            raise TypeError('Unknown type for robot: {}. It must be a string or '
                            'an instance of Robot'.format(type(robot)))

        self.robots[robot.id] = robot
        self.robot_init_states[robot.id] = (robot.getJointPositions(), robot.getJointVelocities())
        return robot

    def isRobotId(self, robotId):
        """
        Check if the given id is a robot id.

        Args:
            robotId (int): the possible robot id

        Returns:
            bool: True if the id is a robot id, False otherwise
        """
        return robotId in self.robots

    def getRobot(self, robotId):
        """
        Return the robot object (instance of Robot) associated to the given robot id.

        Args:
            robotId (int): unique id of the robot

        Raises:
            KeyError: if the given robot id is not in the world.

        Returns:
            Robot: robot instance
        """
        return self.robots[robotId]

    def resetRobots(self):
        """
        Reset the joint states of each robot
        """
        for robotId, robot in self.robots.items():
            pos, vel = self.robot_init_states[robotId]
            for jointId, p, v in zip(robot.joints, pos, vel):
                self.sim.resetJointState(robotId, jointId, p, v)

    def loadURDF(self, filename, position, orientation, useFixedBase=False, scaling=1., objectName=None):
        """
        Load URDF specified by the given path. This is basically a wrapper around the simulator's `loadURDF` method.

        Args:
            filename (str): path to the URDF file
            position (float[3]): position of the object described in the URDF
            orientation (float[4]): orientation represented as a quaternion
            usedFixedBase (bool): if the base of the object should be fixed or not
            scaling (float): scale factor for the object
            objectName (str, None): name of the object. If None, it will extract it from the URDF.

        Returns:
            int: unique id of the loaded body.
        """
        obj = self.sim.loadURDF(filename, position, orientation, useFixedBase=useFixedBase, globalScaling=scaling)
        self.movable_objects[obj] = self.sim.getBodyInfo(obj) if objectName is None else objectName
        return obj

    def loadSDF(self, filename, scaling=1.):
        """
        Load the given SDF file; this will thus load all the object described in a SDF file.

        Args:
            filename (str): path to the SDF file
            scaling (float): scale factor for the object

        Returns:
            list(int): list of ids
        """
        objects = self.sim.loadSDF(filename, globalScaling=scaling)
        for obj in objects:
            self.movable_objects[obj] = self.sim.getBodyInfo(obj)
        return objects

    def loadMJCF(self, filename, scaling=1.):
        """
        Load the given MJCF file; this will thus load all the object described in a MJCF file.

        Args:
            filename (str): path to the MJCF file
            scaling (float): scale factor for the object

        Returns:
            list(int): list of ids
        """
        objects = self.sim.loadMJCF(filename) #, globalScaling=scaling)
        for obj in objects:
            self.movable_objects[obj] = self.sim.getBodyInfo(obj)
        return objects

    def _loadSDForURDF(self, path, position, orientation, scaling, objectType=None):
        extension_name = path.split('.')[-1]
        if extension_name == 'urdf':
            object_id = self.sim.loadURDF(path, position, orientation, globalScaling=scaling)
            self.movable_objects[object_id] = 'urdf' if objectType is None else objectType
        elif extension_name == 'sdf':
            object_id = self.sim.loadSDF(path, globalScaling=scaling) # list of ids
            for i in object_id: # assume for now that the objects are movable...
                self.movable_objects[i] = 'sdf' if objectType is None else objectType
        else:
            raise ValueError('Extension name of the file is not known; this method only accepts URDF/SDF files.')
        return object_id

    def loadObject(self, objectType, path=None, position=(0,0,0), orientation=(0,0,0,1), scaling=1.):
        """
        Load the specified object. This is a method that allows you to quickly load stuffs however it is less
        accurate than other methods in this class.

        Args:
            objectType (str): type of the object (name, 'sphere',
            path:
            position:
            orientation:
            scaling:

        Returns:
            int or int[]: object ids
        """
        # check if an object has already been loaded at that place.

        if path is not None:
            objectId = self._loadSDForURDF(path, position, orientation, scaling=1., objectType=objectType)
        else:
            if objectType == 'sphere':
                objectId = self.loadSphere(position)
            elif objectType == 'box':
                objectId = self.loadBox(position, orientation)
            elif objectType == 'cylinder':
                objectId = self.loadCylinder(position, orientation)
            elif objectType == 'capsule':
                objectId = self.loadCapsule(position, orientation)
            else:
                raise TypeError("Object type not known...")

        return objectId

    def moveObject(self, objectId, position=None, orientation=None):
        """
        Move the given object at the specified position and orientation.

        Args:
            objectId (int): object id
            position (float[3]): new position of the object. If None, it will keep the old position.
            orientation (float[4]): new orientation of the object. If None, it will keep the old orientation.

        Returns:
            None
        """
        if position is None:
            position = self.sim.getBasePositionAndOrientation(objectId)[0]
        if orientation is None:
            orientation = self.sim.getBasePositionAndOrientation(objectId)[1]
        self.sim.resetBasePositionAndOrientation(objectId, position, orientation)

    def applyForce(self, objectId, linkId=-1, force=(0.,0.,0.), position=None, frameFlag=2):
        """
        Apply the given force on the specified object or link of the object.

        Warnings:
            - after each simulation step, the external forces are cleared to 0.
            - this does not work when using `sim.setRealTimeSimulation(1)`.

        Args:
            objectId (int): object id to apply the force on
            linkId (int): link id to apply the force, if -1 it will apply the force on the base
            force (float[3]): Cartesian forces to be applied on the body
            position (float[3]): position on the link where the force is applied. If None, it is the center of mass
                of the object (or the link if specified)
            frameFlag (int): allows to specify the coordinate system of force/position. sim.LINK_FRAME (=1) for local
                link frame, and sim.WORLD_FRAME (=2) for world frame. By default, it is the world frame.

        Returns:
            None
        """
        if position is None:
            if linkId != -1:
                position = self.sim.getBasePositionAndOrientation(objectId)[0]
            else:
                position = self.sim.getLinkState(objectId, linkId)[0]
        self.sim.applyExternalForce(objectId, linkId, force, position, frameFlag)

    def getObjectColor(self, objectId):
        """
        Return the RGBA color of the given object.

        Args:
            objectId (int): object id

        Returns:
            float[4]: RGBA color
        """
        return self.sim.getVisualShapeData(objectId)[-1]

    def changeObjectColor(self, objectId, color=(1,1,1,1), linkId=-1):
        """
        Change the color of the given object.

        Args:
            objectId (int): object id
            color (float[4]): RGBA color
            linkId (int): link id

        Returns:
            None
        """
        self.sim.changeVisualShape(objectId, linkId, rgbaColor=color)

    def getObjectPosition(self, objectId):
        """
        Return the position of the given object.

        Args:
            objectId (int): object id

        Returns:
            float[3]: position of the object
        """
        return np.array(self.sim.getBasePositionAndOrientation(objectId)[0])

    def getObjectOrientation(self, objectId):
        """
        Return the orientation of the given object.

        Args:
            objectId (int): object id

        Returns:
            float[4]: orientation of the object
        """
        return np.array(self.sim.getBasePositionAndOrientation(objectId)[1])

    def getObjectVelocity(self, objectId):
        """
        Return the linear and angular velocities of the given object.

        Args:
            objectId (int): object id

        Returns:
            float[6]: linear and angular velocities of the object
        """
        lin_vel, ang_vel = self.sim.getBaseVelocity(objectId)
        return np.array(lin_vel + ang_vel)

    def getObjectLinearVelocity(self, objectId):
        """
        Return the linear velocity of the given object.

        Args:
            objectId (int): object id

        Returns:
            float[3]: linear velocity of the object
        """
        return np.array(self.sim.getBaseVelocity(objectId)[0])

    def getObjectAngularVelocity(self, objectId):
        """
        Return the angular velocity of the given object.

        Args:
            objectId (int): object id

        Returns:
            float[3]: angular velocity of the object
        """
        return np.array(self.sim.getBaseVelocity(objectId)[1])

    def hideObject(self, objectId):
        """
        Hide (visually) the given object; by making it transparent.

        Args:
            objectId (int): object id

        Returns:
            None
        """
        color = self.getObjectColor(objectId)
        color[-1] = 0.
        self.changeObjectColor(objectId, color=color)

    def showObject(self, objectId):
        """
        Show (visually) a hidden object; by making it opaque.

        Args:
            objectId (int): object id

        Returns:
            None
        """
        color = self.getObjectColor(objectId)
        color[-1] = 1.
        self.changeObjectColor(objectId, color=color)

    def removeObject(self, objectId):
        """
        Remove the object specified by its unique id from the world/simulator.

        Args:
            objectId (int, Robot): unique id of the object in the simulator.

        Returns:
            bool: True if succeeded, False if not. This method does not raise any errors.
        """
        if isinstance(objectId, Robot):
            objectId = objectId.id
        if objectId in self.robots:
            self.robots.pop(objectId)
        elif objectId in self.movable_objects:
            self.movable_objects.pop(objectId)
        elif objectId in self.immovable_objects:
            self.immovable_objects.pop(objectId)
        elif objectId in self.visual_objects:
            self.visual_objects.pop(objectId)
        else:
            return False

        self.sim.removeBody(objectId)
        return True

    def getObjectDimensions(self, objectId):
        """
        Return the object dimensions of the given object.

        Args:
            objectId (int): object id

        Returns:
            float[3]: dimensions of the object
        """
        return np.array(self.sim.getVisualShapeData[3])

    def changeObjectScale(self, objectId, scale=(1.,1.,1.)):
        """
        Change the scale of the given object; it changes the scale for the visual and collision shapes.

        Args:
            objectId (int): object id
            scale (float[3]): scaling factors in each direction

        Returns:
            None
        """
        # TODO: currently not possible in PyBullet
        raise NotImplementedError

    def getObjectAABB(self, objectId, linkId=-1):
        """
        Return the axis-aligned bounding box (AABB) in world space of the given object.

        Args:
            objectId (int): object id
            linkId (int): optional link id

        Returns:
            float[3]: coordinates in world space of the min corner of the AABB
            float[3]: coordinates in world space of the max corner of the AABB
        """
        bbMin, bbMax = self.sim.getAABB(objectId, linkId)
        return np.array(bbMin), np.array(bbMax)

    def getObjectIdsInAABB(self, bbMin, bbMax):
        """
        Get the list of object ids that have AABB overlap with a given AABB.

        Args:
            bbMin (float[3]): coordinates of the min corner of the bounding box
            bbMax (float[3]): coordinates of the max corner of the bounding box

        Returns:
            int[N]: list of object ids
        """
        overlapping_objects = self.sim.getOverlappingObjects(bbMin, bbMax)
        if overlapping_objects is None:
            return []
        return overlapping_objects

    def isThereAnObject(self, bbMin, bbMax, except_floor=True):
        """
        Return True if there is an object in the bounding box defined by bbMin and bbMax.

        Args:
            bbMin (float[3]): minimum coordinates of the bounding box
            bbMax (float[3]): maximum coordinates of the bounding box
            except_floor (bool): if the floor should be counted as an object

        Returns:
            bool: True if there is an object in the specified bounding box
        """
        objects = self.sim.getOverlappingObjects(bbMin, bbMax)
        if len(objects) > 2:
            return True
        if len(objects) == 0:
            return False
        idx = objects[0]
        if idx == self.floor_id and except_floor:
            return False
        return True

    def getClosestObject(self): # Not possible for now
        raise NotImplementedError

    def getClosestObjects(self, radius): # Not possible for now
        raise NotImplementedError

    def loadFloor(self, scaling=1.):
        """
        Load a basic floor in the world.

        Args:
            scaling (float): scaling for the floor.

        Returns:
            int: unique id of the floor in the world
        """
        # self.floor_id = self.sim.loadURDF('plane100.urdf', useFixedBase=True, globalScaling=scaling)
        self.floor_id = self.sim.loadURDF('plane.urdf', useFixedBase=True, globalScaling=scaling)
        return self.floor_id

    def loadTerrain(self, heightmap, position=(0.,0.,0.), scaling=1., replace_floor=True):
        """
        Load the given terrain/heightmap.

        Args:
            heightmap (str, np.array[heigth,width]): path to the urdf, sdf, xml, or obj file of the terrain. It can
                also be the path to a heightmap in tif, jpg, or png format. Alternatively, it can represents
                the heightmap as a 2D numpy array where the values represent the height in meters.
            position (float[3]): position of the terrain. By default, origin of the world.
            scaling (float): scaling factor of the terrain
            replace_floor (bool): if True, it will replace the existing floor. Be careful, that it can cause
                problems with collision.

        Returns:
            int: unique id of the terrain.

        Modules to create mesh files (.obj):
        - `mayavi`: https://docs.enthought.com/mayavi/mayavi/
        - `openmesh`: https://www.openmesh.org/media/Documentations/OpenMesh-6.2-Documentation/a00036.html
        - `bpy`: Blender python API - https://docs.blender.org/api/current/
        """
        if self.floor_id > -1:  # there is already a floor defined
            if replace_floor:
                self.sim.removeBody(self.floor_id)

        if heightmap[-4:] == 'obj':  # obj (mesh)
            self.floor_id = self.loadMesh(heightmap, position, mass=0., scale=[scaling]*3,
                                          flags=self.sim.GEOM_FORCE_CONCAVE_TRIMESH, objectType='terrain')
        elif heightmap[-4:] == '.sdf':  # SDF
            self.floor_id = self.loadSDF(filename=heightmap, scaling=scaling)
        elif heightmap[-4:] == '.xml':  # MJCF
            self.floor_id = self.loadMJCF(filename=heightmap, scaling=scaling)
        elif heightmap[-5:] == '.urdf':  # URDF
            self.floor_id = self.sim.loadURDF(heightmap, position, useFixedBase=True, globalScaling=scaling)
        else: # heightmap (.tif, .jpg, .png, etc)
            def create_mesh(heightmap):
                # create 3D mesh
                # create3DMesh(heightmap, filename=, subsample=, interpolate_fct=)
                pass

            # create process to create the 3D mesh
            process = multiprocessing.Process(target=create_mesh, args=(heightmap,))
            process.start()
            process.join()

            # remove mesh from memory
            os.remove(filename + '.obj')  # remove mesh from memory
            # os.remove(filename + '.mtl')

            # apply the given texture if provided
            if isinstance(texture, str):
                texture = self.sim.loadTexture(texture)
                self.sim.changeVisualShape(heightmap, -1, textureUniqueId=texture)

        return self.floor_id

    def loadHeightmap(self, heightmap, texture=None, position=(0.,0.,0.), scale=1.):
        """
        Load a heightmap for the terrain.

        Args:
            heightmap (str, np.ndarray[M,M]): if string, filename containing the heightmap in  the png, jpg, obj format
                if a 2D numpy arrays, the values represent the height in meters.

        Returns:
            int: unique id of the floor

        Modules to create mesh files (.obj):
        - `mayavi`: https://docs.enthought.com/mayavi/mayavi/
        - `openmesh`: https://www.openmesh.org/media/Documentations/OpenMesh-6.2-Documentation/a00036.html
        - `bpy`: Blender python API - https://docs.blender.org/api/current/
        """

        extension, filename = None, 'generated_file'

        # if string, get the extension and name of the file
        if isinstance(heightmap, str):
            extension = heightmap.split('.')[-1]
            filename = heightmap[:-4]

            # if picture (png/jpg), load 2D array (grayscale values)
            if extension == 'png' or extension == 'jpg':
                heightmap = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

        # if 2D numpy array, create the mesh (in the .obj format)
        if isinstance(heightmap, np.ndarray):
            heightmap = heightmap.astype(np.float)
            mlab.surf(heightmap)
            mlab.savefig(filename + '.obj')
            mlab.close()
            # TODO: set the map of the world

        elif extension != 'obj':
            raise ValueError("Expecting heightmap in a png/jpg/obj format")

        # load the mesh of the terrain
        heightmap = self.loadTerrain(filename, position=position, scaling=scale)

        # change the dynamic properties of the terrain based on the given type (grass, mud, bumpy
        # WARNING: only 1 type can be specified. Currently, it is not possible to have different dynamic properties
        # for different parts of the terrain. Loading multiple terrains is currently not supported (there is only
        # 1 unique id for the floor).

        # apply the given texture if provided
        if isinstance(texture, str):
            texture = p.loadTexture(texture)
            self.sim.changeVisualShape(heightmap, -1, textureUniqueId=texture)

        # remove mesh from memory
        os.remove(filename + '.obj') # remove mesh from memory
        os.remove(filename + '.mtl')

        # replace the floor if there is already one present
        if self.floor_id > -1:
            self.sim.removeBody(self.floor_id)
        self.floor_id = heightmap

        return self.floor_id

    # aliases
    loadDEM = loadHeightmap

    def generateHeightmap(self, filename=None, algo=None):
        """
        Generate a heightmap (png) using the specified algorithm. We provide 4 algorithms to generate this last one:
        1. Random
        2. Diamond algorithm
        3.
        4.

        By default, the diamond algorithm is used.

        Args:
            filename (None, str): if not None, it will save the heightmap in the format specified by the filename.
                The format is inferred from the filename. Supported ones include '.png', '.jpg', and '.obj'.

        Returns:
            np.ndarray: heightmap
        """
        pass

    def generateTerrain(self, filename=None):
        pass

    def loadStadium(self, scaling=1.):
        """
        Load a stadium as the floor.

        Args:
            scaling (float): scaling for the stadium.

        Returns:
            int: unique id of the floor/stadium in the world
        """
        if self.floor_id > -1:
            self.sim.removeBody(self.floor_id)
        self.floor_id = self.sim.loadURDF('stadium.urdf', useFixedBase=True, globalScaling=scaling)
        return self.floor_id

    def loadJapaneseMonastery(self, scaling=1.):
        """
        Load a japanese monastery.

        Args:
            scaling (float): scaling for the japanese monastery

        Returns:
            int: unique id of the monastery
        """
        # replace the floor if there is already one present
        if self.floor_id > -1:
            self.sim.removeBody(self.floor_id)
        self.floor_id = self.sim.loadURDF('samurai.urdf', useFixedBase=True, globalScaling=scaling)
        return self.floor_id

    def loadBotLab(self, scaling=2.):
        return self.loadSDF('sdf/botlab/botlab.sdf', scaling=scaling)

    def loadStairs(self):
        pass

    def createCity(self):
        pass

    def createParkour(self):
        pass

    def createMountainWithPath(self):
        pass

    def loadTable(self, position, scaling=1.):
        """
        Load a table in the world.

        Args:
            position (float[3]): position of the table
            scaling (float): scaling for the table

        Returns:
            int: unique id of the table
        """
        table = self.sim.loadURDF('table/table.urdf', globalScaling=scaling)
        self.movable_objects[table] = 'table'
        return table

    def loadKivaShelf(self, scaling=1.):
        """
        Load a Kiva shelf.

        Args:
            scaling (float): scaling of the shelf

        Returns:
            int: unique id of the shelf
        """
        shelf = self.sim.loadSDF('kiva_shelf/model.sdf', globalScaling=scaling)[0]
        self.movable_objects[shelf] = 'shelf'
        return shelf

    def loadVisualSphere(self, position, radius=0.5, color=(1,1,1,1)):
        """
        Load a visual sphere in the world (only available in the simulator).

        Args:
            position (float[3]): position of the sphere in Cartesian world space (in meters)
            radius (float): radius of the sphere (in meters)
            color (int[4]): color of the sphere (by default: white and opaque)

        Returns:
            int: unique id of the visual sphere in the world
        """
        visualShape = self.sim.createVisualShape(self.sim.GEOM_SPHERE, radius=radius, rgbaColor = color)
        sphere = self.sim.createMultiBody(baseMass = 0.,
                                          baseVisualShapeIndex = visualShape,
                                          basePosition = position)
        self.visual_objects[sphere] = 'sphere'
        return sphere

    def loadSphere(self, position, mass=1., radius=0.5, color=(1,1,1,1)):
        """
        Load a sphere in the world (only available in the simulator).

        Args:
            position (float[3]): position of the sphere in Cartesian world space (in meters)
            mass (float): mass of the sphere (in kg). If mass = 0, the sphere won't move even if there is a collision.
            radius (float): radius of the sphere (in meters).
            color (int[4]): color of the sphere (by default: white and opaque)

        Returns:
            int: unique id of the sphere in the world
        """
        collisionShape = self.sim.createCollisionShape(self.sim.GEOM_SPHERE,
                                                       radius = radius)
        visualShape = self.sim.createVisualShape(self.sim.GEOM_SPHERE,
                                                 radius = radius,
                                                 rgbaColor = color)
        sphere = self.sim.createMultiBody(baseMass = mass,
                                          baseCollisionShapeIndex = collisionShape,
                                          baseVisualShapeIndex = visualShape,
                                          basePosition = position)
        if mass == 0.0:
            self.immovable_objects[sphere] = 'sphere'
        else:
            self.movable_objects[sphere] = 'sphere'

        return sphere

    def loadVisualBox(self, position, orientation=(0,0,0,1), dimensions=(1.,1.,1.), color=(1,1,1,1)):
        """
        Load a visual box in the world (only available in the simulator).

        Args:
            position (float[3]): position of the box in the Cartesian world space (in meters)
            orientation (float[4], np.quaternion): orientation of the box using quaternion. If np.quaternion then it
                uses the convention (w,x,y,z). If float[4], it uses the convention (x,y,z,w).
            dimensions (float[3]): dimensions of the box
            color (int[4]): color of the box (by default: white and opaque)

        Returns:
            int: unique id of the box in the world
        """
        dimensions = np.array(dimensions) / 2.
        visualShape = self.sim.createVisualShape(self.sim.GEOM_BOX, halfExtents=dimensions, rgbaColor = color)
        orientation = self.quaternion_converter.convertFrom(orientation) # convert to list
        box = self.sim.createMultiBody(baseMass = 0.,
                                       baseVisualShapeIndex = visualShape,
                                       basePosition = position,
                                       baseOrientation = orientation)
        self.visual_objects[box] = 'box'
        return box

    def loadBox(self, position, orientation=(0,0,0,1), mass=1., dimensions=(1.,1.,1.), color=(1,1,1,1)):
        """
        Load a box in the world (only available in the simulator).

        Args:
            position (float[3]): position of the box in the Cartesian world space (in meters)
            orientation (float[4], np.quaternion): orientation of the box using quaternion. If np.quaternion then it
                uses the convention (w,x,y,z). If float[4], it uses the convention (x,y,z,w).
            mass (float): mass of the box (in kg). If mass = 0, the box won't move even if there is a collision.
            dimensions (float[3]): dimensions of the box
            color (int[4]): color of the box (by default: white and opaque)

        Returns:
            int: unique id of the box in the world
        """
        dimensions = np.array(dimensions) / 2.
        collisionShape = self.sim.createCollisionShape(self.sim.GEOM_BOX, halfExtents=dimensions)
        visualShape = self.sim.createVisualShape(self.sim.GEOM_BOX, halfExtents=dimensions, rgbaColor=color)
        orientation = self.quaternion_converter.convertFrom(orientation)  # convert to list
        box = self.sim.createMultiBody(baseMass = mass,
                                       baseCollisionShapeIndex = collisionShape,
                                       baseVisualShapeIndex = visualShape,
                                       basePosition = position,
                                       baseOrientation = orientation)
        if mass == 0.0:
            self.immovable_objects[box] = 'box'
        else:
            self.movable_objects[box] = 'box'
        return box

    def loadVisualCylinder(self, position, orientation=(0,0,0,1), radius=0.5, height=1., color=(1,1,1,1)):
        """
        Load a visual cylinder in the world (only available in the simulator).

        Args:
            position (float[3]): position of the cylinder in the Cartesian world space (in meters)
            orientation (float[4], np.quaternion): orientation of the cylinder using quaternion.
                If np.quaternion then it uses the convention (w,x,y,z). If float[4], it uses the convention (x,y,z,w)
            radius (float): radius of the cylinder (in meters)
            height (float): height of the cylinder (in meters)
            color (int[4]): color of the cylinder (by default: white and opaque)

        Returns:
            int: unique id of the cylinder in the world
        """
        visualShape = self.sim.createVisualShape(self.sim.GEOM_CYLINDER, radius=radius, length=height,
                                                 rgbaColor = color)
        orientation = self.quaternion_converter.convertFrom(orientation) # convert to list
        cylinder = self.sim.createMultiBody(baseMass = 0.,
                                            baseVisualShapeIndex = visualShape,
                                            basePosition = position,
                                            baseOrientation = orientation)
        self.visual_objects[cylinder] = 'cylinder'
        return cylinder

    def loadCylinder(self, position, orientation=(0,0,0,1), mass=1., radius=0.5, height=1., color=(1,1,1,1)):
        """
        Load a cylinder in the world (only available in the simulator).

        Args:
            position (float[3]): position of the cylinder in the Cartesian world space (in meters)
            orientation (float[4], np.quaternion): orientation of the cylinder using quaternion.
                If np.quaternion then it uses the convention (w,x,y,z). If float[4], it uses the convention (x,y,z,w)
            mass (float): mass of the cylinder (in kg). If mass = 0, it won't move even if there is a collision.
            radius (float): radius of the cylinder (in meters)
            height (float): height of the cylinder (in meters)
            color (int[4]): color of the cylinder (by default: white and opaque)

        Returns:
            int: unique id of the cylinder in the world
        """
        collisionShape = self.sim.createCollisionShape(self.sim.GEOM_CYLINDER, radius=radius, height=height)
        visualShape = self.sim.createVisualShape(self.sim.GEOM_CYLINDER, radius=radius, length=height,
                                                 rgbaColor = color)
        orientation = self.quaternion_converter.convertFrom(orientation) # convert to list
        cylinder = self.sim.createMultiBody(baseMass = mass,
                                            baseCollisionShapeIndex = collisionShape,
                                            baseVisualShapeIndex = visualShape,
                                            basePosition = position,
                                            baseOrientation = orientation)
        if mass == 0.0:
            self.immovable_objects[cylinder] = 'cylinder'
        else:
            self.movable_objects[cylinder] = 'cylinder'
        return cylinder

    def loadVisualCapsule(self, position, orientation=(0,0,0,1), radius=0.5, height=1., color=(1,1,1,1)):
        """
        Load a visual capsule in the world (only available in the simulator).

        Args:
            position (float[3]): position of the capsule in the Cartesian world space (in meters)
            orientation (float[4], np.quaternion): orientation of the capsule using quaternion.
                If np.quaternion then it uses the convention (w,x,y,z). If float[4], it uses the convention (x,y,z,w)
            radius (float): radius of the capsule (in meters)
            height (float): height of the capsule (in meters)
            color (int[4]): color of the capsule (by default: white and opaque)

        Returns:
            int: unique id of the capsule in the world
        """
        height = height/2.
        visualShape = self.sim.createVisualShape(self.sim.GEOM_CAPSULE, radius=radius, length=height,
                                                 rgbaColor=color)
        orientation = self.quaternion_converter.convertFrom(orientation) # convert to list
        capsule = self.sim.createMultiBody(baseMass = 0.,
                                            baseVisualShapeIndex = visualShape,
                                            basePosition = position,
                                            baseOrientation = orientation)
        self.visual_objects[capsule] = 'capsule'
        return capsule

    def loadCapsule(self, position, orientation=(0,0,0,1), mass=1., radius=0.5, height=1., color=(1,1,1,1)):
        """
        Load a capsule in the world (only available in the simulator).

        Args:
            position (float[3]): position of the capsule in the Cartesian world space (in meters)
            orientation (float[4], np.quaternion): orientation of the capsule using quaternion.
                If np.quaternion then it uses the convention (w,x,y,z). If float[4], it uses the convention (x,y,z,w)
            mass (float): mass of the capsule (in kg). If mass = 0, it won't move even if there is a collision.
            radius (float): radius of the capsule (in meters)
            height (float): height of the capsule (in meters)
            color (int[4]): color of the capsule (by default: white and opaque)

        Returns:
            int: unique id of the capsule in the world
        """
        height = height / 2.
        collisionShape = self.sim.createCollisionShape(self.sim.GEOM_CAPSULE, radius=radius, height=height)
        visualShape = self.sim.createVisualShape(self.sim.GEOM_CAPSULE, radius=radius, length=height,
                                                 rgbaColor = color)
        orientation = self.quaternion_converter.convertFrom(orientation) # convert to list
        capsule = self.sim.createMultiBody(baseMass = mass,
                                            baseCollisionShapeIndex = collisionShape,
                                            baseVisualShapeIndex = visualShape,
                                            basePosition = position,
                                            baseOrientation = orientation)
        if mass == 0.0:
            self.immovable_objects[capsule] = 'capsule'
        else:
            self.movable_objects[capsule] = 'capsule'
        return capsule

    def loadVisualMesh(self, filename, position, orientation=(0,0,0,1), scale=(1.,1.,1.), color=(1,1,1,1),
                       objectType='mesh'):
        """
        Load a visual mesh in the world (only available in the simulator).

        Args:
            filename (str): path to file for the mesh. Currently, only Wavefront .obj. It will create convex hulls
                for each object (marked as 'o') in the .obj file.
            position (float[3]): position of the mesh in the Cartesian world space (in meters)
            orientation (float[4], np.quaternion): orientation of the mesh using quaternion.
                If np.quaternion then it uses the convention (w,x,y,z). If float[4], it uses the convention (x,y,z,w)
            scale (float[3]): scale the mesh in the (x,y,z) directions
            color (int[4]): color of the mesh (by default: white and opaque)

        Returns:
            int: unique id of the mesh in the world
        """
        visualShape = self.sim.createVisualShape(self.sim.GEOM_MESH, fileName=filename, meshScale=scale,
                                                 rgbaColor=color)
        orientation = self.quaternion_converter.convertFrom(orientation) # convert to list
        mesh = self.sim.createMultiBody(baseMass = 0.,
                                        baseVisualShapeIndex = visualShape,
                                        basePosition = position,
                                        baseOrientation = orientation)
        self.visual_objects[mesh] = objectType
        return mesh

    def loadMesh(self, filename, position, orientation=(0,0,0,1), mass=1., scale=(1.,1.,1.), color=(1,1,1,1),
                 flags=None, objectType='mesh'):
        """
        Load a mesh in the world (only available in the simulator).

        Args:
            filename (str): path to file for the mesh. Currently, only Wavefront .obj. It will create convex hulls
                for each object (marked as 'o') in the .obj file.
            position (float[3]): position of the mesh in the Cartesian world space (in meters)
            orientation (float[4], np.quaternion): orientation of the mesh using quaternion.
                If np.quaternion then it uses the convention (w,x,y,z). If float[4], it uses the convention (x,y,z,w)
            mass (float): mass of the mesh (in kg). If mass = 0, it won't move even if there is a collision.
            scale (float[3]): scale the mesh in the (x,y,z) directions
            color (int[4]): color of the mesh (by default: white and opaque)
            flags (int, None): if flag = `sim.GEOM_FORCE_CONCAVE_TRIMESH` (=1), this will create a concave static
                triangle mesh. This should not be used with dynamic/moving objects, only for static (mass=0) terrain.

        Returns:
            int: unique id of the mesh in the world
        """
        if flags is None:
            collisionShape = self.sim.createCollisionShape(self.sim.GEOM_MESH, fileName=filename, meshScale=scale)
        else:
            collisionShape = self.sim.createCollisionShape(self.sim.GEOM_MESH, fileName=filename, meshScale=scale,
                                                           flags=flags)
        visualShape = self.sim.createVisualShape(self.sim.GEOM_MESH, fileName=filename, meshScale=scale,
                                                 rgbaColor=color)
        orientation = self.quaternion_converter.convertFrom(orientation) # convert to list
        mesh = self.sim.createMultiBody(baseMass = mass,
                                        baseCollisionShapeIndex=collisionShape,
                                        baseVisualShapeIndex = visualShape,
                                        basePosition = position,
                                        baseOrientation = orientation)
        if mass == 0.0:
            self.immovable_objects[mesh] = objectType
        else:
            self.movable_objects[mesh] = objectType
        return mesh

    # The following commented code does not work currently because URDF_GEOM_PLANE is not set in Bullet
    # Note that a plane can be seen as a thin box.
    # def loadVisualPlane(self, position, orientation, normal=(0.,0.,1.), color=(1,1,1,1)):
    #     """
    #     Load a visual plane in the world (only available in the simulator).
    #
    #     Args:
    #         position (float[3]): position of the plane in the Cartesian world space (in meters)
    #         orientation (float[4], np.quaternion): orientation of the plane using quaternion.
    #             If np.quaternion then it uses the convention (w,x,y,z). If float[4], it uses the convention (x,y,z,w)
    #         normal (float[3]): normal to the plane
    #         color (int[4]): color of the plane (by default: white and opaque)
    #
    #     Returns:
    #         int: unique id of the plane in the world
    #     """
    #     visualShape = self.sim.createVisualShape(self.sim.GEOM_PLANE, planeNormal=normal, rgbaColor=color)
    #     orientation = self.quaternion_converter.convertFrom(orientation)  # convert to list
    #     plane = self.sim.createMultiBody(baseMass=0.,
    #                                     baseVisualShapeIndex=visualShape,
    #                                     basePosition=position,
    #                                     baseOrientation=orientation)
    #     self.visual_objects[plane] = 'plane'
    #     return plane
    #
    # def loadPlane(self, position, orientation, mass=1., normal=(0.,0.,1.), color=(1,1,1,1)):
    #     """
    #     Load a plane in the world (only available in the simulator).
    #
    #     Args:
    #         position (float[3]): position of the plane in the Cartesian world space (in meters)
    #         orientation (float[4], np.quaternion): orientation of the plane using quaternion.
    #             If np.quaternion then it uses the convention (w,x,y,z). If float[4], it uses the convention (x,y,z,w)
    #         mass (float): mass of the plane (in kg). If mass = 0, it won't move even if there is a collision.
    #         normal (float[3]): normal to the plane
    #         color (int[4]): color of the plane (by default: white and opaque)
    #
    #     Returns:
    #         int: unique id of the plane in the world
    #     """
    #     collisionShape = self.sim.createCollisionShape(self.sim.GEOM_PLANE, planeNormal=normal)
    #     visualShape = self.sim.createVisualShape(self.sim.GEOM_PLANE, planeNormal=normal, rgbaColor=color)
    #     orientation = self.quaternion_converter.convertFrom(orientation)  # convert to list
    #     plane = self.sim.createMultiBody(baseMass=mass,
    #                                      baseCollisionShapeIndex=collisionShape,
    #                                      baseVisualShapeIndex=visualShape,
    #                                      basePosition=position,
    #                                      baseOrientation=orientation)
    #     if mass == 0.0:
    #         self.immovable_objects[plane] = 'plane'
    #     else:
    #         self.movable_objects[plane] = 'plane'
    #     return plane

    # Temporary because the code above doesn't work
    def loadPlane(self, position=(0.,0.,0.), orientation=(0.,0.,0.,1.), scaling=1.):
        """
        Load a plane in the world (only available in the simulator)

        Args:
            position (float[3]): position of the plane
            orientation (float[4]): orientation of the plane
            scaling (float): scaling of the plane

        Returns:
            int: unique id of the plane
        """
        plane = self.sim.loadURDF('plane.urdf', position, orientation, useFixedBase=True, globalScaling=scaling)
        self.immovable_objects[plane] = plane
        return plane

    def loadVisualEllipsoid(self, position, orientation=(0,0,0,1), scale=(1., 1., 1.), color=(1,1,1,1)):
        """
        Load a visual ellipsoid (using a mesh) in the world (only available in the simulator).

        Args:
            position (float[3]): position in the Cartesian world space (in meters)
            orientation (float[4], np.quaternion): orientation using quaternion.
                If np.quaternion then it uses the convention (w,x,y,z). If float[4], it uses the convention (x,y,z,w)
            scale (float[3]): scale in the (x,y,z) directions
            color (int[4]): color (by default: white and opaque)

        Returns:
            int: unique id of the ellipsoid in the world
        """
        filename = os.path.dirname(__file__) + '/meshes/ellipsoid.obj'
        return self.loadVisualMesh(filename, position, orientation, scale=scale, color=color)

    def loadEllipsoid(self, position, orientation=(0,0,0,1), mass=1., scale=(1., 1., 1.), color=(1,1,1,1)):
        """
        Load a ellipsoid (using a mesh) in the world (only available in the simulator).

        Args:
            position (float[3]): position in the Cartesian world space (in meters)
            orientation (float[4], np.quaternion): orientation using quaternion.
                If np.quaternion then it uses the convention (w,x,y,z). If float[4], it uses the convention (x,y,z,w)
            mass (float): mass [kg]
            scale (float[3]): scale in the (x,y,z) directions
            color (int[4]): color (by default: white and opaque)

        Returns:
            int: unique id of the ellipsoid in the world
        """
        filename = os.path.dirname(__file__) + '/meshes/ellipsoid.obj'
        return self.loadMesh(filename, position, orientation, mass=mass, scale=scale, color=color)

    def loadVisualRightTriangularPrism(self, position, orientation=(0,0,0,1), scale=(1., 1., 1.), color=(1,1,1,1)):
        """
        Load a visual right triangular prism (using a mesh) in the world (only available in the simulator).

        Args:
            position (float[3]): position in the Cartesian world space (in meters)
            orientation (float[4], np.quaternion): orientation using quaternion.
                If np.quaternion then it uses the convention (w,x,y,z). If float[4], it uses the convention (x,y,z,w)
            scale (float[3]): scale in the (x,y,z) directions
            color (int[4]): color (by default: white and opaque)

        Returns:
            int: unique id of the triangular prism in the world
        """
        filename = os.path.dirname(__file__) + '/meshes/right_triangular_prism.obj'
        return self.loadVisualMesh(filename, position, orientation, scale=scale, color=color)

    def loadRightTriangularPrism(self, position, orientation=(0,0,0,1), mass=1., scale=(1., 1., 1.), color=(1,1,1,1)):
        """
        Load a right triangular prism (using a mesh) in the world (only available in the simulator).

        Args:
            position (float[3]): position in the Cartesian world space (in meters)
            orientation (float[4], np.quaternion): orientation using quaternion.
                If np.quaternion then it uses the convention (w,x,y,z). If float[4], it uses the convention (x,y,z,w)
            mass (float): mass [kg]
            scale (float[3]): scale in the (x,y,z) directions
            color (int[4]): color (by default: white and opaque)

        Returns:
            int: unique id of the triangular prism in the world
        """
        filename = os.path.dirname(__file__) + '/meshes/right_triangular_prism.obj'
        return self.loadMesh(filename, position, orientation, mass=mass, scale=scale, color=color)

    def loadVisualCone(self, position, orientation=(0,0,0,1), scale=(1., 1., 1.), color=(1,1,1,1)):
        """
        Load a visual cone (using a mesh) in the world (only available in the simulator).

        Args:
            position (float[3]): position in the Cartesian world space (in meters)
            orientation (float[4], np.quaternion): orientation using quaternion.
                If np.quaternion then it uses the convention (w,x,y,z). If float[4], it uses the convention (x,y,z,w)
            scale (float[3]): scale in the (x,y,z) directions
            color (int[4]): color (by default: white and opaque)

        Returns:
            int: unique id of the cone in the world
        """
        filename = os.path.dirname(__file__) + '/meshes/cone.obj'
        return self.loadVisualMesh(filename, position, orientation, scale=scale, color=color)

    def loadCone(self, position, orientation=(0,0,0,1), mass=1., scale=(1.,1.,1.), color=(1,1,1,1)):
        """
        Load a visual cone (using a mesh) in the world (only available in the simulator).

        Args:
            position (float[3]): position in the Cartesian world space (in meters)
            orientation (float[4], np.quaternion): orientation using quaternion.
                If np.quaternion then it uses the convention (w,x,y,z). If float[4], it uses the convention (x,y,z,w)
            scale (float[3]): scale in the (x,y,z) directions
            color (int[4]): color (by default: white and opaque)

        Returns:
            int: unique id of the cone in the world
        """
        filename = os.path.dirname(__file__) + '/meshes/cone.obj'
        return self.loadMesh(filename, position, orientation, mass=mass, scale=scale, color=color)

    def loadVisualArrow(self, position, orientation=(0,0,0,1), scale=(1.,1.,1.), color=(1,1,1,1)):
        pass

    def loadArrow(self, position, orientation=(0,0,0,1), mass=1., scale=(1.,1.,1.), color=(1,1,1,1)):
        pass

    def distribute_objects(self, distributor, objects):
        pass

    def getDynamicsInfo(self, bodyId, linkId):
        """
        Return the dynamics information about objects that are in the world.

        Args:
            bodyId: object unique id
            linkId: link index (or -1 for the base)

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
        info = self.sim.getDynamicsInfo(bodyId, linkId)
        local_inertia_diag, local_inertial_pos, local_inertial_orn = info[2:5]
        local_inertia_diag = np.array(local_inertia_diag)
        local_inertial_pos = np.array(local_inertial_pos)
        local_inertial_orn = np.array(local_inertial_orn)
        return info[:2] + [local_inertia_diag, local_inertial_pos, local_inertial_orn] + info[5:]

    def changeDynamics(self, lateralFriction, spinningFriction, rollingFriction, linearDamping, angularDamping,
                       contactStiffness=-1, contactDamping=-1):
        self.sim.changeDynamics(bodyUniqueId=self.floor_id, linkIndex=-1, lateralFriction=lateralFriction,
                                spinningFriction=spinningFriction, rollingFriction=rollingFriction,
                                linearDamping=linearDamping, angularDamping=angularDamping)


class BasicWorld(World):
    r"""Basic World class.

    It creates a basic world with a floor and set the gravity.
    """

    def __init__(self, simulator, floor_path=None, set_gravity=True, scaling=1., lateralFriction=.9,
                 spinningFriction=0., rollingFriction=0., contactStiffness=-1, contactDamping=-1):
        super(BasicWorld, self).__init__(simulator, set_gravity=set_gravity)

        if floor_path is None:
            self.loadFloor(scaling=scaling)
            self.changeDynamics(lateralFriction=lateralFriction, spinningFriction=spinningFriction,
                                rollingFriction=rollingFriction, linearDamping=0, angularDamping=0,
                                contactStiffness=30000)
            self.simulator.setDefaultContactERP(0.9)
        else:
            self.loadTerrain(floor_path, replace_floor=True)

        print(self.sim.getDynamicsInfo(self.floor_id, -1))


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
    from pyrobolearn.simulators import BulletSim

    # create simulator
    sim = BulletSim()

    # create world
    world = BasicWorld(sim)
    # world = World(sim)
    # world.loadBotLab()

    # world.loadSDF('/home/brian/Downloads/cobblestones_origin/model.sdf', scaling=1)

    # world.loadMesh('/home/brian/Downloads/cobblestones_origin/mesh/cobblestones.obj',
    #                position=[0, 0, 0],
    #                orientation=[.707, 0, 0, .707],
    #                mass=0.,
    #                scale=(1., 1., 1.),
    #                # color=[1, 0, 0, 1],
    #                flags=1)

    # world.loadMesh('/home/brian/save/code/random-terrain-generator-master/terrain.obj',
    #                position=[0, 0, -2],
    #                orientation=[.707, 0, 0, .707],
    #                mass=0.,
    #                scale=(.1, .1, .1),
    #                # color=[1, 0, 0, 1],
    #                flags=1)

    # world.loadMesh('bedroom.obj', [0, 0, 0], mass=0., color=[0.4, 0.4, 0.4, 1], flags=1) #, scale=(0.01, 0.01, 0.01))
    # world.loadMesh('mtsthelens.obj', [0, 0, -8], mass=0., color=[0.2, 0.5, 0.2, 1], flags=1, scale=(0.01,0.01,0.01))
    # world.loadMesh('meshes/terrain.obj', [0,0,0], mass=0., color=[1,1,1,1], flags=1)
    # world.loadMesh('/home/brian/heightmap_old.obj', [0,0,0], mass=0., scale=(0.1,0.1,0.01), color=[1,1,1,1], flags=1)
    # world.loadMesh('/home/brian/Downloads/arab_desert/desert.obj',
    #                position=[0, 0, -10.8], orientation=(0.707,0,0,0.707), mass=0., scale=(1, 1, 1),
    #                color=[1, 1, 1, 1], flags=1)
    # world.loadMesh('/home/brian/PhD-repos/pyrobolearn/tests/heightmap_test_exp.obj', [0, 0, 0], mass=0.,
    #                scale=(0.1, 0.1, 0.015),
    #                color=[1, 0, 0, 1],
    #                flags=1)

    # world.loadRobot('Cogimon', position=[0,0,1.])

    # world.loadRobot('coman', useFixedBase=False)
    # sphere = world.loadVisualSphere([1.,0,1.], color=(1,0,0,0.5))
    # world.loadVisualBox([-1,0,1], dimensions=[1.,1.,1.], color=[0,0,1,0.5])
    # world.loadCylinder([0, -1, 1], color=[1, 0, 0, 1])
    #  world.loadCapsule([0, 1, 1],  color=[1, 0, 0, 1])
    #  world.loadMesh(filename='duck.obj', [1, 0, 2], [0.707, 0, 0, 0.707], mass=0.1, scale=[0.1,0.1,0.1],
    #                 color=[1, 0, 0, 1])

    # from utils.orientation import RotX, RotY, RotZ, getQuaternionFromMatrix
    # R = RotZ(np.deg2rad(90.))
    # q = tuple(getQuaternionFromMatrix(R))
    # world.loadEllipsoid([0,0,2], orientation=q, mass=0, scale=[2.,1.,1.], color=(0,0,1,1))
    # world.loadCone([1,1,2])
    world.loadRightTriangularPrism([-1,-1,2])
    # floor = world.loadMesh(filename='box', [1, 0, 2], mass=0, color=(1, 1, 1, 1))

    # floor = world.loadFloor()
    # print(p.getDynamicsInfo(floor, -1))
    # floor = world.loadMesh([1,0,0], [0,0,0,1], filename='grass.obj', mass=0, color=(1,1,1,1))
    # texture = sim.loadTexture('grass.png')
    # sim.changeVisualShape(floor, -1, textureUniqueId=texture)

    # grass = sim.loadURDF('grass.urdf')
    # texture = sim.loadTexture('grass.png')
    # sim.changeVisualShape(grass, -1, textureUniqueId=texture)

    # vs = world.loadVisualSphere([0,0,2], radius=0.1, color=(0,0,1,1))

    # path = '/home/brian/bullet3/data/'
    # objects = p.loadMJCF(path+"MPL/mpl2.xml")

    # world.loadPlane([1.,0.,1.], [0,0,0,1], color=(1,0,0,1))

    T = 1000
    w = 2.*np.pi / T
    red = True
    # loop
    for t in count():
        # p = world.getObjectPosition(sphere)
        # p -= 0.001 * np.array([1.,0,0])
        p = np.array([np.cos(w*t), np.sin(w*t), 1.])
        # world.moveObject(sphere, p)
        # if t % T == 0:
        #     if red:
        #         world.changeObjectColor(sphere, (1,0,0,0.5))
        #     else:
        #         world.changeObjectColor(sphere, (0,0,1,0.5))
        #     red = not red
        # world.moveObject(sphere)
        world.step(sleep_dt=1./240)
