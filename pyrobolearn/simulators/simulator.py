#!/usr/bin/env python
"""Define the Simulator API.

All the simulators inherit from the interface defined here. This acts as a bridge between the simulator and
the PyRoboLearn framework. The signature of each method presents in this interface were inspired by the ones defined
in PyBullet [1,2], but in accordance with the PEP8 style guide [3].

Because the simulator is based on the PyBullet API and we want all the simulator APIs to be similar, all the other
simulators would have to be able to carry out operations such as querying the state of the robots, kinematics and
dynamics, etc.

Dependencies in PRL: None

References:
    [1] PyBullet: https://pybullet.org
    [2] PyBullet Quickstart Guide: https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA
    [3] PEP8: https://www.python.org/dev/peps/pep-0008/
"""


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class Simulator(object):
    r"""Simulator (abstract class)

    All the simulators inherits from the Simulator defined here. This acts as a bridge between the simulator and
    the PyRoboLearn framework. This avoids the PyRoboLearn framework to depends on a particular simulator.
    The signature of each method presents in this interface were inspired by the ones defined in PyBullet [1].

    Examples::
        sim = Bullet()
        sim = ROS_RBDL()
        sim = GazeboROS()

    References:
    [1] PyBullet: https://pybullet.org
    [2] PEP8: https://www.python.org/dev/peps/pep-0008/
    """

    # TODO: this is really bad to have attributes like that... It doesn't generalize well to other simulators...

    B3G_ALT = 65308
    B3G_BACKSPACE = 65305
    B3G_CONTROL = 65307
    B3G_DELETE = 65304
    B3G_DOWN_ARROW = 65298
    B3G_END = 65301
    B3G_F1 = 65280
    B3G_F10 = 65289
    B3G_F11 = 65290
    B3G_F12 = 65291
    B3G_F13 = 65292
    B3G_F14 = 65293
    B3G_F15 = 65294
    B3G_F2 = 65281
    B3G_F3 = 65282
    B3G_F4 = 65283
    B3G_F5 = 65284
    B3G_F6 = 65285
    B3G_F7 = 65286
    B3G_F8 = 65287
    B3G_F9 = 65288
    B3G_HOME = 65302
    B3G_INSERT = 65303
    B3G_LEFT_ARROW = 65295
    B3G_PAGE_DOWN = 65300
    B3G_PAGE_UP = 65299
    B3G_RETURN = 65309
    B3G_RIGHT_ARROW = 65296
    B3G_SHIFT = 65306
    B3G_UP_ARROW = 65297

    COV_ENABLE_DEPTH_BUFFER_PREVIEW = 14
    COV_ENABLE_GUI = 1
    COV_ENABLE_KEYBOARD_SHORTCUTS = 9
    COV_ENABLE_MOUSE_PICKING = 10
    COV_ENABLE_PLANAR_REFLECTION = 16
    COV_ENABLE_RENDERING = 7
    COV_ENABLE_RGB_BUFFER_PREVIEW = 13
    COV_ENABLE_SEGMENTATION_MARK_PREVIEW = 15
    COV_ENABLE_SHADOWS = 2
    COV_ENABLE_SINGLE_STEP_RENDERING = 17
    COV_ENABLE_TINY_RENDERER = 12
    COV_ENABLE_WIREFRAME = 3
    COV_ENABLE_Y_AXIS_UP = 11

    DIRECT = 2
    ER_BULLET_HARDWARE_OPENGL = 131072
    ER_NO_SEGMENTATION_MASK = 4
    ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX = 1
    ER_TINY_RENDERER = 65536
    ER_USE_PROJECTIVE_TEXTURE = 2

    GEOM_FORCE_CONCAVE_TRIMESH = 1
    GEOM_SPHERE = 2
    GEOM_CONCAVE_INTERNAL_EDGE = 2
    GEOM_BOX = 3
    GEOM_CYLINDER = 4
    GEOM_MESH = 5
    GEOM_PLANE = 6
    GEOM_CAPSULE = 7

    GUI = 1
    GUI_MAIN_THREAD = 8
    GUI_SERVER = 7
    IK_DLS = 0
    IK_HAS_JOINT_DAMPING = 128
    IK_HAS_NULL_SPACE_VELOCITY = 64
    IK_HAS_TARGET_ORIENTATION = 32
    IK_HAS_TARGET_POSITION = 16
    IK_SDLS = 1

    JOINT_FEEDBACK_IN_JOINT_FRAME = 2
    JOINT_FEEDBACK_IN_WORLD_SPACE = 1
    JOINT_FIXED = 4
    JOINT_GEAR = 6
    JOINT_PLANAR = 3
    JOINT_POINT2POINT = 5
    JOINT_PRISMATIC = 1
    JOINT_REVOLUTE = 0
    JOINT_SPHERICAL = 2

    KEY_IS_DOWN = 1
    KEY_WAS_RELEASED = 4
    KEY_WAS_TRIGGERED = 2

    LINK_FRAME = 1
    WORLD_FRAME = 2

    MAX_RAY_INTERSECTION_BATCH_SIZE = 16384

    VELOCITY_CONTROL = 0
    TORQUE_CONTROL = 1
    POSITION_CONTROL = 2
    PD_CONTROL = 3

    SENSOR_FORCE_TORQUE = 1
    SHARED_MEMORY = 3
    SHARED_MEMORY_KEY = 12347
    SHARED_MEMORY_KEY2 = 12348
    SHARED_MEMORY_SERVER = 9
    STATE_LOGGING_ALL_COMMANDS = 7
    STATE_LOGGING_CONTACT_POINTS = 5
    STATE_LOGGING_CUSTOM_TIMER = 9
    STATE_LOGGING_GENERIC_ROBOT = 1
    STATE_LOGGING_MINITAUR = 0
    STATE_LOGGING_PROFILE_TIMINGS = 6
    STATE_LOGGING_VIDEO_MP4 = 3
    STATE_LOGGING_VR_CONTROLLERS = 2
    STATE_LOG_JOINT_MOTOR_TORQUES = 1
    STATE_LOG_JOINT_TORQUES = 3
    STATE_LOG_JOINT_USER_TORQUES = 2
    STATE_REPLAY_ALL_COMMANDS = 8

    TCP = 5
    UDP = 4

    URDF_ENABLE_CACHED_GRAPHICS_SHAPES = 1024
    URDF_ENABLE_SLEEPING = 2048
    URDF_GLOBAL_VELOCITIES_MB = 256
    URDF_INITIALIZE_SAT_FEATURES = 4096
    URDF_USE_IMPLICIT_CYLINDER = 128
    URDF_USE_INERTIA_FROM_FILE = 2
    URDF_USE_MATERIAL_COLORS_FROM_MTL = 32768
    URDF_USE_MATERIAL_TRANSPARANCY_FROM_MTL = 65536
    URDF_USE_SELF_COLLISION = 8
    URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS = 32
    URDF_USE_SELF_COLLISION_EXCLUDE_PARENT = 16
    URDF_USE_SELF_COLLISION_INCLUDE_PARENT = 8192

    def __init__(self, render=True, **kwargs):
        self._render = render
        self.real_time = False
        self.kwargs = kwargs

        # main camera in the simulator
        self._camera = None

        # default timestep
        self.default_timestep = 1. / 240
        self.dt = self.default_timestep

        # TODO: this is really bad to have attributes like that... It doesn't generalize well to other simulators...
        # import pybullet
        # for attribute in dir(pybullet):
        #     if attribute[0].isupper():
        #         print('self.{} = {}'.format(attribute, getattr(pybullet, attribute)))

    ##############
    # Properties #
    ##############

    @property
    def version(self):
        """Return the version of the simulator."""
        return 0

    @property
    def gravity(self):
        """Return the gravity in the simulator."""
        return self.get_gravity()

    @gravity.setter
    def gravity(self, gravity):
        """Set the gravity in the simulator."""
        self.set_gravity(gravity)

    @property
    def camera(self):
        """Return the camera (yaw, pitch, distance, target_position) or None."""
        return self._camera

    @property
    def timestep(self):
        """Return the simulator time step."""
        return self.get_time_step()

    #############
    # Operators #
    #############

    def __str__(self):
        """Return a readable string about the class."""
        return self.__class__.__name__

    def __del__(self):
        """Close/Delete the simulator."""
        self.close()

    def __copy__(self):
        """Return a shallow copy of the simulator. This can be overridden in the child class."""
        return self.__class__(render=self._render, **self.kwargs)

    def __deepcopy__(self, memo={}):
        """Return a deep copy of the simulator. This can be overridden in the child class.

        Args:
            memo (dict): memo dictionary of objects already copied during the current copying pass.
        """
        # if the object has already been copied return the reference to the copied object
        if self in memo:
            return memo[self]

        # create a new copy of the simulator
        sim = self.__class__(render=self._render, **self.kwargs)

        memo[self] = sim
        return sim

    ###########
    # Methods #
    ###########

    # Simulators

    def reset(self, *args, **kwargs):
        """Reset the simulator."""
        pass

    def close(self):
        """Close the simulator."""
        pass

    def seed(self, seed=None):
        """Set the given seed in the simulator."""
        pass

    def step(self, sleep_time=0):
        """Perform a step in the simulator, and sleep the specified time.

        Args:
            sleep_time (float): time to sleep after performing one step in the simulation.
        """
        pass

    def is_rendering(self):
        """Return True if the simulator is in the render mode."""
        return self._render

    def reset_scene_camera(self, camera=None):
        """
        Reinitialize/Reset the scene view camera to the previous one.

        Args:
            camera (object): scene view camera. This is let to the user to decide what to do.
        """
        pass

    def render(self, enable=True):
        """Render the simulation.

        Args:
            enable (bool): If True, it will render the simulator by enabling the GUI.
        """
        self._render = enable

    def hide(self):
        """Hide the GUI."""
        self.render(False)

    def get_time_step(self):
        """Get the time step in the simulator.

        Returns:
            float: time step in the simulator
        """
        pass

    def set_time_step(self, time_step):
        """Set the time step in the simulator.

        Args:
            time_step (float): Each time you call 'step' the time step will proceed with 'time_step'.
        """
        pass

    def set_real_time(self, enable=True):
        """Enable real time in the simulator.

        Args:
            enable (bool): If True, it will enable the real-time simulation. If False, it will disable it.
        """
        pass

    def pause(self):
        """Pause the simulator if in real-time."""
        pass

    def unpause(self):
        """Unpause the simulator if in real-time."""
        pass

    def get_physics_properties(self):
        """Get the physics engine parameters."""
        pass

    def set_physics_properties(self, *args, **kwargs):
        """Set the physics engine parameters."""
        pass

    def start_logging(self, *args, **kwargs):
        """Start the logging."""
        pass

    def stop_logging(self, logger_id):
        """Stop the logging."""
        pass

    def get_gravity(self):
        """Return the gravity set in the simulator."""
        pass

    def set_gravity(self, gravity=(0, 0, -9.81)):
        """Set the gravity in the simulator with the given acceleration.

        Args:
            gravity (list, tuple of 3 floats): acceleration in the x, y, z directions.
        """
        pass

    def save(self, filename=None, *args, **kwargs):
        """Save the state of the simulator.

        Args:
            filename (None, str): path to file to store the state of the simulator. If None, it will save it in
                memory instead of the disk.

        Returns:
            int: unique state id. This id can be used to load the state.
        """
        pass

    def load(self, state, *args, **kwargs):
        """Load / Restore the simulator to a previous state.

        Args:
            state (int, str): unique state id, or path to the file containing the state.
        """
        pass

    def load_plugin(self, plugin_path, name, *args, **kwargs):
        """Load a certain plugin in the simulator.

        Args:
            plugin_path (str): path, location on disk where to find the plugin
            name (str): postfix name of the plugin that is appended to each API

        Returns:
             int: unique plugin id. If this id is negative, the plugin is not loaded. Once a plugin is loaded, you can
                send commands to the plugin using `execute_plugin_commands`
        """
        pass

    def execute_plugin_commands(self, plugin_id, *args, **kwargs):
        """Execute the commands on the specified plugin.

        Args:
            plugin_id (int): unique plugin id.
            *args (list): list of argument values to be interpreted by the plugin. One can be a string, while the
                others must be integers or float.
        """
        pass

    def unload_plugin(self, plugin_id, *args, **kwargs):
        """Unload the specified plugin from the simulator.

        Args:
            plugin_id (int): unique plugin id.
        """
        pass

    # loading URDFs, SDFs, MJCFs

    def load_urdf(self, filename, position, orientation, use_fixed_base=0, scale=1.0, *args, **kwargs):
        """Load a URDF file in the simulator.

        Args:
            filename (str): a relative or absolute path to the URDF file on the file system of the physics server.
            position (vec3): create the base of the object at the specified position in world space coordinates [x,y,z]
            orientation (quat): create the base of the object at the specified orientation as world space quaternion
                [x,y,z,w]
            use_fixed_base (bool): force the base of the loaded object to be static
            scale (float): scale factor to the URDF model.

        Returns:
            int (non-negative): unique id associated to the load model.
        """
        pass

    def load_sdf(self, filename, scaling=1., *args, **kwargs):
        """Load a SDF file in the simulator.

        Args:
            filename (str): a relative or absolute path to the SDF file on the file system of the physics server.
            scaling (float): scale factor for the object

        Returns:
            list(int): list of object unique id for each object loaded
        """
        pass

    def load_mjcf(self, filename, scaling=1., *args, **kwargs):
        """Load a Mujoco file in the simulator.

        Args:
            filename (str): a relative or absolute path to the MJCF file on the file system of the physics server.
            scaling (float): scale factor for the object

        Returns:
            list(int): list of object unique id for each object loaded
        """
        pass

    def load_mesh(self, filename, position, orientation=(0, 0, 0, 1), mass=1., scale=(1., 1., 1.), color=None,
                  with_collision=True, flags=None, *args, **kwargs):
        """Load a mesh into the simulator.

        Args:
            filename (str): path to file for the mesh. Currently, only Wavefront .obj. It will create convex hulls
                for each object (marked as 'o') in the .obj file.
            position (float[3]): position of the mesh in the Cartesian world space (in meters)
            orientation (float[4], np.quaternion): orientation of the mesh using quaternion.
                If np.quaternion then it uses the convention (w,x,y,z). If float[4], it uses the convention (x,y,z,w)
            mass (float): mass of the mesh (in kg). If mass = 0, it won't move even if there is a collision.
            scale (float[3]): scale the mesh in the (x,y,z) directions
            color (int[4], None): color of the mesh for red, green, blue, and alpha, each in range [0,1].
            with_collision (bool): If True, it will also create the collision mesh, and not only a visual mesh.
            flags (int, None): if flag = `sim.GEOM_FORCE_CONCAVE_TRIMESH` (=1), this will create a concave static
                triangle mesh. This should not be used with dynamic/moving objects, only for static (mass=0) terrain.

        Returns:
            int: unique id of the mesh in the world
        """
        pass

    @staticmethod
    def get_available_sdfs(fullpath=False):
        """Return the list of available SDFs in the simulator.

        Args:
            fullpath (bool): If True, it will return the full path to the SDFs. If False, it will just return the
                name of the SDF files (without the extension).
        """
        return []

    @staticmethod
    def get_available_urdfs(fullpath=False):
        """Return the list of available URDFs in the simulator.

        Args:
            fullpath (bool): If True, it will return the full path to the URDFs. If False, it will just return the
                name of the URDF files (without the extension).
        """
        return []

    @staticmethod
    def get_available_mjcfs(fullpath=False):
        """Return the list of available MJCFs in the simulator.

        Args:
            fullpath (bool): If True, it will return the full path to the MJCFs. If False, it will just return the
            name of the MJCF files (without the extension).
        """
        return []

    @staticmethod
    def get_available_objs(fullpath=False):
        """Return the list of available OBJs in the simulator.

        Args:
            fullpath (bool): If True, it will return the full path to the OBJs. If False, it will just return the
                name of the OBJ files (without the extension).
        """
        return []

    # bodies

    def create_body(self, visual_shape_id=-1, collision_shape_id=-1, mass=0., position=(0., 0., 0.),
                    orientation=(0., 0., 0., 1.), *args, **kwargs):
        """Create a body in the simulator.

        Args:
            visual_shape_id (int): unique id from createVisualShape or -1. You can reuse the visual shape (instancing)
            collision_shape_id (int): unique id from createCollisionShape or -1. You can re-use the collision shape
                for multiple multibodies (instancing)
            mass (float): mass of the base, in kg (if using SI units)
            position (np.float[3]): Cartesian world position of the base
            orientation (np.float[4]): Orientation of base as quaternion [x,y,z,w]

        Returns:
            int: non-negative unique id or -1 for failure.
        """
        pass

    def remove_body(self, body_id):
        """Remove a particular body in the simulator.

        Args:
            body_id (int): unique body id.
        """
        pass

    def num_bodies(self):
        """Return the number of bodies present in the simulator.

        Returns:
            int: number of bodies
        """
        pass

    def get_body_info(self, body_id):
        """Get the specified body information.

        Args:
            body_id (int): unique body id.

        Returns:
            dict, list: info
        """
        pass

    def get_body_id(self, index):
        """Get the body id associated to the index which is between 0 and `num_bodies()`.

        Args:
            index (int): index between [0, `num_bodies()`]

        Returns:
            int: unique body id.
        """
        pass

    # constraint

    def create_constraint(self, parent_body_id, parent_link_id, child_body_id, child_link_id, joint_type,
                          joint_axis, parent_frame_position, child_frame_position,
                          parent_frame_orientation=(0., 0., 0., 1.), child_frame_orientation=(0., 0., 0., 1.),
                          *args, **kwargs):
        """
        Create a constraint.

        Args:
            parent_body_id (int): parent body unique id
            parent_link_id (int): parent link index (or -1 for the base)
            child_body_id (int): child body unique id, or -1 for no body (specify a non-dynamic child frame in world
                coordinates)
            child_link_id (int): child link index, or -1 for the base
            joint_type (int): joint type: JOINT_PRISMATIC (=1), JOINT_FIXED (=4), JOINT_POINT2POINT (=5),
                JOINT_GEAR (=6)
            joint_axis (np.float[3]): joint axis, in child link frame
            parent_frame_position (np.float[3]): position of the joint frame relative to parent CoM frame.
            child_frame_position (np.float[3]): position of the joint frame relative to a given child CoM frame (or
                world origin if no child specified)
            parent_frame_orientation (np.float[4]): the orientation of the joint frame relative to parent CoM
                coordinate frame
            child_frame_orientation (np.float[4]): the orientation of the joint frame relative to the child CoM
                coordinate frame (or world origin frame if no child specified)

        Returns:
            int: constraint unique id.
        """
        pass

    def remove_constraint(self, constraint_id):
        """
        Remove the specified constraint.

        Args:
            constraint_id (int): constraint unique id.
        """
        pass

    def change_constraint(self, constraint_id, *args, **kwargs):
        """
        Change the parameters of an existing constraint.

        Args:
            constraint_id (int): constraint unique id.
        """
        pass

    def num_constraints(self):
        """
        Get the number of constraints created.

        Returns:
            int: number of constraints created.
        """
        pass

    def get_constraint_id(self, index):
        """
        Get the constraint unique id associated with the index which is between 0 and `num_constraints()`.

        Args:
            index (int): index between [0, `num_constraints()`]

        Returns:
            int: constraint unique id.
        """
        pass

    def get_constraint_info(self, constraint_id):
        """
        Get information about the given constaint id.

        Args:
            constraint_id (int): constraint unique id.

        Returns:
            dict, list: info
        """
        pass

    def get_constraint_state(self, constraint_id):
        """
        Get the state of the given constraint.

        Args:
            constraint_id (int): constraint unique id.

        Returns:
            dict, list: state
        """
        pass

    # objects

    def get_mass(self, body_id):
        """
        Return the total mass of the robot (=sum of all mass links).

        Args:
            body_id (int): unique object id, as returned from `load_urdf`.

        Returns:
            float: total mass of the robot [kg]
        """
        pass

    def get_base_mass(self, body_id):
        """Return the base mass of the robot.

        Args:
            body_id (int): unique object id.
        """
        pass

    def get_base_name(self, body_id):
        """
        Return the base name.

        Args:
            body_id (int): unique object id.

        Returns:
            str: base name
        """
        pass

    def get_center_of_mass_position(self, body_id, link_ids=None):
        """
        Return the center of mass position.

        Args:
            body_id (int): unique body id.
            link_ids (list of int): link ids associated with the given body id. If None, it will take all the links
                of the specified body.

        Returns:
            np.float[3]: center of mass position in the Cartesian world coordinates
        """
        pass

    def get_center_of_mass_velocity(self, body_id, link_ids=None):
        """
        Return the center of mass linear velocity.

        Args:
            body_id (int): unique body id.
            link_ids (list of int): link ids associated with the given body id. If None, it will take all the links
                of the specified body.

        Returns:
            np.float[3]: center of mass linear velocity.
        """
        pass

    def get_base_pose(self, body_id):
        """
        Get the current position and orientation of the base (or root link) of the body in Cartesian world coordinates.

        Args:
            body_id (int): object unique id, as returned from `load_urdf`.

        Returns:
            np.float[3]: base position
            np.float[4]: base orientation (quaternion [x,y,z,w])
        """
        pass

    def get_base_position(self, body_id):
        """
        Return the base position of the specified body.

        Args:
            body_id (int): object unique id, as returned from `load_urdf`.

        Returns:
            np.float[3]: base position.
        """
        pass

    def get_base_orientation(self, body_id):
        """
        Get the base orientation of the specified body.

        Args:
            body_id (int): object unique id, as returned from `load_urdf`.

        Returns:
            np.float[4]: base orientation in the form of a quaternion (x,y,z,w)
        """
        pass

    def reset_base_pose(self, body_id, position, orientation):
        """
        Reset the base position and orientation of the specified object id.

        Args:
            body_id (int): unique object id.
            position (np.float[3]): new base position.
            orientation (np.float[4]): new base orientation (expressed as a quaternion [x,y,z,w])
        """
        pass

    def reset_base_position(self, body_id, position):
        """
        Reset the base position of the specified body/object id while preserving its orientation.

        Args:
            body_id (int): unique object id.
            position (np.float[3]): new base position.
        """
        pass

    def reset_base_orientation(self, body_id, orientation):
        """
        Reset the base orientation of the specified body/object id while preserving its position.

        Args:
            body_id (int): unique object id.
            orientation (np.float[4]): new base orientation (expressed as a quaternion [x,y,z,w])
        """
        pass

    def get_base_velocity(self, body_id):
        """
        Return the base linear and angular velocities.

        Args:
            body_id (int): object unique id, as returned from `load_urdf`.

        Returns:
            np.float[3]: linear velocity of the base in Cartesian world space coordinates
            np.float[3]: angular velocity of the base in Cartesian world space coordinates
        """
        pass

    def get_base_linear_velocity(self, body_id):
        """
        Return the linear velocity of the base.

        Args:
            body_id (int): object unique id, as returned from `load_urdf`.

        Returns:
            np.float[3]: linear velocity of the base in Cartesian world space coordinates
        """
        pass

    def get_base_angular_velocity(self, body_id):
        """
        Return the angular velocity of the base.

        Args:
            body_id (int): object unique id, as returned from `load_urdf`.

        Returns:
            np.float[3]: angular velocity of the base in Cartesian world space coordinates
        """
        pass

    def reset_base_velocity(self, body_id, linear_velocity=None, angular_velocity=None):
        """
        Reset the base velocity.

        Args:
            body_id (int): unique object id.
            linear_velocity (np.float[3]): new linear velocity of the base.
            angular_velocity (np.float[3]): new angular velocity of the base.
        """
        pass

    def reset_base_linear_velocity(self, body_id, linear_velocity):
        """
        Reset the base linear velocity.

        Args:
            body_id (int): unique object id.
            linear_velocity (np.float[3]): new linear velocity of the base
        """
        pass

    def reset_base_angular_velocity(self, body_id, angular_velocity):
        """
        Reset the base angular velocity.

        Args:
            body_id (int): unique object id.
            angular_velocity (np.float[3]): new angular velocity of the base
        """
        pass

    def apply_external_force(self, body_id, link_id=-1, force=(0., 0., 0.), position=(0., 0., 0.), frame=1):
        """
        Apply the specified external force on the specified position on the body / link.

        Args:
            body_id (int): unique body id.
            link_id (int): unique link id. If -1, it will be the base.
            force (np.float[3]): external force to be applied.
            position (np.float[3]): position on the link where the force is applied. See `flags` for coordinate
                systems. If None, it is the center of mass of the body (or the link if specified).
            frame (int): if frame = 1, then the force / position is described in the link frame. If frame = 2, they
                are described in the world frame.
        """
        pass

    def apply_external_torque(self, body_id, link_id=-1, torque=(0., 0., 0.), frame=1):
        """
        Apply an external torque on a body, or a link of the body. Note that after each simulation step, the external
        torques are cleared to 0.

        Args:
            body_id (int): unique body id.
            link_id (int): link id to apply the torque, if -1 it will apply the torque on the base
            torque (float[3]): Cartesian torques to be applied on the body
            frame (int): Specify the coordinate system of force/position: either `pybullet.WORLD_FRAME` (=2) for
                Cartesian world coordinates or `pybullet.LINK_FRAME` (=1) for local link coordinates.
        """
        pass

    # robots (joints and links)

    def num_joints(self, body_id):
        """
        Return the total number of joints of the specified body. This is the same as calling `num_links`.

        Args:
            body_id (int): unique body id.

        Returns:
            int: number of joints with the associated body id.
        """
        pass

    def num_actuated_joints(self, body_id):
        """
        Return the total number of actuated joints associated with the given body id.

        Args:
            body_id (int): unique body id.

        Returns:
            int: number of actuated joints of the specified body.
        """
        pass

    def num_links(self, body_id):
        """
        Return the total number of links of the specified body. This is the same as calling `num_joints`.

        Args:
            body_id (int): unique body id.

        Returns:
            int: number of links with the associated body id.
        """
        return self.num_joints(body_id)

    def get_joint_info(self, body_id, joint_id):
        """
        Return information about the given joint about the specified body.

        Note that this method returns a lot of information, so specific methods have been implemented that return
        only the desired information. Also, note that we do not convert the data here.

        Args:
            body_id (int): unique body id.
            joint_id (int): joint id is included in [0..`num_joints(body_id)`].

        Returns:
            dict, list: joint info
        """
        pass

    def get_joint_state(self, body_id, joint_id):
        """
        Get the joint state.

        Args:
            body_id (int): unique body id.
            joint_id (int): joint index in range [0..num_joints(body_id)]

        Returns:
            float: The position value of this joint.
            float: The velocity value of this joint.
            np.float[6]: These are the joint reaction forces, if a torque sensor is enabled for this joint it is
                [Fx, Fy, Fz, Mx, My, Mz]. Without torque sensor, it is [0, 0, 0, 0, 0, 0].
            float: This is the motor torque applied during the last stepSimulation. Note that this only applies in
                VELOCITY_CONTROL and POSITION_CONTROL. If you use TORQUE_CONTROL then the applied joint motor torque
                is exactly what you provide, so there is no need to report it separately.
        """
        pass

    def get_joint_states(self, body_id, joint_ids):
        """
        Get the joint state of the specified joints.

        Args:
            body_id (int): unique body id.
            joint_ids (list of int): list of joint ids.

        Returns:
            list:
                float: The position value of this joint.
                float: The velocity value of this joint.
                np.float[6]: These are the joint reaction forces, if a torque sensor is enabled for this joint it is
                    [Fx, Fy, Fz, Mx, My, Mz]. Without torque sensor, it is [0, 0, 0, 0, 0, 0].
                float: This is the motor torque applied during the last `step`. Note that this only applies in
                    VELOCITY_CONTROL and POSITION_CONTROL. If you use TORQUE_CONTROL then the applied joint motor
                    torque is exactly what you provide, so there is no need to report it separately.
        """
        pass

    def reset_joint_state(self, body_id, joint_id, position, velocity=0.):
        """
        Reset the state of the joint. It is best only to do this at the start, while not running the simulation:
        `reset_joint_state` overrides all physics simulation.

        Args:
            body_id (int): unique body id.
            joint_id (int): joint index in range [0..num_joints(body_id)]
            position (float): the joint position (angle in radians [rad] or position [m])
            velocity (float): the joint velocity (angular [rad/s] or linear velocity [m/s])
        """
        pass

    def enable_joint_force_torque_sensor(self, body_id, joint_ids, enable=True):
        """
        You can enable or disable a joint force/torque sensor in each joint.

        Args:
            body_id (int): body unique id.
            joint_ids (int, int[N]): joint index in range [0..num_joints(body_id)], or list of joint ids.
            enable (bool): True to enable, False to disable the force/torque sensor
        """
        pass

    def set_joint_motor_control(self, body_id, joint_ids, control_mode=2, positions=None,
                                velocities=None, forces=None, kp=None, kd=None, max_velocity=None):
        r"""
        Set the joint motor control.

        In position control:
        .. math:: error = Kp (x_{des} - x) + Kd (\dot{x}_{des} - \dot{x})

        In velocity control:
        .. math:: error = \dot{x}_{des} - \dot{x}

        Note that the maximum forces and velocities are not automatically used for the different control schemes.

        Args:
            body_id (int): body unique id.
            joint_ids (int): joint/link id, or list of joint ids.
            control_mode (int): POSITION_CONTROL (=2) (which is in fact CONTROL_MODE_POSITION_VELOCITY_PD),
                VELOCITY_CONTROL (=0), TORQUE_CONTROL (=1) and PD_CONTROL (=3).
            positions (float, np.float[N]): target joint position(s) (used in POSITION_CONTROL).
            velocities (float, np.float[N]): target joint velocity(ies). In VELOCITY_CONTROL and POSITION_CONTROL,
                the target velocity(ies) is(are) the desired velocity of the joint. Note that the target velocity(ies)
                is(are) not the maximum joint velocity(ies). In PD_CONTROL and
                POSITION_CONTROL/CONTROL_MODE_POSITION_VELOCITY_PD, the final target velocities are computed using:
                `kp*(erp*(desiredPosition-currentPosition)/dt)+currentVelocity+kd*(m_desiredVelocity - currentVelocity)`
            forces (float, list of float): in POSITION_CONTROL and VELOCITY_CONTROL, these are the maximum motor
                forces used to reach the target values. In TORQUE_CONTROL these are the forces / torques to be applied
                each simulation step.
            kp (float, list of float): position (stiffness) gain(s) (used in POSITION_CONTROL).
            kd (float, list of float): velocity (damping) gain(s) (used in POSITION_CONTROL).
            max_velocity (float): in POSITION_CONTROL this limits the velocity to a maximum.
        """
        pass

    def get_link_state(self, body_id, link_id, compute_velocity=False, compute_forward_kinematics=False):
        """
        Get the state of the associated link.

        Args:
            body_id (int): body unique id.
            link_id (int): link index.
            compute_velocity (bool): If True, the Cartesian world velocity will be computed and returned.
            compute_forward_kinematics (bool): if True, the Cartesian world position/orientation will be recomputed
                using forward kinematics.

        Returns:
            np.float[3]: Cartesian position of CoM
            np.float[4]: Cartesian orientation of CoM, in quaternion [x,y,z,w]
            np.float[3]: local position offset of inertial frame (center of mass) expressed in the URDF link frame
            np.float[4]: local orientation (quaternion [x,y,z,w]) offset of the inertial frame expressed in URDF link
                frame
            np.float[3]: world position of the URDF link frame
            np.float[4]: world orientation of the URDF link frame
            np.float[3]: Cartesian world linear velocity. Only returned if `compute_velocity` is True.
            np.float[3]: Cartesian world angular velocity. Only returned if `compute_velocity` is True.
        """
        pass

    def get_link_states(self, body_id, link_ids, compute_velocity=False, compute_forward_kinematics=False):
        """
        Get the state of the associated links.

        Args:
            body_id (int): body unique id.
            link_ids (list of int): list of link index.
            compute_velocity (bool): If True, the Cartesian world velocity will be computed and returned.
            compute_forward_kinematics (bool): if True, the Cartesian world position/orientation will be recomputed
                using forward kinematics.

        Returns:
            list:
                np.float[3]: Cartesian position of CoM
                np.float[4]: Cartesian orientation of CoM, in quaternion [x,y,z,w]
                np.float[3]: local position offset of inertial frame (center of mass) expressed in the URDF link frame
                np.float[4]: local orientation (quaternion [x,y,z,w]) offset of the inertial frame expressed in URDF
                    link frame
                np.float[3]: world position of the URDF link frame
                np.float[4]: world orientation of the URDF link frame
                np.float[3]: Cartesian world linear velocity. Only returned if `compute_velocity` is True.
                np.float[3]: Cartesian world angular velocity. Only returned if `compute_velocity` is True.
        """
        pass

    def get_link_names(self, body_id, link_ids):
        """
        Return the name of the given link(s).

        Args:
            body_id (int): unique body id.
            link_ids (int, list of int): link id, or list of link ids.

        Returns:
            if 1 link:
                str: link name
            if multiple links:
                str[N]: link names
        """
        pass

    def get_link_masses(self, body_id, link_ids):
        """
        Return the mass of the given link(s).

        Args:
            body_id (int): unique body id.
            link_ids (int, list of int): link id, or list of link ids.

        Returns:
            if 1 link:
                float: mass of the given link
            else:
                float[N]: mass of each link
        """
        pass

    def get_link_frames(self, body_id, link_ids):
        pass

    def get_link_world_positions(self, body_id, link_ids):
        """
        Return the CoM position (in the Cartesian world space coordinates) of the given link(s).

        Args:
            body_id (int): unique body id.
            link_ids (list of int): list of link indices.

        Returns:
            if 1 link:
                np.float[3]: the link CoM position in the world space
            if multiple links:
                np.float[N,3]: CoM position of each link in world space
        """
        pass

    def get_link_positions(self, body_id, link_ids):
        pass

    def get_link_world_orientations(self, body_id, link_ids):
        """
        Return the CoM orientation (in the Cartesian world space) of the given link(s).

        Args:
            body_id (int): unique body id.
            link_ids (list of int): list of link indices.

        Returns:
            if 1 link:
                np.float[4]: Cartesian orientation of the link CoM (x,y,z,w)
            if multiple links:
                np.float[N,4]: CoM orientation of each link (x,y,z,w)
        """
        pass

    def get_link_orientations(self, body_id, link_ids):
        pass

    def get_link_world_linear_velocities(self, body_id, link_ids):
        """
        Return the linear velocity of the link(s) expressed in the Cartesian world space coordinates.

        Args:
            body_id (int): unique body id.
            link_ids (list of int): list of link indices.

        Returns:
            if 1 link:
                np.float[3]: linear velocity of the link in the Cartesian world space
            if multiple links:
                np.float[N,3]: linear velocity of each link
        """
        pass

    def get_link_world_angular_velocities(self, body_id, link_ids):
        """
        Return the angular velocity of the link(s) in the Cartesian world space coordinates.

        Args:
            body_id (int): unique body id.
            link_ids (list of int): list of link indices.

        Returns:
            if 1 link:
                np.float[3]: angular velocity of the link in the Cartesian world space
            if multiple links:
                np.float[N,3]: angular velocity of each link
        """
        pass

    def get_link_world_velocities(self, body_id, link_ids):
        """
        Return the linear and angular velocities (expressed in the Cartesian world space coordinates) for the given
        link(s).

        Args:
            body_id (int): unique body id.
            link_ids (list of int): list of link indices.

        Returns:
            if 1 link:
                np.float[6]: linear and angular velocity of the link in the Cartesian world space
            if multiple links:
                np.float[N,6]: linear and angular velocity of each link
        """
        pass

    def get_link_velocities(self, body_id, link_ids):
        pass

    def get_q_indices(self, body_id, joint_ids):
        """
        Get the corresponding q index of the given joint(s).

        Args:
            body_id (int): unique body id.
            joint_ids (int, list of int): a joint id, or list of joint ids.

        Returns:
            if 1 joint:
                int: q index
            if multiple joints:
                np.int[N]: q indices
        """
        pass

    def get_actuated_joint_ids(self, body_id):
        """
        Get the actuated joint ids associated with the given body id.

        Args:
            body_id (int): unique body id.

        Returns:
            list of int: actuated joint ids.
        """
        pass

    def get_joint_names(self, body_id, joint_ids):
        """
        Return the name of the given joint(s).

        Args:
            body_id (int): unique body id.
            joint_ids (int, list of int): a joint id, or list of joint ids.

        Returns:
            if 1 joint:
                str: name of the joint
            if multiple joints:
                str[N]: name of each joint
        """
        pass

    def get_joint_type_ids(self, body_id, joint_ids):
        """
        Get the joint type ids.

        Args:
            body_id (int): unique body id.
            joint_ids (int, list of int): a joint id, or list of joint ids.

        Returns:
            if 1 joint:
                int: joint type id.
            if multiple joints: list of above
        """
        pass

    def get_joint_type_names(self, body_id, joint_ids):
        """
        Get joint type names.

        Args:
            body_id (int): unique body id.
            joint_ids (int, list of int): a joint id, or list of joint ids.

        Returns:
            if 1 joint:
                str: joint type name.
            if multiple joints: list of above
        """
        pass

    def get_joint_dampings(self, body_id, joint_ids):
        """
        Get the damping coefficient of the given joint(s).

        Args:
            body_id (int): unique body id.
            joint_ids (int, list of int): a joint id, or list of joint ids.

        Returns:
            if 1 joint:
                float: damping coefficient of the given joint
            if multiple joints:
                np.float[N]: damping coefficient for each specified joint
        """
        pass

    def get_joint_frictions(self, body_id, joint_ids):
        """
        Get the friction coefficient of the given joint(s).

        Args:
            body_id (int): unique body id.
            joint_ids (int, list of int): a joint id, or list of joint ids.

        Returns:
            if 1 joint:
                float: friction coefficient of the given joint
            if multiple joints:
                np.float[N]: friction coefficient for each specified joint
        """
        pass

    def get_joint_limits(self, body_id, joint_ids):
        """
        Get the joint limits of the given joint(s).

        Args:
            body_id (int): unique body id.
            joint_ids (int, list of int): a joint id, or list of joint ids.

        Returns:
            if 1 joint:
                np.float[2]: lower and upper limit
            if multiple joints:
                np.float[N,2]: lower and upper limit for each specified joint
        """
        pass

    def get_joint_max_forces(self, body_id, joint_ids):
        """
        Get the maximum force that can be applied on the given joint(s).

        Warning: Note that this is not automatically used in position, velocity, or torque control.

        Args:
            body_id (int): unique body id.
            joint_ids (int, list of int): a joint id, or list of joint ids.

        Returns:
            if 1 joint:
                float: maximum force [N]
            if multiple joints:
                np.float[N]: maximum force for each specified joint [N]
        """
        pass

    def get_joint_max_velocities(self, body_id, joint_ids):
        """
        Get the maximum velocity that can be applied on the given joint(s).

        Warning: Note that this is not automatically used in position, velocity, or torque control.

        Args:
            body_id (int): unique body id.
            joint_ids (int, list of int): a joint id, or list of joint ids.

        Returns:
            if 1 joint:
                float: maximum velocity [rad/s]
            if multiple joints:
                np.float[N]: maximum velocities for each specified joint [rad/s]
        """
        pass

    def get_joint_axes(self, body_id, joint_ids):
        """
        Get the joint axis about the given joint(s).

        Args:
            body_id (int): unique body id.
            joint_ids (int, list of int): a joint id, or list of joint ids.

        Returns:
            if 1 joint:
                np.float[3]: joint axis
            if multiple joint:
                np.float[N,3]: list of joint axis
        """
        pass

    def set_joint_positions(self, body_id, joint_ids, positions, velocities=None, kps=None, kds=None, forces=None):
        """
        Set the position of the given joint(s) (using position control).

        Args:
            body_id (int): unique body id.
            joint_ids (int, list of int): joint id, or list of joint ids.
            positions (float, np.float[N]): desired position, or list of desired positions [rad]
            velocities (None, float, np.float[N]): desired velocity, or list of desired velocities [rad/s]
            kps (None, float, np.float[N]): position gain(s)
            kds (None, float, np.float[N]): velocity gain(s)
            forces (None, float, np.float[N]): maximum motor force(s)/torque(s) used to reach the target values.
        """
        pass

    def get_joint_positions(self, body_id, joint_ids):
        """
        Get the position of the given joint(s).

        Args:
            body_id (int): unique body id.
            joint_ids (int, list of int): joint id, or list of joint ids.

        Returns:
            if 1 joint:
                float: joint position [rad]
            if multiple joints:
                np.float[N]: joint positions [rad]
        """
        pass

    def set_joint_velocities(self, body_id, joint_ids, velocities, max_force=None):
        """
        Set the velocity of the given joint(s) (using velocity control).

        Args:
            body_id (int): unique body id.
            joint_ids (int, list of int): joint id, or list of joint ids.
            velocities (float, np.float[N]): desired velocity, or list of desired velocities [rad/s]
            max_force (None, float, np.float[N]): maximum motor forces/torques
        """
        pass

    def get_joint_velocities(self, body_id, joint_ids):
        """
        Get the velocity of the given joint(s).

        Args:
            body_id (int): unique body id.
            joint_ids (int, list of int): joint id, or list of joint ids.

        Returns:
            if 1 joint:
                float: joint velocity [rad/s]
            if multiple joints:
                np.float[N]: joint velocities [rad/s]
        """
        pass

    def set_joint_accelerations(self, body_id, joint_ids, accelerations, q=None, dq=None):
        """
        Set the acceleration of the given joint(s) (using force control). This is achieved by performing inverse
        dynamic which given the joint accelerations compute the joint torques to be applied.

        Args:
            body_id (int): unique body id.
            joint_ids (int, list of int): joint id, or list of joint ids.
            accelerations (float, np.float[N]): desired joint acceleration, or list of desired joint accelerations
                [rad/s^2]
        """
        pass

    def get_joint_accelerations(self, body_id, joint_ids, q=None, dq=None):
        """
        Get the acceleration at the given joint(s). This is carried out by first getting the joint torques, then
        performing forward dynamics to get the joint accelerations from the joint torques.

        Args:
            body_id (int): unique body id.
            joint_ids (int, list of int): joint id, or list of joint ids.
            q (list of int, None): all the joint positions. If None, it will compute it.
            dq (list of int, None): all the joint velocities. If None, it will compute it.

        Returns:
            if 1 joint:
                float: joint acceleration [rad/s^2]
            if multiple joints:
                np.float[N]: joint accelerations [rad/s^2]
        """
        pass

    def set_joint_torques(self, body_id, joint_ids, torques):
        """
        Set the torque/force to the given joint(s) (using force/torque control).

        Args:
            body_id (int): unique body id.
            joint_ids (int, list of int): joint id, or list of joint ids.
            torques (float, list of float): desired torque(s) to apply to the joint(s) [N].
        """
        pass

    def get_joint_torques(self, body_id, joint_ids):
        """
        Get the applied torque(s) on the given joint(s).

        Args:
            body_id (int): unique body id.
            joint_ids (int, list of int): a joint id, or list of joint ids.

        Returns:
            if 1 joint:
                float: torque [Nm]
            if multiple joints:
                np.float[N]: torques associated to the given joints [Nm]
        """
        pass

    def get_joint_reaction_forces(self, body_id, joint_ids):
        """Return the joint reaction forces at the given joint. Note that the torque sensor must be enabled, otherwise
        it will always return [0,0,0,0,0,0].

        Args:
            body_id (int): unique body id.
            joint_ids (int, int[N]): joint id, or list of joint ids

        Returns:
            if 1 joint:
                np.float[6]: joint reaction force (fx,fy,fz,mx,my,mz) [N,Nm]
            if multiple joints:
                np.float[N,6]: joint reaction forces [N, Nm]
        """
        pass

    def get_joint_powers(self, body_id, joint_ids):
        """Return the applied power at the given joint(s). Power = torque * velocity.

        Args:
            body_id (int): unique body id.
            joint_ids (int, int[N]): joint id, or list of joint ids

        Returns:
            if 1 joint:
                float: joint power [W]
            if multiple joints:
                np.float[N]: power at each joint [W]
        """
        pass

    # visualization

    def create_visual_shape(self, shape_type, radius=0.5, half_extents=(1., 1., 1.), length=1., filename=None,
                            mesh_scale=(1., 1., 1.), plane_normal=(0., 0., 1.), flags=-1, rgba_color=None,
                            specular_color=None, visual_frame_position=None, vertices=None, indices=None, uvs=None,
                            normals=None, visual_frame_orientation=None):
        """
        Create a visual shape in the simulator.

        Args:
            shape_type (int): type of shape; GEOM_SPHERE (=2), GEOM_BOX (=3), GEOM_CAPSULE (=7), GEOM_CYLINDER (=4),
                GEOM_PLANE (=6), GEOM_MESH (=5)
            radius (float): only for GEOM_SPHERE, GEOM_CAPSULE, GEOM_CYLINDER
            half_extents (np.float[3], list/tuple of 3 floats): only for GEOM_BOX.
            length (float): only for GEOM_CAPSULE, GEOM_CYLINDER (length = height).
            filename (str): Filename for GEOM_MESH, currently only Wavefront .obj. Will create convex hulls for each
                object (marked as 'o') in the .obj file.
            mesh_scale (np.float[3], list/tuple of 3 floats): scale of mesh (only for GEOM_MESH).
            plane_normal (np.float[3], list/tuple of 3 floats): plane normal (only for GEOM_PLANE).
            flags (int): unused / to be decided
            rgba_color (list/tuple of 4 floats): color components for red, green, blue and alpha, each in range [0..1].
            specular_color (list/tuple of 3 floats): specular reflection color, red, green, blue components in range
                [0..1]
            visual_frame_position (np.float[3]): translational offset of the visual shape with respect to the link frame
            vertices (list of np.float[3]): Instead of creating a mesh from obj file, you can provide vertices, indices,
                uvs and normals
            indices (list of int): triangle indices, should be a multiple of 3.
            uvs (list of np.float[2]): uv texture coordinates for vertices. Use changeVisualShape to choose the
                texture image. The number of uvs should be equal to number of vertices
            normals (list of np.float[3]): vertex normals, number should be equal to number of vertices.
            visual_frame_orientation (np.float[4]): rotational offset (quaternion x,y,z,w) of the visual shape with
                respect to the link frame

        Returns:
            int: The return value is a non-negative int unique id for the visual shape or -1 if the call failed.
        """
        pass

    def get_visual_shape_data(self, object_id, flags=-1):
        """
        Get the visual shape data associated with the given object id. It will output a list of visual shape data.

        Args:
            object_id (int): object unique id.
            flags (int, None): VISUAL_SHAPE_DATA_TEXTURE_UNIQUE_IDS (=1) will also provide `texture_unique_id`.

        Returns:
            list:
                int: object unique id.
                int: link index or -1 for the base
                int: visual geometry type (TBD)
                np.float[3]: dimensions (size, local scale) of the geometry
                str: path to the triangle mesh, if any. Typically relative to the URDF, SDF or MJCF file location, but
                    could be absolute
                np.float[3]: position of local visual frame, relative to link/joint frame
                np.float[4]: orientation of local visual frame relative to link/joint frame
                list of 4 floats: URDF color (if any specified) in Red / Green / Blue / Alpha
                int: texture unique id of the shape or -1 if None. This field only exists if using
                    VISUAL_SHAPE_DATA_TEXTURE_UNIQUE_IDS (=1) flag.
        """
        pass

    def change_visual_shape(self, object_id, link_id, shape_id=None, texture_id=None, rgba_color=None,
                            specular_color=None):
        """
        Allows to change the texture of a shape, the RGBA color and other properties.

        Args:
            object_id (int): unique object id.
            link_id (int): link id.
            shape_id (int): shape id.
            texture_id (int): texture id.
            rgba_color (float[4]): RGBA color. Each is in the range [0..1]. Alpha has to be 0 (invisible) or 1
                (visible) at the moment.
            specular_color (int[3]): specular color components, RED, GREEN and BLUE, can be from 0 to large number
                (>100).
        """
        pass

    def load_texture(self, filename):
        """
        Load a texture from file and return a non-negative texture unique id if the loading succeeds.
        This unique id can be used with changeVisualShape.

        Args:
            filename (str): path to the file.

        Returns:
            int: texture unique id. If non-negative, the texture was loaded successfully.
        """
        pass

    def compute_view_matrix(self, eye_position, target_position, up_vector):
        """Compute the view matrix.

        The view matrix is the 4x4 matrix that maps the world coordinates into the camera coordinates. Basically,
        it applies a rotation and translation such that the world is in front of the camera. That is, instead
        of turning the camera to capture what we want in the world, we keep the camera fixed and turn the world.

        Args:
            eye_position (np.float[3]): eye position in Cartesian world coordinates
            target_position (np.float[3]): position of the target (focus) point in Cartesian world coordinates
            up_vector (np.float[3]): up vector of the camera in Cartesian world coordinates

        Returns:
            np.float[4,4]: the view matrix
        """
        pass

    def compute_view_matrix_from_ypr(self, target_position, distance, yaw, pitch, roll, up_axis_index=2):
        """Compute the view matrix from the yaw, pitch, and roll angles.

        The view matrix is the 4x4 matrix that maps the world coordinates into the camera coordinates. Basically,
        it applies a rotation and translation such that the world is in front of the camera. That is, instead
        of turning the camera to capture what we want in the world, we keep the camera fixed and turn the world.

        Args:
            target_position (np.float[3]): target focus point in Cartesian world coordinates
            distance (float): distance from eye to focus point
            yaw (float): yaw angle in radians left/right around up-axis
            pitch (float): pitch in radians up/down.
            roll (float): roll in radians around forward vector
            up_axis_index (int): either 1 for Y or 2 for Z axis up.

        Returns:
            np.float[4,4]: the view matrix
        """
        pass

    def compute_projection_matrix(self, left, right, bottom, top, near, far):
        """Compute the orthographic projection matrix.

        The projection matrix is the 4x4 matrix that maps from the camera/eye coordinates to clipped coordinates.
        It is applied after the view matrix.

        There are 2 projection matrices:
        * orthographic projection
        * perspective projection

        For the perspective projection, see `computeProjectionMatrixFOV(self)`.

        Args:
            left (float): left screen (canvas) coordinate
            right (float): right screen (canvas) coordinate
            bottom (float): bottom screen (canvas) coordinate
            top (float): top screen (canvas) coordinate
            near (float): near plane distance
            far (float): far plane distance

        Returns:
            np.float[4,4]: the perspective projection matrix
        """
        pass

    def compute_projection_matrix_fov(self, fov, aspect, near, far):
        """Compute the perspective projection matrix using the field of view (FOV).

        Args:
            fov (float): field of view
            aspect (float): aspect ratio
            near (float): near plane distance
            far (float): far plane distance

        Returns:
            np.float[4,4]: the perspective projection matrix
        """
        pass

    def get_camera_image(self, width, height, view_matrix=None, projection_matrix=None, light_direction=None,
                         light_color=None, light_distance=None, shadow=None, light_ambient_coeff=None,
                         light_diffuse_coeff=None, light_specular_coeff=None, renderer=None, flags=None):
        """
        The `get_camera_image` API will return a RGB image, a depth buffer and a segmentation mask buffer with body
        unique ids of visible objects for each pixel.

        Args:
            width (int): horizontal image resolution in pixels
            height (int): vertical image resolution in pixels
            view_matrix (np.float[4,4]): 4x4 view matrix, see `compute_view_matrix`
            projection_matrix (np.float[4,4]): 4x4 projection matrix, see `compute_projection`
            light_direction (np.float[3]): `light_direction` specifies the world position of the light source,
                the direction is from the light source position to the origin of the world frame.
            light_color (np.float[3]): directional light color in [RED,GREEN,BLUE] in range 0..1
            light_distance (float): distance of the light along the normalized `light_direction`
            shadow (bool): True for shadows, False for no shadows
            light_ambient_coeff (float): light ambient coefficient
            light_diffuse_coeff (float): light diffuse coefficient
            light_specular_coeff (float): light specular coefficient
            renderer (int): renderer
            flags (int): flags

        Returns:
            int: width image resolution in pixels (horizontal)
            int: height image resolution in pixels (vertical)
            np.int[width, height, 4]: RBGA pixels (each pixel is in the range [0..255] for each channel R, G, B, A)
            np.float[width, heigth]: Depth buffer.
            np.int[width, height]: Segmentation mask buffer. For each pixels the visible object unique id.
        """
        pass

    def get_rgba_image(self, width, height, view_matrix=None, projection_matrix=None, light_direction=None,
                       light_color=None, light_distance=None, shadow=None, light_ambient_coeff=None,
                       light_diffuse_coeff=None, light_specular_coeff=None, renderer=None, flags=None):
        """
        The `get_rgba_image` API will return a RGBA image.

        Args:
            width (int): horizontal image resolution in pixels
            height (int): vertical image resolution in pixels
            view_matrix (np.float[4,4]): 4x4 view matrix, see `compute_view_matrix`
            projection_matrix (np.float[4,4]): 4x4 projection matrix, see `compute_projection`
            light_direction (np.float[3]): `light_direction` specifies the world position of the light source,
                the direction is from the light source position to the origin of the world frame.
            light_color (np.float[3]): directional light color in [RED,GREEN,BLUE] in range 0..1
            light_distance (float): distance of the light along the normalized `light_direction`
            shadow (bool): True for shadows, False for no shadows
            light_ambient_coeff (float): light ambient coefficient
            light_diffuse_coeff (float): light diffuse coefficient
            light_specular_coeff (float): light specular coefficient
            renderer (int): renderer.
            flags (int): flags.

        Returns:
            np.int[width, height, 4]: RBGA pixels (each pixel is in the range [0..255] for each channel R, G, B, A)
        """
        pass

    def get_depth_image(self, width, height, view_matrix=None, projection_matrix=None, light_direction=None,
                        light_color=None, light_distance=None, shadow=None, light_ambient_coeff=None,
                        light_diffuse_coeff=None, light_specular_coeff=None, renderer=None, flags=None):
        """
        The `get_depth_image` API will return a depth buffer.

        Args:
            width (int): horizontal image resolution in pixels
            height (int): vertical image resolution in pixels
            view_matrix (np.float[4,4]): 4x4 view matrix, see `compute_view_matrix`
            projection_matrix (np.float[4,4]): 4x4 projection matrix, see `compute_projection`
            light_direction (np.float[3]): `light_direction` specifies the world position of the light source,
                the direction is from the light source position to the origin of the world frame.
            light_color (np.float[3]): directional light color in [RED,GREEN,BLUE] in range 0..1
            light_distance (float): distance of the light along the normalized `light_direction`
            shadow (bool): True for shadows, False for no shadows
            light_ambient_coeff (float): light ambient coefficient
            light_diffuse_coeff (float): light diffuse coefficient
            light_specular_coeff (float): light specular coefficient
            renderer (int): renderer.
            flags (int): flags.

        Returns:
            np.float[width, heigth]: Depth buffer.
        """
        pass

    def get_segmentation_image(self, width, height, view_matrix=None, projection_matrix=None, light_direction=None,
                               light_color=None, light_distance=None, shadow=None, light_ambient_coeff=None,
                               light_diffuse_coeff=None, light_specular_coeff=None, renderer=None, flags=None):
        """
        The `get_segmentation_image` API will return a segmentation mask buffer with body unique ids of visible objects
        for each pixel.

        Args:
            width (int): horizontal image resolution in pixels
            height (int): vertical image resolution in pixels
            view_matrix (np.float[4,4]): 4x4 view matrix, see `compute_view_matrix`
            projection_matrix (np.float[4,4]): 4x4 projection matrix, see `compute_projection`
            light_direction (np.float[3]): `light_direction` specifies the world position of the light source,
                the direction is from the light source position to the origin of the world frame.
            light_color (np.float[3]): directional light color in [RED,GREEN,BLUE] in range 0..1
            light_distance (float): distance of the light along the normalized `light_direction`
            shadow (bool): True for shadows, False for no shadows
            light_ambient_coeff (float): light ambient coefficient
            light_diffuse_coeff (float): light diffuse coefficient
            light_specular_coeff (float): light specular coefficient
            renderer (int): renderer
            flags (int): flags

        Returns:
            np.int[width, height]: Segmentation mask buffer. For each pixels the visible object unique id.
        """
        pass

    # collisions

    def create_collision_shape(self, shape_type, radius=0.5, half_extents=(1., 1., 1.), height=1., filename=None,
                               mesh_scale=(1., 1., 1.), plane_normal=(0., 0., 1.), flags=-1,
                               collision_frame_position=None, collision_frame_orientation=None):
        """
        Create collision shape in the simulator.

        Args:
            shape_type (int): type of shape; GEOM_SPHERE (=2), GEOM_BOX (=3), GEOM_CAPSULE (=7), GEOM_CYLINDER (=4),
                GEOM_PLANE (=6), GEOM_MESH (=5)
            radius (float): only for GEOM_SPHERE, GEOM_CAPSULE, GEOM_CYLINDER
            half_extents (np.float[3], list/tuple of 3 floats): only for GEOM_BOX.
            height (float): only for GEOM_CAPSULE, GEOM_CYLINDER (length = height).
            filename (str): Filename for GEOM_MESH, currently only Wavefront .obj. Will create convex hulls for each
                object (marked as 'o') in the .obj file.
            mesh_scale (np.float[3], list/tuple of 3 floats): scale of mesh (only for GEOM_MESH).
            plane_normal (np.float[3], list/tuple of 3 floats): plane normal (only for GEOM_PLANE).
            flags (int): unused / to be decided
            collision_frame_position (np.float[3]): translational offset of the collision shape with respect to the
                link frame
            collision_frame_orientation (np.float[4]): rotational offset (quaternion x,y,z,w) of the collision shape
                with respect to the link frame

        Returns:
            int: The return value is a non-negative int unique id for the collision shape or -1 if the call failed.
        """
        pass

    def get_collision_shape_data(self, object_id, link_id=-1):
        """
        Get the collision shape data associated with the specified object id and link id.

        Args:
            object_id (int): object unique id.
            link_id (int): link index or -1 for the base.

        Returns:
            int: object unique id.
            int: link id.
            int: geometry type; GEOM_BOX (=3), GEOM_SPHERE (=2), GEOM_CAPSULE (=7), GEOM_MESH (=5), GEOM_PLANE (=6)
            np.float[3]: depends on geometry type:
                for GEOM_BOX: extents,
                for GEOM_SPHERE: dimensions[0] = radius,
                for GEOM_CAPSULE and GEOM_CYLINDER: dimensions[0] = height (length), dimensions[1] = radius.
                For GEOM_MESH: dimensions is the scaling factor.
            str: Only for GEOM_MESH: file name (and path) of the collision mesh asset.
            np.float[3]: Local position of the collision frame with respect to the center of mass/inertial frame
            np.float[4]: Local orientation of the collision frame with respect to the inertial frame
        """
        pass

    def get_overlapping_objects(self, aabb_min, aabb_max):
        """
        This query will return all the unique ids of objects that have Axis Aligned Bounding Box (AABB) overlap with
        a given axis aligned bounding box. Note that the query is conservative and may return additional objects that
        don't have actual AABB overlap. This happens because the acceleration structures have some heuristic that
        enlarges the AABBs a bit (extra margin and extruded along the velocity vector).

        Args:
            aabb_min (np.float[3]): minimum coordinates of the aabb
            aabb_max (np.float[3]): maximum coordinates of the aabb

        Returns:
            list of int: list of object unique ids.
        """
        pass

    def get_aabb(self, body_id, link_id=-1):
        """
        You can query the axis aligned bounding box (in world space) given an object unique id, and optionally a link
        index. (when you don't pass the link index, or use -1, you get the AABB of the base).

        Args:
            body_id (int): object unique id as returned by creation methods
            link_id (int): link index in range [0..`getNumJoints(..)]

        Returns:
            np.float[3]: minimum coordinates of the axis aligned bounding box
            np.float[3]: maximum coordinates of the axis aligned bounding box
        """
        pass

    def get_contact_points(self, body1, body2=None, link1_id=None, link2_id=None):
        """
        Returns the contact points computed during the most recent call to `step`.

        Args:
            body1 (int): only report contact points that involve body A
            body2 (int, None): only report contact points that involve body B. Important: you need to have a valid
                body A if you provide body B
            link1_id (int, None): only report contact points that involve link index of body A
            link2_id (int, None): only report contact points that involve link index of body B

        Returns:
            list:
                int: contact flag (reserved)
                int: body unique id of body A
                int: body unique id of body B
                int: link index of body A, -1 for base
                int: link index of body B, -1 for base
                np.float[3]: contact position on A, in Cartesian world coordinates
                np.float[3]: contact position on B, in Cartesian world coordinates
                np.float[3]: contact normal on B, pointing towards A
                float: contact distance, positive for separation, negative for penetration
                float: normal force applied during the last `step`
                float: lateral friction force in the first lateral friction direction (see next returned value)
                np.float[3]: first lateral friction direction
                float: lateral friction force in the second lateral friction direction (see next returned value)
                np.float[3]: second lateral friction direction
        """
        pass

    def get_closest_points(self, body1, body2, distance, link1_id=None, link2_id=None):
        """
        Computes the closest points, independent from `step`. This also lets you compute closest points of objects
        with an arbitrary separating distance. In this query there will be no normal forces reported.

        Args:
            body1 (int): only report contact points that involve body A
            body2 (int): only report contact points that involve body B. Important: you need to have a valid body A
                if you provide body B
            distance (float): If the distance between objects exceeds this maximum distance, no points may be returned.
            link1_id (int): only report contact points that involve link index of body A
            link2_id (int): only report contact points that involve link index of body B

        Returns:
            list:
                int: contact flag (reserved)
                int: body unique id of body A
                int: body unique id of body B
                int: link index of body A, -1 for base
                int: link index of body B, -1 for base
                np.float[3]: contact position on A, in Cartesian world coordinates
                np.float[3]: contact position on B, in Cartesian world coordinates
                np.float[3]: contact normal on B, pointing towards A
                float: contact distance, positive for separation, negative for penetration
                float: normal force applied during the last `step`. Always equal to 0.
                float: lateral friction force in the first lateral friction direction (see next returned value)
                np.float[3]: first lateral friction direction
                float: lateral friction force in the second lateral friction direction (see next returned value)
                np.float[3]: second lateral friction direction
        """
        pass

    def ray_test(self, from_position, to_position):
        """
        Performs a single raycast to find the intersection information of the first object hit.

        Args:
            from_position (np.float[3]): start of the ray in world coordinates
            to_position (np.float[3]): end of the ray in world coordinates

        Returns:
            list:
                int: object unique id of the hit object
                int: link index of the hit object, or -1 if none/parent
                float: hit fraction along the ray in range [0,1] along the ray.
                np.float[3]: hit position in Cartesian world coordinates
                np.float[3]: hit normal in Cartesian world coordinates
        """
        pass

    def ray_test_batch(self, from_positions, to_positions, parent_object_id=None, parent_link_id=None):
        """Perform a batch of raycasts to find the intersection information of the first objects hit.

        This is similar to the ray_test, but allows you to provide an array of rays, for faster execution. The size of
        'rayFromPositions' needs to be equal to the size of 'rayToPositions'. You can one ray result per ray, even if
        there is no intersection: you need to use the objectUniqueId field to check if the ray has hit anything: if
        the objectUniqueId is -1, there is no hit. In that case, the 'hit fraction' is 1.

        Args:
            from_positions (np.array[N,3]): list of start points for each ray, in world coordinates
            to_positions (np.array[N,3]): list of end points for each ray in world coordinates
            parent_object_id (int): ray from/to is in local space of a parent object
            parent_link_id (int): ray from/to is in local space of a parent object

        Returns:
            list:
                int: object unique id of the hit object
                int: link index of the hit object, or -1 if none/parent
                float: hit fraction along the ray in range [0,1] along the ray.
                np.float[3]: hit position in Cartesian world coordinates
                np.float[3]: hit normal in Cartesian world coordinates
        """
        pass

    def set_collision_filter_group_mask(self, body_id, link_id, filter_group, filter_mask):
        """
        Enable/disable collision detection between groups of objects. Each body is part of a group. It collides with
        other bodies if their group matches the mask, and vise versa. The following check is performed using the group
        and mask of the two bodies involved. It depends on the collision filter mode.

        Args:
            body_id (int): unique id of the body to be configured
            link_id (int): link index of the body to be configured
            filter_group (int): bitwise group of the filter
            filter_mask (int): bitwise mask of the filter
        """
        pass

    def set_collision_filter_pair(self, body1, body2, link1=-1, link2=-1, enable=True):
        """
        Enable/disable collision between two bodies/links.

        Args:
            body1 (int): unique id of body A to be filtered
            body2 (int): unique id of body B to be filtered, A==B implies self-collision
            link1 (int): link index of body A
            link2 (int): link index of body B
            enable (bool): True to enable collision, False to disable collision
        """
        pass

    # kinematics and dynamics

    def get_dynamics_info(self, body_id, link_id=-1):
        """
        Get dynamic information about the mass, center of mass, friction and other properties of the base and links.

        Args:
            body_id (int): body/object unique id.
            link_id (int): link/joint index or -1 for the base.

        Returns:
            float: mass in kg
            float: lateral friction coefficient
            np.float[3]: local inertia diagonal. Note that links and base are centered around the center of mass and
                aligned with the principal axes of inertia.
            np.float[3]: position of inertial frame in local coordinates of the joint frame
            np.float[4]: orientation of inertial frame in local coordinates of joint frame
            float: coefficient of restitution
            float: rolling friction coefficient orthogonal to contact normal
            float: spinning friction coefficient around contact normal
            float: damping of contact constraints. -1 if not available.
            float: stiffness of contact constraints. -1 if not available.
        """
        pass

    def change_dynamics(self, body_id, link_id=-1, mass=None, lateral_friction=None, spinning_friction=None,
                        rolling_friction=None, restitution=None, linear_damping=None, angular_damping=None,
                        contact_stiffness=None, contact_damping=None, friction_anchor=None,
                        local_inertia_diagonal=None, joint_damping=None):
        """
        Change dynamic properties of the given body (or link) such as mass, friction and restitution coefficients, etc.

        Args:
            body_id (int): object unique id, as returned by `load_urdf`, etc.
            link_id (int): link index or -1 for the base.
            mass (float): change the mass of the link (or base for link index -1)
            lateral_friction (float): lateral (linear) contact friction
            spinning_friction (float): torsional friction around the contact normal
            rolling_friction (float): torsional friction orthogonal to contact normal
            restitution (float): bouncyness of contact. Keep it a bit less than 1.
            linear_damping (float): linear damping of the link (0.04 by default)
            angular_damping (float): angular damping of the link (0.04 by default)
            contact_stiffness (float): stiffness of the contact constraints, used together with `contact_damping`
            contact_damping (float): damping of the contact constraints for this body/link. Used together with
                `contact_stiffness`. This overrides the value if it was specified in the URDF file in the contact
                section.
            friction_anchor (int): enable or disable a friction anchor: positional friction correction (disabled by
                default, unless set in the URDF contact section)
            local_inertia_diagonal (np.float[3]): diagonal elements of the inertia tensor. Note that the base and
                links are centered around the center of mass and aligned with the principal axes of inertia so there
                are no off-diagonal elements in the inertia tensor.
            joint_damping (float): joint damping coefficient applied at each joint. This coefficient is read from URDF
                joint damping field. Keep the value close to 0.
                `joint_damping_force = -damping_coefficient * joint_velocity`.
        """
        pass

    def calculate_jacobian(self, body_id, link_id, local_position, q, dq, des_ddq):
        r"""
        Return the full geometric Jacobian matrix :math:`J(q) = [J_{lin}(q), J_{ang}(q)]^T`, such that:

        .. math:: v = [\dot{p}, \omega]^T = J(q) \dot{q}

        where :math:`\dot{p}` is the Cartesian linear velocity of the link, and :math:`\omega` is its angular velocity.

        Warnings: if we have a floating base then the Jacobian will also include columns corresponding to the root
            link DoFs (at the beginning). If it is a fixed base, it will only have columns associated with the joints.

        Args:
            body_id (int): unique body id.
            link_id (int): link id.
            local_position (np.float[3]): the point on the specified link to compute the Jacobian (in link local
                coordinates around its center of mass). If None, it will use the CoM position (in the link frame).
            q (np.float[N]): joint positions of size N, where N is the number of DoFs.
            dq (np.float[N]): joint velocities of size N, where N is the number of DoFs.
            des_ddq (np.float[N]): desired joint accelerations of size N.

        Returns:
            np.float[6,N], np.float[6,(6+N)]: full geometric (linear and angular) Jacobian matrix. The number of
                columns depends if the base is fixed or floating.
        """
        pass

    def calculate_mass_matrix(self, body_id, q):
        r"""
        Return the mass/inertia matrix :math:`H(q)`, which is used in the rigid-body equation of motion (EoM) in joint
        space given by (see [1]):

        .. math:: \tau = H(q)\ddot{q} + C(q,\dot{q})

        where :math:`\tau` is the vector of applied torques, :math:`H(q)` is the inertia matrix, and
        :math:`C(q,\dot{q}) \dot{q}` is the vector accounting for Coriolis, centrifugal forces, gravity, and any
        other forces acting on the system except the applied torques :math:`\tau`.

        Warnings: If the base is floating, it will return a [6+N,6+N] inertia matrix, where N is the number of actuated
            joints. If the base is fixed, it will return a [N,N] inertia matrix

        Args:
            body_id (int): body unique id.
            q (np.float[N]): joint positions of size N, where N is the total number of DoFs.

        Returns:
            np.float[N,N], np.float[6+N,6+N]: inertia matrix
        """
        pass

    def calculate_inverse_kinematics(self, body_id, link_id, position, orientation=None, lower_limits=None,
                                     upper_limits=None, joint_ranges=None, rest_poses=None, joint_dampings=None,
                                     solver=None, q_curr=None, max_iters=None, threshold=None):
        r"""
        Compute the FULL Inverse kinematics; it will return a position for all the actuated joints.

        "You can compute the joint angles that makes the end-effector reach a given target position in Cartesian world
        space. Internally, Bullet uses an improved version of Samuel Buss Inverse Kinematics library. At the moment
        only the Damped Least Squares method with or without Null Space control is exposed, with a single end-effector
        target. Optionally you can also specify the target orientation of the end effector. In addition, there is an
        option to use the null-space to specify joint limits and rest poses. This optional null-space support requires
        all 4 lists (lower_limits, upper_limits, joint_ranges, rest_poses), otherwise regular IK will be used." [1]

        Args:
            body_id (int): body unique id, as returned by `load_urdf`, etc.
            link_id (int): end effector link index.
            position (np.float[3]): target position of the end effector (its link coordinate, not center of mass
                coordinate!). By default this is in Cartesian world space, unless you provide `q_curr` joint angles.
            orientation (np.float[4]): target orientation in Cartesian world space, quaternion [x,y,w,z]. If not
                specified, pure position IK will be used.
            lower_limits (np.float[N], list of N floats): lower joint limits. Optional null-space IK.
            upper_limits (np.float[N], list of N floats): upper joint limits. Optional null-space IK.
            joint_ranges (np.float[N], list of N floats): range of value of each joint.
            rest_poses (np.float[N], list of N floats): joint rest poses. Favor an IK solution closer to a given rest
                pose.
            joint_dampings (np.float[N], list of N floats): joint damping factors. Allow to tune the IK solution using
                joint damping factors.
            solver (int): p.IK_DLS (=0) or p.IK_SDLS (=1), Damped Least Squares or Selective Damped Least Squares, as
                described in the paper by Samuel Buss "Selectively Damped Least Squares for Inverse Kinematics".
            q_curr (np.float[N]): list of joint positions. By default PyBullet uses the joint positions of the body.
                If provided, the target_position and targetOrientation is in local space!
            max_iters (int): maximum number of iterations. Refine the IK solution until the distance between target
                and actual end effector position is below this threshold, or the `max_iters` is reached.
            threshold (float): residual threshold. Refine the IK solution until the distance between target and actual
                end effector position is below this threshold, or the `max_iters` is reached.

        Returns:
            np.float[N]: joint positions (for each actuated joint).
        """
        pass

    def calculate_inverse_dynamics(self, body_id, q, dq, des_ddq):
        r"""
        Starting from the specified joint positions :math:`q` and velocities :math:`\dot{q}`, it computes the joint
        torques :math:`\tau` required to reach the desired joint accelerations :math:`\ddot{q}_{des}`. That is,
        :math:`\tau = ID(model, q, \dot{q}, \ddot{q}_{des})`.

        Specifically, it uses the rigid-body equation of motion in joint space given by (see [1]):

        .. math:: \tau = H(q)\ddot{q} + C(q,\dot{q})

        where :math:`\tau` is the vector of applied torques, :math:`H(q)` is the inertia matrix, and
        :math:`C(q,\dot{q}) \dot{q}` is the vector accounting for Coriolis, centrifugal forces, gravity, and any
        other forces acting on the system except the applied torques :math:`\tau`.

        Normally, a more popular form of this equation of motion (in joint space) is given by:

        .. math:: H(q) \ddot{q} + S(q,\dot{q}) \dot{q} + g(q) = \tau + J^T(q) F

        which is the same as the first one with :math:`C = S\dot{q} + g(q) - J^T(q) F`. However, this last formulation
        is useful to understand what happens when we set some variables to 0.
        Assuming that there are no forces acting on the system, and giving desired joint accelerations of 0, this
        method will return :math:`\tau = S(q,\dot{q}) \dot{q} + g(q)`. If in addition joint velocities are also 0,
        it will return :math:`\tau = g(q)` which can for instance be useful for gravity compensation.

        For forward dynamics, which computes the joint accelerations given the joint positions, velocities, and
        torques (that is, :math:`\ddot{q} = FD(model, q, \dot{q}, \tau)`, this can be computed using
        :math:`\ddot{q} = H^{-1} (\tau - C)` (see also `computeFullFD`). For more information about different
        control schemes (position, force, impedance control and others), or about the formulation of the equation
        of motion in task/operational space (instead of joint space), check the references [1-4].

        Args:
            body_id (int): body unique id.
            q (np.float[N]): joint positions
            dq (np.float[N]): joint velocities
            des_ddq (np.float[N]): desired joint accelerations

        Returns:
            np.float[N]: joint torques computed using the rigid-body equation of motion

        References:
            [1] "Rigid Body Dynamics Algorithms", Featherstone, 2008, chap1.1
            [2] "Robotics: Modelling, Planning and Control", Siciliano et al., 2010
            [3] "Springer Handbook of Robotics", Siciliano et al., 2008
            [4] Lecture on "Impedance Control" by Prof. De Luca, Universita di Roma,
                http://www.diag.uniroma1.it/~deluca/rob2_en/15_ImpedanceControl.pdf
        """
        pass

    def calculate_forward_dynamics(self, body_id, q, dq, torques):
        r"""
        Given the specified joint positions :math:`q` and velocities :math:`\dot{q}`, and joint torques :math:`\tau`,
        it computes the joint accelerations :math:`\ddot{q}`. That is, :math:`\ddot{q} = FD(model, q, \dot{q}, \tau)`.

        Specifically, it uses the rigid-body equation of motion in joint space given by (see [1]):

        .. math:: \ddot{q} = H(q)^{-1} (\tau - C(q,\dot{q}))

        where :math:`\tau` is the vector of applied torques, :math:`H(q)` is the inertia matrix, and
        :math:`C(q,\dot{q}) \dot{q}` is the vector accounting for Coriolis, centrifugal forces, gravity, and any
        other forces acting on the system except the applied torques :math:`\tau`.

        Normally, a more popular form of this equation of motion (in joint space) is given by:

        .. math:: H(q) \ddot{q} + S(q,\dot{q}) \dot{q} + g(q) = \tau + J^T(q) F

        which is the same as the first one with :math:`C = S\dot{q} + g(q) - J^T(q) F`. However, this last formulation
        is useful to understand what happens when we set some variables to 0.
        Assuming that there are no forces acting on the system, and giving desired joint torques of 0, this
        method will return :math:`\ddot{q} = - H(q)^{-1} (S(q,\dot{q}) \dot{q} + g(q))`. If in addition
        the joint velocities are also 0, it will return :math:`\ddot{q} = - H(q)^{-1} g(q)` which are
        the accelerations due to gravity.

        For inverse dynamics, which computes the joint torques given the joint positions, velocities, and
        accelerations (that is, :math:`\tau = ID(model, q, \dot{q}, \ddot{q})`, this can be computed using
        :math:`\tau = H(q)\ddot{q} + C(q,\dot{q})`. For more information about different
        control schemes (position, force, impedance control and others), or about the formulation of the equation
        of motion in task/operational space (instead of joint space), check the references [1-4].

        Args:
            body_id (int): unique body id.
            q (np.float[N]): joint positions
            dq (np.float[N]): joint velocities
            torques (np.float[N]): desired joint torques

        Returns:
            np.float[N]: joint accelerations computed using the rigid-body equation of motion

        References:
            [1] "Rigid Body Dynamics Algorithms", Featherstone, 2008, chap1.1
            [2] "Robotics: Modelling, Planning and Control", Siciliano et al., 2010
            [3] "Springer Handbook of Robotics", Siciliano et al., 2008
            [4] Lecture on "Impedance Control" by Prof. De Luca, Universita di Roma,
                http://www.diag.uniroma1.it/~deluca/rob2_en/15_ImpedanceControl.pdf
        """
        pass

    # debug

    def add_user_debug_line(self, from_pos, to_pos, rgb_color=None, width=None, lifetime=None, parent_object_id=None,
                            parent_link_id=None, line_id=None):
        """Add a user debug line in the simulator.

        You can add a 3d line specified by a 3d starting point (from) and end point (to), a color [red,green,blue],
        a line width and a duration in seconds.

        Args:
            from_pos (np.float[3]): starting point of the line in Cartesian world coordinates
            to_pos (np.float[3]): end point of the line in Cartesian world coordinates
            rgb_color (np.float[3]): RGB color (each channel in range [0,1])
            width (float): line width (limited by OpenGL implementation).
            lifetime (float): use 0 for permanent line, or positive time in seconds (afterwards the line with be
                removed automatically)
            parent_object_id (int): draw line in local coordinates of a parent object.
            parent_link_id (int): draw line in local coordinates of a parent link.
            line_id (int): replace an existing line item (to avoid flickering of remove/add).

        Returns:
            int: unique user debug line id.
        """
        pass

    def add_user_debug_text(self, text, position, rgb_color=None, size=None, lifetime=None, orientation=None,
                            parent_object_id=None, parent_link_id=None, text_id=None):
        """
        Add 3D text at a specific location using a color and size.

        Args:
            text (str): text.
            position (np.float[3]): 3d position of the text in Cartesian world coordinates.
            rgb_color (list/tuple of 3 floats): RGB color; each component in range [0..1]
            size (float): text size
            lifetime (float): use 0 for permanent text, or positive time in seconds (afterwards the text with be
                removed automatically)
            orientation (np.float[4]): By default, debug text will always face the camera, automatically rotation.
                By specifying a text orientation (quaternion), the orientation will be fixed in world space or local
                space (when parent is specified). Note that a different implementation/shader is used for camera
                facing text, with different appearance: camera facing text uses bitmap fonts, text with specified
                orientation uses TrueType font.
            parent_object_id (int): draw text in local coordinates of a parent object.
            parent_link_id (int): draw text in local coordinates of a parent link.
            text_id (int): replace an existing text item (to avoid flickering of remove/add).

        Returns:
            int: unique user debug text id.
        """
        pass

    def add_user_debug_parameter(self, name, min_range, max_range, start_value):
        """
        Add custom sliders to tune parameters.

        Args:
            name (str): name of the parameter.
            min_range (float): minimum value.
            max_range (float): maximum value.
            start_value (float): starting value.

        Returns:
            int: unique user debug parameter id.
        """
        pass

    def read_user_debug_parameter(self, parameter_id):
        """
        Read the value of the parameter / slider.

        Args:
            parameter_id: unique user debug parameter id.

        Returns:
            float: reading of the parameter.
        """
        pass

    def remove_user_debug_item(self, item_id):
        """
        Remove the specified user debug item (line, text, parameter) from the simulator.

        Args:
            item_id (int): unique id of the debug item to be removed (line, text etc)
        """
        pass

    def remove_all_user_debug_items(self):
        """
        Remove all user debug items from the simulator.
        """
        pass

    def set_debug_object_color(self, object_id, link_id, rgb_color=(1, 0, 0)):
        """
        Override the color of a specific object and link.

        Args:
            object_id (int): unique object id.
            link_id (int): link id.
            rgb_color (float[3]): RGB debug color.
        """
        pass

    def add_user_data(self, object_id, key, value):
        """
        Add user data (at the moment text strings) attached to any link of a body. You can also override a previous
        given value. You can add multiple user data to the same body/link.

        Args:
            object_id (int): unique object/link id.
            key (str): key string.
            value (str): value string.

        Returns:
            int: user data id.
        """
        pass

    def num_user_data(self, object_id):
        """
        Return the number of user data associated with the specified object/link id.

        Args:
            object_id (int): unique object/link id.

        Returns:
            int: the number of user data
        """
        pass

    def get_user_data(self, user_data_id):
        """
        Get the specified user data value.

        Args:
            user_data_id (int): unique user data id.

        Returns:
            str: value string.
        """
        pass

    def get_user_data_id(self, object_id, key):
        """
        Get the specified user data id.

        Args:
            object_id (int): unique object/link id.
            key (str): key string.

        Returns:
            int: user data id.
        """
        pass

    def get_user_data_info(self, object_id, index):
        """
        Get the user data info associated with the given object and index.

        Args:
            object_id (int): unique object id.
            index (int): index (should be between [0, self.num_user_data(object_id)]).

        Returns:
            int: user data id.
            str: key.
            int: body id.
            int: link index
            int: visual shape index.
        """
        pass

    def remove_user_data(self, user_data_id):
        """
        Remove the specified user data.

        Args:
            user_data_id (int): user data id.
        """
        pass

    def sync_user_data(self):
        """
        Synchronize the user data.
        """
        pass

    def configure_debug_visualizer(self, flag, enable):
        """Configure the debug visualizer camera.

        Configure some settings of the built-in OpenGL visualizer, such as enabling or disabling wireframe,
        shadows and GUI rendering.

        Args:
            flag (int): The feature to enable or disable, such as
                        COV_ENABLE_WIREFRAME (=3): show/hide the collision wireframe
                        COV_ENABLE_SHADOWS (=2): show/hide shadows
                        COV_ENABLE_GUI (=1): enable/disable the GUI
                        COV_ENABLE_VR_PICKING (=5): enable/disable VR picking
                        COV_ENABLE_VR_TELEPORTING (=4): enable/disable VR teleporting
                        COV_ENABLE_RENDERING (=7): enable/disable rendering
                        COV_ENABLE_TINY_RENDERER (=12): enable/disable tiny renderer
                        COV_ENABLE_VR_RENDER_CONTROLLERS (=6): render VR controllers
                        COV_ENABLE_KEYBOARD_SHORTCUTS (=9): enable/disable keyboard shortcuts
                        COV_ENABLE_MOUSE_PICKING (=10): enable/disable mouse picking
                        COV_ENABLE_Y_AXIS_UP (Z is default world up axis) (=11): enable/disable Y axis up
                        COV_ENABLE_RGB_BUFFER_PREVIEW (=13): enable/disable RGB buffer preview
                        COV_ENABLE_DEPTH_BUFFER_PREVIEW (=14): enable/disable Depth buffer preview
                        COV_ENABLE_SEGMENTATION_MARK_PREVIEW (=15): enable/disable segmentation mark preview
            enable (bool): False (disable) or True (enable)
        """
        pass

    def get_debug_visualizer(self):
        """Get information about the debug visualizer camera.

        Returns:
            float: width of the visualizer camera
            float: height of the visualizer camera
            np.float[4,4]: view matrix [4,4]
            np.float[4,4]: perspective projection matrix [4,4]
            np.float[3]: camera up vector expressed in the Cartesian world space
            np.float[3]: forward axis of the camera expressed in the Cartesian world space
            np.float[3]: This is a horizontal vector that can be used to generate rays (for mouse picking or creating
                a simple ray tracer for example)
            np.float[3]: This is a vertical vector that can be used to generate rays (for mouse picking or creating a
                simple ray tracer for example)
            float: yaw angle (in radians) of the camera, in Cartesian local space coordinates
            float: pitch angle (in radians) of the camera, in Cartesian local space coordinates
            float: distance between the camera and the camera target
            np.float[3]: target of the camera, in Cartesian world space coordinates
        """
        pass

    def reset_debug_visualizer(self, distance, yaw, pitch, target_position):
        """Reset the debug visualizer camera.

        Reset the 3D OpenGL debug visualizer camera distance (between eye and camera target position), camera yaw and
        pitch and camera target position

        Args:
            distance (float): distance from eye to camera target position
            yaw (float): camera yaw angle (in radians) left/right
            pitch (float): camera pitch angle (in radians) up/down
            target_position (np.float[3]): target focus point of the camera
        """
        pass

    # events (mouse, keyboard)

    def get_keyboard_events(self):
        """Get the key events.

        Returns:
            dict: {keyId: keyState}
                * `keyID` is an integer (ascii code) representing the key. Some special keys like shift, arrows,
                and others are are defined in pybullet such as `B3G_SHIFT`, `B3G_LEFT_ARROW`, `B3G_UP_ARROW`,...
                * `keyState` is an integer. 3 if the button has been pressed, 1 if the key is down, 2 if the key has
                been triggered.
        """
        pass

    def get_mouse_events(self):
        """Get the mouse events.

        Returns:
            list of mouse events:
                eventType (int): 1 if the mouse is moving, 2 if a button has been pressed or released
                mousePosX (float): x-coordinates of the mouse pointer
                mousePosY (float): y-coordinates of the mouse pointer
                buttonIdx (int): button index for left/middle/right mouse button. It is -1 if nothing,
                                 0 if left button, 1 if scroll wheel (pressed), 2 if right button
                buttonState (int): 0 if nothing, 3 if the button has been pressed, 4 is the button has been released,
                                   1 if the key is down (never observed), 2 if the key has been triggered (never
                                   observed).
        """
        pass

    def get_mouse_and_keyboard_events(self):
        """Get the mouse and key events.

        Returns:
            list: list of mouse events
            dict: dictionary of key events
        """
        pass
