#!/usr/bin/env python
"""Define the Simulator API.

All the simulators inherit from the interface defined here. This acts as a bridge between the simulator and
the PyRoboLearn framework. The signature of each method presents in this interface were inspired by the ones defined
in PyBullet [1,2], but in accordance with the PEP8 style guide [3].

Because the simulator is based on the PyBullet API and we want all the simulator APIs to be similar, all the other
simulators would have to be able to carry out operations such as querying the state of the robots, kinematics and
dynamics, .

Dependencies in PRL: None

References:
    [1] PyBullet: https://pybullet.org
    [2] PyBullet Quickstart Guide: https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA
    [3] PEP8: https://www.python.org/dev/peps/pep-0008/
"""

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "MIT"
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

    def __init__(self, render=True):
        self._render = render
        self.real_time = False

    ##############
    # Properties #
    ##############

    @property
    def version(self):
        """Return the version of the simulator."""
        return 0

    #############
    # Operators #
    #############

    def __repr__(self):
        """Return a string about the class for debugging and development."""
        return self.__class__.__name__

    def __str__(self):
        """Return a readable string about the class."""
        return self.__class__.__name__

    def __del__(self):
        """Close/Delete the simulator."""
        self.close()

    ###########
    # Methods #
    ###########

    # Simulators

    def reset(self):
        """Reset the simulator."""
        pass

    def close(self):
        """Close the simulator."""
        pass

    def seed(self, seed=None):
        """Set the given seed in the simulator."""
        pass

    def step(self, sleep_time=0):
        """Perform a step in the simulator, and sleep the specified time."""
        pass

    def render(self, flag=True):
        """Render the simulation."""
        pass

    def hide(self):
        """Hide the GUI."""
        self.render(False)

    def set_time_step(self, time_step):
        """Set the time step in the simulator."""
        pass

    def set_real_time(self):
        """Enable real time in the simulator."""
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

    def set_gravity(self, gravity=(0, 0, -9.81)):
        """Set the gravity in the simulator."""
        pass

    def save(self, on_disk=False):
        """Save the state of the simulator."""
        pass

    def load(self, state):
        """Load the simulator to a previous state."""
        pass

    def load_plugin(self, plugin):
        """Load a certain plugin in the simulator."""
        pass

    def execute_plugin_commands(self, plugin_id, commands):
        """Execute the commands on the specified plugin."""
        pass

    def unload_plugin(self, plugin_id):
        """Unload the specified plugin from the simulator."""
        pass

    # loading URDFs, SDFs, MJCFs

    def load_urdf(self, filename, position, orientation):
        """Load a URDF file in the simulator."""
        pass

    def load_sdf(self, filename):
        """Load a SDF file in the simulator."""
        pass

    def load_mjcf(self, filename):
        """Load a Mujoco file in the simulator."""
        pass

    def load_mesh(self, filename, position, orientation=(0, 0, 0, 1), mass=1., scale=(1., 1., 1.), color=(1, 1, 1, 1),
                 flags=None):
        """Load a mesh into the simulator.

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
        pass

    # bodies

    def create_visual_shape(self, shape_type, radius=0.5, half_extents=(1, 1, 1), length=1, filename='.obj'):
        pass

    def get_visual_shape_data(self, object_id):
        pass

    def create_collision_shape(self, shape_type, radius=0.5, half_extents=(1, 1, 1), length=1):
        pass

    def get_collision_shape_data(self):
        pass

    def create_body(self):
        """Create a body in the simulator."""
        pass

    def remove_body(self, body_id):
        """Remove a particular body in the simulator."""
        pass

    def num_bodies(self):
        """Return the number of bodies present in the simulator."""
        pass

    def get_body_info(self, body_id):
        """Get the specified body information."""
        pass

    def get_body_id(self):
        pass

    # constraint

    def create_constraint(self):
        pass

    def remove_constraint(self):
        pass

    def change_constraint(self):
        pass

    def get_num_constraint(self):
        pass

    def get_constraint_id(self):
        pass

    def get_constraint_info(self):
        pass

    def get_constraint_state(self):
        pass

    # objects

    def get_base_pose(self):
        pass

    def reset_base_pose(self):
        pass

    def get_base_position(self):
        pass

    def reset_base_position(self):
        pass

    def get_base_orientation(self):
        pass

    def reset_base_orientation(self):
        pass

    def get_base_velocity(self):
        pass

    def reset_base_velocity(self):
        pass

    def apply_external_force(self):
        pass

    def apply_external_torque(self):
        pass

    # robots (joints and links)

    def get_num_joints(self):
        pass

    def get_joint_info(self):
        pass

    def get_joint_state(self):
        pass

    def get_joint_states(self):
        pass

    def reset_joint_state(self):
        pass

    def enable_joint_force_torque_sensor(self):
        pass

    def set_joint_motor_control(self):
        pass

    def set_joint_motor_control_array(self):
        pass

    def get_link_state(self):
        pass

    # visualization

    def compute_view_matrix(self):
        pass

    def compute_projection_matrix(self):
        pass

    def get_camera_image(self):
        pass

    def load_texture(self):
        pass

    # collisions

    def get_overlapping_objects(self):
        pass

    def get_aabb(self):
        pass

    def get_contact_points(self):
        pass

    def get_closest_points(self):
        pass

    def ray_test(self):
        pass

    def ray_test_batch(self):
        pass

    # kinematics and dynamics

    def get_dynamics_info(self):
        pass

    def change_dynamics(self):
        pass

    def calculate_jacobian(self):
        pass

    def calculate_mass_matrix(self):
        pass

    def calculate_inverse_kinematics(self):
        pass

    def calculate_inverse_dynamics(self):
        pass

    def calculate_forward_dynamics(self):
        pass

    # debug

    def add_user_debug_line(self):
        pass

    def add_user_debug_text(self):
        pass

    def add_user_debug_parameter(self):
        pass

    def add_user_data(self):
        pass

    def configure_debug_visualizer(self):
        pass

    def get_debug_visualizer(self):
        pass

    def reset_debug_visualizer(self):
        pass

    # events (mouse, keyboard)

    def get_keyboard_events(self):
        pass

    def get_mouse_events(self):
        pass

    def get_mouse_and_keyboard_events(self):
        pass
