#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Define the Isaac SDK simulator API.

This is the main interface that communicates with the Isaac SDK/Gym simulator [1]. By defining this interface, it
allows to decouple the PyRoboLearn framework from the simulator. It also converts some data types to the ones required
by Isaac Gym. For instance, some methods in `isaacgym` do not accept numpy arrays but only `gymapi` data types such
as `gymapi.Vec3`, `gymapi.Quat`, and others. The interface provided here makes the necessary conversions.

The signature of each method defined here are inspired by [2] but in accordance with the PEP8 style guide [3].
Parts of the documentation for the methods have been copied-pasted from [2] for completeness purposes.

Dependencies in PRL:
* `pyrobolearn.simulators.simulator.Simulator`

References:
    - [1] Isaac SDK: https://developer.nvidia.com/isaac-sdk
    - [2] PyBullet Quickstart Guide: https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA
    - [3] PEP8: https://www.python.org/dev/peps/pep-0008/
    - [4] Isaac gym slides:
      https://developer.download.nvidia.com/video/gputechconf/gtc/2019/presentation/s9918-isaac-gym.pdf
"""

# TODO: waiting for its release in September

import os
import time
import numpy as np
from collections import OrderedDict

try:
    import isaacgym
    from isaacgym import gymapi
except ImportError as e:
    raise ImportError(str(e) + "\nTry to install `Isaac Gym`!")

from pyrobolearn.simulators.simulator import Simulator


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Nvidia Isaac", "Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class Isaac(Simulator):
    r"""Isaac simulator.

    "Isaac Sim is a virtual robotics laboratory, a high-fidelity 3D world simulator, that accelerates the research,
    design and development of robots by reducing both cost and risk. Developers can quickly and easily train and test
    their robots created with the Isaac SDK, in detailed, highly realistic scenarios resulting robots that can safely
    operate and cooperate with humans." [1]

    References:
        - [1] https://developer.nvidia.com/isaac-sdk
        - [2] https://www.nvidia.com/en-au/deep-learning-ai/industries/robotics/
        - [3] "GPU-Accelerated Robotic Simulation for Distributed Reinforcement Learning", Liang et al., 2018
        - [4] Slides: https://developer.download.nvidia.com/video/gputechconf/gtc/2019/presentation/s9918-isaac-gym.pdf
    """

    def __init__(self, render=True, num_instances=1, middleware=None, **kwargs):
        """
        Initialize Isaac gym simulator.

        Args:
            render (bool): if True, it will open the GUI, otherwise, it will just run the server.
            num_instances (int): number of simulator instances.
            middleware (MiddleWare, None): middleware instance.
            **kwargs (dict): optional arguments (this is not used here).
        """
        super(Isaac, self).__init__(render=render, num_instances=num_instances, middleware=middleware, **kwargs)

        # define variables
        self.gym = gymapi.acquire_gym()
        self.sim = self.gym.create_sim()
        self.params = gymapi.SimParams()
        self.gym.get_sim_params(self.sim, self.params)
        spacing = 10
        lower, upper = gymapi.Vec3(-spacing, -spacing, 0.), gymapi.Vec3(spacing, spacing, spacing)
        self.env = self.gym.create_env(self.sim, lower, upper)
        self.viewer = None

        self.dt = 1./60
        self.num_substeps = 2

        if render:  # gui
            self.viewer = self.gym.create_viewer(None, 1920, 1080)

        # keep track of loaded bodies
        self.bodies = OrderedDict()  # {body_id: Body}
        self._body_cnt = 0

    ##################
    # Static methods #
    ##################

    @staticmethod
    def simulate_gas_dynamics():
        """Return True if the simulator can simulate gases."""
        return False

    @staticmethod
    def simulate_liquid_dynamics():
        """Return True if the simulator can simulate liquids."""
        return True  # using Flex

    @staticmethod
    def simulate_fluid_dynamics():
        """Return True if the simulator can simulate fluids (gases and liquids)."""
        return Simulator.simulate_gas_dynamics() and Simulator.simulate_liquid_dynamics()

    @staticmethod
    def simulate_soft_bodies():
        """Return True if the simulator can simulate soft bodies."""
        return True  # using Flex

    @staticmethod
    def has_middleware_communication_layer():
        """Return True if the simulator has a middleware communication layer (like ROS, YARP, etc)."""
        return False

    @staticmethod
    def supports_dynamic_loading():
        """Return True if the simulator supports the dynamic loading of models."""
        return True

    @staticmethod
    def supports_acceleration():
        """Return True if the simulator provides acceleration (dynamic) information (such as joint accelerations, link
        Cartesian accelerations, etc). If not, the `Robot` class will have to implement these using finite
        difference."""
        return False

    @staticmethod
    def supports_sensors(sensor_type=None):
        """Return True if the simulator provides supports for the specified sensor."""
        return False

    @staticmethod
    def supports_urdf():
        """Return True if we can use URDFs."""
        return True

    @staticmethod
    def supports_light():
        """Return True if we can define and access to the lights in the simulator."""
        return False

    @staticmethod
    def supports_depth_image():
        """Return True if we can get depth images from the simulator."""
        return False

    @staticmethod
    def supports_segmentation_images():
        """Return True if we can get segmentation images from the simulator."""
        return False

    @staticmethod
    def supports_visualization():
        """Return True if there is a graphical user interface (GUI)."""
        return True

    @staticmethod
    def supports_interactive_gui():
        """Return True if the simulator has an interactive GUI."""
        return False

    @staticmethod
    def supports_mousekeyboard_events():
        """Return True if the simulator allows to capture mouse and keyboard events."""
        return False

    @staticmethod
    def supports_visual_objects():
        """Return True if we can simulate objects that do not have collision shapes."""
        return False

    @staticmethod
    def supports_plugins():
        """Return True if we can use plugins."""
        return False

    @staticmethod
    def supports_constraints(constraint_type):
        """Return True if we can support the specified constraint type."""
        return False

    @staticmethod
    def supports_realtime():
        """Return True if the simulator supports real-time (meaning we don't need to step manually in the simulator).
        Note that if we can step in the simulator, we can use threads to simulate the real-time. So the return value
        should always be True."""
        return True

    @staticmethod
    def supports_ray_casting():
        """Return True if the simulator supports ray casting."""
        return False

    @staticmethod
    def can_step():
        """Return True if we can step manually in the simulator."""
        return True

    @staticmethod
    def can_load_heightmap():
        """Return True if the simulator can load a heightmap."""
        return False

    ###########
    # Methods #
    ###########

    ##############
    # Simulators #
    ##############

    def step(self, sleep_time=0):
        """Perform a step in the simulator, and sleep the specified amount of time.

        Args:
            sleep_time (float): amount of time to sleep after performing one step in the simulation.
        """
        # step the simulation
        self.gym.simulate(self.sim, self.dt, self.num_substeps)
        self.gym.fetch_results(self.sim, True)

        # update the viewer
        if self.viewer is not None:
            if self.gym.query_viewer_has_closed(self.viewer):
                exit()
            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, True)

            # wait for dt to elapse in real-time.
            # This synchronizes the physics simulation with the rendering rate.
            self.gym.sync_frame_time(self.sim)

        # time.sleep(sleep_time)

    def render(self, enable=True):
        """Render the simulation.

        Args:
            enable (bool): If True, it will render the simulator by enabling the GUI.
        """
        self._render = enable
        if self._render:
            if self.viewer is None:
                self.viewer = self.gym.create_viewer(None, 1920, 1080)
        else:
            if self.viewer is not None:
                pass  # close the viewer set viewer to None

    def get_physics_properties(self):
        """Get the physics engine parameters.

        Returns:
            dict: dictionary containing the physics simulator properties.
        """
        properties = dict()
        gravity = self.params.gravity  # return gymapi.Vec3
        gravity = np.array([gravity[0], gravity[1], gravity[2]])
        properties['gravity'] = gravity
        properties['solver_type'] = self.params.solver_type
        properties['num_outer_iterations'] = self.params.num_outer_iterations
        properties['num_inner_iterations'] = self.params.num_inner_iterations
        properties['relaxation'] = self.params.relaxation
        properties['warm_start'] = self.params.warm_start
        properties['num_substeps'] = self.num_substeps
        return properties

    def set_physics_properties(self, solver_type=None, num_outer_iterations=None, num_inner_iterations=None,
                               relaxation=None, warm_start=None, num_substeps=None, *args, **kwargs):
        """Set the physics engine parameters."""
        if solver_type is not None:
            self.params.solver_type = int(solver_type)
        if num_outer_iterations is not None:
            self.params.num_outer_iterations = int(num_outer_iterations)
        if num_inner_iterations is not None:
            self.params.num_inner_iterations = int(num_inner_iterations)
        if relaxation is not None:
            self.params.relaxation = float(relaxation)
        if warm_start is not None:
            self.params.warm_start = float(warm_start)
        if num_substeps is not None:
            self.num_substeps = int(num_substeps) if num_substeps >= 1 else 1

    def get_gravity(self):
        """Return the gravity set in the simulator."""
        gravity = self.params.gravity  # return gymapi.Vec3
        return np.asarray([gravity[0], gravity[1], gravity[2]])

    def set_gravity(self, gravity=(0, 0, -9.81)):
        """Set the gravity in the simulator with the given acceleration.

        By default, there is no gravitational force enabled in the simulator.

        Args:
            gravity (list, tuple of 3 floats): acceleration in the x, y, z directions.
        """
        gravity = gymapi.Vec3(*gravity)
        self.params.gravity = gravity
        self.gym.set_sim_params(self.sim, self.params)

    ######################################
    # Loading URDFs, SDFs, MJCFs, meshes #
    ######################################

    def load_urdf(self, filename, position, orientation, use_fixed_base=0, scale=1.0, *args, **kwargs):
        """Load a URDF file in the simulator.

        Args:
            filename (str): a relative or absolute path to the URDF file on the file system of the physics server.
            position (np.array[float[3]]): create the base of the object at the specified position in world space
              coordinates [x,y,z].
            orientation (np.array[float[4]]): create the base of the object at the specified orientation as world
              space quaternion [x,y,z,w].
            use_fixed_base (bool): force the base of the loaded object to be static
            scale (float): scale factor to the URDF model.

        Returns:
            int (non-negative): unique id associated to the load model.
        """
        dirname = os.path.dirname(filename)
        filename = os.path.basename(filename)
        name = ''.join(filename.split('.')[:-1])
        asset = self.gym.load_asset(dirname, filename)

        position = gymapi.Vec3(*position)
        orientation = gymapi.Quat(*orientation)
        pose = gymapi.Transform(position, orientation)
        self.gym.create_actor(self.env, asset, pose, name)

        self._body_cnt += 1
        self.bodies[self._body_cnt] = name
        return self._body_cnt

    def load_mjcf(self, filename, scaling=1., *args, **kwargs):
        """Load a Mujoco file in the simulator.

        Args:
            filename (str): a relative or absolute path to the MJCF file on the file system of the physics server.
            scaling (float): scale factor for the object

        Returns:
            list(int): list of object unique id for each object loaded
        """
        dirname = os.path.dirname(filename)
        filename = os.path.basename(filename)
        name = ''.join(filename.split('.')[:-1])
        asset = self.gym.load_asset(dirname, filename)

        self.gym.create_actor(self.env, asset, name)

        self._body_cnt += 1
        self.bodies[self._body_cnt] = name
        return self._body_cnt
