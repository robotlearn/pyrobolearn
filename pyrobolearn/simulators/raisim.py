#!/usr/bin/env python
"""Define the RaiSim Simulator API.

This is the main interface that communicates with the RaiSim simulator [1-5]. By defining this interface, it
allows to decouple the PyRoboLearn framework from the simulator. It also converts some data types to the ones required
by RaiSim. Because it didn't have a Python wrapper, one has been written using ``pybind11`` [6] and is available at:
https://github.com/robotlearn/raisimpy

The signature of each method defined here are inspired by [1,2] but in accordance with the PEP8 style guide [7].
Parts of the documentation for the methods have been copied-pasted from [2-5] for completeness purposes.

RaiSim is distributed under the End-User License Agreement (EULA) [8], and officially works on Ubuntu 16.04 and 18.04.

- Supported Python versions: Python 2.7 and 3.5
- Python wrappers: Pybind11 [10]

Dependencies in PRL:
* `pyrobolearn.simulators.simulator.Simulator`

References:
    - [1] "Per-Contact Iteration Method for Solving Contact Dynamics", Hwangbo et al., 2018
    - [2] RaiSim benchmarks: https://leggedrobotics.github.io/SimBenchmark/about/sims.html
    - [3] RaiSim, a physics engine for robotics and AI research: https://github.com/leggedrobotics/raisimLib
    - [4] raisimOgre - Visualizer for raisim: https://github.com/leggedrobotics/raisimOgre
    - [5] raisimGym - RL examples using raisim: https://github.com/leggedrobotics/raisimGym
    - [6] pybind11 (documentation): https://pybind11.readthedocs.io/en/stable/
    - [7] PEP8: https://www.python.org/dev/peps/pep-0008/
    - [8] RaiSim license: https://github.com/leggedrobotics/raisimLib/blob/master/LICENSE.md
    - [9] RaiSimPy - A Python wrapper for Raisim: https://github.com/robotlearn/raisimpy
    - [10] Pybind11: https://pybind11.readthedocs.io/en/stable/
        - Cython, pybind11, cffi – which tool should you choose?:
          http://blog.behnel.de/posts/cython-pybind11-cffi-which-tool-to-choose.html
"""

import os
import time
from collections import OrderedDict
import numpy as np


# import raisim
try:
    import raisimpy as raisim
except ImportError as e:
    print(e.__str__() + "\nHINT: you need to install `raisimLib` and `raisimOgre`, and build the Python wrappers "
                        "that are located in the `raisim_wrapper` folder.")

# import PRL simulator
from pyrobolearn.simulators.simulator import Simulator
from pyrobolearn.utils.decorator import keyboard_interrupt


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2019, PyRoboLearn"
__credits__ = ["RaiSim (ETHz, Hwangbo, Kang, Lee)", "Brian Delhaisse (raisimpy + PRL)"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class Raisim(Simulator):
    r"""RaiSim

    This is a wrapper around ``raisimpy`` [6] which is a Python around the RaiSim simulator [1-5].

    Examples:
        sim = Raisim()

    References:
        - [1] "Per-Contact Iteration Method for Solving Contact Dynamics", Hwangbo et al., 2018
        - [2] RaiSim, a physics engine for robotics and AI research: https://github.com/leggedrobotics/raisimLib
        - [3] raisimOgre - Visualizer for raisim: https://github.com/leggedrobotics/raisimOgre
        - [4] raisimGym - RL examples using raisim: https://github.com/leggedrobotics/raisimGym
        - [5] RaiSim benchmark:  https://leggedrobotics.github.io/SimBenchmark/about/sims.html
        - [6] RaiSimPy - A Python wrapper for Raisim: https://github.com/robotlearn/raisimpy
    """

    def __init__(self, render=True, num_instances=1, **kwargs):
        """
        Initialize the Raisim simulator.

        Args:
            render (bool): if True, it will open the GUI, otherwise, it will just run the server.
            num_instances (int): number of simulator instances.
            **kwargs (dict): optional arguments (this is not used here).
        """
        super(Raisim, self).__init__(render, **kwargs)

        # create world
        self.world = raisim.World()
        self.sim = self.world  # alias

        # define default timestep
        self.default_timestep = self.world.get_time_step()  # 0.005
        self.dt = self.default_timestep

        # create visualizer and render if specified
        self.visualizer = None
        self._desired_fps = 60
        self.visualization_cnt = 0

        if self._render:
            self._init_visualization()

        # keep track of the loaded bodies
        self._bodies = OrderedDict()  # {body_id: Body}
        self._body_cnt = 0  # 0 is for the world

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
        return self.world.get_gravity()

    @gravity.setter
    def gravity(self, gravity):
        """Set the gravity in the simulator."""
        self.world.set_gravity(gravity)

    @property
    def camera(self):
        """Return the camera (yaw, pitch, distance, target_position) or None."""
        return self._camera

    @property
    def timestep(self):
        """Return the simulator time step."""
        return self.dt

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

    ##################
    # Static methods #
    ##################

    @staticmethod
    def in_simulation():
        """Return True if we are running in simulation instead of the real-world."""
        return True

    @staticmethod
    def simulate_gas_dynamics():
        """Return True if the simulator can simulate gases."""
        return False

    @staticmethod
    def simulate_liquid_dynamics():
        """Return True if the simulator can simulate liquids."""
        return False

    @staticmethod
    def simulate_fluid_dynamics():
        """Return True if the simulator can simulate fluids (gases and liquids)."""
        return Simulator.simulate_gas_dynamics() and Simulator.simulate_liquid_dynamics()

    @staticmethod
    def simulate_soft_bodies():
        """Return True if the simulator can simulate soft bodies."""
        return False

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
        return True

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
        return True

    @staticmethod
    def supports_mousekeyboard_events():
        """Return True if the simulator allows to capture mouse and keyboard events."""
        return True  # however this requires some code

    @staticmethod
    def supports_visual_objects():
        """Return True if we can simulate objects that do not have collision shapes."""
        return True  # however it has to be coded in the wrapper

    @staticmethod
    def supports_plugins():
        """Return True if we can use plugins."""
        return False

    @staticmethod
    def supports_constraints(constraint_type):
        """Return True if we can support the specified constraint type."""
        if constraint_type == 'wire':
            return True
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
        return True

    ###########
    # Methods #
    ###########

    def _init_visualization(self):

        def normalize(array):
            return np.asarray(array) / np.linalg.norm(array)

        def setup_callback():
            vis = raisim.OgreVis.get()

            # light
            light = vis.get_light()
            light.set_diffuse_color(1, 1, 1)
            light.set_cast_shadows(True)
            # light.set_direction(normalize([-3., -3., -0.5]))
            vis.get_light_node().set_position(3, 3, 3)

            # load textures
            vis.add_resource_directory(vis.get_resource_dir() + "/material/gravel")
            vis.load_material("gravel.material")

            vis.add_resource_directory(vis.get_resource_dir() + "/material/checkerboard")
            vis.load_material("checkerboard.material")

            # shadow setting
            manager = vis.get_scene_manager()
            manager.set_shadow_technique(raisim.ogre.ShadowTechnique.SHADOWTYPE_TEXTURE_ADDITIVE)
            manager.set_shadow_texture_settings(2048, 3)

            # scale related settings!! Please adapt it depending on your map size
            # beyond this distance, shadow disappears
            manager.set_shadow_far_distance(10)
            # size of contact points and contact forces
            vis.set_contact_visual_object_size(0.03, 0.6)
            # speed of camera motion in freelook mode
            vis.get_camera_man().set_top_speed(10)

        # these methods must be called before initApp
        vis = raisim.OgreVis.get()
        vis.set_world(self.world)
        vis.set_window_size(1800, 1000)
        vis.set_default_callbacks()
        vis.set_setup_callback(setup_callback)
        vis.set_anti_aliasing(2)

        # init
        vis.init_app()

        # set visualizer
        self.visualizer = vis

        # camera
        camera = self.visualizer.get_camera_man().get_camera()
        camera.set_position(8, -12, 6)
        camera.pitch(1.2)
        camera.yaw(0.6, raisim.ogre.Node.TransformSpace.TS_WORLD)

    #############
    # Simulator #
    #############

    def reset(self, *args, **kwargs):
        """Reset the simulator."""
        pass

    def close(self):
        """Close the simulator."""
        if self.visualizer is not None:
            self.visualizer.close_app()
            self.visualizer = None

    def seed(self, seed=None):
        """Set the given seed in the simulator."""
        # if seed is not None:
        #     self.world.set_seed(seed)
        pass

    @keyboard_interrupt
    def step(self, sleep_time=0):
        """Perform a step in the simulator, and sleep the specified amount of time.

        Args:
            sleep_time (float): amount of time to sleep after performing one step in the simulation.
        """
        # update world/simulator
        self.world.integrate()

        # if we need to render
        if self._render:  # TODO: should we create the visualizer in a thread??
            # if the visualizer is not defined, create one
            if self.visualizer is None:
                self._init_visualization()

            # update the frame if time to update it
            vis_decimation = int(1. / (self._desired_fps * self.dt) + 1.e-10)
            if self.visualization_cnt % vis_decimation == 0:
                self.visualizer.render_one_frame()
                self.visualization_cnt = 0

            # update visualization counter
            self.visualization_cnt += 1

        if sleep_time:
            time.sleep(sleep_time)

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
        return self.world.get_time_step()

    def set_time_step(self, time_step):
        """Set the time step in the simulator.

        Args:
            time_step (float): Each time you call 'step' the time step will proceed with 'time_step'.
        """
        self.dt = time_step
        self.world.set_time_step(time_step)

    def set_real_time(self, enable=True):
        """Enable real time in the simulator.

        Args:
            enable (bool): If True, it will enable the real-time simulation. If False, it will disable it.
        """
        self.real_time = True

    def use_real_time(self):
        """Return True if the simulator is in real-time mode."""
        return self.real_time

    def pause(self):
        """Pause the simulator if in real-time."""
        raisim.gui.manual_stepping = True

    def unpause(self):
        """Unpause the simulator if in real-time."""
        raisim.gui.manual_stepping = False

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
        return self.world.get_gravity()

    def set_gravity(self, gravity=(0, 0, -9.81)):
        """Set the gravity in the simulator with the given acceleration.

        Args:
            gravity (list, tuple of 3 floats): acceleration in the x, y, z directions.
        """
        self.world.set_gravity(gravity)

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

    ######################################
    # Loading URDFs, SDFs, MJCFs, meshes #
    ######################################

    def load_urdf(self, filename, position, orientation=None, use_fixed_base=0, scale=1.0, *args, **kwargs):
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
        # load body
        body = self.world.add_articulated_system(filename)

        # set the position and orientation
        body.set_base_position(position)
        if orientation is not None:
            body.set_base_orientation(orientation)

        # set initial gains
        num_dof = body.get_dof()
        body.set_control_mode(raisim.ControlMode.PD_PLUS_FEEDFORWARD_TORQUE)
        joint_p_gain, joint_d_gain = np.zeros(num_dof), np.zeros(num_dof)
        joint_p_gain[-num_dof+6:] = 200.
        joint_d_gain[-num_dof+6:] = 10.
        body.set_pd_gains(joint_p_gain, joint_d_gain)

        # increment body counter and remember the body
        self._body_cnt += 1
        self._bodies[self._body_cnt] = body

        # if we need to render create visual shape
        if self._render:
            self.visualizer.create_graphical_object(body, name="body_" + str(self._body_cnt))

        return self._body_cnt

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

    ##########
    # Bodies #
    ##########

    def create_primitive_object(self, shape_type, position, mass, orientation=(0., 0., 0., 1.), radius=0.5,
                                half_extents=(.5, .5, .5), height=1., filename=None, mesh_scale=(1., 1., 1.),
                                plane_normal=(0., 0., 1.), rgba_color=None, specular_color=None, frame_position=None,
                                frame_orientation=None, vertices=None, indices=None, uvs=None, normals=None, flags=-1):
        """Create a primitive object in the simulator. This is basically the combination of `create_visual_shape`,
        `create_collision_shape`, and `create_body`.

        Args:
            shape_type (int): type of shape; GEOM_SPHERE (=2), GEOM_BOX (=3), GEOM_CAPSULE (=7), GEOM_CYLINDER (=4),
                GEOM_PLANE (=6), GEOM_MESH (=5)
            position (np.array[float[3]]): Cartesian world position of the base
            mass (float): mass of the base, in kg (if using SI units)
            orientation (np.array[float[4]]): Orientation of base as quaternion [x,y,z,w]
            radius (float): only for GEOM_SPHERE, GEOM_CAPSULE, GEOM_CYLINDER
            half_extents (np.array[float[3]], list/tuple of 3 floats): only for GEOM_BOX.
            height (float): only for GEOM_CAPSULE, GEOM_CYLINDER (height = length).
            filename (str): Filename for GEOM_MESH, currently only Wavefront .obj. Will create convex hulls for each
                object (marked as 'o') in the .obj file.
            mesh_scale (np.array[float[3]], list/tuple of 3 floats): scale of mesh (only for GEOM_MESH).
            plane_normal (np.array[float[3]], list/tuple of 3 floats): plane normal (only for GEOM_PLANE).
            rgba_color (list/tuple of 4 floats): color components for red, green, blue and alpha, each in range [0..1].
            specular_color (list/tuple of 3 floats): specular reflection color, red, green, blue components in range
                [0..1]
            frame_position (np.array[float[3]]): translational offset of the visual and collision shape with respect
              to the link frame.
            frame_orientation (np.array[float[4]]): rotational offset (quaternion x,y,z,w) of the visual and collision
              shape with respect to the link frame.
            vertices (list[np.array[float[3]]]): Instead of creating a mesh from obj file, you can provide vertices,
              indices, uvs and normals
            indices (list[int]): triangle indices, should be a multiple of 3.
            uvs (list of np.array[2]): uv texture coordinates for vertices. Use `changeVisualShape` to choose the
              texture image. The number of uvs should be equal to number of vertices.
            normals (list[np.array[float[3]]]): vertex normals, number should be equal to number of vertices.
            flags (int): unused / to be decided

        Returns:
            int: non-negative unique id for primitive object, or -1 for failure
        """
        material = "white"
        if rgba_color is not None:
            rgb = rgba_color[:3]
            if rgb == (1, 0, 0):
                material = "red"
            elif rgb == (0, 1, 0):
                material = "green"
            elif rgb == (0, 0, 1):
                material = "blue"
            elif rgb == (1, 1, 1):
                material = "white"
            else:
                print("Material not defined for the given color")

        if shape_type == self.GEOM_BOX:
            x, y, z = half_extents
            body = self.world.add_box(2*x, 2*y, 2*z, mass=mass, material=material)
        elif shape_type == self.GEOM_SPHERE:
            body = self.world.add_sphere(radius=radius, mass=mass, material=material)
        elif shape_type == self.GEOM_CAPSULE:
            body = self.world.add_capsule(radius=radius, height=height, mass=mass, material=material)
        elif shape_type == self.GEOM_CYLINDER:
            body = self.world.add_cylinder(radius=radius, height=height, mass=mass, material=material)
        elif shape_type == self.GEOM_CONE:
            body = self.world.add_cone(radius=radius, height=height, mass=mass, material=material)
        # elif shape_type == self.GEOM_MESH:
            # body = self.world.add_mesh(file_name=filename, mass=mass)
        else:
            raise NotImplementedError("Primitive object not defined for the given shape type.")

        # set the position and orientation
        body.set_position(position)
        if orientation is not None:
            body.set_orientation(orientation)

        # increment body counter and remember the body
        self._body_cnt += 1
        self._bodies[self._body_cnt] = body

        # if we need to render create visual shape
        if self._render:
            self.visualizer.create_graphical_object(body, name="body_" + str(self._body_cnt), material=material)

        return self._body_cnt

    def load_floor(self, dimension=20):
        """Load a floor in the simulator.

        Args:
            dimension (float): dimension of the floor.

        Returns:
            int: non-negative unique id for the floor, or -1 for failure.
        """
        ground = self.world.add_ground()
        if self._render:
            self.visualizer.create_graphical_object(ground, dimension=dimension, name="floor",
                                                    material="checkerboard_green")
        self._body_cnt += 1
        self._bodies[self._body_cnt] = ground
        return self._body_cnt

    def create_body(self, visual_shape_id=-1, collision_shape_id=-1, mass=0., position=(0., 0., 0.),
                    orientation=(0., 0., 0., 1.), *args, **kwargs):
        """Create a body in the simulator.

        Args:
            visual_shape_id (int): unique id from createVisualShape or -1. You can reuse the visual shape (instancing)
            collision_shape_id (int): unique id from createCollisionShape or -1. You can re-use the collision shape
                for multiple multibodies (instancing)
            mass (float): mass of the base, in kg (if using SI units)
            position (np.array[float[3]]): Cartesian world position of the base
            orientation (np.array[float[4]]): Orientation of base as quaternion [x,y,z,w]

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

    ###############
    # Constraints #
    ###############

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
            joint_axis (np.array[float[3]]): joint axis, in child link frame
            parent_frame_position (np.array[float[3]]): position of the joint frame relative to parent CoM frame.
            child_frame_position (np.array[float[3]]): position of the joint frame relative to a given child CoM frame
                (or world origin if no child specified)
            parent_frame_orientation (np.array[float[4]]): the orientation of the joint frame relative to parent CoM
                coordinate frame
            child_frame_orientation (np.array[float[4]]): the orientation of the joint frame relative to the child CoM
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

    ###########
    # Objects #
    ###########

    #############################
    # Robots (joints and links) #
    #############################

    #################
    # Visualization #
    #################

    ##############
    # Collisions #
    ##############


# Tests
if __name__ == '__main__':
    from itertools import count

    sim = Raisim(render=True)
    print("Gravity: {}".format(sim.get_gravity()))

    # load floor
    floor = sim.load_floor(dimension=20)

    # create box
    box = sim.create_primitive_object(sim.GEOM_BOX, position=(0, 0, 2), mass=1, rgba_color=(1, 0, 0, 1))
    sphere = sim.create_primitive_object(sim.GEOM_SPHERE, position=(2, 0, 2), mass=1, rgba_color=(0, 1, 0, 1))
    capsule = sim.create_primitive_object(sim.GEOM_CAPSULE, position=(0, -2, 2), mass=1, rgba_color=(0, 0, 1, 1))
    cylinder = sim.create_primitive_object(sim.GEOM_CYLINDER, position=(0, 2, 2), mass=1)

    # load robot
    path = os.path.dirname(os.path.abspath(__file__)) + '/../robots/urdfs/anymal/anymal.urdf'
    robot = sim.load_urdf(path, position=(3, -3, 2))

    # perform step
    for t in count():
        sim.step(sleep_time=sim.dt)
