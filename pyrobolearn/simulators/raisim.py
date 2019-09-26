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
from pyrobolearn.utils.mesh import convert_mesh
from pyrobolearn.utils.parsers.robots import URDFParser


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

    def __init__(self, render=True, num_instances=1, middleware=None, **kwargs):
        """
        Initialize the Raisim simulator.

        Args:
            render (bool): if True, it will open the GUI, otherwise, it will just run the server.
            num_instances (int): number of simulator instances.
            middleware (MiddleWare, None): middleware instance.
            **kwargs (dict): optional arguments (this is not used here).
        """
        super(Raisim, self).__init__(render=render, num_instances=num_instances, middleware=middleware, **kwargs)

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

    #################
    # utils methods #
    #################

    @staticmethod
    def _convert_wxyz_to_xyzw(q):
        """Convert a quaternion in the (w,x,y,z) format to (x,y,z,w)."""
        q = np.asarray(q)
        return np.roll(q, shift=-1, axis=q.ndim - 1)

    @staticmethod
    def _convert_xyzw_to_wxyz(q):
        """Convert a quaternion in the (x,y,z,w) format to (w,x,y,z)."""
        q = np.asarray(q)
        return np.roll(q, shift=1, axis=q.ndim - 1)

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

        # if sleep_time:
        #     time.sleep(sleep_time)

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

    @staticmethod
    def _convert_mesh(filename, format='obj'):
        extension = filename.split('.')[-1]
        if extension.lower() != format:  # if different file format than obj convert it
            basename = os.path.basename(filename)
            basename_without_extension = ''.join(basename.split('.')[:-1])
            # dirname = os.path.dirname(os.path.abspath(__file__)) + '/meshes/'  # Raisim uses relative paths
            new_filename = basename_without_extension + '.' + format
            if not os.path.isfile(new_filename):
                convert_mesh(filename, 'meshes/' + new_filename, library='pyassimp')
            return True, new_filename
        return False, filename

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
        # parse the URDF file
        urdf_parser = URDFParser(filename=filename)
        tree = urdf_parser.tree

        # Raisim only accepts collision bodies in the obj format, so check that each mesh is in the correct format.
        # If not, convert them.
        urdf_changed = False
        for body in tree.bodies.values():
            for visual in body.visuals:
                if visual.dtype == 'mesh':
                    urdf_changed, new_filename = self._convert_mesh(visual.filename)
                    visual.filename = new_filename
            for collision in body.collisions:
                if collision.dtype == 'mesh':
                    urdf_changed, new_filename = self._convert_mesh(collision.filename)
                    collision.filename = new_filename

        # if we had to convert some meshes, just create a new URDF with the converted meshes
        if urdf_changed:
            root = urdf_parser.generate(tree)
            basename = os.path.basename(filename)
            dirname = os.path.dirname(os.path.abspath(__file__)) + '/meshes/'
            filename = dirname + 'prl_generated_' + basename
            urdf_parser.write(filename, root=root)

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
        body = self._bodies.pop(body_id)

        # remove body from the world
        self.world.remove_object(body)

        # remove body from visualization
        if self._render:
            self.visualizer.remove(body)

    def num_bodies(self):
        """Return the number of bodies present in the simulator.

        Returns:
            int: number of bodies
        """
        # return len(self._bodies)
        # return len(self.world.get_object_list)
        return self.world.get_configuration_number()

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

    def get_mass(self, body_id):
        """
        Return the total mass of the robot (=sum of all mass links).

        Args:
            body_id (int): unique object id, as returned from `load_urdf`.

        Returns:
            float: total mass of the robot [kg]
        """
        body = self._bodies[body_id]
        if isinstance(body, raisim.SingleBodyObject):
            return body.get_mass(0)
        return sum(body.get_masses())

    def get_base_mass(self, body_id):
        """Return the base mass of the robot.

        Args:
            body_id (int): unique object id.
        """
        body = self._bodies[body_id]
        if isinstance(body, raisim.SingleBodyObject):
            return body.get_mass(0)
        return body.get_masses()[0]

    def get_base_name(self, body_id):
        """
        Return the base name.

        Args:
            body_id (int): unique object id.

        Returns:
            str: base name
        """
        body = self._bodies[body_id]
        if isinstance(body, raisim.SingleBodyObject):
            return body.get_name()
        return body.get_body_names()[0]  # body.get_name()

    def get_center_of_mass_position(self, body_id, link_ids=None):
        """
        Return the center of mass position.

        Args:
            body_id (int): unique body id.
            link_ids (list[int]): link ids associated with the given body id. If None, it will take all the links
                of the specified body.

        Returns:
            np.array[float[3]]: center of mass position in the Cartesian world coordinates
        """
        body = self._bodies[body_id]
        if isinstance(body, raisim.SingleBodyObject):
            return body.get_com_position()
        return body.get_composite_com()

    def get_center_of_mass_velocity(self, body_id, link_ids=None):
        """
        Return the center of mass linear velocity.

        Args:
            body_id (int): unique body id.
            link_ids (list[int]): link ids associated with the given body id. If None, it will take all the links
                of the specified body.

        Returns:
            np.array[float[3]]: center of mass linear velocity.
        """
        body = self._bodies[body_id]
        if isinstance(body, raisim.SingleBodyObject):
            return body.get_linear_velocity()
        return body.get_world_linear_velocity(body_id=body_id, body_pos=np.zeros(3))  # TODO: correct body_id

    def get_base_pose(self, body_id):
        """
        Get the current position and orientation of the base (or root link) of the body in Cartesian world coordinates.

        Args:
            body_id (int): object unique id, as returned from `load_urdf`.

        Returns:
            np.array[float[3]]: base position
            np.array[float[4]]: base orientation (quaternion [x,y,z,w])
        """
        body = self._bodies[body_id]
        if isinstance(body, raisim.SingleBodyObject):
            return body.get_position(), self._convert_wxyz_to_xyzw(body.get_quaternion())
        pos, quat = body.get_body_pose(0)
        return pos, self._convert_wxyz_to_xyzw(quat)

    def get_base_position(self, body_id):
        """
        Return the base position of the specified body.

        Args:
            body_id (int): object unique id, as returned from `load_urdf`.

        Returns:
            np.array[float[3]]: base position.
        """
        body = self._bodies[body_id]
        if isinstance(body, raisim.SingleBodyObject):
            return body.get_position()
        return body.get_world_position(0)

    def get_base_orientation(self, body_id):
        """
        Get the base orientation of the specified body.

        Args:
            body_id (int): object unique id, as returned from `load_urdf`.

        Returns:
            np.array[float[4]]: base orientation in the form of a quaternion (x,y,z,w)
        """
        body = self._bodies[body_id]
        if isinstance(body, raisim.SingleBodyObject):
            return self._convert_wxyz_to_xyzw(body.get_quaternion())
        return self._convert_wxyz_to_xyzw(body.get_base_quaternion())

    def reset_base_pose(self, body_id, position, orientation):
        """
        Reset the base position and orientation of the specified object id.

        Args:
            body_id (int): unique object id.
            position (np.array[float[3]]): new base position.
            orientation (np.array[float[4]]): new base orientation (expressed as a quaternion [x,y,z,w])
        """
        body = self._bodies[body_id]
        if isinstance(body, raisim.SingleBodyObject):
            body.set_position(position)
            body.set_orientation(self._convert_xyzw_to_wxyz(orientation))
        else:
            body.set_base_position(position)
            body.set_base_orientation(self._convert_xyzw_to_wxyz(orientation))

    def reset_base_position(self, body_id, position):
        """
        Reset the base position of the specified body/object id while preserving its orientation.

        Args:
            body_id (int): unique object id.
            position (np.array[float[3]]): new base position.
        """
        body = self._bodies[body_id]
        if isinstance(body, raisim.SingleBodyObject):
            body.set_position(position)
        else:
            body.set_base_position(position)

    def reset_base_orientation(self, body_id, orientation):
        """
        Reset the base orientation of the specified body/object id while preserving its position.

        Args:
            body_id (int): unique object id.
            orientation (np.array[float[4]]): new base orientation (expressed as a quaternion [x,y,z,w])
        """
        body = self._bodies[body_id]
        if isinstance(body, raisim.SingleBodyObject):
            body.set_orientation(self._convert_xyzw_to_wxyz(orientation))
        else:
            body.set_base_orientation(self._convert_xyzw_to_wxyz(orientation))

    def get_base_velocity(self, body_id):
        """
        Return the base linear and angular velocities.

        Args:
            body_id (int): object unique id, as returned from `load_urdf`.

        Returns:
            np.array[float[3]]: linear velocity of the base in Cartesian world space coordinates
            np.array[float[3]]: angular velocity of the base in Cartesian world space coordinates
        """
        body = self._bodies[body_id]
        if isinstance(body, raisim.SingleBodyObject):
            return body.get_linear_velocity(), body.get_angular_velocity()
        return body.get_world_linear_velocity(0), body.get_world_angular_velocity(0)

    def get_base_linear_velocity(self, body_id):
        """
        Return the linear velocity of the base.

        Args:
            body_id (int): object unique id, as returned from `load_urdf`.

        Returns:
            np.array[float[3]]: linear velocity of the base in Cartesian world space coordinates
        """
        body = self._bodies[body_id]
        if isinstance(body, raisim.SingleBodyObject):
            return body.get_linear_velocity()
        return body.get_world_linear_velocity(0)

    def get_base_angular_velocity(self, body_id):
        """
        Return the angular velocity of the base.

        Args:
            body_id (int): object unique id, as returned from `load_urdf`.

        Returns:
            np.array[float[3]]: angular velocity of the base in Cartesian world space coordinates
        """
        body = self._bodies[body_id]
        if isinstance(body, raisim.SingleBodyObject):
            return body.get_angular_velocity()
        return body.get_world_angular_velocity(0)

    def reset_base_velocity(self, body_id, linear_velocity=None, angular_velocity=None):
        """
        Reset the base velocity.

        Args:
            body_id (int): unique object id.
            linear_velocity (np.array[float[3]]): new linear velocity of the base.
            angular_velocity (np.array[float[3]]): new angular velocity of the base.
        """
        body = self._bodies[body_id]
        if isinstance(body, raisim.SingleBodyObject):
            body.set_velocity(linear_velocity, angular_velocity)
        else:
            # TODO: request feature on Raisim github
            pass

    def reset_base_linear_velocity(self, body_id, linear_velocity):
        """
        Reset the base linear velocity.

        Args:
            body_id (int): unique object id.
            linear_velocity (np.array[float[3]]): new linear velocity of the base
        """
        body = self._bodies[body_id]
        if isinstance(body, raisim.SingleBodyObject):
            body.set_velocity(linear_velocity, np.zeros(3))
        else:
            # TODO: request feature on Raisim github
            pass

    def reset_base_angular_velocity(self, body_id, angular_velocity):
        """
        Reset the base angular velocity.

        Args:
            body_id (int): unique object id.
            angular_velocity (np.array[float[3]]): new angular velocity of the base
        """
        body = self._bodies[body_id]
        if isinstance(body, raisim.SingleBodyObject):
            body.set_velocity(np.zeros(3), angular_velocity)
        else:
            # TODO: request feature on Raisim github
            pass

    def get_base_acceleration(self, body_id):
        """
        Get the base acceleration. This is only valid if the simulator `supports_acceleration`.

        Args:
            body_id (int): unique object id.

        Returns:
            np.array[float[3]]: linear acceleration [m/s^2]
            np.array[float[3]]: angular acceleration [rad/s^2]
        """
        pass  # Raisim does not support accelerations

    def apply_external_force(self, body_id, link_id=-1, force=(0., 0., 0.), position=(0., 0., 0.),
                             frame=Simulator.LINK_FRAME):
        """
        Apply the specified external force on the specified position on the body / link.

        Args:
            body_id (int): unique body id.
            link_id (int): unique link id. If -1, it will be the base.
            force (np.array[float[3]]): external force to be applied.
            position (np.array[float[3]]): position on the link where the force is applied. See `flags` for coordinate
                systems. If None, it is the center of mass of the body (or the link if specified).
            frame (int): if frame = 1, then the force / position is described in the link frame. If frame = 2, they
                are described in the world frame.
        """
        body = self._bodies[body_id]
        if isinstance(body, raisim.SingleBodyObject):
            body.setExternalForce(link_id + 1, force)
        else:
            if frame == Simulator.LINK_FRAME:
                frame = raisim.ArticulatedSystem.Frame.BODY_FRAME
            elif frame == Simulator.WORLD_FRAME:
                frame = raisim.ArticulatedSystem.Frame.WORLD_FRAME
            else:
                raise ValueError("Unknown specified frame.")
            body.setExternalForce(link_id + 1, frame, force, frame, position)

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
        body = self._bodies[body_id]
        body.setExternalTorque(link_id + 1, torque)

    #############################
    # Robots (joints and links) #
    #############################

    def num_joints(self, body_id):
        """
        Return the total number of joints of the specified body. This is the same as calling `num_links`.

        Args:
            body_id (int): unique body id.

        Returns:
            int: number of joints with the associated body id.
        """
        body = self._bodies[body_id]
        if isinstance(body, raisim.SingleBodyObject):
            return 0
        return len(body.get_body_names())

    def num_actuated_joints(self, body_id):
        """
        Return the total number of actuated joints associated with the given body id.

        Args:
            body_id (int): unique body id.

        Returns:
            int: number of actuated joints of the specified body.
        """
        body = self._bodies[body_id]
        if isinstance(body, raisim.SingleBodyObject):
            return 0
        return body.get_num_dof()

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
            np.array[float[6]]: These are the joint reaction forces, if a torque sensor is enabled for this joint it is
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
            joint_ids (list[int]): list of joint ids.

        Returns:
            list:
                float: The position value of this joint.
                float: The velocity value of this joint.
                np.array[float[6]]: These are the joint reaction forces, if a torque sensor is enabled for this joint
                    it is [Fx, Fy, Fz, Mx, My, Mz]. Without torque sensor, it is [0, 0, 0, 0, 0, 0].
                float: This is the motor torque applied during the last `step`. Note that this only applies in
                    VELOCITY_CONTROL and POSITION_CONTROL. If you use TORQUE_CONTROL then the applied joint motor
                    torque is exactly what you provide, so there is no need to report it separately.
        """
        pass

    def reset_joint_state(self, body_id, joint_id, position, velocity=None):
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
            positions (float, np.array[float[N]]): target joint position(s) (used in POSITION_CONTROL).
            velocities (float, np.array[float[N]]): target joint velocity(ies). In VELOCITY_CONTROL and
                POSITION_CONTROL, the target velocity(ies) is(are) the desired velocity of the joint. Note that the
                target velocity(ies) is(are) not the maximum joint velocity(ies). In PD_CONTROL and
                POSITION_CONTROL/CONTROL_MODE_POSITION_VELOCITY_PD, the final target velocities are computed using:
                `kp*(erp*(desiredPosition-currentPosition)/dt)+currentVelocity+kd*(m_desiredVelocity - currentVelocity)`
            forces (float, list[float]): in POSITION_CONTROL and VELOCITY_CONTROL, these are the maximum motor
                forces used to reach the target values. In TORQUE_CONTROL these are the forces / torques to be applied
                each simulation step.
            kp (float, list[float]): position (stiffness) gain(s) (used in POSITION_CONTROL).
            kd (float, list[float]): velocity (damping) gain(s) (used in POSITION_CONTROL).
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
            np.array[float[3]]: Cartesian world position of CoM
            np.array[float[4]]: Cartesian world orientation of CoM, in quaternion [x,y,z,w]
            np.array[float[3]]: local position offset of inertial frame (center of mass) expressed in the URDF link
                frame
            np.array[float[4]]: local orientation (quaternion [x,y,z,w]) offset of the inertial frame expressed in
                URDF link frame
            np.array[float[3]]: world position of the URDF link frame
            np.array[float[4]]: world orientation of the URDF link frame
            np.array[float[3]]: Cartesian world linear velocity. Only returned if `compute_velocity` is True.
            np.array[float[3]]: Cartesian world angular velocity. Only returned if `compute_velocity` is True.
        """
        pass

    def get_link_states(self, body_id, link_ids, compute_velocity=False, compute_forward_kinematics=False):
        """
        Get the state of the associated links.

        Args:
            body_id (int): body unique id.
            link_ids (list[int]): list of link index.
            compute_velocity (bool): If True, the Cartesian world velocity will be computed and returned.
            compute_forward_kinematics (bool): if True, the Cartesian world position/orientation will be recomputed
                using forward kinematics.

        Returns:
            list:
                np.array[float[3]]: Cartesian position of CoM
                np.array[float[4]]: Cartesian orientation of CoM, in quaternion [x,y,z,w]
                np.array[float[3]]: local position offset of inertial frame (center of mass) expressed in the URDF
                    link frame
                np.array[float[4]]: local orientation (quaternion [x,y,z,w]) offset of the inertial frame expressed
                    in URDF link frame
                np.array[float[3]]: world position of the URDF link frame
                np.array[float[4]]: world orientation of the URDF link frame
                np.array[float[3]]: Cartesian world linear velocity. Only returned if `compute_velocity` is True.
                np.array[float[3]]: Cartesian world angular velocity. Only returned if `compute_velocity` is True.
        """
        pass

    def get_link_names(self, body_id, link_ids):
        """
        Return the name of the given link(s).

        Args:
            body_id (int): unique body id.
            link_ids (int, list[int]): link id, or list of link ids.

        Returns:
            if 1 link:
                str: link name
            if multiple links:
                str[N]: link names
        """
        body = self._bodies[body_id]
        if isinstance(body, raisim.SingleBodyObject):
            return body.get_name()
        names = body.get_body_names()
        if isinstance(link_ids, int):
            return names[link_ids]
        return np.asarray(names)[link_ids]

    def get_link_masses(self, body_id, link_ids):
        """
        Return the mass of the given link(s).

        Args:
            body_id (int): unique body id.
            link_ids (int, list[int]): link id, or list of link ids.

        Returns:
            if 1 link:
                float: mass of the given link
            else:
                float[N]: mass of each link
        """
        body = self._bodies[body_id]
        if isinstance(body, raisim.SingleBodyObject):
            return body.get_mass(0)
        masses = body.get_masses()
        if isinstance(link_ids, int):
            return masses[link_ids]
        return np.asarray(masses)[link_ids]

    def get_link_frames(self, body_id, link_ids):
        r"""
        Return the link world frame position(s) and orientation(s).

        Args:
            body_id (int): body id.
            link_ids (int, int[N]): link id, or list of desired link ids.

        Returns:
            if 1 link:
                np.array[float[3]]: the link frame position in the world space
                np.array[float[4]]: Cartesian orientation of the link frame [x,y,z,w]
            if multiple links:
                np.array[float[N,3]]: link frame position of each link in world space
                np.array[float[N,4]]: orientation of each link frame [x,y,z,w]
        """
        body = self._bodies[body_id]
        # get_frame_world_position
        # get_frame_world_quaternion

    def get_link_world_positions(self, body_id, link_ids):
        """
        Return the CoM position (in the Cartesian world space coordinates) of the given link(s).

        Args:
            body_id (int): unique body id.
            link_ids (int, list[int]): link index, or list of link indices.

        Returns:
            if 1 link:
                np.array[float[3]]: the link CoM position in the world space
            if multiple links:
                np.array[float[N,3]]: CoM position of each link in world space
        """
        body = self._bodies[body_id]
        # get_link_coms  # in body frame
        # get_frame_world_position

    def get_link_positions(self, body_id, link_ids):
        pass

    def get_link_world_orientations(self, body_id, link_ids):
        """
        Return the CoM orientation (in the Cartesian world space) of the given link(s).

        Args:
            body_id (int): unique body id.
            link_ids (int, list[int]): link index, or list of link indices.

        Returns:
            if 1 link:
                np.array[float[4]]: Cartesian orientation of the link CoM (x,y,z,w)
            if multiple links:
                np.array[float[N,4]]: CoM orientation of each link (x,y,z,w)
        """
        pass

    def get_link_orientations(self, body_id, link_ids):
        pass

    def get_link_world_linear_velocities(self, body_id, link_ids):
        """
        Return the linear velocity of the link(s) expressed in the Cartesian world space coordinates.

        Args:
            body_id (int): unique body id.
            link_ids (int, list[int]): link index, or list of link indices.

        Returns:
            if 1 link:
                np.array[float[3]]: linear velocity of the link in the Cartesian world space
            if multiple links:
                np.array[float[N,3]]: linear velocity of each link
        """
        body = self._bodies[body_id]
        # get_frame_linear_velocity
        # get_world_linear_velocity

    def get_link_world_angular_velocities(self, body_id, link_ids):
        """
        Return the angular velocity of the link(s) in the Cartesian world space coordinates.

        Args:
            body_id (int): unique body id.
            link_ids (int, list[int]): link index, or list of link indices.

        Returns:
            if 1 link:
                np.array[float[3]]: angular velocity of the link in the Cartesian world space
            if multiple links:
                np.array[float[N,3]]: angular velocity of each link
        """
        body = self._bodies[body_id]
        # get_frame_angular_velocity
        # get_world_angular_velocity

    def get_link_world_velocities(self, body_id, link_ids):
        """
        Return the linear and angular velocities (expressed in the Cartesian world space coordinates) for the given
        link(s).

        Args:
            body_id (int): unique body id.
            link_ids (int, list[int]): link index, or list of link indices.

        Returns:
            if 1 link:
                np.array[float[6]]: linear and angular velocity of the link in the Cartesian world space
            if multiple links:
                np.array[float[N,6]]: linear and angular velocity of each link
        """
        body = self._bodies[body_id]
        # get_frame_linear_velocity
        # get_world_linear_velocity
        # get_frame_angular_velocity
        # get_world_angular_velocity

    def get_link_velocities(self, body_id, link_ids):
        pass

    def get_link_world_linear_accelerations(self, body_id, link_ids):
        """
        Return the linear acceleration of the link(s) expressed in the Cartesian world space coordinates.

        Args:
            body_id (int): unique body id.
            link_ids (int, list[int]): link index, or list of link indices.

        Returns:
            if 1 link:
                np.array[float[3]]: linear acceleration of the link in the Cartesian world space
            if multiple links:
                np.array[float[N,3]]: linear acceleration of each link
        """
        pass  # Raisim does not support accelerations

    def get_link_world_angular_accelerations(self, body_id, link_ids):
        """
        Return the angular acceleration of the link(s) in the Cartesian world space coordinates.

        Args:
            body_id (int): unique body id.
            link_ids (int, list[int]): link index, or list of link indices.

        Returns:
            if 1 link:
                np.array[float[3]]: angular acceleration of the link in the Cartesian world space
            if multiple links:
                np.array[float[N,3]]: angular acceleration of each link
        """
        pass  # Raisim does not support accelerations

    def get_link_world_accelerations(self, body_id, link_ids):
        """
        Return the linear and angular accelerations (expressed in the Cartesian world space coordinates) for the given
        link(s). This is only valid if the simulator `supports_acceleration`.

        Args:
            body_id (int): unique body id.
            link_ids (int, list[int]): link index, or list of link indices.

        Returns:
            if 1 link:
                np.array[float[6]]: linear and angular acceleration of the link in the Cartesian world space
            if multiple links:
                np.array[float[N,6]]: linear and angular acceleration of each link
        """
        pass  # Raisim does not support accelerations

    def get_q_indices(self, body_id, joint_ids):
        """
        Get the corresponding q index of the given joint(s).

        Args:
            body_id (int): unique body id.
            joint_ids (int, list[int]): a joint id, or list of joint ids.

        Returns:
            if 1 joint:
                int: q index
            if multiple joints:
                list[int]: q indices
        """
        pass

    def get_actuated_joint_ids(self, body_id):
        """
        Get the actuated joint ids associated with the given body id.

        Args:
            body_id (int): unique body id.

        Returns:
            list[int]: actuated joint ids.
        """
        pass

    def get_joint_names(self, body_id, joint_ids):
        """
        Return the name of the given joint(s).

        Args:
            body_id (int): unique body id.
            joint_ids (int, list[int]): a joint id, or list of joint ids.

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
            joint_ids (int, list[int]): a joint id, or list of joint ids.

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
            joint_ids (int, list[int]): a joint id, or list of joint ids.

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
            joint_ids (int, list[int]): a joint id, or list of joint ids.

        Returns:
            if 1 joint:
                float: damping coefficient of the given joint
            if multiple joints:
                np.array[float[N]]: damping coefficient for each specified joint
        """
        pass

    def get_joint_frictions(self, body_id, joint_ids):
        """
        Get the friction coefficient of the given joint(s).

        Args:
            body_id (int): unique body id.
            joint_ids (int, list[int]): a joint id, or list of joint ids.

        Returns:
            if 1 joint:
                float: friction coefficient of the given joint
            if multiple joints:
                np.array[float[N]]: friction coefficient for each specified joint
        """
        pass

    def get_joint_limits(self, body_id, joint_ids):
        """
        Get the joint limits of the given joint(s).

        Args:
            body_id (int): unique body id.
            joint_ids (int, list[int]): a joint id, or list of joint ids.

        Returns:
            if 1 joint:
                np.array[float[2]]]: lower and upper limit
            if multiple joints:
                np.array[N,2]: lower and upper limit for each specified joint
        """
        pass

    def get_joint_max_forces(self, body_id, joint_ids):
        """
        Get the maximum force that can be applied on the given joint(s).

        Warning: Note that this is not automatically used in position, velocity, or torque control.

        Args:
            body_id (int): unique body id.
            joint_ids (int, list[int]): a joint id, or list of joint ids.

        Returns:
            if 1 joint:
                float: maximum force [N]
            if multiple joints:
                np.array[float[N]]: maximum force for each specified joint [N]
        """
        pass

    def get_joint_max_velocities(self, body_id, joint_ids):
        """
        Get the maximum velocity that can be applied on the given joint(s).

        Warning: Note that this is not automatically used in position, velocity, or torque control.

        Args:
            body_id (int): unique body id.
            joint_ids (int, list[int]): a joint id, or list of joint ids.

        Returns:
            if 1 joint:
                float: maximum velocity [rad/s]
            if multiple joints:
                np.array[float[N]]: maximum velocities for each specified joint [rad/s]
        """
        pass

    def get_joint_axes(self, body_id, joint_ids):
        """
        Get the joint axis about the given joint(s).

        Args:
            body_id (int): unique body id.
            joint_ids (int, list[int]): a joint id, or list of joint ids.

        Returns:
            if 1 joint:
                np.array[float[3]]: joint axis
            if multiple joint:
                np.array[float[N,3]]: list of joint axis
        """
        pass

    def set_joint_positions(self, body_id, joint_ids, positions, velocities=None, kps=None, kds=None, forces=None):
        """
        Set the position of the given joint(s) (using position control).

        Args:
            body_id (int): unique body id.
            joint_ids (int, list[int]): joint id, or list of joint ids.
            positions (float, np.array[float[N]]): desired position, or list of desired positions [rad]
            velocities (None, float, np.array[float[N]]): desired velocity, or list of desired velocities [rad/s]
            kps (None, float, np.array[float[N]]): position gain(s)
            kds (None, float, np.array[float[N]]): velocity gain(s)
            forces (None, float, np.array[float[N]]): maximum motor force(s)/torque(s) used to reach the target values.
        """
        pass

    def get_joint_positions(self, body_id, joint_ids):
        """
        Get the position of the given joint(s).

        Args:
            body_id (int): unique body id.
            joint_ids (int, list[int]): joint id, or list of joint ids.

        Returns:
            if 1 joint:
                float: joint position [rad]
            if multiple joints:
                np.array[float[N]]: joint positions [rad]
        """
        pass

    def set_joint_velocities(self, body_id, joint_ids, velocities, max_force=None):
        """
        Set the velocity of the given joint(s) (using velocity control).

        Args:
            body_id (int): unique body id.
            joint_ids (int, list[int]): joint id, or list of joint ids.
            velocities (float, np.array[float[N]]): desired velocity, or list of desired velocities [rad/s]
            max_force (None, float, np.array[float[N]]): maximum motor forces/torques
        """
        pass

    def get_joint_velocities(self, body_id, joint_ids):
        """
        Get the velocity of the given joint(s).

        Args:
            body_id (int): unique body id.
            joint_ids (int, list[int]): joint id, or list of joint ids.

        Returns:
            if 1 joint:
                float: joint velocity [rad/s]
            if multiple joints:
                np.array[float[N]]: joint velocities [rad/s]
        """
        pass

    def set_joint_accelerations(self, body_id, joint_ids, accelerations, q=None, dq=None):
        """
        Set the acceleration of the given joint(s) (using force control). This is achieved by performing inverse
        dynamic which given the joint accelerations compute the joint torques to be applied.

        Args:
            body_id (int): unique body id.
            joint_ids (int, list[int]): joint id, or list of joint ids.
            accelerations (float, np.array[float[N]]): desired joint acceleration, or list of desired joint
                accelerations [rad/s^2]
        """
        pass

    def get_joint_accelerations(self, body_id, joint_ids):  # , q=None, dq=None):
        """
        Get the acceleration of the specified joint(s). This is only valid if the simulator `supports_acceleration`.

        Args:
            body_id (int): unique body id.
            joint_ids (int, list[int]): joint id, or list of joint ids.

        Returns:
            if 1 joint:
                float: joint acceleration [rad/s^2]
            if multiple joints:
                np.array[float[N]]: joint accelerations [rad/s^2]
        """
        pass

    def set_joint_torques(self, body_id, joint_ids, torques):
        """
        Set the torque/force to the given joint(s) (using force/torque control).

        Args:
            body_id (int): unique body id.
            joint_ids (int, list[int]): joint id, or list of joint ids.
            torques (float, list[float], np.array[float]): desired torque(s) to apply to the joint(s) [N].
        """
        pass

    def get_joint_torques(self, body_id, joint_ids):
        """
        Get the applied torque(s) on the given joint(s).

        Args:
            body_id (int): unique body id.
            joint_ids (int, list[int]): a joint id, or list of joint ids.

        Returns:
            if 1 joint:
                float: torque [Nm]
            if multiple joints:
                np.array[float[N]]: torques associated to the given joints [Nm]
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
                np.array[float[6]]: joint reaction force (fx,fy,fz,mx,my,mz) [N,Nm]
            if multiple joints:
                np.array[float[N,6]]: joint reaction forces [N, Nm]
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
                np.array[float[N]]: power at each joint [W]
        """
        pass

    #################
    # Visualization #
    #################

    ##############
    # Collisions #
    ##############

    ###########################
    # Kinematics and Dynamics #
    ###########################

    def get_dynamics_info(self, body_id, link_id=-1):
        """
        Get dynamic information about the mass, center of mass, friction and other properties of the base and links.

        Args:
            body_id (int): body/object unique id.
            link_id (int): link/joint index or -1 for the base.

        Returns:
            float: mass in kg
            float: lateral friction coefficient
            np.array[float[3]]: local inertia diagonal. Note that links and base are centered around the center of
                mass and aligned with the principal axes of inertia.
            np.array[float[3]]: position of inertial frame in local coordinates of the joint frame
            np.array[float[4]]: orientation of inertial frame in local coordinates of joint frame
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
                        local_inertia_diagonal=None, inertia_position=None, inertia_orientation=None,
                        joint_damping=None, joint_friction=None):
        """
        Change dynamic properties of the given body (or link) such as mass, friction and restitution coefficients, etc.

        Args:
            body_id (int): object unique id, as returned by `load_urdf`, etc.
            link_id (int): link index or -1 for the base.
            mass (float): change the mass of the link (or base for link index -1)
            lateral_friction (float): lateral (linear) contact friction
            spinning_friction (float): torsional friction around the contact normal
            rolling_friction (float): torsional friction orthogonal to contact normal
            restitution (float): bounciness of contact. Keep it a bit less than 1.
            linear_damping (float): linear damping of the link (0.04 by default)
            angular_damping (float): angular damping of the link (0.04 by default)
            contact_stiffness (float): stiffness of the contact constraints, used together with `contact_damping`
            contact_damping (float): damping of the contact constraints for this body/link. Used together with
              `contact_stiffness`. This overrides the value if it was specified in the URDF file in the contact
              section.
            friction_anchor (int): enable or disable a friction anchor: positional friction correction (disabled by
              default, unless set in the URDF contact section)
            local_inertia_diagonal (np.array[float[3]]): diagonal elements of the inertia tensor. Note that the base
              and links are centered around the center of mass and aligned with the principal axes of inertia so
              there are no off-diagonal elements in the inertia tensor.
            inertia_position (np.array[float[3]]): new inertia position with respect to the link frame.
            inertia_orientation (np.array[float[4]]): new inertia orientation (expressed as a quaternion [x,y,z,w]
              with respect to the link frame.
            joint_damping (float): joint damping coefficient applied at each joint. This coefficient is read from URDF
              joint damping field. Keep the value close to 0.
              `joint_damping_force = -damping_coefficient * joint_velocity`.
            joint_friction (float): joint friction coefficient.
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
            local_position (np.array[float[3]]): the point on the specified link to compute the Jacobian (in link local
                coordinates around its center of mass). If None, it will use the CoM position (in the link frame).
            q (np.array[float[N]]): joint positions of size N, where N is the number of DoFs.
            dq (np.array[float[N]]): joint velocities of size N, where N is the number of DoFs.
            des_ddq (np.array[float[N]]): desired joint accelerations of size N.

        Returns:
            np.array[float[6,N]], np.array[float[6,6+N]]: full geometric (linear and angular) Jacobian matrix. The
                number of columns depends if the base is fixed or floating.
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
            q (np.array[float[N]]): joint positions of size N, where N is the total number of DoFs.

        Returns:
            np.array[float[N,N]], np.array[float[6+N,6+N]]: inertia matrix
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
            position (np.array[float[3]]): target position of the end effector (its link coordinate, not center of mass
                coordinate!). By default this is in Cartesian world space, unless you provide `q_curr` joint angles.
            orientation (np.array[float[4]]): target orientation in Cartesian world space, quaternion [x,y,w,z]. If not
                specified, pure position IK will be used.
            lower_limits (np.array[float[N]], list of N floats): lower joint limits. Optional null-space IK.
            upper_limits (np.array[float[N]], list of N floats): upper joint limits. Optional null-space IK.
            joint_ranges (np.array[float[N]], list of N floats): range of value of each joint.
            rest_poses (np.array[float[N]], list of N floats): joint rest poses. Favor an IK solution closer to a
                given rest pose.
            joint_dampings (np.array[float[N]], list of N floats): joint damping factors. Allow to tune the IK solution
                using joint damping factors.
            solver (int): p.IK_DLS (=0) or p.IK_SDLS (=1), Damped Least Squares or Selective Damped Least Squares, as
                described in the paper by Samuel Buss "Selectively Damped Least Squares for Inverse Kinematics".
            q_curr (np.array[float[N]]): list of joint positions. By default PyBullet uses the joint positions of the
                body. If provided, the target_position and targetOrientation is in local space!
            max_iters (int): maximum number of iterations. Refine the IK solution until the distance between target
                and actual end effector position is below this threshold, or the `max_iters` is reached.
            threshold (float): residual threshold. Refine the IK solution until the distance between target and actual
                end effector position is below this threshold, or the `max_iters` is reached.

        Returns:
            np.array[float[N]]: joint positions (for each actuated joint).
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
            q (np.array[float[N]]): joint positions
            dq (np.array[float[N]]): joint velocities
            des_ddq (np.array[float[N]]): desired joint accelerations

        Returns:
            np.array[float[N]]: joint torques computed using the rigid-body equation of motion

        References:
            - [1] "Rigid Body Dynamics Algorithms", Featherstone, 2008, chap1.1
            - [2] "Robotics: Modelling, Planning and Control", Siciliano et al., 2010
            - [3] "Springer Handbook of Robotics", Siciliano et al., 2008
            - [4] Lecture on "Impedance Control" by Prof. De Luca, Universita di Roma,
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
            q (np.array[float[N]]): joint positions
            dq (np.array[float[N]]): joint velocities
            torques (np.array[float[N]]): desired joint torques

        Returns:
            np.array[float[N]]: joint accelerations computed using the rigid-body equation of motion

        References:
            - [1] "Rigid Body Dynamics Algorithms", Featherstone, 2008, chap1.1
            - [2] "Robotics: Modelling, Planning and Control", Siciliano et al., 2010
            - [3] "Springer Handbook of Robotics", Siciliano et al., 2008
            - [4] Lecture on "Impedance Control" by Prof. De Luca, Universita di Roma,
                http://www.diag.uniroma1.it/~deluca/rob2_en/15_ImpedanceControl.pdf
        """
        pass


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
    # path = os.path.dirname(os.path.abspath(__file__)) + '/../robots/urdfs/kuka/kuka_iiwa/iiwa14.urdf'
    robot = sim.load_urdf(path, position=(3, -3, 2))

    print(sim.get_base_name(box))
    print(sim.get_mass(sphere))
    print(sim.get_mass(capsule))
    print(sim.get_mass(cylinder))
    print(sim.get_base_name(robot))

    # perform step
    for t in count():
        sim.step(sleep_time=sim.dt)
