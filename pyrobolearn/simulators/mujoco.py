#!/usr/bin/env python
"""Define the MuJoCo Simulator API.

This is the main interface that communicates with the MuJoCo simulator [1]. By defining this interface, it allows to
decouple the PyRoboLearn framework from the simulator. It also converts some data types to the ones required by
MuJoCo.

Warnings:
    - The MuJoCo simulator requires a license in order to use it: https://www.roboti.us/license.html
    - You have to install the MuJoCo simulator beforehand: https://www.roboti.us/index.html
    - You have to install ``mujoco_py``:
        - for Python 2: https://github.com/openai/mujoco-py/tree/0.5
        - for Python 3: https://github.com/openai/mujoco-py
    - The authors of MuJoCo are working on another simulator called ``Optico`` so it is likely that MuJoCo won't
      be upgraded anymore.
    - This wrapper works only with Python 3 as they provide more functionalities. Also, the Python 3 API is pretty
      different from the Python 2.
    - You might have several errors if you don't export the correct environment variables beforehand.

- Supported Python versions: Python 3.*
- Python wrappers: Cython [4]

Dependencies in PRL:
* `pyrobolearn.simulators.simulator.Simulator`

References:
    - [1] MuJoCo: http://www.mujoco.org/
        - Documentation: http://mujoco.org/book
    - [2] MuJoCo Python: https://github.com/openai/mujoco-py
    - [3] DeepMind Control Suite: https://github.com/deepmind/dm_control/tree/master/dm_control/mujoco
    - [4] Cython: cython.org
        - Cython, pybind11, cffi – which tool should you choose?:
          http://blog.behnel.de/posts/cython-pybind11-cffi-which-tool-to-choose.html
"""

# import standard libraries
import os
import time
import numpy as np
import pickle
from collections import OrderedDict

# import XML parsers to parse / create XML for MuJoCo
import xml.etree.ElementTree as ET  # XML parser
from xml.dom import minidom  # to print in a pretty way the XML file

# import mesh converter (from .obj to .stl)
try:
    import pymesh    # rapid prototyping platform focused on geometry processing
    # doc: https://pymesh.readthedocs.io/en/latest/user_guide.html

    import pyassimp  # library to import and export various 3d-model-formats
    # doc: http://www.assimp.org/index.php
    # github: https://github.com/assimp/assimp
except ImportError as e:
    raise ImportError(str(e) + "\nTry to install pymesh pyassimp: `pip install pymesh pyassimp`")

# import image converter
from PIL import Image

# import MuJoCo
try:
    import mujoco_py as mujoco
    # from dm_control import mujoco
except ImportError as e:
    raise ImportError(str(e) + "\nTry to install `MuJoCo` and `mujoco_py`!")


# import glfw
# GLFW: OpenGL library for creating windows, contexts and surfaces, receiving input and events. This is used in
# mujoco_py
import glfw


# import pyrobolearn related functionalities
from pyrobolearn.simulators.simulator import Simulator
from pyrobolearn.utils.parsers.robots import URDFParser, MuJoCoParser, SDFParser
import pyrobolearn.utils.parsers.robots.data_structures as struct


# check Python version
import sys
if sys.version_info[0] < 3:
    raise RuntimeError("You must use Python 3 with the MuJoCo simulator.")


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2019, PyRoboLearn"
__credits__ = ["MuJoCo (Emo Todorov et al.)", "Open AI (MuJoCo Python API)", "Brian Delhaisse (PyRoboLearn interface)"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class Texture(object):
    """Texture information"""

    def __init__(self, texture_id, texture, material):
        """
        Initialize the texture.

        Args:
            texture_id (int): texture id.
        """
        self.id = texture_id
        self.texture = texture
        self.material = material


# class Body(object):
#     """Body."""
#
#     def __init__(self, body_id, body_tag):
#         """
#         Initialize the Body.
#
#         Args:
#             body_id (int): unique body id.
#             body_tag (xml.etree.ElementTree.Element): body tag element in the xml file.
#         """
#         self.id = body_id
#         if not isinstance(body_tag, ET.Element):
#             raise TypeError("Expecting the given 'body_tag' to be an instance of `ET.Element`, but got instead: "
#                             "{}".format(type(body_tag)))
#         self.body = body_tag
#
#         self.q_start = 0  # starting index in the whole state
#         self.q_end = 0    # end index in the whole state
#         self.fixed_base = False
#
#         # list of inner bodies (=links)
#         self.bodies = []
#         self.joints = []
#         self.joint = None
#
#     @property
#     def name(self):
#         """Return the body name."""
#         return self.body.attrib.get("name")


class MultiBody(object):
    """MultiBody."""

    def __init__(self, tree):
        """
        Initialize the MultiBody.

        Args:
            tree (Tree): tree / multi-body data structure.
        """
        if not isinstance(tree, struct.Tree):
            raise TypeError("Expecting the given 'tree' to be an instance of `Tree`, but got instead: "
                            "{}".format(type(tree)))
        self.tree = tree

        self.q_start = 0  # starting index in the whole state
        self.q_end = 0  # end index in the whole state
        static = self.tree.static
        self.fixed_base = static if static is not None else False


class Mujoco(Simulator):
    r"""Mujoco Simulator interface.

    This is the main interface that communicates with the MuJoCo simulator [1]. By defining this interface, it allows
    to decouple the PyRoboLearn framework from the simulator. It also converts some data types to the ones required by
    MuJoCo.

    Warnings:
        - The MuJoCo simulator requires a license in order to use it: https://www.roboti.us/license.html
        - You have to install the MuJoCo simulator beforehand: https://www.roboti.us/index.html
        - You have to install ``mujoco_py``:
            - for Python 2: https://github.com/openai/mujoco-py/tree/0.5
            - for Python 3: https://github.com/openai/mujoco-py
        - The authors of MuJoCo are working on another simulator called ``Optico`` so it is likely that MuJoCo won't
          be upgraded anymore.

    Note that initially MuJoCo doesn't allow to load dynamically objects in the world. This is carried out here, where
    we remember the current state of the world, and when asked to dynamically load an object we create a new world
    which is the old world + the new object. This can cause glitches on the GUI side.

    There are few important concepts:

    - model (mjModel, sim.model): model description and is constant. After giving it to the simulator, it can be
      accessed from the simulator `sim.model`.
    - sim (MjSim): the simulation instance. This takes as input the model.
    - data (mjData, sim.data): contains all the dynamic variables and intermediate results.
    - opt (mjOption, sim.opt): contains all options that affect the physics simulation.
    - viewer (MjViewer): the viewer instance which allows to visualize the simulation. It uses the GLFW library.
    - state (sim.get_state): snapshot of the simulator state (which includes time, qpos, qvel, act, and udd_state).

    References:
        - [1] MuJoCo: http://www.mujoco.org/
        - [2] MuJoCo Python: https://github.com/openai/mujoco-py
        - [3] DeepMind Control Suite: https://github.com/deepmind/dm_control/tree/master/dm_control/mujoco
    """

    def __init__(self, render=True, num_instances=1, update_dynamically=False, middleware=None, **kwargs):
        """
        Initialize the MuJoCo simulator.

        Args:
            render (bool): if True, it will open the GUI, otherwise, it will just run the server (i.e. in a headless
              mode, i.e. without a GUI).
            num_instances (int): number of simulator instances.
            update_dynamically (bool): if True, it will load dynamically the models each time we add or remove a model
              in the world. Otherwise, it will load the models when calling the `step()` method for the first time.
              The reason for that parameter is because the MuJoCo simulator does not allow to load dynamically models
              into the world.
            middleware (MiddleWare, None): middleware instance.
        """
        super(Mujoco, self).__init__(render=render, num_instances=num_instances, middleware=middleware, **kwargs)

        # define variables
        self._update_dynamically = update_dynamically
        self._floor_id = None
        self.model = None
        self.sim = None
        self.viewer = None
        self._parser = MuJoCoParser()  # main parser & generator that also contains the XML tags

        # keep track of visual and collision shapes
        self._visual_shapes = {}  # {visual_id: Visual}
        self._collision_shapes = {}  # {collision_id: Collision}
        self._bodies = OrderedDict()  # {body_id: Body}
        self._textures = {}  # {texture_id: Texture}
        self._constraints = OrderedDict()  # {constraint_id: Constraint}

        # create counters
        self._visual_cnt = 0
        self._collision_cnt = 0
        self._body_cnt = 0  # 0 is for the world
        self._texture_cnt = 0
        self._constraint_cnt = 0
        self._q_cnt = 0

        self.default_timestep = 0.002
        self.dt = self.default_timestep

        # each time we add or remove an object, the following variable is set to True. When calling the `step` method,
        # it will reload the model in the simulator.
        self._model_changed = False

        # create dynamically an empty world (XML)
        self._root = self._parser.root
        self._worldbody = self._parser.worldbody

        # add light
        self._parser.add_element("light", self._worldbody,
                                 attributes={"diffuse": ".5 .5 .5", "pos": "0 0 3", "dir": "0 0 -1"})
        # add floor
        self.load_floor()

        # create model, simulator, and viewer
        self._create_sim(render=render and update_dynamically)

        # define saving states
        self.__simulator_saving_states = {}

    ##############
    # Properties #
    ##############

    @property
    def version(self):
        """Return the version of the simulator in a year-month-day format."""
        return mujoco.get_version()

    @property
    def timestep(self):
        """Return the simulator time step."""
        return self.dt

    # @property
    # def dt(self):
    #     """Return the simulator time step."""
    #     return self.sim.model.opt.timestep

    #############
    # Operators #
    #############

    ##################
    # Static methods #
    ##################

    @staticmethod
    def simulate_soft_bodies():
        """Return True if the simulator can simulate soft bodies."""
        return True

    @staticmethod
    def supports_acceleration():
        """Return True if the simulator provides acceleration (dynamic) information (such as joint accelerations, link
        Cartesian accelerations, etc). If not, the `Robot` class will have to implement these using finite
        difference."""
        return True

    @staticmethod
    def supports_sensors(sensor_type=None):  # TODO: map the names
        """Return True if the simulator provides supports for the specified sensor."""
        sensor_type = sensor_type.lower()
        return sensor_type in {'touch', 'accelerometer', 'velocimeter', 'gyro', 'force', 'torque', 'magnetometer',
                               'rangefinder', 'jointpos', 'jointvel', 'tendonpos', 'tendonvel', 'actuatorpos',
                               'actuatorvel', 'actuatorfrc', 'ballquat', 'ballangvel', 'jointlimitpos',
                               'jointlimitvel', 'jointlimitfrc', 'tendonlimitpos', 'tendonlimitvel', 'tendonlimitfrc',
                               'framepos', 'framequat', 'framexaxis', 'frameyaxis', 'framelinvel', 'frameangvel',
                               'framelinacc', 'frameangacc', 'subtreecom', 'subtreelinvel', 'subtreeangmom', 'user'}

    ###########
    # Methods #
    ###########

    ###########
    # Private #
    ###########

    def _create_sim(self, render=False):
        """
        Instantiate the model, simulator, and viewer.

        Args:
            root (str, ET.Element, None): xml string containing the definition of the Mujoco file, or root XML element.
              If None, it will take the root defined in the simulator.
            render (bool): if we should render or not.
        """
        # create the model
        # self.model = mujoco.load_model_from_path(path)
        root = self._parser.get_string(pretty_format=False)
        self.model = mujoco.load_model_from_xml(root)

        # create the simulator from the model
        self.sim = mujoco.MjSim(self.model)

        # if we need to render
        if render:
            # self.render(enable=False)  # to delete the previous viewer instance if defined
            # self.render(enable=True)   # to instantiate the viewer
            if self.viewer is None:
                self.render(enable=True)
            else:
                print("Update the viewer's sim")
                self.viewer.update_sim(self.sim)

    #################
    # utils methods #
    #################

    @staticmethod
    def _convert_wxyz_to_xyzw(q):
        """Convert a quaternion in the (w,x,y,z) format to (x,y,z,w)."""
        return np.roll(q, shift=-1)

    @staticmethod
    def _convert_xyzw_to_wxyz(q):
        """Convert a quaternion in the (x,y,z,w) format to (w,x,y,z)."""
        return np.roll(q, shift=1)

    ##############
    # Simulators #
    ##############

    def reset(self):
        """Reset the simulator.

        Resets the simulation data and clears buffers.
        """
        self.sim.reset()

    def close(self):
        """Close the simulator."""
        # delete the simulator
        del self.sim

    def seed(self, seed=None):
        """Set the given seed in the simulator."""
        # It seems this is not possible in MuJoCo
        pass

    def step(self, sleep_time=0.):
        """Perform a step in the simulator, and sleep the specified amount of time.

        Args:
            sleep_time (float): amount of time to sleep after performing one step in the simulation.
        """
        # if the simulator/model has not been created
        if self.sim is None or self._model_changed:
            self._create_sim()
            self._model_changed = False

        # computes forward kinematics in the simulator
        self.sim.forward()

        # advance the simulation
        self.sim.step()

        # render if necessary
        if self.is_rendering():
            if self.viewer is None:
                self.viewer = mujoco.MjViewer(self.sim)
            self.viewer.render()

        # sleep the specified amount of time
        # time.sleep(sleep_time)

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
        if enable:
            # if the viewer has not been instantiated, instantiate it
            if self.viewer is None:
                self.viewer = mujoco.MjViewer(self.sim)
        else:
            # if the viewer has been set, close it
            if self.viewer is not None:
                glfw.set_window_should_close(self.viewer.window, True)
                glfw.destroy_window(self.viewer.window)
                self.viewer = None

    def get_time_step(self):
        """Get the time step in the simulator.

        Returns:
            float: time step in the simulator
        """
        return self.sim.model.opt.timestep

    def set_time_step(self, time_step):
        """Set the specified time step in the simulator.

        "Warning: in many cases it is best to leave the timeStep to default, which is 240Hz. Several parameters are
        tuned with this value in mind. For example the number of solver iterations and the error reduction parameters
        (erp) for contact, friction and non-contact joints are related to the time step. If you change the time step,
        you may need to re-tune those values accordingly, especially the erp values.
        You can set the physics engine timestep that is used when calling 'stepSimulation'. It is best to only call
        this method at the start of a simulation. Don't change this time step regularly. setTimeStep can also be
        achieved using the new setPhysicsEngineParameter API." [1]

        Args:
            time_step (float): Each time you call 'step' the time step will proceed with 'time_step'.
        """
        self.sim.model.opt.timestep = time_step

    def get_gravity(self):
        """Return the gravity set in the simulator."""
        return self.sim.model.opt.gravity

    def set_gravity(self, gravity=(0, 0, -9.81)):
        """Set the gravity in the simulator with the given acceleration.

        By default, there is no gravitational force enabled in the simulator.

        Args:
            gravity (list/tuple[float[3]]): acceleration in the x, y, z directions.
        """
        self.sim.model.opt.gravity = gravity

    def save(self, filename=None, *args, **kwargs):
        """
        Save the state of the simulator.

        Args:
            filename (None, str): path to file to store the state of the simulator. If None, it will save it in
                memory instead of the disk.

        Returns:
            int / str: unique state id, or filename. This id / filename can be used to load the state.
        """
        id_ = None
        if filename is None:
            # create unique state id
            id_ = len(self.__simulator_saving_states)

        # self.sim.save(filename, format='mjb')  # format='xml'
        model = self.model.get_mjb()
        state = self.sim.get_state()
        if id_ is None:
            with open(filename, 'wb') as f:
                pickle.dump((model, state), f)
            return filename
        self.__simulator_saving_states[id_] = (model, state)
        return id_

    def load(self, state, *args, **kwargs):
        """
        Load/Restore the simulator to a previous state.

        Args:
            state (int, str): unique state id, or path to the file containing the state.
        """
        if isinstance(state, int):
            if state not in self.__simulator_saving_states:
                raise ValueError("The given state (int) has not been saved.")
            model, state = self.__simulator_saving_states.pop(state)
        elif isinstance(state, str):
            with open(state, 'wb') as f:
                model, state = pickle.load(f)
        else:
            raise TypeError("Expecting the given state to be an int or string, instead got: {}".format(type(state)))

        # restore the simulator
        self.sim = mujoco.MjSim(model)
        self.sim.set_state(state)

    ######################################
    # Loading URDFs, SDFs, MJCFs, meshes #
    ######################################

    def load_urdf(self, filename, position, orientation=(0., 0., 0., 1.), use_fixed_base=0, scale=1.0, *args, **kwargs):
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
        print(filename)
        print(os.path.dirname(filename))
        # parse URDF file
        urdf_parser = URDFParser(filename=filename)
        tree = urdf_parser.tree

        # update position and orientation
        tree.position = position
        tree.orientation = orientation

        # add the parse tree to the MJCF parser/generator
        self._parser.add_multibody(tree, mesh_directory_path=os.path.dirname(os.path.abspath(__file__)) + '/meshes/')

        print(self._parser.get_string(pretty_format=True))

    def load_sdf(self, filename, scaling=1., *args, **kwargs):
        """Load a SDF file in the simulator.

        Args:
            filename (str): a relative or absolute path to the SDF file on the file system of the physics server.
            scaling (float): scale factor for the object

        Returns:
            list(int): list of object unique id for each object loaded
        """
        # parse sdf file
        sdf_parser = SDFParser(filename=filename)

        for tree in sdf_parser.world.trees:
            # # update the position and orientation
            # tree.position = position
            # tree.orientation = orientation

            self._parser.add_multibody(tree)

    def load_mjcf(self, filename, scaling=1., *args, **kwargs):
        """Load a Mujoco file in the simulator.

        Warnings: this only loads the bodies, joints, and assets. It does not load other elements such as the physical
        engine properties (number of iterations, solver, etc), physical properties (gravity, friction, viscosity, etc),
        and others.

        Args:
            filename (str): a relative or absolute path to the MJCF file on the file system of the physics server.
            scaling (float): scale factor for the object

        Returns:
            list(int): list of object unique id for each object loaded
        """
        # load MJCF  # TODO: check if empty world
        # self.model = mujoco.load_model_from_path(filename)
        # self.sim = mujoco.MjSim(self.model)

        raise NotImplementedError

        # # parse MJCF file
        # parser = MuJoCoParser(filename=filename)
        #
        # # generate XML elements
        # elements = [parser.generate(tree) for tree in parser.world.trees]
        #
        # # append each element to worldbody
        # for element in elements:
        #     self._worldbody.append(element)

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
        # convert file '.obj' to '.stl' as MuJoCo only supports STL formats.

        # try to look for textures and colors in the '.mtl' file

        # create collision shape if specified

        # create visual shape

        # create body
        pass

    def load_soft_body(self, shape=None, filename=None):  # TODO
        pass

    ##########
    # Bodies #
    ##########

    def load_floor(self, dimension=20):
        """Load a floor in the simulator.

        Args:
            dimension (float): dimension of the floor.

        Returns:
            int: non-negative unique id for the floor, or -1 for failure.
        """
        # if floor already loaded, remove it
        if self._floor_id is not None:
            floor = self._bodies.pop(self._floor_id)
            self._worldbody.remove(floor)
            self._model_changed = True

        # create floor
        dim = dimension/2.
        floor = self._parser.add_element(name="geom", parent_element=self._worldbody,
                                         attributes={"type": "plane", "size": str(dim) + " " + str(dim) + " 1."})
        self._body_cnt += 1
        self._bodies[self._body_cnt] = floor
        self._floor_id = self._body_cnt
        return self._floor_id

    def create_body(self, visual_shape_id=-1, collision_shape_id=-1, mass=0., position=(0., 0., 0.),
                    orientation=(0., 0., 0., 1.), *args, **kwargs):  # DONE
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
        # check that at least the visual or collision shape is provided
        if visual_shape_id == -1 and collision_shape_id == -1:
            raise ValueError("Expecting the visual shape or collision shape id to be specified.")

        # get the corresponding visual / collision object
        visual = self._visual_shapes.get(visual_shape_id, None)
        collision = self._collision_shapes.get(collision_shape_id, None)

        # create body data structure
        self._body_cnt += 1
        static = (mass == 0)
        inertial = struct.Inertial(mass=mass)
        # name = "prl_body_" + str(self._body_cnt)  # parser will automatically prefix 'prl_' and suffix '_idx'
        name = "body"
        body = struct.Body(body_id=self._body_cnt, name=name, inertials=inertial, visuals=visual, collisions=collision,
                           position=position, orientation=orientation, static=static)

        # if not static, add free joint to body
        if not static:
            # name = "prl_joint_" + str(self._body_cnt)  # parser will automatically prefix 'prl_' and suffix '_idx'
            name = "joint"
            joint = struct.Joint(joint_id=self._body_cnt, name=name, dtype='free', child=body)
            body.add_parent_joint(joint)

        # wrap body into multi-body data structure
        tree = struct.Tree(name=body.name, root=body, position=body.position, orientation=body.orientation)

        # save multi-body data structure
        self._parser.add_multibody(tree)
        body = MultiBody(tree)
        self._bodies[self._body_cnt] = body

        # set generalized coordinates counter
        if not static:
            body.q_start = self._q_cnt
            body.q_end = self._q_cnt + 7
            self._q_cnt += 7

        # update mujoco model if necessary
        if self._update_dynamically:
            self._create_sim(render=self._render)
        else:
            # notify that the Mujoco model has changed (this will be checked in the `step` method)
            self._model_changed = True

        # print(self._parser.get_string(pretty_format=True))

        # return body id
        return self._body_cnt

    def remove_body(self, body_id):  # DONE
        """Remove a particular body in the simulator.

        Args:
            body_id (int): unique body id.
        """
        # remove body from the bodies
        body = self._bodies.pop(body_id)

        # remove it from the worldbody
        self._worldbody.remove(body)

        # if the model / sim were loaded, notify that the Mujoco has changed
        if self.sim is not None and self.model is not None:
            self._model_changed = True

    def num_bodies(self):  # DONE
        """Return the number of bodies present in the simulator.

        Returns:
            int: number of bodies
        """
        return len(self._bodies)

    def get_body_info(self, body_id):  # DONE
        """Get the specified body information.

        Specifically, it returns the base name extracted from the URDF, SDF, MJCF, or other file.

        Args:
            body_id (int): unique body id.

        Returns:
            str: base name
        """
        return self._bodies[body_id].name

    def get_body_id(self, index):  # DONE
        """
        Get the body id associated to the index which is between 0 and `num_bodies()`.

        Args:
            index (int): index between [0, `num_bodies()`]

        Returns:
            int: unique body id.
        """
        return list(self._bodies.items())[index][0]

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
        # check <equality> tag in xml (which are used for constraints).
        asset = self._root.find("equality")

        # if no <equality> tag, create one
        if asset is None:
            asset = ET.SubElement(self._root, "equality")

        # Constraints:
        # - connect: constraint that connects two bodies at a point (ball joints)
        # - weld:
        # - joint:
        # - tendon:
        # - distance:

        # create constraint tag
        constraint = ET.SubElement(asset, "connect", attrib={"name": "constraint_" + str(self._constraint_cnt),
                                                             "body1": None, "body2": None})

        # remember constraint
        self._constraints[self._constraint_cnt] = constraint

        # increment constraint counter
        self._constraint_cnt += 1

        # return constraint id
        return self._constraint_cnt - 1

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
        return len(self._constraints)

    def get_constraint_id(self, index):
        """
        Get the constraint unique id associated with the index which is between 0 and `num_constraints()`.

        Args:
            index (int): index between [0, `num_constraints()`]

        Returns:
            int: constraint unique id.
        """
        return list(self._constraints.items())[index][0]

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
        mass = self.get_base_mass(body_id)
        for b in body.bodies:
            mass += sim.get_base_mass(b.id)
        return mass

    def get_base_mass(self, body_id):
        """Return the base mass of the robot.

        Args:
            body_id (int): unique object id.
        """
        return self.sim.model.body_mass[body_id]

    def get_base_name(self, body_id):
        """
        Return the base name.

        Args:
            body_id (int): unique object id.

        Returns:
            str: base name
        """
        return self.sim.model.body_id2name(box2)

    def get_center_of_mass_position(self, body_id, link_ids=None):  # TODO
        """
        Return the center of mass position.

        Args:
            body_id (int): unique body id.
            link_ids (list[int]): link ids associated with the given body id. If None, it will take all the links
                of the specified body.

        Returns:
            np.array[float[3]]: center of mass position in the Cartesian world coordinates
        """
        return self.sim.data.subtree_com[body_id]

    def get_center_of_mass_velocity(self, body_id, link_ids=None):  # TODO
        """
        Return the center of mass linear velocity.

        Args:
            body_id (int): unique body id.
            link_ids (list[int]): link ids associated with the given body id. If None, it will take all the links
                of the specified body.

        Returns:
            np.array[float[3]]: center of mass linear velocity.
        """
        return self.sim.data.subtree_linvel[body_id]

    def get_base_pose(self, body_id):
        """
        Get the current position and orientation of the base (or root link) of the body in Cartesian world coordinates.

        Args:
            body_id (int): object unique id, as returned from `load_urdf`.

        Returns:
            np.array[float[3]]: base position
            np.array[float[4]]: base orientation (quaternion [x,y,z,w])
        """
        # WARNING: body_xpos is one step late compared to qpos
        # return self.sim.data.body_xpos[body_id], self._convert_wxyz_to_xyzw(self.sim.data.body_xquat[body_id])

        # print(self.sim.data.body_xpos, q[body.q_start:body.q_start+3])
        # print(self.sim.data.body_xquat, q[body.q_start+3:body.q_end])

        body = self._bodies[body_id]
        q = self.sim.data.qpos
        return q[body.q_start:body.q_start+3], self._convert_wxyz_to_xyzw(q[body.q_start+3:body.q_start+7])

    def get_base_position(self, body_id):
        """
        Return the base position of the specified body.

        Args:
            body_id (int): object unique id, as returned from `load_urdf`.

        Returns:
            np.array[float[3]]: base position.
        """
        # return self.sim.data.body_xpos[body_id]
        body = self._bodies[body_id]
        q = self.sim.data.qpos
        return q[body.q_start:body.q_start + 3]

    def get_base_orientation(self, body_id):
        """
        Get the base orientation of the specified body.

        Args:
            body_id (int): object unique id, as returned from `load_urdf`.

        Returns:
            np.array[float[4]]: base orientation in the form of a quaternion (x,y,z,w)
        """
        # return self._convert_wxyz_to_xyzw(self.sim.data.body_xquat[body_id])
        body = self._bodies[body_id]
        q = self.sim.data.qpos
        return self._convert_wxyz_to_xyzw(q[body.q_start+3:body.q_start+7])

    def reset_base_pose(self, body_id, position, orientation):
        """
        Reset the base position and orientation of the specified object id.

        "It is best only to do this at the start, and not during a running simulation, since the command will override
        the effect of all physics simulation. The linear and angular velocity is set to zero. You can use
        `reset_base_velocity` to reset to a non-zero linear and/or angular velocity." [1]

        Args:
            body_id (int): unique object id.
            position (np.array[float[3]]): new base position.
            orientation (np.array[float[4]]): new base orientation (expressed as a quaternion [x,y,z,w])
        """
        body = self._bodies[body_id]
        q = self.sim.data.qpos
        q[body.q_start:body.q_start + 3] = position
        q[body.q_start + 3:body.q_start + 7] = self._convert_xyzw_to_wxyz(orientation)

    def reset_base_position(self, body_id, position):
        """
        Reset the base position of the specified body/object id while preserving its orientation.

        Args:
            body_id (int): unique object id.
            position (np.array[float[3]]): new base position.
        """
        # self.sim.data.body_xpos[body_id] = position
        # self.sim.forward()
        body = self._bodies[body_id]
        q = self.sim.data.qpos
        q[body.q_start:body.q_start + 3] = position

    def reset_base_orientation(self, body_id, orientation):
        """
        Reset the base orientation of the specified body/object id while preserving its position.

        Args:
            body_id (int): unique object id.
            orientation (np.array[float[4]]): new base orientation (expressed as a quaternion [x,y,z,w])
        """
        body = self._bodies[body_id]
        q = self.sim.data.qpos
        q[body.q_start+3:body.q_start+7] = self._convert_xyzw_to_wxyz(orientation)

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
        dq = self.sim.data.qvel
        return dq[body.q_start:body.q_start+3], dq[body.q_start+3:body.q_start+6]

    def get_base_linear_velocity(self, body_id):
        """
        Return the linear velocity of the base.

        Args:
            body_id (int): object unique id, as returned from `load_urdf`.

        Returns:
            np.array[float[3]]: linear velocity of the base in Cartesian world space coordinates
        """
        body = self._bodies[body_id]
        dq = self.sim.data.qvel
        return dq[body.q_start:body.q_start + 3]

    def get_base_angular_velocity(self, body_id):
        """
        Return the angular velocity of the base.

        Args:
            body_id (int): object unique id, as returned from `load_urdf`.

        Returns:
            np.array[float[3]]: angular velocity of the base in Cartesian world space coordinates
        """
        body = self._bodies[body_id]
        dq = self.sim.data.qvel
        return dq[body.q_start+3:body.q_start+6]

    def reset_base_velocity(self, body_id, linear_velocity=None, angular_velocity=None):
        """
        Reset the base velocity.

        Args:
            body_id (int): unique object id.
            linear_velocity (np.array[float[3]]): new linear velocity of the base.
            angular_velocity (np.array[float[3]]): new angular velocity of the base.
        """
        body = self._bodies[body_id]
        dq = self.sim.data.qvel
        dq[body.q_start:body.q_start + 3] = linear_velocity
        dq[body.q_start + 3:body.q_start + 6] = angular_velocity

    def reset_base_linear_velocity(self, body_id, linear_velocity):
        """
        Reset the base linear velocity.

        Args:
            body_id (int): unique object id.
            linear_velocity (np.array[float[3]]): new linear velocity of the base
        """
        body = self._bodies[body_id]
        dq = self.sim.data.qvel
        dq[body.q_start:body.q_start + 3] = linear_velocity

    def reset_base_angular_velocity(self, body_id, angular_velocity):
        """
        Reset the base angular velocity.

        Args:
            body_id (int): unique object id.
            angular_velocity (np.array[float[3]]): new angular velocity of the base
        """
        body = self._bodies[body_id]
        dq = self.sim.data.qvel
        dq[body.q_start + 3:body.q_start + 6] = angular_velocity

    def get_base_acceleration(self, body_id):
        """
        Get the base acceleration. This is only valid if the simulator `supports_acceleration`.

        Args:
            body_id (int): unique object id.

        Returns:
            np.array[float[3]]: linear acceleration [m/s^2]
            np.array[float[3]]: angular acceleration [rad/s^2]
        """
        body = self._bodies[body_id]
        ddq = self.sim.data.qacc
        return ddq[body.q_start:body.q_start+3], ddq[body.q_start+3:body.q_start+6]

    def apply_external_force(self, body_id, link_id=-1, force=(0., 0., 0.), position=(0., 0., 0.), frame=1):
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
        return len(body.links)

    def num_actuated_joints(self, body_id):
        """
        Return the total number of actuated joints associated with the given body id.

        Args:
            body_id (int): unique body id.

        Returns:
            int: number of actuated joints of the specified body.
        """
        body = self._bodies[body_id]
        return len(body.joints)

    #################
    # Visualization #
    #################

    def create_visual_shape(self, shape_type, radius=0.5, half_extents=(1., 1., 1.), length=1., filename=None,
                            mesh_scale=(1., 1., 1.), plane_normal=(0., 0., 1.), flags=-1, rgba_color=None,
                            specular_color=None, visual_frame_position=None, vertices=None, indices=None, uvs=None,
                            normals=None, visual_frame_orientation=None):
        """
        Create a visual shape in the simulator.

        Args:
            shape_type (int): type of shape; GEOM_SPHERE (=2), GEOM_BOX (=3), GEOM_CAPSULE (=7), GEOM_CYLINDER (=4),
                GEOM_PLANE (=6), GEOM_MESH (=5), GEOM_ELLIPSOID (=9)
            radius (float): only for GEOM_SPHERE, GEOM_CAPSULE, GEOM_CYLINDER
            half_extents (np.array[float[3]], list/tuple of 3 floats): only for GEOM_BOX, and GEOM_ELLIPSOID
            length (float): only for GEOM_CAPSULE, GEOM_CYLINDER (length = height).
            filename (str): Filename for GEOM_MESH, currently only Wavefront .obj. Will create convex hulls for each
                object (marked as 'o') in the .obj file.
            mesh_scale (np.array[float[3]], list/tuple of 3 floats): scale of mesh (only for GEOM_MESH).
            plane_normal (np.array[float[3]], list/tuple of 3 floats): plane normal (only for GEOM_PLANE).
            flags (int): unused / to be decided
            rgba_color (list/tuple of 4 floats): color components for red, green, blue and alpha, each in range [0..1].
            specular_color (list/tuple of 3 floats): specular reflection color, red, green, blue components in range
                [0..1]
            visual_frame_position (np.array[float[3]]): translational offset of the visual shape with respect to the
                link frame.
            vertices (list of np.array[float[3]]): Instead of creating a mesh from obj file, you can provide vertices,
                indices, uvs and normals
            indices (list[int]): triangle indices, should be a multiple of 3.
            uvs (list of np.array[2]): uv texture coordinates for vertices. Use changeVisualShape to choose the
                texture image. The number of uvs should be equal to number of vertices
            normals (list of np.array[float[3]]): vertex normals, number should be equal to number of vertices.
            visual_frame_orientation (np.array[float[4]]): rotational offset (quaternion x,y,z,w) of the visual shape
                with respect to the link frame

        Returns:
            int: The return value is a non-negative int unique id for the visual shape or -1 if the call failed.
        """
        if shape_type == self.GEOM_SPHERE:
            size = radius
            shape_type = "sphere"
        elif shape_type == self.GEOM_BOX:
            size = 2 * np.asarray(half_extents)
            shape_type = "box"
        elif shape_type == self.GEOM_CAPSULE:
            size = (radius, length)
            shape_type = "capsule"
        elif shape_type == self.GEOM_CYLINDER:
            size = (radius, length)
            shape_type = "cylinder"
        elif shape_type == self.GEOM_PLANE:
            size = (0., 0., 1.)  # (X half-size, Y half-size, spacing between square grid lines)
            # compute the orientation based on the normal
            shape_type = "plane"
        elif shape_type == self.GEOM_ELLIPSOID:
            size = half_extents  # (X radius, Y radius, Z radius)
            shape_type = "ellipsoid"
        elif shape_type == self.GEOM_MESH:
            size = mesh_scale
            shape_type = "mesh"
        else:
            raise ValueError("Unknown visual shape type.")

        # create visual
        visual = struct.Visual(dtype=shape_type, size=size, filename=filename, color=rgba_color,
                               position=visual_frame_position, orientation=visual_frame_orientation)

        # save visual and return visual counter
        self._visual_cnt += 1
        self._visual_shapes[self._visual_cnt] = visual
        return self._visual_cnt

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
                np.array[float[3]]: dimensions (size, local scale) of the geometry
                str: path to the triangle mesh, if any. Typically relative to the URDF, SDF or MJCF file location, but
                    could be absolute
                np.array[float[3]]: position of local visual frame, relative to link/joint frame
                np.array[float[4]]: orientation of local visual frame relative to link/joint frame
                list of 4 floats: URDF color (if any specified) in Red / Green / Blue / Alpha
                int: texture unique id of the shape or -1 if None. This field only exists if using
                    VISUAL_SHAPE_DATA_TEXTURE_UNIQUE_IDS (=1) flag.
        """
        shapes = []
        return shapes

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

    def load_texture(self, filename):  # DONE
        """
        Load a texture from file and return a non-negative texture unique id if the loading succeeds.
        This unique id can be used with change_visual_shape.

        Args:
            filename (str): path to the file.

        Returns:
            int: texture unique id. If non-negative, the texture was loaded successfully.
        """
        # check <asset> tag in xml
        asset = self._root.find("asset")

        # if no <asset> tag, create one
        if asset is None:
            asset = ET.SubElement(self._root, "asset")

        # create texture tag
        texture = ET.SubElement(asset, "texture", attrib={"name": "texture_" + str(self._texture_cnt),
                                                          "type": "2d",
                                                          "file": filename})
        material = ET.SubElement(asset, "material", attrib={"name": "material_" + str(self._texture_cnt),
                                                            "texture": "texture_" + str(self._texture_cnt)})

        # remember texture_id --> (texture, material)
        self._textures[self._texture_cnt] = Texture(texture_id=self._texture_cnt, texture=texture, material=material)

        # increment texture counter
        self._texture_cnt += 1

        # return texture id
        return self._texture_cnt - 1

    # TODO: change such that we don't return the width and height (the user already knows them)
    # TODO: check for segmentation image
    def get_camera_image(self, width, height, view_matrix=None, projection_matrix=None, light_direction=None,
                         light_color=None, light_distance=None, shadow=None, light_ambient_coeff=None,
                         light_diffuse_coeff=None, light_specular_coeff=None, renderer=None, flags=None):
        """
        The `get_camera_image` API will return a RGB image, a depth buffer and a segmentation mask buffer with body
        unique ids of visible objects for each pixel.

        Args:
            width (int): horizontal image resolution in pixels
            height (int): vertical image resolution in pixels
            view_matrix (np.array[float[4,4]]): 4x4 view matrix, see `compute_view_matrix`
            projection_matrix (np.array[float[4,4]]): 4x4 projection matrix, see `compute_projection`
            light_direction (np.array[float[3]]): `light_direction` specifies the world position of the light source,
                the direction is from the light source position to the origin of the world frame.
            light_color (np.array[float[3]]): directional light color in [RED,GREEN,BLUE] in range 0..1
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
            np.array[int[width, height, 4]]: RBGA pixels (each pixel is in the range [0..255] for each channel).
            np.array[float[width, height]]: Depth buffer.
            np.array[int[width, height]]: Segmentation mask buffer. For each pixels the visible object unique id.
        """
        # based on the arguments, check the camera name
        camera_name = None

        return width, height, self.sim.render(width, height, camera_name, depth=True), None

    def get_rgba_image(self, width, height, view_matrix=None, projection_matrix=None, light_direction=None,
                       light_color=None, light_distance=None, shadow=None, light_ambient_coeff=None,
                       light_diffuse_coeff=None, light_specular_coeff=None, renderer=None, flags=None):
        """
        The `get_rgba_image` API will return a RGBA image.

        Args:
            width (int): horizontal image resolution in pixels
            height (int): vertical image resolution in pixels
            view_matrix (np.array[float[4,4]]): 4x4 view matrix, see `compute_view_matrix`
            projection_matrix (np.array[float[4,4]]): 4x4 projection matrix, see `compute_projection`
            light_direction (np.array[float[3]]): `light_direction` specifies the world position of the light source,
                the direction is from the light source position to the origin of the world frame.
            light_color (np.array[float[3]]): directional light color in [RED,GREEN,BLUE] in range 0..1
            light_distance (float): distance of the light along the normalized `light_direction`
            shadow (bool): True for shadows, False for no shadows
            light_ambient_coeff (float): light ambient coefficient
            light_diffuse_coeff (float): light diffuse coefficient
            light_specular_coeff (float): light specular coefficient
            renderer (int): renderer.
            flags (int): flags.

        Returns:
            np.array[int[width, height, 4]]: RBGA pixels (each pixel is in the range [0..255] for each channel).
        """
        # based on the arguments, check the camera name
        camera_name = None

        # return the RGB image
        return self.sim.render(width, height, camera_name, depth=False)

    def get_depth_image(self, width, height, view_matrix=None, projection_matrix=None, light_direction=None,
                        light_color=None, light_distance=None, shadow=None, light_ambient_coeff=None,
                        light_diffuse_coeff=None, light_specular_coeff=None, renderer=None, flags=None):
        """
        The `get_depth_image` API will return a depth buffer.

        Args:
            width (int): horizontal image resolution in pixels
            height (int): vertical image resolution in pixels
            view_matrix (np.array[float[4,4]]): 4x4 view matrix, see `compute_view_matrix`
            projection_matrix (np.array[float[4,4]]): 4x4 projection matrix, see `compute_projection`
            light_direction (np.array[float[3]]): `light_direction` specifies the world position of the light source,
                the direction is from the light source position to the origin of the world frame.
            light_color (np.array[float[3]]): directional light color in [RED,GREEN,BLUE] in range 0..1
            light_distance (float): distance of the light along the normalized `light_direction`
            shadow (bool): True for shadows, False for no shadows
            light_ambient_coeff (float): light ambient coefficient
            light_diffuse_coeff (float): light diffuse coefficient
            light_specular_coeff (float): light specular coefficient
            renderer (int): renderer.
            flags (int): flags.

        Returns:
            np.array[float[width, height]]: Depth buffer.
        """
        # based on the arguments, check the camera name
        camera_name = None

        # return the depth image
        rgb, depth = self.sim.render(width, height, camera_name, depth=True)
        return depth

    ##############
    # Collisions #
    ##############

    def create_collision_shape(self, shape_type, radius=0.5, half_extents=(1., 1., 1.), height=1., filename=None,
                               mesh_scale=(1., 1., 1.), plane_normal=(0., 0., 1.), flags=-1,
                               collision_frame_position=None, collision_frame_orientation=None):
        """
        Create collision shape in the simulator.

        Args:
            shape_type (int): type of shape; GEOM_SPHERE (=2), GEOM_BOX (=3), GEOM_CAPSULE (=7), GEOM_CYLINDER (=4),
                GEOM_PLANE (=6), GEOM_MESH (=5), GEOM_ELLIPSOID (=9)
            radius (float): only for GEOM_SPHERE, GEOM_CAPSULE, GEOM_CYLINDER
            half_extents (np.array[float[3]], list/tuple of 3 floats): only for GEOM_BOX.
            height (float): only for GEOM_CAPSULE, GEOM_CYLINDER (length = height).
            filename (str): Filename for GEOM_MESH, currently only Wavefront .obj. Will create convex hulls for each
                object (marked as 'o') in the .obj file.
            mesh_scale (np.array[float[3]], list/tuple of 3 floats): scale of mesh (only for GEOM_MESH).
            plane_normal (np.array[float[3]], list/tuple of 3 floats): plane normal (only for GEOM_PLANE).
            flags (int): unused / to be decided
            collision_frame_position (np.array[float[3]]): translational offset of the collision shape with respect to
                the link frame
            collision_frame_orientation (np.array[float[4]]): rotational offset (quaternion x,y,z,w) of the collision
                shape with respect to the link frame

        Returns:
            int: The return value is a non-negative int unique id for the collision shape or -1 if the call failed.
        """
        if shape_type == self.GEOM_SPHERE:
            size = radius
            shape_type = "sphere"
        elif shape_type == self.GEOM_BOX:
            size = 2 * np.asarray(half_extents)
            shape_type = "box"
        elif shape_type == self.GEOM_CAPSULE:
            size = (radius, height)
            shape_type = "capsule"
        elif shape_type == self.GEOM_CYLINDER:
            size = (radius, height)
            shape_type = "cylinder"
        elif shape_type == self.GEOM_PLANE:
            size = (0., 0., 1.)  # (X half-size, Y half-size, spacing between square grid lines)
            # compute the orientation based on the normal
            shape_type = "plane"
        elif shape_type == self.GEOM_ELLIPSOID:
            size = half_extents  # (X radius, Y radius, Z radius)
            shape_type = "ellipsoid"
        elif shape_type == self.GEOM_MESH:
            size = mesh_scale
            shape_type = "mesh"
        else:
            raise ValueError("Unknown collision shape type.")

        # create collision
        collision = struct.Collision(dtype=shape_type, size=size, filename=filename,
                                     position=collision_frame_position, orientation=collision_frame_orientation)

        # save collision and return collision counter
        self._collision_cnt += 1
        self._collision_shapes[self._collision_cnt] = collision
        return self._collision_cnt

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
            np.array[float[3]]: depends on geometry type:
                for GEOM_BOX: extents,
                for GEOM_SPHERE: dimensions[0] = radius,
                for GEOM_CAPSULE and GEOM_CYLINDER: dimensions[0] = height (length), dimensions[1] = radius.
                For GEOM_MESH: dimensions is the scaling factor.
            str: Only for GEOM_MESH: file name (and path) of the collision mesh asset.
            np.array[float[3]]: Local position of the collision frame with respect to the center of mass/inertial frame
            np.array[float[4]]: Local orientation of the collision frame with respect to the inertial frame
        """
        pass

    def ray_test(self, from_position, to_position):
        """
        Performs a single raycast to find the intersection information of the first object hit.

        Args:
            from_position (np.array[float[3]]): start of the ray in world coordinates
            to_position (np.array[float[3]]): end of the ray in world coordinates

        Returns:
            list:
                int: object unique id of the hit object
                int: link index of the hit object, or -1 if none/parent
                float: hit fraction along the ray in range [0,1] along the ray.
                np.array[float[3]]: hit position in Cartesian world coordinates
                np.array[float[3]]: hit normal in Cartesian world coordinates
        """
        vec = to_position - from_position
        return self.sim.ray(pnt=from_position, vec=vec)  # this return the distance and id of the geom


# Test
if __name__ == '__main__':
    from itertools import count

    sim = Mujoco(render=True)

    # create box in simulator
    dimensions = 1. * np.ones(3)
    collision_shape = sim.create_collision_shape(sim.GEOM_BOX, half_extents=dimensions / 2.)
    visual_shape = sim.create_visual_shape(sim.GEOM_BOX, half_extents=dimensions / 2., rgba_color=(0.8, 0.2, 0.2, 1.))
    visual_shape2 = sim.create_visual_shape(sim.GEOM_BOX, half_extents=dimensions / 2., rgba_color=(0.2, 0.2, 0.8, 1.))

    box = sim.create_body(mass=1., collision_shape_id=collision_shape, visual_shape_id=visual_shape,
                          position=[0., 0., 1.])

    box2 = sim.create_body(mass=2, visual_shape_id=visual_shape2, collision_shape_id=collision_shape,
                           position=[0., 1., 2.])

    print(dir(sim.sim.model))
    print(dir(sim.sim.data))

    for t in count():
        # sim.get_base_pose(box)
        # print(sim.get_base_velocity(box))
        # print("base pose: {}".format(sim.get_base_pose(box)))
        # if t == 200:
        #     print("Reset")
        #     sim.reset_base_position(box, [0., 0., 2.])

        # print(sim.sim.data.subtree_com)
        # print("Mass: ", sim.sim.model.body_mass)
        # print("Subtree mass: ", sim.sim.model.body_subtreemass)
        # print(sim.get_base_mass(box))

        # print(sim.sim.data.subtree_com)

        # if t == 200:
        #     sim.viewer.finish()
        #     input()

        sim.step(sim.dt)
