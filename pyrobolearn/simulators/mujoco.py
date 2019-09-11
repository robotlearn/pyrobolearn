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
from pyrobolearn.utils.transformation import get_homogeneous_matrix, get_quaternion_from_matrix


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


class Body(object):
    """Body."""

    def __init__(self, body_id, body_tag, body):
        """
        Initialize the MultiBody.

        Args:
            body_id (int): unique body id in the Mujoco model.
            body_tag (ET.Element): body tag element in the XML file.
            body (MultiBody, None): multibody data structure.
        """
        self.id = body_id
        self.tag = body_tag

        # check given body
        if body is None:
            body = struct.MultiBody()
        if not isinstance(body, struct.MultiBody):
            raise TypeError("Expecting the given 'body' to be an instance of `MultiBody` but got instead: "
                            "{}".format(type(body)))

        # get information from body
        self.num_bodies = body.num_bodies  # number of links/bodies
        self.num_joints = body.num_joints  # nb of joints (include fixed joints but exclude free joints) = num_bodies
        self.num_actuated_joints = body.num_actuated_joints  # nb of actuated joints (exclude fixed and free joints)
        self.num_free_joints = body.num_free_joints  # nb of free joints (including free but excluding fixed joints)
        self.num_dofs = body.num_dofs  # nb of DoFs
        self.q_length = self.num_dofs  # length of q
        self.fixed = body.fixed_base if body.fixed_base is not None else True
        if not self.fixed:  # if free joint, add 1 because in Mujoco the pose is represented as position vector
            self.q_length += 1   # (3) + quaternion (4) = 7, so one more than 6 DoFs

        # define variables for indices that appears in the various vectors and matrices returned by mjModel and mjData
        self._q_idx0, self._q_idxf = 0, 0  # initial and final q indices
        self._b_idx0, self._b_idxf = 0, 0  # initial and final body indices
        self._j_idx0, self._j_idxf = 0, 0  # initial and final free joint indices
        self._v_idx0, self._v_idxf = 0, 0  # initial and final dq (velocity) indices

        # keep in memory the body
        # self.body = body
        self.joints = list(body.joints.values())
        self.links = list(body.bodies.values())

        # compute mapping from joint ids to q indices
        idx, jnt_to_q = 0, []
        for joint in body.joints.values():
            if joint.dtype == 'fixed':
                jnt_to_q.append(-1)
            else:
                jnt_to_q.append(idx)
                idx += 1
        self.jnt_to_q = np.array(jnt_to_q)

    @property
    def num_links(self):
        """Alias to `num_bodies`."""
        return self.num_bodies

    @property
    def q_idx0(self):
        """Return the initial q index."""
        return self._q_idx0

    @q_idx0.setter
    def q_idx0(self, q):
        """Set the the initial q index."""
        q = int(q)
        if q < 0:
            raise ValueError("Error while setting the initial q index, this index has to be bigger than 0!")
        self._q_idx0 = q
        self._q_idxf = q + self.q_length  # set the final q index

    @property
    def q_idxf(self):
        """Return the final q index."""
        return self._q_idxf

    @q_idxf.setter
    def q_idxf(self, q):
        """Set the final q index."""
        q = int(q)
        if q < 0:
            raise ValueError("Error while setting the final q index, this index has to be bigger than 0!")
        self._q_idxf = q
        self._q_idx0 = q - self.q_length  # set initial q index
        if self._q_idx0 < 0:
            raise ValueError("Error while setting the final q index, by computing automatically the initial q index "
                             "from it, it appears it is smaller than 0. The initial q index has to be bigger than 0!")

    @property
    def b_idx0(self):
        """Return the initial body (link) index."""
        return self._b_idx0

    @b_idx0.setter
    def b_idx0(self, b):
        """Set the initial body (link) index."""
        b = int(b)
        if b < 0:
            raise ValueError("Error while setting the initial body index, this index has to be bigger than 0!")
        self._b_idx0 = b
        self._b_idxf = b + self.num_bodies  # set the final body index

    @property
    def b_idxf(self):
        """Return the final body (link) index."""
        return self._b_idxf

    @b_idxf.setter
    def b_idxf(self, b):
        """Set the final body (link) index."""
        b = int(b)
        if b < 0:
            raise ValueError("Error while setting the final body index, this index has to be bigger than 0!")
        self._b_idxf = b
        self._b_idx0 = b - self.num_bodies  # set initial body index
        if self._b_idx0 < 0:
            raise ValueError("Error while setting the final body index, by computing automatically the initial body "
                             "index from it, it appears it is smaller than 0. The initial body index has to be bigger "
                             "than 0!")

    @property
    def j_idx0(self):
        """Return the initial free joint index."""
        return self._j_idx0

    @j_idx0.setter
    def j_idx0(self, j):
        """Set the initial free joint index."""
        j = int(j)
        if j < 0:
            raise ValueError("Error while setting the initial free joint index, this index has to be bigger than 0!")
        self._j_idx0 = j
        self._j_idxf = j + self.num_free_joints  # set the final free joint index

    @property
    def j_idxf(self):
        """Return the initial free joint index."""
        return self._j_idxf

    @j_idxf.setter
    def j_idxf(self, j):
        """Set the final free joint index."""
        j = int(j)
        if j < 0:
            raise ValueError("Error while setting the final free joint index, this index has to be bigger than 0!")
        self._j_idxf = j
        self._j_idx0 = j - self.num_free_joints  # set initial free joint index
        if self._j_idx0 < 0:
            raise ValueError("Error while setting the final joint index, by computing automatically the initial joint "
                             "index from it, it appears it is smaller than 0. The initial joint index has to be bigger "
                             "than 0!")

    @property
    def v_idx0(self):
        """Return the initial velocity index."""
        return self._v_idx0

    @v_idx0.setter
    def v_idx0(self, v):
        """Set the initial velocity index."""
        v = int(v)
        if v < 0:
            raise ValueError("Error while setting the initial velocity index, this index has to be bigger than 0!")
        self._v_idx0 = v
        self._v_idxf = v + self.num_dofs  # set the final velocity index

    @property
    def v_idxf(self):
        """Return the initial velocity index."""
        return self._v_idxf

    @v_idxf.setter
    def v_idxf(self, v):
        """Set the final velocity index."""
        v = int(v)
        if v < 0:
            raise ValueError("Error while setting the final velocity index, this index has to be bigger than 0!")
        self._v_idxf = v
        self._v_idx0 = v - self.num_dofs  # set initial velocity index
        if self._v_idx0 < 0:
            raise ValueError("Error while setting the final velocity index, by computing automatically the initial "
                             "velocity index from it, it appears it is smaller than 0. The initial velocity index has "
                             "to be bigger than 0!")

    @property
    def name(self):
        """Return the body name."""
        # note that we remove the generated prefix and suffix by the parser/generator
        return '_'.join(self.tag_name.split('_')[1:-1])

    @property
    def tag_name(self):
        """Return the body tag name."""
        return self.tag.attrib.get("name")

    def get_q_idx(self, joint_id, keep=False):
        q = self.jnt_to_q[joint_id]
        if keep:  # keep fixed joints (-1)
            return q
        if isinstance(q, float):
            if q!=-1:
                return q
            return []
        return q[q!=-1]  # remove fixed joints

    def get_dq_idx(self, joint_id, keep=False):
        return self.get_q_idx(joint_id, keep)

    def check_joint_id(self, joint_id):
        if joint_id < 0 or joint_id > (self.num_joints - 1):
            raise ValueError("joint_id should belong to [0, `num_joints-1`].")

    def check_link_id(self, link_id):
        if link_id < -1 or link_id > (self.num_bodies - 2):  # -1 is for the base
            raise ValueError("link_id should belong to [-1, `num_links-2`].")

    def get_joint_type(self, joint_id):
        return self.joints[joint_id].dtype


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
        self._body_cnt = 0  # 0 is for the world floor
        self._texture_cnt = 0
        self._constraint_cnt = 0

        # counters for Mujoco
        self._q_cnt = 0
        self._dq_cnt = 0
        self._joint_cnt = 0
        self._link_cnt = 1    # this is the number of bodies (=links) in Mujoco, 0 is for the worldbody.
        self._mjc_body_id = 0

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
                                 attributes={"diffuse": "0.5 0.5 0.5", "pos": "0 0 3", "directional": "true",
                                             "dir": "0 0 -1"})
        # add floor
        self.load_floor()

        # create model, simulator, and viewer
        self._create_sim(render=render and update_dynamically)

        # define saving states
        self.__simulator_saving_states = {}

        # define directory mesh path (to write the converted mesh files as Mujoco only accepts STL)
        self.mesh_directory_path = os.path.dirname(os.path.abspath(__file__)) + '/meshes/'

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

    def set_time_step(self, time_step):  # TODO: modify the option tag
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
        # self.sim.model.opt.timestep = time_step
        time_step = self._parser.convert_attribute_to_string(time_step)
        self._parser.option_tag.attrib.setdefault('timestep', time_step)
        self._update_sim()

    def get_gravity(self):
        """Return the gravity set in the simulator."""
        return self.sim.model.opt.gravity

    def set_gravity(self, gravity=(0, 0, -9.81)):  # TODO: modify the option tag
        """Set the gravity in the simulator with the given acceleration.

        By default, there is no gravitational force enabled in the simulator.

        Args:
            gravity (list/tuple[float[3]]): acceleration in the x, y, z directions.
        """
        # self.sim.model.opt.gravity = gravity
        gravity = self._parser.convert_attribute_to_string(gravity)
        self._parser.option_tag.attrib.setdefault('gravity', gravity)
        self._update_sim()

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
        # parse URDF file
        urdf_parser = URDFParser(filename=filename)
        tree = urdf_parser.tree

        # update tree position and orientation
        position = np.zeros(3) if position is None else np.asarray(position)
        orientation = np.array([0., 0., 0., 1.]) if orientation is None else np.asarray(orientation)
        h = get_homogeneous_matrix(position, orientation)
        tree.homogeneous = h.dot(tree.homogeneous)

        # update tree base if it is static or not
        tree.static = use_fixed_base
        # if not static, add free joint to body
        if not use_fixed_base:
            name, body = "joint", tree.root
            joint = struct.Joint(joint_id=-1, name=name, dtype='free', child=body)
            body.add_parent_joint(joint)
            tree.add_joint(joint, idx=0)

        return self._create_body(tree, verbose=2)

    def load_sdf(self, filename, scaling=1., *args, **kwargs):  # TODO
        """Load a SDF file in the simulator.

        Args:
            filename (str): a relative or absolute path to the SDF file on the file system of the physics server.
            scaling (float): scale factor for the object

        Returns:
            list(int): list of object unique id for each object loaded
        """
        raise NotImplementedError

        # # parse sdf file
        # sdf_parser = SDFParser(filename=filename)
        #
        # for tree in sdf_parser.world.trees:
        #     # # update the position and orientation
        #     # tree.position = position
        #     # tree.orientation = orientation
        #
        #     self._parser.add_multibody(tree)

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
        filename = self._parser.convert_mesh(filename, mesh_dirname=self.mesh_directory_path)

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
            self._worldbody.remove(floor.tag)
            self._model_changed = True

        # create floor
        dim = dimension/2.
        floor = self._parser.add_element(name="geom", parent_element=self._worldbody,
                                         attributes={"type": "plane", "size": str(dim) + " " + str(dim) + " 1."})

        body = Body(body_id=0, body_tag=floor, body=None)  # 0 is only for the floor
        self._bodies[0] = body  # the floor is always the first body in the ordered dict
        self._floor_id = 0

        # update mujoco model if necessary
        self._update_sim()

        return self._floor_id

    def _update_sim(self):
        # update mujoco model if necessary
        if self._update_dynamically:
            self._create_sim(render=self._render)
        else:
            # notify that the Mujoco model has changed (this will be checked in the `step` method)
            self._model_changed = True

    def _create_body(self, tree, body_id=None, verbose=2):
        """
        Create inner body in the simulator; given the tree data structure, it generates all the XML tags using
        the inner parser/generator, wraps the returned tree XML tag to a Body structure, sets its attributes (such as
        its q indices), update the simulator if necessary and return the unique body id.

        Args:
            tree (struct.MultiBody): the tree / multi-body data structure.
            body_id (int, None): unique body id to save the wrapped body.
            verbose (bool, int): if True, it will print the current Mujoco XML file that we have which is useful for
              debug. If int, it represents the level of verbosity.

        Returns:
            int: unique body id.
        """
        # check body id
        if body_id is None:
            self._body_cnt += 1
            body_id = self._body_cnt

        # create tree tag
        tree_tag = self._parser.add_multibody(tree, mesh_directory_path=self.mesh_directory_path)

        # save body
        self._mjc_body_id += 1
        body = Body(body_id=self._mjc_body_id, body_tag=tree_tag, body=tree)
        self._bodies[body_id] = body

        if verbose > 0:
            print("\nNum DoFs: {}".format(tree.num_dofs))
            print("Length of q: {}".format(body.q_length))
            print("Num links/bodies: {}".format(tree.num_bodies))
            print("Num joints: {}".format(tree.num_joints))
            print("Num actuated joints: {}\n".format(tree.num_actuated_joints))

        # update indices
        body.q_idx0 = self._q_cnt
        self._q_cnt += body.q_length
        body.b_idx0 = self._link_cnt
        self._link_cnt += body.num_bodies
        body.j_idx0 = self._joint_cnt
        self._joint_cnt += body.num_free_joints
        body.v_idx0 = self._dq_cnt
        self._dq_cnt += body.num_dofs

        # update mujoco model if necessary
        self._update_sim()

        # if verbose, print current xml file
        if verbose > 1:
            print(self._parser.get_string(pretty_format=True))

        # return body id
        return body_id

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
        tree.add_body(body)
        if not static:
            tree.add_joint(joint)

        return self._create_body(tree, body_id=self._body_cnt, verbose=2)

    def remove_body(self, body_id):  # DONE
        """Remove a particular body in the simulator.

        Args:
            body_id (int): unique body id.
        """
        # remove body from the bodies
        # This is an O(N) operation because I have to modify the id, q_idx0, and q_idxf of the bodies that appears
        # after the given body
        body_to_remove = self._bodies[body_id]
        found, q_length, num_links, num_free_joints = False, 0, 0, 0
        for body in self._bodies:
            if body_to_remove == body:
                found = True
                q_length = body.q_length
                num_links = body.num_bodies
                num_free_joints = body.num_free_joints
                self._q_cnt -= q_length
                self._joint_cnt -= num_free_joints
                self._link_cnt -= num_links
            if found:  # for the bodies after body_to_remove, shift their indices to the left
                body.id -= 1
                body.q_idx0 -= q_length
                body.b_idx0 -= num_links
                body.j_idx0 -= num_free_joints

        body = self._bodies.pop(body_id)
        self._mjc_body_id -= 1

        # remove it from the worldbody
        self._worldbody.remove(body.tag)

        # if the model / sim were loaded, notify that the Mujoco has changed
        self._update_sim()

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
        # return self._bodies[body_id].name
        return self.get_base_name(body_id)

    def get_body_id(self, index):  # DONE
        """
        Get the body id associated to the index which is between 0 and `num_bodies()`.

        Args:
            index (int): index between [0, `num_bodies()`]

        Returns:
            int: unique body id.
        """
        # O(N)
        return list(self._bodies.keys())[index][0]

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
        return sum(self.sim.model.body_mass[body.b_idx0:body.b_idxf])

    def get_base_mass(self, body_id):
        """Return the base mass of the robot.

        Args:
            body_id (int): unique object id.
        """
        body = self._bodies[body_id]
        return self.sim.model.body_mass[body.b_idx0]

    def get_base_name(self, body_id):
        """
        Return the base name.

        Args:
            body_id (int): unique object id.

        Returns:
            str: base name
        """
        body = self._bodies[body_id]
        return self.sim.model.body_id2name(body.b_idx0)

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
        body = self._bodies[body_id]
        return self.sim.data.subtree_com[body.b_idx0]  # sim.data.body_xipos

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
        body = self._bodies[body_id]
        return self.sim.data.subtree_linvel[body.b_idx0]  # sim.data.cvel[3:]

    def get_center_of_mass_angular_momentum(self, body_id, link_ids=None):  # TODO
        """
        Return the link angular momentum around its CoM.

        Args:
            body_id (int): unique body id.
            link_ids (list[int]): link ids associated with the given body id. If None, it will take all the links
                of the specified body.

        Returns:
            np.array[float[3]]: angular momentum.
        """
        body = self._bodies[body_id]
        return self.sim.data.subtree_angmom[body.b_idx0]  # sim.data.cvel[:3]

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
        body = self._bodies[body_id]
        position = self.sim.data.body_xpos[body.b_idx0]
        orientation = self._convert_wxyz_to_xyzw(self.sim.data.body_xquat[body.b_idx0])
        return position, orientation

    def get_base_position(self, body_id):
        """
        Return the base position of the specified body.

        Args:
            body_id (int): object unique id, as returned from `load_urdf`.

        Returns:
            np.array[float[3]]: base position.
        """
        body = self._bodies[body_id]
        return self.sim.data.body_xpos[body.b_idx0]

    def get_base_orientation(self, body_id):
        """
        Get the base orientation of the specified body.

        Args:
            body_id (int): object unique id, as returned from `load_urdf`.

        Returns:
            np.array[float[4]]: base orientation in the form of a quaternion (x,y,z,w)
        """
        body = self._bodies[body_id]
        return self._convert_wxyz_to_xyzw(self.sim.data.body_xquat[body.b_idx0])

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
        if body.fixed:
            self.model.body_pos[body.b_idx0] = position
            self.model.body_quat[body.b_idx0] = self._convert_xyzw_to_wxyz(orientation)
        else:
            self.sim.data.qpos[body.q_idx0:body.q_idx0 + 3] = position
            self.sim.data.qpos[body.q_idx0 + 3:body.q_idx0 + 7] = self._convert_xyzw_to_wxyz(orientation)

    def reset_base_position(self, body_id, position):
        """
        Reset the base position of the specified body/object id while preserving its orientation.

        Args:
            body_id (int): unique object id.
            position (np.array[float[3]]): new base position.
        """
        body = self._bodies[body_id]
        if body.fixed:  # fixed base
            self.model.body_pos[body.b_idx0] = position
        else:  # free joint
            self.sim.data.qpos[body.q_idx0:body.q_idx0 + 3] = position

    def reset_base_orientation(self, body_id, orientation):
        """
        Reset the base orientation of the specified body/object id while preserving its position.

        Args:
            body_id (int): unique object id.
            orientation (np.array[float[4]]): new base orientation (expressed as a quaternion [x,y,z,w])
        """
        body = self._bodies[body_id]
        if body.fixed:
            self.model.body_quat[body.b_idx0] = self._convert_xyzw_to_wxyz(orientation)
        else:
            self.sim.data.qpos[body.q_idx0+3:body.q_idx0+7] = self._convert_xyzw_to_wxyz(orientation)

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
        if body.fixed:
            return np.zeros(3), np.zeros(3)
        dq = self.sim.data.qvel
        return dq[body.q_idx0:body.q_idx0+3], dq[body.q_idx0+3:body.q_idx0+6]

    def get_base_linear_velocity(self, body_id):
        """
        Return the linear velocity of the base.

        Args:
            body_id (int): object unique id, as returned from `load_urdf`.

        Returns:
            np.array[float[3]]: linear velocity of the base in Cartesian world space coordinates
        """
        body = self._bodies[body_id]
        if body.fixed:
            return np.zeros(3)
        return self.sim.data.qvel[body.q_idx0:body.q_idx0 + 3]

    def get_base_angular_velocity(self, body_id):
        """
        Return the angular velocity of the base.

        Args:
            body_id (int): object unique id, as returned from `load_urdf`.

        Returns:
            np.array[float[3]]: angular velocity of the base in Cartesian world space coordinates
        """
        body = self._bodies[body_id]
        if body.fixed:
            return np.zeros(3)
        return self.sim.data.qvel[body.q_idx0+3:body.q_idx0+6]

    def reset_base_velocity(self, body_id, linear_velocity=None, angular_velocity=None):
        """
        Reset the base velocity.

        Args:
            body_id (int): unique object id.
            linear_velocity (np.array[float[3]]): new linear velocity of the base.
            angular_velocity (np.array[float[3]]): new angular velocity of the base.
        """
        body = self._bodies[body_id]
        if not body.fixed:
            dq = self.sim.data.qvel
            dq[body.q_idx0:body.q_idx0 + 3] = linear_velocity
            dq[body.q_idx0 + 3:body.q_idx0 + 6] = angular_velocity

    def reset_base_linear_velocity(self, body_id, linear_velocity):
        """
        Reset the base linear velocity.

        Args:
            body_id (int): unique object id.
            linear_velocity (np.array[float[3]]): new linear velocity of the base
        """
        body = self._bodies[body_id]
        if not body.fixed:
            self.sim.data.qvel[body.q_idx0:body.q_idx0 + 3] = linear_velocity

    def reset_base_angular_velocity(self, body_id, angular_velocity):
        """
        Reset the base angular velocity.

        Args:
            body_id (int): unique object id.
            angular_velocity (np.array[float[3]]): new angular velocity of the base
        """
        body = self._bodies[body_id]
        if not body.fixed:
            self.sim.data.qvel[body.q_idx0 + 3:body.q_idx0 + 6] = angular_velocity

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
        if body.fixed:
            return np.zeros(3), np.zeros(3)
        ddq = self.sim.data.qacc
        return ddq[body.q_idx0:body.q_idx0+3], ddq[body.q_idx0+3:body.q_idx0+6]

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
        if link_id < -1 or link_id > (body.num_bodies - 2):
            raise ValueError("link_id should belong to [-1, `num_links-2`].")
        idx = body.b_idx0 + link_id + 1
        # self.sim.data.xfrc_applied[idx, :3] = force

        # if position expressed in link frame, map it to world frame
        if frame == Simulator.LINK_FRAME:
            xpos = np.zeros(3)
            xmat = np.zeros(9)
            quat = np.array([1., 0., 0., 0.])  # TODO: get orientation of the link
            sameframe = 1
            mujoco.functions.mj_local2Global(self.sim.data, xpos, xmat, position, quat, idx, sameframe)
            position = xpos

        # apply force
        qfrc_target = np.zeros(self.model.nv)
        mujoco.functions.mj_applyFT(self.model, self.sim.data, force, np.zeros(3), position, idx, qfrc_target)
        return qfrc_target[body.v_idx0:body.v_idxf]

    def apply_external_torque(self, body_id, link_id=-1, torque=(0., 0., 0.), frame=Simulator.LINK_FRAME):
        """
        Apply an external torque on a body, or a link of the body. Note that after each simulation step, the external
        torques are cleared to 0.

        Args:
            body_id (int): unique body id.
            link_id (int): link id to apply the torque, if -1 it will apply the torque on the base
            torque (float[3]): Cartesian torques to be applied on the body
            frame (int): Specify the coordinate system of force/position: either `Simulator.WORLD_FRAME` (=2) for
                Cartesian world coordinates or `Simulator.LINK_FRAME` (=1) for local link coordinates.
        """
        body = self._bodies[body_id]
        if link_id < -1 or link_id > (body.num_bodies - 2):
            raise ValueError("link_id should belong to [-1, `num_links-2`].")
        idx = body.b_idx0 + link_id + 1
        # self.sim.data.xfrc_applied[idx, 3:] = torque
        qfrc_target = np.zeros(self.model.nv)
        position = np.zeros(3)
        mujoco.functions.mj_applyFT(self.model, self.sim.data, np.zeros(3), torque, position, idx, qfrc_target)
        return qfrc_target[body.v_idx0:body.v_idxf]

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
        return self._bodies[body_id].num_joints

    def num_actuated_joints(self, body_id):
        """
        Return the total number of actuated joints associated with the given body id.

        Args:
            body_id (int): unique body id.

        Returns:
            int: number of actuated joints of the specified body.
        """
        return self._bodies[body_id].num_actuated_joints

    def num_links(self, body_id):
        """
        Return the total number of links of the specified body. This is the same as calling `num_joints`.

        Args:
            body_id (int): unique body id.

        Returns:
            int: number of links with the associated body id.
        """
        return self._bodies[body_id].num_links

    def get_joint_info(self, body_id, joint_id):
        """
        Return information about the given joint about the specified body.

        Note that this method returns a lot of information, so specific methods have been implemented that return
        only the desired information. Also, note that we do not convert the data here.

        Args:
            body_id (int): unique body id.
            joint_id (int): joint id is included in [0..`num_joints(body_id)`].

        Returns:
            [0] int:        the same joint id as the input parameter
            [1] str:        name of the joint (as specified in the URDF/SDF/etc file)
            [2] int:        type of the joint which implies the number of position and velocity variables.
                            The types include JOINT_REVOLUTE (=0), JOINT_PRISMATIC (=1), JOINT_SPHERICAL (=2),
                            JOINT_PLANAR (=3), and JOINT_FIXED (=4).
            [3] int:        q index - the first position index in the positional state variables for this body
            [4] int:        dq index - the first velocity index in the velocity state variables for this body
            [5] int:        flags (reserved)
            [6] float:      the joint damping value (as specified in the URDF file)
            [7] float:      the joint friction value (as specified in the URDF file)
            [8] float:      the positional lower limit for slider and revolute joints
            [9] float:      the positional upper limit for slider and revolute joints
            [10] float:     maximum force specified in URDF. Note that this value is not automatically used.
                            You can use maxForce in 'setJointMotorControl2'.
            [11] float:     maximum velocity specified in URDF. Note that this value is not used in actual
                            motor control commands at the moment.
            [12] str:       name of the link (as specified in the URDF/SDF/etc file)
            [13] np.array[float[3]]:  joint axis in local frame (ignored for JOINT_FIXED)
            [14] np.array[float[3]]:  joint position in parent frame
            [15] np.array[float[4]]:  joint orientation in parent frame
            [16] int:       parent link index, -1 for base
        """
        body = self._bodies[body_id]

        if joint_id < 0 or joint_id > (body.num_joints - 1):
            raise ValueError("joint_id should belong to [0, `num_joints-1`].")

        dtype = body.get_joint_type(joint_id)
        if dtype == 'fixed':  # special care for fixed joints (as they are usually not specified in Mujoco models)
            pass
        idx = body.j_idx0 + joint_id

        name = self.model.joint_id2name()
        dtype = self.model.jnt_type  # ['free', 'ball', 'slide', 'hinge']
        q_idx = body.get_q_idx(joint_id)
        dq_idx = body.get_dq_idx(joint_id)
        flag = -1
        damping = self.model.dof_damping
        friction = self.model.dof_frictionloss
        limited = self.model.limited
        limits = self.model.jnt_range
        axis = self.model.jnt_axis
        pos = self.model.jnt_pos
        orientation = 0

        axis_pos = self.sim.data.xaxis
        # xanchor

        # stiffness

        return joint_id

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
        body = self._bodies[body_id]
        if not isinstance(joint_id, int):
            raise TypeError("Expecting the given joint id to be an int, but got instead: {}".format(type(joint_id)))
        if joint_id < 0 or joint_id > (body.num_joints - 1):
            raise ValueError("joint_id should belong to [0, `num_joints-1`].")
        q = body.get_q_idx(joint_id, keep=True)
        if q == -1:
            return 0, 0, np.zeros(6), 0
        qpos_idx = body.q_idx0 + q
        qvel_idx = body.v_idx0 + q
        if not body.fixed:
            qpos_idx += 7
            qvel_idx += 6

        pos = self.sim.data.qpos[qpos_idx]
        vel = self.sim.data.qvel[qvel_idx]
        reaction_forces = np.zeros(6)
        torque = self.sim.data.qfrc_applied[qvel_idx]
        return pos, vel, reaction_forces, torque

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
        return [self.get_joint_state(body_id, joint_id) for joint_id in joint_ids]

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
        body = self._bodies[body_id]
        if not isinstance(joint_id, int):
            raise TypeError("Expecting the given joint id to be an int, but got instead: {}".format(type(joint_id)))
        if joint_id < 0 or joint_id > (body.num_joints - 1):
            raise ValueError("joint_id should belong to [0, `num_joints-1`].")
        q = body.get_q_idx(joint_id, keep=True)
        if q == -1:
            return
        qpos_idx = body.q_idx0 + q
        qvel_idx = body.v_idx0 + q
        if not body.fixed:
            qpos_idx += 7
            qvel_idx += 6

        self.sim.data.qpos[qpos_idx] = position
        self.sim.data.qvel[qvel_idx] = velocity

    def enable_joint_force_torque_sensor(self, body_id, joint_ids, enable=True):
        """
        You can enable or disable a joint force/torque sensor in each joint.

        Args:
            body_id (int): body unique id.
            joint_ids (int, int[N]): joint index in range [0..num_joints(body_id)], or list of joint ids.
            enable (bool): True to enable, False to disable the force/torque sensor
        """
        pass  # TODO

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
            [0] np.array[float[3]]: Cartesian world position of CoM
            [1] np.array[float[4]]: Cartesian world orientation of CoM, in quaternion [x,y,z,w]
            [2] np.array[float[3]]: local position offset of inertial frame (center of mass) expressed in the URDF
              link frame
            [3] np.array[float[4]]: local orientation (quaternion [x,y,z,w]) offset of the inertial frame expressed
              in URDF link frame
            [4] np.array[float[3]]: Cartesian world position of the URDF link frame
            [5] np.array[float[4]]: Cartesian world orientation of the URDF link frame
            [6] np.array[float[3]]: Cartesian world linear velocity. Only returned if `compute_velocity` is True.
            [7] np.array[float[3]]: Cartesian world angular velocity. Only returned if `compute_velocity` is True.
        """
        body = self._bodies[body_id]
        if link_id < -1 or link_id > (body.num_bodies - 2):  # -1 is for the base
            raise ValueError("link_id should belong to [-1, `num_links-2`].")
        idx = body.b_idx0 + 1 + link_id

        pos = self.sim.data.body_xpos[idx]  # Cartesian position of body frame (same as xipos)
        quat = self._convert_wxyz_to_xyzw(self.sim.data.body_xquat[idx])  # Cartesian orientation of body frame
        # Note that in Mujoco the body frame is defined at the CoM

        # link frame in URDF is the joint frame in Mujoco
        # TODO: read from the Tree data structure and perform the correct transformations

        if compute_velocity:
            vel = self.sim.data.cvel[idx]  # com-based velocity [3D rot; 3D tran]
            return pos, quat, vel[3:], vel[:3]

        return pos, quat

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
        return [self.get_link_state(body_id, link_id, compute_velocity, compute_forward_kinematics)
                for link_id in link_ids]

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
        pass

    def _get_link_result(self, body_id, link_ids, mujoco_data, fct=None, slice=None):
        body = self._bodies[body_id]  # TODO: maybe use the tree data structure instead...
        one_link = isinstance(link_ids, int)
        if one_link:
            link_ids = [link_ids]
        results = []
        for link_id in link_ids:
            if link_id < -1 or link_id > (body.num_bodies - 2):  # -1 is for the base
                raise ValueError("link_id should belong to [-1, `num_links-2`].")
            idx = body.b_idx0 + 1 + link_id
            data = mujoco_data[idx]
            if slice is not None:
                data = data[slice]
            if fct is not None:
                data = fct(data)
            results.append(data)
        if one_link and len(results) > 0:
            return results[0]
        return results

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
        # TODO: maybe use the tree data structure instead...
        return self._get_link_result(body_id, link_ids, self.sim.model.body_mass)

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
        pass

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
        # Cartesian position of body frame (same as xipos)
        return np.asarray(self._get_link_result(body_id, link_ids, self.sim.data.body_xpos))

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
        # Cartesian orientation of body frame
        return np.asarray(self._get_link_result(body_id, link_ids, self.sim.data.body_xquat,
                                                fct=self._convert_wxyz_to_xyzw))

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
        return np.asarray(self._get_link_result(body_id, link_ids, self.sim.data.cvel, slice=slice(3, 6)))

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
        return np.asarray(self._get_link_result(body_id, link_ids, self.sim.data.cvel, slice=slice(3)))

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
        velocities = np.asarray(self._get_link_result(body_id, link_ids, self.sim.data.cvel))
        if velocities.ndim == 1:
            return np.roll(velocities, shift=3)
        return np.roll(velocities, shift=3, axis=1)

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
        # com-based acceleration
        return np.asarray(self._get_link_result(body_id, link_ids, self.sim.data.cacc, slice=slice(3, 6)))

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
        # com-based acceleration
        return np.asarray(self._get_link_result(body_id, link_ids, self.sim.data.cacc, slice=slice(3)))

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
        accelerations = np.asarray(self._get_link_result(body_id, link_ids, self.sim.data.cacc))
        if accelerations.ndim == 1:
            return np.roll(accelerations, shift=3)
        return np.roll(accelerations, shift=3, axis=1)

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
        body = self._bodies[body_id]
        q_idx = body.get_q_idx(joint_ids)
        return q_idx

    def get_actuated_joint_ids(self, body_id):
        """
        Get the actuated joint ids associated with the given body id.

        Args:
            body_id (int): unique body id.

        Returns:
            list[int]: actuated joint ids.
        """
        body = self._bodies[body_id]
        return [i for i, joint in enumerate(body.joints) if joint.dtype != 'fixed' and joint.dtype != 'floating']

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
        body = self._bodies[body_id]
        one_joint = isinstance(joint_ids, int)
        if one_joint:
            joint_ids = [joint_ids]
        names = []
        for joint_id in joint_ids:
            name = body.joints[joint_id].name
            if name.startswith('prl_'):
                name = '_'.join(name.split('_')[1:-1])
            names.append(name)
        if one_joint and len(names) > 1:
            return names[0]
        return names

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
        body = self._bodies[body_id]
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
            [0] float: mass in kg
            [1] float: lateral friction coefficient
            [2] np.array[float[3]]: local inertia diagonal. Note that links and base are centered around the center of
                mass and aligned with the principal axes of inertia.
            [3] np.array[float[3]]: position of inertial frame in local coordinates of the joint frame
            [4] np.array[float[4]]: orientation of inertial frame in local coordinates of joint frame
            [5] float: coefficient of restitution
            [6] float: rolling friction coefficient orthogonal to contact normal
            [7] float: spinning friction coefficient around contact normal
            [8] float: damping of contact constraints. -1 if not available.
            [9] float: stiffness of contact constraints. -1 if not available.
        """
        body = self._bodies[body_id]
        if link_id < -1 or link_id > (body.num_bodies - 2):  # -1 is for the base
            raise ValueError("link_id should belong to [-1, `num_links-2`].")

        idx = body.b_idx0 + 1 + link_id
        mass = self.sim.model.body_mass[idx]

        inertia = self.model.body_inertia[idx]  # diagonal inertia in ipos/iquat frame
        position = self.model.body_ipos[idx]
        orientation = self._convert_wxyz_to_xyzw(self.model.body_iquat[idx])

        slide, spin, roll = self.model.geom_friction[idx]  # TODO: one body can have multiple geoms

        # TODO: see solref and solimp
        # From Todorov: "The solref and solimp parameters can be adjusted to obtain different restitution effects
        # but they are not in one-to-one correspondence with the notion of restitution, and it is generally not
        # possible to guarantee a fixed coefficient of restitution. You can make contacts more bouncy by increasing
        # the second parameter in solref; it corresponds to a damping coefficient: 1 = critical damping, less than
        # 1 = under-damped, more than 1 = over-damped.
        # http://www.mujoco.org/forum/index.php?threads/coefficient-of-restitution.3426/

        # Check: self.model.geom_solmix[idx], self.model.geom_solref[idx], self.model.geom_solimp[idx]
        # Check section 'Restitution' in http://mujoco.org/book/modeling.html
        restitution = 0

        # 1. http://www.mujoco.org/book/modeling.html#CContact
        # 2. http://www.mujoco.org/book/modeling.html#CSolver
        damping = -1
        stiffness = -1

        return mass, slide, inertia, position, orientation, restitution, roll, spin, damping, stiffness

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
            local_inertia_diagonal (np.array[float[3]]): diagonal elements of the inertia tensor. Note that the base
                and links are centered around the center of mass and aligned with the principal axes of inertia so
                there are no off-diagonal elements in the inertia tensor.
            joint_damping (float): joint damping coefficient applied at each joint. This coefficient is read from URDF
                joint damping field. Keep the value close to 0.
                `joint_damping_force = -damping_coefficient * joint_velocity`.
        """
        body = self._bodies[body_id]
        if link_id < -1 or link_id > (body.num_bodies - 2):  # -1 is for the base
            raise ValueError("link_id should belong to [-1, `num_links-2`].")

        idx = body.b_idx0 + 1 + link_id
        if mass is not None:
            self.sim.model.body_mass[idx] = mass

        if lateral_friction is not None:  # TODO: one body can have multiple geoms
            self.model.geom_friction[idx][0] = lateral_friction

        if spinning_friction is not None:
            self.model.geom_friction[idx][1] = spinning_friction

        if rolling_friction is not None:
            self.model.geom_friction[idx][2] = rolling_friction

        if restitution is not None:  # TODO
            # play with solref and solimp
            # self.model.geom_solmix[idx] =
            # self.model.geom_solref[idx] =
            # self.model.geom_solimp[idx] =
            pass

        if linear_damping is not None:
            pass

        if angular_damping is not None:
            pass

        if contact_stiffness is not None:
            pass

        if contact_damping is not None:
            pass

        if friction_anchor is not None:
            pass

        # inertia
        if local_inertia_diagonal is not None:
            self.model.body_inertia[idx] = local_inertia_diagonal
        if inertia_position is not None:
            self.model.body_ipos[idx] = inertia_position
        if inertia_orientation is not None:
            self.model.body_iquat[idx] = inertia_orientation

        if joint_damping is not None:
            if link_id < 0 or link_id > (body.num_joints - 1):
                raise ValueError("link_id should belong to [0, `num_joints-1`] when setting the joint damping.")
            idx = body.v_idx0 + link_id
            if not body.fixed:
                idx += 6  #
            self.model.dof_damping[idx] = joint_damping

    def calculate_jacobian(self, body_id, link_id, local_position, q=None, dq=None, des_ddq=None):
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
        body = self._bodies[body_id]
        if link_id < -1 or link_id > (body.num_bodies - 2):  # -1 is for the base
            raise ValueError("link_id should belong to [-1, `num_links-2`].")

        # TODO: use q, dq, des_ddq by setting it in the data and then restoring the data

        idx = body.b_idx0 + 1 + link_id
        jacp, jacr = np.zeros(3 * self.model.nv), np.zeros(3 * self.model.nv)
        if local_position is None:
            local_position = np.zeros(3)
        mujoco.functions.mj_jac(self.model, self.sim.data, jacp, jacr, local_position, idx)
        jacp = jacp.reshape(3, self.model.nv)[:, body.v_idx0:body.v_idxf]
        jacr = jacr.reshape(3, self.model.nv)[:, body.v_idx0:body.v_idxf]
        return np.vstack((jacp, jacr))

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
        body = self._bodies[body_id]

        # TODO: use q

        # get sparse matrix
        sparse_inertia = self.sim.data.qM

        # Convert sparse inertia matrix M into full (i.e. dense) matrix
        inertia = np.zeros(self.model.nv * self.model.nv)
        mujoco.functions.mj_fullM(self.model, inertia, sparse_inertia)
        inertia = inertia.reshape(self.model.nv, self.model.nv)[body.v_idx0:body.v_idxf, body.v_idx0:body.v_idxf]
        return inertia

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
        body = self._bodies[body_id]

        # copy data
        dest = mujoco.cymj.PyMjData()
        mujoco.functions.mj_copyData(dest, self.model, self.sim.data)
        dest.qpos[body.q_idx0:body.q_idxf] = q
        dest.qvel[body.v_idx0:body.v_idxf] = dq
        dest.qacc[body.v_idx0:body.v_idxf] = des_ddq

        # inverse dynamics
        mujoco.functions.mj_inverse(self.model, dest)

        # get resulting torques and return it
        torques = dest.qfrc_applied[body.v_idx0:body.v_idxf]
        return torques

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
        body = self._bodies[body_id]

        # copy data and set q, dq, tau
        dest = mujoco.cymj.PyMjData()
        mujoco.functions.mj_copyData(dest, self.model, self.sim.data)
        dest.qpos[body.q_idx0:body.q_idxf] = q
        dest.qvel[body.v_idx0:body.v_idxf] = dq
        dest.qfrc_applied[body.v_idx0:body.v_idxf] = torques

        # forward dynamics
        mujoco.functions.mj_forward(self.model, dest)

        # get ddq and return it
        qacc = dest.qacc[body.v_idx0:body.v_idxf]
        return qacc


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
