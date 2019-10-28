# -*- coding: utf-8 -*-
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
# try:
#     import pymesh    # rapid prototyping platform focused on geometry processing
#     # doc: https://pymesh.readthedocs.io/en/latest/user_guide.html
#
#     import pyassimp  # library to import and export various 3d-model-formats
#     # doc: http://www.assimp.org/index.php
#     # github: https://github.com/assimp/assimp
# except ImportError as e:
#     raise ImportError(str(e) + "\nTry to install pymesh pyassimp: `pip install pymesh pyassimp`")

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
try:
    import glfw
except ImportError as e:
    raise ImportError(str(e) + "\nTry to install GLFW: `pip install glfw`")

# import pyrobolearn related functionalities
from pyrobolearn.simulators.simulator import Simulator
from pyrobolearn.utils.parsers.robots import URDFParser, MuJoCoParser, SDFParser
import pyrobolearn.utils.parsers.robots.data_structures as struct
from pyrobolearn.utils.transformation import get_homogeneous_matrix, get_quaternion_from_matrix
from pyrobolearn.utils.parsers.robots.data_structures import transform_child_joint_frame_to_parent_inertial_frame, \
    transform_inertial_frame_to_child_inertial_frame, transform_inertial_frame_to_child_link_frame, \
    transform_inertial_frame_to_joint_frame


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
    """MuJoCo Body

    This class describes a MuJoCo multi-body system. MuJoCo works with vectors, matrices that contains all the
    variables for the various multi-body systems. So this class allows to remember the corresponding indices
    associated with this multi-body for each of these MuJoCo vectors / matrices.

    For instance, in MuJoCo, `data.qpos` will return a long vector that contains the position values of all the
    actuated joints for all the multi-bodies that are loaded in the simulator. So if you have several robots, they are
    all in that long `data.qpos` vector. It becomes then important to remember the indices that are associated with
    each multi-body.

    It also keep in memory all the joints and bodies/links data structures that are defined in
    `pyrobolearn.utils.parsers.robots.data_structures.py`. These data structures are instantiated when parsing the
    various robotic files (URDF, MuJoCo XML, SDF, etc). From these structures you can instantiated the various
    information that were present in these files.
    """

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
            body = struct.MultiBody(name='prl_dummy_' + str(body_id), root=struct.Body(body_id=body_id))
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
        if not self.fixed:  # if free joint, add 1 because in Mujoco the pose is represented as position vector (3)
            self.q_length += 1   # + quaternion (4) = 7, so one more than 6 DoFs

        # define variables for indices that appears in the various vectors and matrices returned by mjModel and mjData
        self._q_idx0, self._q_idxf = 0, 0  # initial and final q indices
        self._b_idx0, self._b_idxf = 0, 0  # initial and final body indices
        self._j_idx0, self._j_idxf = 0, 0  # initial and final free joint indices
        self._v_idx0, self._v_idxf = 0, 0  # initial and final dq (velocity) indices
        self._q_idx1, self._v_idx1 = 0, 0  # initial q and dq indices (which don't take into account virtual joints)
        self._u_idx0, self._u_idxf = 0, 0  # initial and final ctrl indices
        self._u_p_indices = np.array([])  # ctrl ids for position motors (do +1 to get velocities, +2 to get torques)

        # keep in memory the body
        # self.body = body
        self.joints = np.array(list(body.joints.values()))
        self.links = np.array(list(body.bodies.values()))

        # compute mapping from joint ids to q indices
        idx, jnt_to_q = 0, []
        for joint in body.joints.values():
            if joint.dtype == 'fixed':
                jnt_to_q.append(-1)
            else:
                jnt_to_q.append(idx)
                idx += 1
        self.jnt_to_q = np.array(jnt_to_q)  # joint ids to q indices, e.g. [0, -1, -1, 1, 2, -1, 3] (-1 = fixed)

        # keep in memory the link ids

        # define variables related to actuators and control
        self.ctrl_modes = np.array([struct.ControlMode.NULL] * self.num_actuated_joints)  # one for each joint
        self.gains = None   # original gains
        self.biases = None  # original biases
        self.ctrl_limited = None  # original binary vector (nu,) to specify if the control inputs are limited
        self.force_limited = None  # original binary vector (nu,) to specify if the forces are limited
        self.ctrl_range = None   # original range of control inputs (nu, 2)
        self.force_range = None  # original range of forces (nu, 2)

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
        self._q_idx1 = q if self.fixed else q + 7  # set the initial q index (doesn't take into account virtual joints)

    @property
    def q_idx1(self):
        """Return the initial q index that does not take into account the virtual joints for the base."""
        return self._q_idx1

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
        self._v_idx1 = v if self.fixed else v + 6  # set the initial q index (doesn't take into account virtual joints)

    @property
    def v_idx1(self):
        """Return the initial velocity index that does not take into account the virtual joints for the base."""
        return self._v_idx1

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
    def u_idx0(self):
        """Return the initial ctrl index."""
        return self._u_idx0

    @u_idx0.setter
    def u_idx0(self, u):
        """Set the initial ctrl index."""
        u = int(u)
        if u < 0:
            raise ValueError("Error while setting the initial ctrl index, this index has to be bigger than 0!")
        self._u_idx0 = u
        self._u_idxf = u + 3 * self.num_dofs  # set the final ctrl index (the factor 3 is because we create 3 motors)
        if not self.fixed:
            self._u_idxf -= 3 * 6  # remove the first 6 DoFs
        self._u_p_indices = np.array(range(self._u_idx0, self._u_idxf, 3))

    @property
    def u_idxf(self):
        """Return the final ctrl index."""
        return self._u_idxf

    @u_idxf.setter
    def u_idxf(self, u):
        """Set the final ctrl index."""
        u = int(u)
        if u < 0:
            raise ValueError("Error while setting the final body index, this index has to be bigger than 0!")
        self._u_idxf = u
        self._u_idx0 = u - 3 * self.num_dofs  # set initial ctrl index (the factor 3 is because we create 3 motors)
        if not self.fixed:
            self._u_idx0 += 3 * 6  # remove the first 6 DoFs
        if self._u_idx0 < 0:
            raise ValueError("Error while setting the final ctrl index, by computing automatically the initial ctrl "
                             "index from it, it appears it is smaller than 0. The initial ctrl index has to be bigger "
                             "than 0!")
        self._u_p_indices = np.array(range(self._u_idx0, self._u_idxf, 3))

    @property
    def u_p_indices(self):
        """Return the ctrl indices for the position motors. To get the velocities, just add +1, and to get the efforts
        just add +2."""
        return self._u_p_indices

    @property
    def u_v_indices(self):
        """Return the ctrl indices for the velocity motors. To get the positions, just subtract 1, and to get the
        efforts just add 1."""
        return self._u_p_indices + 1

    @property
    def u_e_indices(self):
        """Return the ctrl indices for the effort (torque/force) motors. To get the positions, just subtract 2, and
        to get the velocities, subtract 1."""
        return self._u_p_indices + 2

    @property
    def num_ctrl_inputs(self):
        """Return the number of control inputs."""
        return self._u_idxf - self._u_idx0

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
        """
        Return the q index(ices) associated with the given joint id(s). The indices are between -1 and the number of
        actuated joints. If the provided `keep` argument is True, then it will return also the q indices for the
        fixed joints. By default, for the fixed joints, the corresponding q index is set to -1.

        For instance, if we have a body with 5 joints such that in the order we have 1 revolute, 2 fixed, and 2
        revolute joints, the method `get_q_idx` (with `keep=True` and specifying all the joint ids even the fixed
        ones) will return [0, -1, -1, 1, 2]. If `keep=False`, then it will return [0, 1, 2].

        Args:
            joint_id (np.array[int], int): joint id(s) (each joint id should be between [0, num_joints[).
            keep (bool): if True, keep the fixed joints.

        Returns:
            np.array[int], int, None: q index(ices) associated with the given joint id(s).
        """
        q = self.jnt_to_q[joint_id]
        if keep:  # keep fixed joints (-1)
            return q
        if isinstance(q, int):
            if q != -1:
                return q
            return None
        return q[q != -1]  # remove fixed joints

    def get_dq_idx(self, joint_id, keep=False):
        """Return the dq index(ices) associated with the given joint id(s).

        Args:
            joint_id (np.array[int], int): joint id(s) (each joint id should be between [0, num_joints[).
            keep (bool): if True, keep the fixed joints.

        Returns:
            np.array[int], int, None: dq index(ices) associated with the given joint id(s).
        """
        return self.get_q_idx(joint_id, keep)

    def get_joint(self, joint_id):
        """Get the joint data structure from the joint id (which is between [0, num_joints[)."""
        return self.joints[joint_id]

    def get_joint_type(self, joint_id):
        return self.joints[joint_id].dtype

    def get_link(self, link_id):
        """Get the link data structure from the link id (which is between [0, num_links[)."""
        return self.links[link_id]

    def transform_inertial_frame_to_joint_frame(self, body_id):
        """Return the homogeneous transform from the inertial frame to the joint frame."""
        body = self.links[body_id]
        return transform_inertial_frame_to_joint_frame(body)

    def transform_child_joint_frame_to_parent_inertial_frame(self, child_body_id):
        """Return the homogeneous transform from the child joint frame to the parent inertial frame."""
        body = self.links[child_body_id]
        return transform_child_joint_frame_to_parent_inertial_frame(body)

    def transform_inertial_frame_to_child_link_frame(self, child_body_id):
        """Return the homogeneous transform from the parent inertial frame to the child link/joint frame."""
        body = self.links[child_body_id]
        return transform_inertial_frame_to_child_link_frame(body)

    def transform_inertial_frame_to_child_inertial_frame(self, child_body_id):
        """Return the homogeneous transform from the parent inertial frame to the child inertial frame."""
        body = self.links[child_body_id]
        return transform_inertial_frame_to_child_inertial_frame(body)


class StateIndices(object):
    """Mujoco state indices.

    This class allows to remember the indices associated with a given state.
    """

    def __init__(self):
        self.qpos = None
        self.qvel = None
        self.act = None
        self.mocap_pos = None
        self.mocap_quat = None
        self.userdata = None
        self.qacc_warmstart = None

    def reset(self):
        self.qpos, self.qvel, self.act, self.mocap_pos, self.mocap_quat = None, None, None, None, None
        self.userdata, self.qacc_warmstart = None, None


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
        self._ctrl_cnt = 0  # this is for the motors

        self.default_timestep = 0.002
        self.dt = self.default_timestep

        # each time we add or remove an object, the following variable is set to True. When calling the `step` method,
        # it will reload the model in the simulator.
        self._model_changed = False

        # create dynamically an empty world (XML)
        self._root = self._parser.root
        self._worldbody = self._parser.worldbody

        self._state_indices = StateIndices()

        # add light
        self._parser.add_element("light", self._worldbody,
                                 attributes={"diffuse": "0.5 0.5 0.5", "pos": "0 0 3", "directional": "true",
                                             "dir": "0 0 -1"})

        # add world camera
        self._parser.add_element("camera", self._worldbody, attributes={"name": "prl_world_camera",
                                                                        "fovy": "45", "pos": "0 0 0"})

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
            render (bool): if we should render or not.
        """
        # self.render(enable=False)  # to delete the previous viewer instance if defined
        state = None if self.sim is None else self._save_state()

        # create the model
        # self.model = mujoco.load_model_from_path(path)
        root = self._parser.get_string(pretty_format=False)
        self.model = mujoco.load_model_from_xml(root)

        # create the simulator from the model
        self.sim = mujoco.MjSim(self.model)

        # load the state
        if state is not None:
            self._load_state(state, self._state_indices)
            self._state_indices.reset()

        # if we need to render
        if render:
            # self.render(enable=True)   # to instantiate the viewer
            if self.viewer is None:
                self.render(enable=True)
            else:
                self.viewer.update_sim(self.sim)

    @staticmethod
    def _check_joint_id(body, joint_id):
        """
        Check that the given joint_id are between [0, num_joints[.

        Args:
            body (Body): MuJoCo body instance.
            joint_id (int): unique joint id.

        Returns:
            int: joint id (same as the one given as input).
        """
        if not isinstance(joint_id, int):
            raise TypeError("Expecting the given joint id to be an int, but got instead: {}".format(type(joint_id)))
        if joint_id < 0 or joint_id > (body.num_joints - 1):
            raise ValueError("joint_id should belong to [0, {}], but got: {}".format(body.num_joints - 1, joint_id))
        return joint_id

    @staticmethod
    def _check_joint_ids(body, joint_ids):
        """
        Check that all the given joint ids are between [0, num_joints[.

        Args:
            body (Body): MuJoCo body instance.
            joint_ids (np.array[int], int): unique joint ids.

        Returns:
            np.array[int], int: joint id(s) (same as the ones given as inputs).
        """
        joint_ids = np.asarray(joint_ids)
        if np.any(joint_ids < 0) or np.any(joint_ids > (body.num_joints - 1)):
            raise ValueError("joint_ids should belong to [0, {}], but got: {}".format(body.num_joints - 1, joint_ids))
        if joint_ids.ndim == 0:
            return int(joint_ids)
        return joint_ids

    @staticmethod
    def _check_link_id(body, link_id):
        """
        Check that the given link_id is between [-1, num_links-2], and return the converted link id such that
        it is between [0, num_links-1].

        Args:
            body (Body): MuJoCo body instance.
            link_id (int): unique link id.

        Returns:
            int: converted link id
        """
        if not isinstance(link_id, int):
            raise TypeError("Expecting the given link id to be an int, but got instead: {}".format(type(link_id)))
        if link_id < -1 or link_id > (body.num_bodies - 2):  # -1 is for the base
            raise ValueError("link_id should belong to [-1, {}], but got: {}".format(body.num_bodies - 2, link_id))
        return link_id + 1  # shift such that between [0, `body.num_bodies`[.

    @staticmethod
    def _check_link_ids(body, link_ids):
        """
        Check that the given link_ids are between [-1, num_links-2], and return the converted link ids such that
        they are between [0, num_links-1].

        Args:
            body (Body): MuJoCo body instance.
            link_ids (np.array[int], int): unique link id(s).

        Returns:
            np.array[int], int: converted link id(s).
        """
        link_ids = np.asarray(link_ids)
        if np.any(link_ids < -1) or np.any(link_ids > (body.num_bodies - 2)):  # -1 is for the base
            raise ValueError("link_ids should belong to [-1, {}], but got: {}".format(body.num_bodies - 2, link_ids))
        if link_ids.ndim == 0:
            return int(link_ids) + 1
        return link_ids + 1  # shift such that between [0, `body.num_bodies`[.

    @staticmethod
    def _get_joint_type_id(joint_type):
        """
        Return the joint type id given the joint type string.

        Args:
            joint_type (str): joint type string.

        Returns:
            int: unique joint type id.
        """
        if joint_type == 'fixed':
            return Simulator.JOINT_FIXED
        if joint_type == 'revolute':
            return Simulator.JOINT_REVOLUTE
        if joint_type == 'prismatic':
            return Simulator.JOINT_PRISMATIC
        elif joint_type == 'ball':
            return Simulator.JOINT_SPHERICAL
        elif joint_type == 'floating':
            return Simulator.JOINT_FREE
        elif joint_type == 'gear':
            return Simulator.JOINT_GEAR
        else:
            return -1

    @staticmethod
    def _process_name(name):
        """
        Process the given name. By default, the MuJoCo parser add the prefix `prl_` and the suffix `_str(cnt)`.
        This is to avoid collisions between different names and making them unique. Here, we remove these prefix and
        suffix and return the original name (of the joint/body).

        Args:
            name (str): name with a possible prefix `prl_` and suffix `_str(cnt)`.

        Returns:
            str: processed name.
        """
        if name.startswith('prl_'):
            return '_'.join(name.split('_')[1:-1])
        return name

    def _save_state(self):
        """Save current mujoco state.

        Returns:
            float: current time step
            np.array[float[nq]]: joint positions
            np.array[float[nv]]: joint velocities
            np.array[float[na]], None: actuator activation
            np.array[float[nmocap,3]]: positions of mocap bodies
            np.array[float[nmocap,4]]: orientations of mocap bodies
            np.array[float[nuserdata]]: user data (not touched by engine)
            np.array[float[nv]]: acceleration used for warm start

        References:
            - http://www.mujoco.org/book/programming.html#siStateControl
        """
        # check: http://www.mujoco.org/book/programming.html#siStateControl

        # copy simulation state
        t = self.sim.data.time
        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel
        act = self.sim.data.act

        # copy mocap body pose and user data
        mocap_pos = self.sim.data.mocap_pos
        mocap_quat = self.sim.data.mocap_quat
        userdata = self.sim.data.userdata

        # copy warm-start acceleration
        qacc_warmstart = self.sim.data.qacc_warmstart

        return t, qpos, qvel, act, mocap_pos, mocap_quat, userdata, qacc_warmstart

    def _clear_control(self):
        """Clear the control vector in Mujoco given by u = (data.ctrl, data.qfrc_applied, data.xfrc_applied)
        where `ctrl` are the control signals for the actuators, `qfrc_applied` are the applied generalized forces
        in the joint space, and `xfrc_applied` are the applied Cartesian force/torque.

        References:
            - http://www.mujoco.org/book/programming.html#siStateControl
        """
        self.sim.data.ctrl[:] = 0  # (nu,)
        self.sim.data.qfrc_applied[:] = 0  # (nv,)
        self.sim.data.xfrc_applied[:, :] = 0  # (nbody, 6)

    def _load_state(self, state, indices=None):
        """
        Load and set the given mujoco state.

        Args:
            state (list, tuple): the state returned by `_save_state()` method.
            indices (None, StateIndices): indices of the state to change in the whole MuJoCo state vector.
        """
        t, qpos, qvel, act, mocap_pos, mocap_quat, userdata, qacc_warmstart = state
        if indices is None:
            indices = self._state_indices

        self.sim.data.time = t

        if qpos is not None:
            if indices.qpos is None:
                self.sim.data.qpos[:len(qpos)] = qpos
            else:
                self.sim.data.qpos[indices.qpos] = qpos

        if qvel is not None:
            if indices.qvel is None:
                self.sim.data.qvel[:len(qvel)] = qvel
            else:
                self.sim.data.qvel[indices.qvel] = qvel

        if act is not None:
            if indices.act is None:
                self.sim.data.act[:len(act)] = act
            else:
                self.sim.data.act[indices.act] = act

        if mocap_pos is not None:
            if indices.mocap_pos is None:
                self.sim.data.mocap_pos[:len(mocap_pos)] = mocap_pos
            else:
                self.sim.data.mocap_pos[indices.mocap_pos] = mocap_pos

        if mocap_quat is not None:
            if indices.mocap_quat is None:
                self.sim.data.mocap_quat[:len(mocap_quat)] = mocap_quat
            else:
                self.sim.data.mocap_quat[indices.mocap_quat] = mocap_quat

        if userdata is not None:
            if indices.userdata is None:
                self.sim.data.userdata[:len(userdata)] = userdata
            else:
                self.sim.data.userdata[indices.userdata] = userdata

        if qacc_warmstart is not None:
            if indices.qacc_warmstart is None:
                self.sim.data.qacc_warmstart[:len(qacc_warmstart)] = qacc_warmstart
            else:
                self.sim.data.qacc_warmstart[indices.qacc_warmstart] = qacc_warmstart

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

    ##############
    # Simulators #
    ##############

    def print_xml(self):
        """
        Print the generated MuJoCo XML file that is currently in memory.
        """
        print(self._parser.get_string(pretty_format=True))

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

            # # select with the mouse
            # coordinates = np.zeros(3)
            # geomid, skin = 0, 0
            #
            # # mouse selection.
            # mujoco.functions.mjv_select(self.model, self.sim.data, self.viewer.vopt, aspectratio, relx, rely,
            #                             self.viewer.scn, coordinates, geomid, skin)
            #
            # # Move perturb object with mouse; action is mjtMouse.
            # action = 0
            # mujoco.functions.mjv_movePerturb(self.model, self.sim.data, action, reldx, reldy, self.viewer.scn,
            #                                  self.viewer.pert)
            #
            # # Set perturb force,torque in d->xfrc_applied, if selected body is dynamic.
            # mujoco.functions.mjv_applyPerturbForce(self.model, self.sim.data, self.viewer.pert)

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
            int, str: unique state id, or filename. This id / filename can be used to load the state.
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
        # check
        path = os.path.abspath(filename)  # /path/to/pyrobolearn/robots/urdfs/<robot>/robot.urdf
        dirname = str(os.path.dirname(path))  # /path/to/pyrobolearn/robots/urdfs/<robot>/
        basename = str(os.path.basename(path).split('.')[-2])  # robot name without extension

        new_path = dirname + '/' + basename + '_mujoco.urdf'
        if os.path.exists(new_path):
            filename = new_path

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

        return self._create_body(tree, verbose=1)

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
            # self._model_changed = True
            self._create_sim()

    def _create_body(self, tree, body_id=None, verbose=1):
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
        body.u_idx0 = self._ctrl_cnt
        self._ctrl_cnt += body.num_ctrl_inputs

        # update mujoco model if necessary
        self._update_sim()

        # if verbose, print current xml file
        if verbose > 1:
            print(self._parser.get_string(pretty_format=True))

        # save default gain and bias parameters, control inputs range, and force range
        body.gains = self.model.actuator_gainprm[body.u_idx0:body.u_idxf]
        body.biases = self.model.actuator_biasprm[body.u_idx0:body.u_idxf]
        body.ctrl_limited = self.model.actuator_ctrllimited[body.u_idx0:body.u_idxf]
        body.force_limited = self.model.actuator_forcelimited[body.u_idx0:body.u_idxf]
        body.ctrl_range = self.model.actuator_ctrlrange[body.u_idx0:body.u_idxf]
        body.force_range = self.model.actuator_forcerange[body.u_idx0:body.u_idxf]

        # print("Creating body with the following gains: ")
        # print("Position gain = {}, bias = {}".format(self.model.actuator_gainprm[body.u_p_indices],
        #                                              self.model.actuator_biasprm[body.u_p_indices]))
        # print("Velocity gain = {}, bias = {}".format(self.model.actuator_gainprm[body.u_v_indices],
        #                                              self.model.actuator_biasprm[body.u_v_indices]))
        # print("Effort gain = {}, bias = {}".format(self.model.actuator_gainprm[body.u_e_indices],
        #                                            self.model.actuator_biasprm[body.u_e_indices]))
        # print("Force limited?: ", self.model.actuator_forcelimited[body.u_p_indices])
        # print("Force range = ", self.model.actuator_forcerange[body.u_p_indices])

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

        return self._create_body(tree, body_id=self._body_cnt, verbose=1)

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
        return self._bodies[body_id].num_links - 1  # remove the base link

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

        # check joint id
        joint_id = self._check_joint_id(body, joint_id)
        joint = body.get_joint(joint_id)

        name = self._process_name(joint.name)
        dtype = self._get_joint_type_id(joint.dtype)

        # get index
        q = body.get_q_idx(joint_id, keep=True)  # -1 for fixed joint
        flag = 1  # as in Bullet
        if joint_id == body.num_joints - 1:
            flag = 0

        if q == -1:  # fixed joint
            q_idx = -1
            dq_idx = -1
            damping = 0.
            friction = 0.
            axis = np.zeros(3)
            lower, upper = 0., -1.
            max_vel = 0.
            max_force = 0.

        else:
            joint_idx = body.j_idx0 + q
            q_idx = q
            dq_idx = q
            # if not body.fixed:  # compatible with Bullet
            q_idx += 7
            dq_idx += 6
            dof_addr = self.model.jnt_dofadr[joint_idx]
            damping = self.model.dof_damping[dof_addr]
            friction = self.model.dof_frictionloss[dof_addr]
            axis = self.model.jnt_axis[joint_idx]
            limited = self.model.jnt_limited[joint_idx]
            if limited:
                lower, upper = self.model.jnt_range[joint_idx]
            else:
                lower, upper = -np.infty, np.infty
            max_vel = joint.velocity
            max_force = joint.effort

        link = joint.child
        link_name = link.name
        parent_idx = joint.parent.id - 1  # the base is -1 by convention in Bullet

        # position = joint.position  # self.model.jnt_pos[body.j_idx0 + q]
        # orientation = joint.quaternion
        homogeneous = transform_inertial_frame_to_child_link_frame(link)  # Bullet express joint from inertial
        position = homogeneous[:3, 3]
        orientation = get_quaternion_from_matrix(homogeneous[:3, :3])

        if position is None:
            position = np.zeros(3)
        if orientation is None:
            orientation = np.zeros(3)

        # xanchor, stiffness, etc

        return joint_id, name, dtype, q_idx, dq_idx, flag, damping, friction, lower, upper, max_force, max_vel, \
               link_name, axis, position, orientation, parent_idx

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
        if q == -1:  # fixed joint
            return 0, 0, np.zeros(6), 0

        # compute joint position and velocity
        pos = self.sim.data.qpos[body.q_idx1 + q]
        vel = self.sim.data.qvel[body.v_idx1 + q]

        # compute reaction force
        force_parent = self.sim.data.cfrc_int[body.b_idx0 + joint_id]  # com-based interaction force with parent
        force_ext = self.sim.data.cfrc_ext[body.b_idx0 + joint_id]  # com-based external force on body
        force = force_ext + force_parent  # TODO: is it - instead of +?
        np.roll(force, shift=3, axis=force.ndim - 1)  # [torque, force] --> [force, torque]
        # TODO: express it in the joint frame

        # compute the applied torque
        torque = self.sim.data.qfrc_applied[body.v_idx1 + q]

        return pos, vel, force, torque

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
        body = self._bodies[body_id]
        joint_id = self._check_joint_id(body, joint_id)
        q = body.get_q_idx(joint_id, keep=True)
        if q != -1:
            self.model.qpos0[body.q_idx1 + q] = -position
            # self.sim.data.qpos[body.q_idx1 + q] = position
            if velocity is not None:
                self.sim.data.qvel[body.v_idx1 + q] = velocity

    def reset_joint_states(self, body_id, joint_ids=None, positions=None, velocities=None):
        """
        Reset the state of the joint. It is best only to do this at the start, while not running the simulation:
        `reset_joint_state` overrides all physics simulation.

        Args:
            body_id (int): unique body id.
            joint_ids (list[int]): joint index in range [0..num_joints(body_id)]
            positions (np.array[float]): the joint positions (angle in radians [rad] or position [m])
            velocities (np.array[float]): the joint velocities (angular [rad/s] or linear velocity [m/s])
        """
        # WARNING: the angles are reversed when setting qpos0 instead of qpos!!
        body = self._bodies[body_id]
        if joint_ids is None:
            # self.model.qpos0[body.q_idx1:body.q_idxf] = -positions
            self.sim.data.qpos[body.q_idx1:body.q_idxf] = positions
        else:
            joint_ids = self._check_joint_ids(body, joint_ids)
            q = body.get_q_idx(joint_ids, keep=False)
            if q is None:
                return
            # self.model.qpos0[body.q_idx1 + q] = -positions
            self.sim.data.qpos[body.q_idx1 + q] = positions
            if velocities is not None:
                self.sim.data.qvel[body.v_idx1 + q] = velocities

    def enable_joint_force_torque_sensor(self, body_id, joint_ids, enable=True):
        """
        You can enable or disable a joint force/torque sensor in each joint.

        Args:
            body_id (int): body unique id.
            joint_ids (int, int[N]): joint index in range [0..num_joints(body_id)], or list of joint ids.
            enable (bool): True to enable, False to disable the force/torque sensor
        """
        # attach a force sensor to the specified body (site)
        # attach a torque sensor to the specfied body (site)
        # or check cfrc_int and cfrc_ext
        pass

    def set_joint_motor_control(self, body_id, joint_ids, control_mode=Simulator.POSITION_CONTROL, positions=None,
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
        if control_mode == Simulator.POSITION_CONTROL:
            self.set_joint_positions(body_id, joint_ids, positions, velocities, kp, kd, forces)
        elif control_mode == Simulator.VELOCITY_CONTROL:
            self.set_joint_velocities(body_id, joint_ids, velocities, forces)
        elif control_mode == Simulator.TORQUE_CONTROL:
            self.set_joint_torques(body_id, joint_ids, forces)

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
        link_id = self._check_link_id(body, link_id)
        idx = body.b_idx0 + link_id

        pos = self.sim.data.xipos[idx]  # Cartesian position of body CoM
        quat = self._convert_wxyz_to_xyzw(self.sim.data.body_xquat[idx])  # Cartesian orientation of body frame
        # Note that in Mujoco the body frame is defined at the CoM

        link = body.get_link(link_id)
        inertial = link.inertial
        ipos = inertial.position if inertial is not None else np.zeros(3)
        iquat = inertial.quaternion if inertial is not None else np.array([0., 0., 0., 1.])

        lpos = self.sim.data.body_xpos[idx]  # Cartesian position of body frame
        lquat = quat

        if compute_velocity:
            vel = self.sim.data.cvel[idx]  # com-based velocity [3D rot; 3D tran]
            lin_vel, ang_vel = vel[3:], vel[:3]
        else:
            lin_vel, ang_vel = np.zeros(3), np.zeros(3)

        return pos, quat, ipos, iquat, lpos, lquat, lin_vel, ang_vel

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

    def get_link_names(self, body_id, link_ids=None):
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
        one_link = isinstance(link_ids, int)
        if link_ids is None:
            link_ids = range(0, body.num_bodies-1)  # do not return the name of the base link
        elif one_link:
            link_ids = [link_ids]
        names = []
        for link_id in link_ids:
            link_id = self._check_link_id(body, link_id)
            name = self._process_name(body.get_link(link_id).name)
            names.append(name)
        if one_link:
            return names[0]
        return names

    def _get_link_result(self, body_id, link_ids, mujoco_data, fct=None, slice=None):
        body = self._bodies[body_id]  # TODO: maybe use the tree data structure instead...
        if link_ids is None:
            data = mujoco_data[body.b_idx0+1:body.b_idxf]  # +1 to not account for the base as in PyBullet
        else:
            link_ids = self._check_link_ids(body, link_ids)
            idx = body.b_idx0 + link_ids
            data = mujoco_data[idx]
        if slice is not None:
            data = data[slice]
        if fct is not None:
            data = fct(data)
        return data

    def get_link_masses(self, body_id, link_ids=None):
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
        return self._get_link_result(body_id, link_ids, self.sim.model.body_mass)

    def get_link_frames(self, body_id, link_ids=None):
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
        pos = self._get_link_result(body_id, link_ids, self.sim.data.body_xpos)
        quat = self._get_link_result(body_id, link_ids, self.sim.data.body_xquat, fct=self._convert_wxyz_to_xyzw)
        return pos, quat

    def get_link_world_positions(self, body_id, link_ids=None):
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
        return self._get_link_result(body_id, link_ids, self.sim.data.body_xpos)  # TODO: xipos vs body_xpos

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
        return self._get_link_result(body_id, link_ids, self.sim.data.body_xquat, fct=self._convert_wxyz_to_xyzw)

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
        return self._get_link_result(body_id, link_ids, self.sim.data.cvel, slice=slice(3, 6))

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
        return np.roll(velocities, shift=3, axis=velocities.ndim-1)

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
        return np.roll(accelerations, shift=3, axis=accelerations.ndim - 1)

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
        body = self._bodies[body_id]
        one_joint = isinstance(joint_ids, int)
        if one_joint:
            joint_ids = [joint_ids]
        types = []
        for joint_id in joint_ids:
            joint = body.get_joint(joint_id)
            types.append(self._get_joint_type_id(joint.dtype))
        if one_joint and len(types) > 1:
            return types[0]
        return types

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
        body = self._bodies[body_id]
        joint = body.get_joint(joint_ids)
        one_joint = isinstance(joint, struct.Joint)
        if one_joint:
            joint = [joint]
        forces = []
        for j in joint:
            force = j.effort
            if force is None:
                forces.append(0.)
            else:
                forces.append(force)
        if one_joint:
            return forces[0]
        return forces

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
        body = self._bodies[body_id]
        joint = body.get_joint(joint_ids)
        one_joint = isinstance(joint, struct.Joint)
        if one_joint:
            joint = [joint]
        velocities = []
        for j in joint:
            vel = j.velocity
            if vel is None:
                velocities.append(0.)
            else:
                velocities.append(vel)
        if one_joint:
            return velocities[0]
        return velocities

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
        body = self._bodies[body_id]
        q = body.get_q_idx(joint_ids, keep=True)  # -1 for fixed joint
        if isinstance(q, float):
            if q == -1:  # fixed joint
                return np.zeros(3)
            return self.model.jnt_axis[body.j_idx0 + q]
        axes = np.zeros((len(q), 3))
        axes[q != -1] = self.model.jnt_axis[body.j_idx0 + q[q != -1]]
        return axes

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
        # body = self._bodies[body_id]
        #
        # q = self.get_joint_positions(body_id, joint_ids=joint_ids)
        # qvel= self.get_joint_velocities(body_id, joint_ids=joint_ids)
        #
        # if kps is None:
        #     kps = 1000.
        # if kds is None:
        #     kds = 1.
        # if velocities is None:
        #     velocities = 0.
        #
        # tau = kps * (positions - q) + kds * (velocities - qvel)
        #
        # if joint_ids is None:
        #     # self.sim.data.qpos[body.q_idx1:body.q_idxf] = positions
        #     c_q_dq = self.sim.data.qfrc_bias[body.v_idx1:body.v_idxf]
        #     # self.sim.data.qfrc_applied[body.v_idx1:body.v_idxf] = tau + c_q_dq  # DEPRECATED
        #     self.sim.data.qfrc_actuator[body.v_idx1:body.v_idxf] = tau + c_q_dq
        #
        # else:
        #     # check if valid joints
        #     self._check_joint_ids(body, joint_ids)
        #
        #     # if one joint, set its position
        #     if isinstance(joint_ids, int):
        #         # self.sim.data.qpos[body.q_idx1 + joint_ids] = positions
        #         c_q_dq = self.sim.data.qfrc_bias[body.v_idx1 + joint_ids]
        #         # self.sim.data.qfrc_applied[body.v_idx1 + joint_ids] = tau + c_q_dq  # DEPRECATED
        #         self.sim.data.qfrc_actuator[body.v_idx1 + joint_ids] = tau + c_q_dq
        #
        #     # if multiple joints, set their positions
        #     else:
        #         q = body.get_q_idx(joint_ids, keep=True)  # E.g. [0, -1, 1, -1, 2, 3]  (-1 are for fixed joints)
        #
        #         # self.sim.data.qpos[body.q_idx1 + q[q != -1]] = positions
        #         c_q_dq = self.sim.data.qfrc_bias[body.v_idx1 + q[q != -1]]
        #         # self.sim.data.qfrc_applied[body.v_idx1 + q[q != -1]] = tau + c_q_dq  # DEPRECATED
        #         self.sim.data.qfrc_actuator[body.v_idx1 + q[q != -1]] = tau + c_q_dq

        # get body and q indices
        body = self._bodies[body_id]
        if joint_ids is None:
            q_idx = np.array(range(body.num_actuated_joints))
        else:
            # check if valid joints
            self._check_joint_ids(body, joint_ids)

            q_idx = body.get_q_idx(joint_ids, keep=True)  # E.g. [0, -1, 1, -1, 2, 3]  (-1 are for fixed joints)
            q_idx = q_idx[q_idx != -1]

        if len(q_idx) == 0:
            raise ValueError("No actuated joints to set the positions to...")

        # set gains
        if kps is not None:
            self.model.actuator_gainprm[body.u_p_indices[q_idx], 0] = kps  # for desired positions
            self.model.actuator_biasprm[body.u_p_indices[q_idx], 1] = -kps  # for current positions
        if kds is not None:
            if velocities is not None:
                self.model.actuator_gainprm[body.u_v_indices[q_idx], 0] = kds  # for desired velocities
            self.model.actuator_biasprm[body.u_v_indices[q_idx], 2] = -kds  # for current velocities

        # check the current control mode, deactivate the other motors by setting their gains and biases to zero
        if velocities is None and np.any(body.ctrl_modes[q_idx] != struct.ControlMode.POSITION):
            # Switch to position control mode
            body.ctrl_modes[q_idx] = struct.ControlMode.POSITION

            body_p_indices = body.u_p_indices - body.u_idx0
            if kps is None:  # if gains were not provided, set back to the default ones
                self.model.actuator_gainprm[body.u_p_indices[q_idx], 0] = body.gains[body_p_indices[q_idx], 0]
                self.model.actuator_biasprm[body.u_p_indices[q_idx], 1] = body.biases[body_p_indices[q_idx], 1]
            self.model.actuator_gainprm[body.u_v_indices[q_idx]] = 0
            self.model.actuator_biasprm[body.u_v_indices[q_idx]] = 0
            self.model.actuator_gainprm[body.u_e_indices[q_idx]] = 0
            self.model.actuator_biasprm[body.u_e_indices[q_idx]] = 0
        elif velocities is not None and np.any(body.ctrl_mode[q_idx] != struct.ControlMode.PD):
            # Switch to PD control mode
            body.ctrl_modes[q_idx] = struct.ControlMode.PD

            body_p_indices = body.u_p_indices - body.u_idx0
            body_v_indices = body_p_indices + 1
            if kps is None:  # if gains were not provided, set back to the default ones
                self.model.actuator_gainprm[body.u_p_indices[q_idx], 0] = body.gains[body_p_indices[q_idx], 0]
                self.model.actuator_biasprm[body.u_p_indices[q_idx], 1] = body.biases[body_p_indices[q_idx], 1]
            if kds is None:  # if gains were not provided, set back to the default ones
                self.model.actuator_gainprm[body.u_v_indices[q_idx], 0] = body.gains[body_v_indices[q_idx], 0]
                self.model.actuator_biasprm[body.u_v_indices[q_idx], 2] = body.gains[body_v_indices[q_idx], 2]
            self.model.actuator_gainprm[body.u_e_indices[q_idx]] = 0
            self.model.actuator_biasprm[body.u_e_indices[q_idx]] = 0

        # set max forces
        if forces is None:  # TODO: improve this part
            body_p_indices = body.u_p_indices - body.u_idx0
            self.model.actuator_forcelimited[body.u_p_indices[q_idx]] = body.force_limited[body_p_indices[q_idx]]
            self.model.actuator_forcerange[body.u_p_indices[q_idx], 0] = -body.force_range[body_p_indices[q_idx], 0]
            self.model.actuator_forcerange[body.u_p_indices[q_idx], 1] = body.force_range[body_p_indices[q_idx], 1]
        else:
            self.model.actuator_forcelimited[body.u_p_indices[q_idx]] = 1
            self.model.actuator_forcerange[body.u_p_indices[q_idx], 0] = -forces
            self.model.actuator_forcerange[body.u_p_indices[q_idx], 1] = forces

        # set joint positions
        self.sim.data.ctrl[body.u_p_indices[q_idx]] = positions

        # set joint velocities if specified as well
        if velocities is not None:
            self.sim.data.ctrl[body.u_v_indices[q_idx]] = velocities

    def get_joint_positions(self, body_id, joint_ids=None):
        """
        Get the position of the given joint(s).

        Args:
            body_id (int): unique body id.
            joint_ids (int, list[int], None): joint id, or list of joint ids. If None, it will take all the actuated
              joints.

        Returns:
            if 1 joint:
                float: joint position [rad]
            if multiple joints:
                np.array[float[N]]: joint positions [rad]
        """
        body = self._bodies[body_id]

        if joint_ids is None:
            return self.sim.data.qpos[body.q_idx1:body.q_idxf]

        # check if valid joint
        self._check_joint_ids(body, joint_ids)

        # if one joint, return its position
        if isinstance(joint_ids, int):
            return self.sim.data.qpos[body.q_idx1 + joint_ids]

        # if multiple joints, return their positions
        q_idx = body.get_q_idx(joint_ids, keep=True)  # E.g. [0, -1, 1, -1, 2, 3]  (-1 are for fixed joints)
        qpos = np.zeros(len(joint_ids))
        qpos[q_idx != -1] = self.sim.data.qpos[body.q_idx1 + q_idx[q_idx != -1]]
        return qpos

    def set_joint_velocities(self, body_id, joint_ids, velocities, max_force=None):
        """
        Set the velocity of the given joint(s) (using velocity control).

        Args:
            body_id (int): unique body id.
            joint_ids (int, list[int]): joint id, or list of joint ids.
            velocities (float, np.array[float[N]]): desired velocity, or list of desired velocities [rad/s]
            max_force (None, float, np.array[float[N]]): maximum motor forces/torques
        """
        # body = self._bodies[body_id]
        # 
        # if joint_ids is None:
        #     self.sim.data.qvel[body.v_idx1:body.v_idxf] = velocities
        # else:
        #     # check if valid joints
        #     self._check_joint_ids(body, joint_ids)
        # 
        #     # if one joint, set its velocity
        #     if isinstance(joint_ids, int):
        #         self.sim.data.qvel[body.v_idx1 + joint_ids] = velocities
        # 
        #     # if multiple joints, set their velocities
        #     else:
        #         q = body.get_q_idx(joint_ids, keep=True)  # E.g. [0, -1, 1, -1, 2, 3]  (-1 are for fixed joints)
        #         self.sim.data.qvel[body.v_idx1 + q[q != -1]] = velocities

        # get body and q indices
        body = self._bodies[body_id]
        if joint_ids is None:
            q_idx = np.array(range(body.num_actuated_joints))
        else:
            # check if valid joints
            self._check_joint_ids(body, joint_ids)

            q_idx = body.get_q_idx(joint_ids, keep=True)  # E.g. [0, -1, 1, -1, 2, 3]  (-1 are for fixed joints)
            q_idx = q_idx[q_idx != -1]

        if len(q_idx) == 0:
            raise ValueError("No actuated joints to set the velocities to...")

        # check the current control mode, deactivate the other motors by setting their gains and biases to zero
        if np.any(body.ctrl_modes[q_idx] != struct.ControlMode.VELOCITY):
            # Switch to velocity control mode
            body.ctrl_modes[q_idx] = struct.ControlMode.VELOCITY
            body_v_indices = body.u_v_indices - body.u_idx0
            self.model.actuator_gainprm[body.u_p_indices[q_idx]] = 0
            self.model.actuator_biasprm[body.u_p_indices[q_idx]] = 0
            self.model.actuator_gainprm[body.u_v_indices[q_idx], 0] = body.gains[body_v_indices[q_idx], 0]
            self.model.actuator_biasprm[body.u_v_indices[q_idx], 2] = body.biases[body_v_indices[q_idx], 2]
            self.model.actuator_gainprm[body.u_e_indices[q_idx]] = 0
            self.model.actuator_biasprm[body.u_e_indices[q_idx]] = 0

        if max_force is not None:
            self.model.actuator_forcelimited[body.u_v_indices[q_idx]] = 1
            self.model.actuator_forcerange[body.u_v_indices[q_idx], 0] = -max_force
            self.model.actuator_forcerange[body.u_v_indices[q_idx], 1] = max_force
        # TODO: reset force range when None

        # set joint velocities
        self.sim.data.ctrl[body.u_v_indices[q_idx]] = velocities

    def get_joint_velocities(self, body_id, joint_ids=None):
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
        body = self._bodies[body_id]

        if joint_ids is None:
            return self.sim.data.qvel[body.v_idx1:body.v_idxf]

        # check if valid joint
        self._check_joint_ids(body, joint_ids)

        # if one joint, return its velocity
        if isinstance(joint_ids, int):
            return self.sim.data.qvel[body.v_idx1 + joint_ids]

        # if multiple joints, return their velocities
        q_idx = body.get_q_idx(joint_ids, keep=True)  # E.g. [0, -1, 1, -1, 2, 3]  (-1 are for fixed joints)
        qvel = np.zeros(len(joint_ids))
        qvel[q_idx != -1] = self.sim.data.qvel[body.v_idx1 + q_idx[q_idx != -1]]
        return qvel

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

    def get_joint_accelerations(self, body_id, joint_ids=None):  # , q=None, dq=None):
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
        body = self._bodies[body_id]

        if joint_ids is None:
            return self.sim.data.qacc[body.v_idx1:body.v_idxf]

        # check if valid joint
        self._check_joint_ids(body, joint_ids)

        # if one joint, return its velocity
        if isinstance(joint_ids, int):
            return self.sim.data.qacc[body.v_idx1 + joint_ids]

        # if multiple joints, return their velocities
        q_idx = body.get_q_idx(joint_ids, keep=True)  # E.g. [0, -1, 1, -1, 2, 3]  (-1 are for fixed joints)
        qacc = np.zeros(len(joint_ids))
        qacc[q_idx != -1] = self.sim.data.qacc[body.v_idx1 + q_idx[q_idx != -1]]
        return qacc

    def set_joint_torques(self, body_id, joint_ids, torques):
        """
        Set the torque/force to the given joint(s) (using force/torque control).

        Args:
            body_id (int): unique body id.
            joint_ids (int, list[int]): joint id, or list of joint ids.
            torques (float, list[float], np.array[float]): desired torque(s) to apply to the joint(s) [N].
        """
        # body = self._bodies[body_id]
        #
        # if joint_ids is None:
        #     self.sim.data.qfrc_applied[body.v_idx1:body.v_idxf] = torques
        # else:
        #     # check if valid joints
        #     self._check_joint_ids(body, joint_ids)
        #
        #     # if one joint, set its torque
        #     if isinstance(joint_ids, int):
        #         self.sim.data.qfrc_applied[body.v_idx1 + joint_ids] = torques
        #
        #     # if multiple joints, set their torques
        #     else:
        #         q = body.get_q_idx(joint_ids, keep=True)  # E.g. [0, -1, 1, -1, 2, 3]  (-1 are for fixed joints)
        #         self.sim.data.qfrc_applied[body.v_idx1 + q[q != -1]] = torques

        body = self._bodies[body_id]
        if joint_ids is None:
            q_idx = np.array(range(body.num_actuated_joints))
        else:
            # check if valid joints
            self._check_joint_ids(body, joint_ids)

            q_idx = body.get_q_idx(joint_ids, keep=True)  # E.g. [0, -1, 1, -1, 2, 3]  (-1 are for fixed joints)
            q_idx = q_idx[q_idx != -1]

        if len(q_idx) == 0:
            raise ValueError("No actuated joints to set the torques to...")

        # check the current control mode, deactivate the other motors by setting their gains and biases to zero
        if np.any(body.ctrl_modes != struct.ControlMode.EFFORT):
            # Switch to effort control mode
            body.ctrl_modes[q_idx] = struct.ControlMode.EFFORT
            body_e_indices = body.u_e_indices - body.u_idx0
            self.model.actuator_gainprm[body.u_p_indices[q_idx]] = 0
            self.model.actuator_biasprm[body.u_p_indices[q_idx]] = 0
            self.model.actuator_gainprm[body.u_v_indices[q_idx]] = 0
            self.model.actuator_biasprm[body.u_v_indices[q_idx]] = 0
            self.model.actuator_gainprm[body.u_e_indices[q_idx], 0] = body.gains[body_e_indices[q_idx], 0]
            self.model.actuator_biasprm[body.u_e_indices[q_idx]] = 0

            # reset to original control and force ranges
            self.model.actuator_ctrllimited[body.u_e_indices[q_idx]] = body.ctrl_limited[body_e_indices[q_idx]]
            self.model.actuator_ctrlrange[body.u_e_indices[q_idx], 0] = body.ctrl_range[body_e_indices[q_idx], 0]
            self.model.actuator_ctrlrange[body.u_e_indices[q_idx], 1] = body.ctrl_range[body_e_indices[q_idx], 1]
            self.model.actuator_forcelimited[body.u_e_indices[q_idx]] = body.force_limited[body_e_indices[q_idx]]
            self.model.actuator_forcerange[body.u_e_indices[q_idx], 0] = body.force_range[body_e_indices[q_idx], 0]
            self.model.actuator_forcerange[body.u_e_indices[q_idx], 1] = body.force_range[body_e_indices[q_idx], 1]

        # set joint efforts (torques/forces)
        self.sim.data.ctrl[body.u_e_indices[q_idx]] = torques

    def get_joint_torques(self, body_id, joint_ids=None):
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
        body = self._bodies[body_id]

        if joint_ids is None:
            return self.sim.data.qfrc_applied[body.v_idx1:body.v_idxf]

        # check if valid joint
        self._check_joint_ids(body, joint_ids)

        # if one joint, return its velocity
        if isinstance(joint_ids, int):
            return self.sim.data.qfrc_applied[body.v_idx1 + joint_ids]

        # if multiple joints, return their velocities
        q_idx = body.get_q_idx(joint_ids, keep=True)  # E.g. [0, -1, 1, -1, 2, 3]  (-1 are for fixed joints)
        torques = np.zeros(len(joint_ids))
        torques[q_idx != -1] = self.sim.data.qfrc_applied[body.v_idx1 + q_idx[q_idx != -1]]
        return torques

    def get_joint_reaction_forces(self, body_id, joint_ids=None):
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
        body = self._bodies[body_id]

        if joint_ids is None:
            # com-based interaction force with parent [torque, force]
            force_parent = self.sim.data.cfrc_int[body.b_idx0:body.b_idxf]
            # com-based external force on body [torque, force]
            force_ext = self.sim.data.cfrc_ext[body.b_idx0:body.b_idxf]
            force = force_ext - force_parent  # TODO: is it + instead of -?
            np.roll(force, shift=3, axis=force.ndim - 1)  # [torque, force] --> [force, torque]
            # TODO: express that force in the joint frame
            return force

        # check if valid link id
        self._check_link_ids(body, joint_ids)

        # com-based interaction force with parent [torque, force]
        force_parent = self.sim.data.cfrc_int[body.b_idx0 + joint_ids]
        # com-based external force on body [torque, force]
        force_ext = self.sim.data.cfrc_ext[body.b_idx0 + joint_ids]

        force = force_ext + force_parent  # TODO: is it - instead of +?

        np.roll(force, shift=3, axis=force.ndim-1)  # [torque, force] --> [force, torque]

        # TODO: express that force in the joint frame
        return force

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
        torque = self.get_joint_torques(body_id, joint_ids)
        velocity = self.get_joint_velocities(body_id, joint_ids)
        return torque * velocity

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

    def compute_view_matrix(self, eye_position, target_position, up_vector):
        """Compute the view matrix.

        The view matrix is the 4x4 matrix that maps the world coordinates into the camera coordinates. Basically,
        it applies a rotation and translation such that the world is in front of the camera. That is, instead
        of turning the camera to capture what we want in the world, we keep the camera fixed and turn the world.

        Args:
            eye_position (np.array[float[3]]): eye position in Cartesian world coordinates
            target_position (np.array[float[3]]): position of the target (focus) point in Cartesian world coordinates
            up_vector (np.array[float[3]]): up vector of the camera in Cartesian world coordinates

        Returns:
            np.array[float[4,4]]: the view matrix
        """
        pass

    def compute_view_matrix_from_ypr(self, target_position, distance, yaw, pitch, roll, up_axis_index=2):
        """Compute the view matrix from the yaw, pitch, and roll angles.

        The view matrix is the 4x4 matrix that maps the world coordinates into the camera coordinates. Basically,
        it applies a rotation and translation such that the world is in front of the camera. That is, instead
        of turning the camera to capture what we want in the world, we keep the camera fixed and turn the world.

        Args:
            target_position (np.array[float[3]]): target focus point in Cartesian world coordinates
            distance (float): distance from eye to focus point
            yaw (float): yaw angle in radians left/right around up-axis
            pitch (float): pitch in radians up/down.
            roll (float): roll in radians around forward vector
            up_axis_index (int): either 1 for Y or 2 for Z axis up.

        Returns:
            np.array[float[4,4]]: the view matrix
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
            np.array[float[4,4]]: the perspective projection matrix
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
            np.array[float[4,4]]: the perspective projection matrix
        """
        pass

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
        camera_name = "prl_world_camera"

        # based on camera

        # mjvCamera

        rgb, depth = self.sim.render(width, height, camera_name, depth=True)
        segmentation = -np.ones(width, height)
        rgba = np.dstack((rgb, -segmentation))
        return width, height, rgba, depth, segmentation

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

    def get_overlapping_objects(self, aabb_min, aabb_max):
        """
        This query will return all the unique ids of objects that have Axis Aligned Bounding Box (AABB) overlap with
        a given axis aligned bounding box. Note that the query is conservative and may return additional objects that
        don't have actual AABB overlap. This happens because the acceleration structures have some heuristic that
        enlarges the AABBs a bit (extra margin and extruded along the velocity vector).

        Args:
            aabb_min (np.array[float[3]]): minimum coordinates of the aabb
            aabb_max (np.array[float[3]]): maximum coordinates of the aabb

        Returns:
            list[int]: list of object unique ids.
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
            np.array[float[3]]: minimum coordinates of the axis aligned bounding box
            np.array[float[3]]: maximum coordinates of the axis aligned bounding box
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
                [0] int: contact flag (reserved)
                [1] int: body unique id of body A
                [2] int: body unique id of body B
                [3] int: link index of body A, -1 for base
                [4] int: link index of body B, -1 for base
                [5] np.array[float[3]]: contact position on A, in Cartesian world coordinates
                [6] np.array[float[3]]: contact position on B, in Cartesian world coordinates
                [7] np.array[float[3]]: contact normal on B, pointing towards A
                [8] float: contact distance, positive for separation, negative for penetration
                [9] float: normal force applied during the last `step`
                [10] float: lateral friction force in the first lateral friction direction (see next returned value)
                [11] np.array[float[3]]: first lateral friction direction
                [12] float: lateral friction force in the second lateral friction direction (see next returned value)
                [13] np.array[float[3]]: second lateral friction direction
        """
        # mjContact
        # mj_contactForce
        # check sim.data.contact = list of all detected contact
        for contact in sim.data.contact:
            geom_id1 = contact.geom1
            geom_id2 = contact.geom2

            body_id1 = self.model.geom_bodyid[geom_id1]
            body_id2 = self.model.geom_bodyid[geom_id2]

            midpos = contact.pos
            dist = contact.dist

            frame = contact.frame
            normal = frame[:3]

            frictions = contact.friction
            lateral_1 = frictions[0]
            lateral_2 = frictions[1]
            spin = frictions[2]
            roll_1 = frictions[3]
            roll_2 = frictions[4]
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
                np.array[float[3]]: contact position on A, in Cartesian world coordinates
                np.array[float[3]]: contact position on B, in Cartesian world coordinates
                np.array[float[3]]: contact normal on B, pointing towards A
                float: contact distance, positive for separation, negative for penetration
                float: normal force applied during the last `step`. Always equal to 0.
                float: lateral friction force in the first lateral friction direction (see next returned value)
                np.array[float[3]]: first lateral friction direction
                float: lateral friction force in the second lateral friction direction (see next returned value)
                np.array[float[3]]: second lateral friction direction
        """
        pass

    def ray_test(self, from_position, to_position):
        """
        Performs a single raycast to find the intersection information of the first object hit.

        Args:
            from_position (np.array[float[3]]): start of the ray in world coordinates
            to_position (np.array[float[3]]): end of the ray in world coordinates

        Returns:
            [0] int: object unique id of the hit object
            [1] int: link index of the hit object, or -1 if none/parent
            [2] float: hit fraction along the ray in range [0,1] along the ray.
            [3] np.array[float[3]]: hit position in Cartesian world coordinates
            [4] np.array[float[3]]: hit normal in Cartesian world coordinates
        """
        vec = to_position - from_position
        distance, geom_id = self.sim.ray(pnt=from_position, vec=vec)  # this return the distance and id of the geom
        norm = np.linalg.norm(vec)
        fraction = distance / norm
        unit_vec = vec / norm
        position = distance * unit_vec
        normal = -unit_vec  # TODO: this is not correct...
        # TODO: get body_id and link_id from geom_id
        return geom_id, geom_id, fraction, position, normal

    def ray_test_batch(self, from_positions, to_positions, parent_object_id=None, parent_link_id=None):
        """Perform a batch of raycasts to find the intersection information of the first objects hit.

        This is similar to the ray_test, but allows you to provide an array of rays, for faster execution. The size of
        'rayFromPositions' needs to be equal to the size of 'rayToPositions'. You can one ray result per ray, even if
        there is no intersection: you need to use the objectUniqueId field to check if the ray has hit anything: if
        the objectUniqueId is -1, there is no hit. In that case, the 'hit fraction' is 1.

        Args:
            from_positions (np.array[float[N,3]]): list of start points for each ray, in world coordinates
            to_positions (np.array[float[N,3]]): list of end points for each ray in world coordinates
            parent_object_id (int): ray from/to is in local space of a parent object
            parent_link_id (int): ray from/to is in local space of a parent object

        Returns:
            list:
                int: object unique id of the hit object
                int: link index of the hit object, or -1 if none/parent
                float: hit fraction along the ray in range [0,1] along the ray.
                np.array[float[3]]: hit position in Cartesian world coordinates
                np.array[float[3]]: hit normal in Cartesian world coordinates
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
        body = self._bodies[body_id]
        self._check_link_id(body, link_id)

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

    def calculate_jacobian(self, body_id, link_id, local_position=None, q=None, dq=None, des_ddq=None):
        r"""
        Return the full geometric Jacobian matrix :math:`J(q) = [J_{lin}(q), J_{ang}(q)]^T`, such that:

        .. math:: v = [\dot{p}, \omega]^T = J(q) \dot{q}

        where :math:`\dot{p}` is the Cartesian linear velocity of the link, and :math:`\omega` is its angular velocity.

        Warnings: if we have a floating base then the Jacobian will also include columns corresponding to the root
            link DoFs (at the beginning). If it is a fixed base, it will only have columns associated with the joints.

        Args:
            body_id (int): unique body id.
            link_id (int): link id.
            local_position (np.array[float[3]], None): the point on the specified link to compute the Jacobian (in
              link local coordinates around its center of mass). If None, it will use the CoM position (in the link
              frame).
            q (np.array[float[N]]): joint positions of size N, where N is the number of DoFs.
            dq (np.array[float[N]]): joint velocities of size N, where N is the number of DoFs.
            des_ddq (np.array[float[N]]): desired joint accelerations of size N.

        Returns:
            np.array[float[6,N]], np.array[float[6,6+N]]: full geometric (linear and angular) Jacobian matrix. The
                number of columns depends if the base is fixed or floating.
        """
        body = self._bodies[body_id]
        link_id = self._check_link_id(body, link_id)
        idx = body.b_idx0 + link_id

        # TODO: use q, dq, des_ddq by setting it in the data and then restoring the data
        jacp, jacr = np.zeros(3 * self.model.nv), np.zeros(3 * self.model.nv)
        local_position = self.get_link_world_positions(body_id, link_ids=link_id-1)  # in the cartesian world frame
        # TODO: modify the local position
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
        # dest = mujoco.cymj.PyMjData()
        # mujoco.functions.mj_copyData(dest, self.model, self.sim.data)
        # dest.qpos[body.q_idx0:body.q_idxf] = q
        # dest.qvel[body.v_idx0:body.v_idxf] = dq
        # dest.qacc[body.v_idx0:body.v_idxf] = des_ddq

        data = self._save_state()
        self.sim.data.qpos[body.q_idx0:body.q_idxf] = q
        self.sim.data.qvel[body.v_idx0:body.v_idxf] = dq
        self.sim.data.qacc[body.v_idx0:body.v_idxf] = des_ddq

        # inverse dynamics
        mujoco.functions.mj_inverse(self.model, self.sim.data)

        # get resulting torques and return it
        # torques = self.sim.data.qfrc_applied[body.v_idx0:body.v_idxf]
        torques = self.sim.data.qfrc_inverse[body.v_idx0:body.v_idxf]

        # restore data
        self._load_state(data)
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
        # dest = mujoco.cymj.PyMjData()
        # mujoco.functions.mj_copyData(dest, self.model, self.sim.data)
        # dest.qpos[body.q_idx0:body.q_idxf] = q
        # dest.qvel[body.v_idx0:body.v_idxf] = dq
        # dest.qfrc_applied[body.v_idx0:body.v_idxf] = torques

        data = self._save_state()
        self.sim.data.qpos[body.q_idx0:body.q_idxf] = q
        self.sim.data.qvel[body.v_idx0:body.v_idxf] = dq
        self.sim.data.qfrc_applied[body.v_idx0:body.v_idxf] = torques

        # forward dynamics
        mujoco.functions.mj_forward(self.model, self.sim.data)

        # get ddq and return it
        qacc = self.sim.data.qacc[body.v_idx0:body.v_idxf]

        # restore data
        self._load_state(data)
        return qacc

    #########
    # Debug #
    #########

    def add_user_debug_line(self, from_pos, to_pos, rgb_color=None, width=None, lifetime=None,
                            parent_object_id=None,
                            parent_link_id=None, line_id=None):
        """Add a user debug line in the simulator.

        You can add a 3d line specified by a 3d starting point (from) and end point (to), a color [red,green,blue],
        a line width and a duration in seconds.

        Args:
            from_pos (np.array[float[3]]): starting point of the line in Cartesian world coordinates
            to_pos (np.array[float[3]]): end point of the line in Cartesian world coordinates
            rgb_color (np.array[float[3]]): RGB color (each channel in range [0,1])
            width (float): line width (limited by OpenGL implementation).
            lifetime (float): use 0 for permanent line, or positive time in seconds (afterwards the line with be
                removed automatically)
            parent_object_id (int): draw line in local coordinates of a parent object.
            parent_link_id (int): draw line in local coordinates of a parent link.
            line_id (int): replace an existing line item (to avoid flickering of remove/add).

        Returns:
            int: unique user debug line id.
        """
        # mjr_drawPixels
        pass

    def add_user_debug_text(self, text, position, rgb_color=None, size=None, lifetime=None, orientation=None,
                            parent_object_id=None, parent_link_id=None, text_id=None):
        """
        Add 3D text at a specific location using a color and size.

        Args:
            text (str): text.
            position (np.array[float[3]]): 3d position of the text in Cartesian world coordinates.
            rgb_color (list/tuple of 3 floats): RGB color; each component in range [0..1]
            size (float): text size
            lifetime (float): use 0 for permanent text, or positive time in seconds (afterwards the text with be
                removed automatically)
            orientation (np.array[float[4]]): By default, debug text will always face the camera, automatically
                rotation. By specifying a text orientation (quaternion), the orientation will be fixed in world space
                or local space (when parent is specified). Note that a different implementation/shader is used for
                camera facing text, with different appearance: camera facing text uses bitmap fonts, text with
                specified orientation uses TrueType font.
            parent_object_id (int): draw text in local coordinates of a parent object.
            parent_link_id (int): draw text in local coordinates of a parent link.
            text_id (int): replace an existing text item (to avoid flickering of remove/add).

        Returns:
            int: unique user debug text id.
        """
        # mjr_text
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
            np.array[float[4,4]],4]: view matrix [4,4]
            np.array[float[4,4]],4]: perspective projection matrix [4,4]
            np.array[float[3]]: camera up vector expressed in the Cartesian world space
            np.array[float[3]]: forward axis of the camera expressed in the Cartesian world space
            np.array[float[3]]: This is a horizontal vector that can be used to generate rays (for mouse picking or
                creating a simple ray tracer for example)
            np.array[float[3]]: This is a vertical vector that can be used to generate rays (for mouse picking or
                creating a simple ray tracer for example)
            float: yaw angle (in radians) of the camera, in Cartesian local space coordinates
            float: pitch angle (in radians) of the camera, in Cartesian local space coordinates
            float: distance between the camera and the camera target
            np.array[float[3]]: target of the camera, in Cartesian world space coordinates
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
            target_position (np.array[float[3]]): target focus point of the camera
        """
        pass

    ############################
    # Events (mouse, keyboard) #
    ############################

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
