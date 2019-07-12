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

Dependencies in PRL:
* `pyrobolearn.simulators.simulator.Simulator`

References:
    - [1] MuJoCo: http://www.mujoco.org/
        - Documentation: http://mujoco.org/book
    - [2] MuJoCo Python: https://github.com/openai/mujoco-py
    - [3] DeepMind Control Suite: https://github.com/deepmind/dm_control/tree/master/dm_control/mujoco
"""

import os
import time
import pickle
import xml.etree.ElementTree as ET

try:
    import mujoco_py as mujoco
except ImportError as e:
    raise ImportError(str(e) + "\nTry to install `MuJoCo` and `mujoco_py`!")

# from dm_control import mujoco

from pyrobolearn.simulators.simulator import Simulator

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

    References:
        - [1] MuJoCo: http://www.mujoco.org/
        - [2] MuJoCo Python: https://github.com/openai/mujoco-py
        - [3] DeepMind Control Suite: https://github.com/deepmind/dm_control/tree/master/dm_control/mujoco
    """

    def __init__(self, render=True, load_at_the_end=False):
        """
        Initialize the MuJoCo simulator.

        Args:
            render (bool): if True, it will open the GUI, otherwise, it will just run the server (i.e. in a headless
                mode, i.e. without a GUI).
            load_at_the_end (bool): if True, it will load at the end all the models that have been "loaded" in the
                simulator. The reason is that the MuJoCo simulator does not allow to load dynamically models into the
                world.
        """
        super(Mujoco, self).__init__(render=render)

        # define variables
        self.load_at_the_end = load_at_the_end
        self.viewer = None

        # create empty world
        xml_path = os.path.dirname(os.path.abspath(__file__)) + '/mujoco_empty_world.xml'
        model = mujoco.load_model_from_path(xml_path)
        self.model = model
        self.sim = mujoco.MjSim(model)

        # parse the world
        self.world = self._parse()

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

    @property
    def dt(self):
        """Return the simulator time step."""
        return self.sim.model.opt.timestep

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

    ###########
    # Methods #
    ###########

    ##############
    # Simulators #
    ##############

    def _parse(self, xml_path):
        """
        Parse the provided XML file.

        Args:
            xml_path (str): path to the MuJoCo xml file.

        Returns:
            xml.etree.ElementTree.Element: root element in the XML file.
        """
        root = ET.parse(xml_path).getroot()
        return root

    def reset(self):
        """Reset the simulator.

        Resets the simulation data and clears buffers.
        """
        self.sim.reset()

    def step(self, sleep_time=0.):
        """Perform a step in the simulator, and sleep the specified amount of time.

        Args:
            sleep_time (float): amount of time to sleep after performing one step in the simulation.
        """
        self.sim.forward()  # computes forward kinematics
        self.sim.step()     # advance the simulation
        if self.is_rendering():
            if self.viewer is None:
                self.viewer = mujoco.MjViewer(self.sim)
            self.viewer.render()
        time.sleep(sleep_time)

    def render(self, enable=True):
        """Render the simulation.

        Args:
            enable (bool): If True, it will render the simulator by enabling the GUI.
        """
        self._render = enable
        if self.viewer is None:
            self.viewer = mujoco.MjViewer(self.sim)

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
            gravity (list, tuple of 3 floats): acceleration in the x, y, z directions.
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
    # loading URDFs, SDFs, MJCFs, meshes #
    ######################################

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
        # create xml file based on URDF file
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
        # update the world

        # load MJCF
        self.model = mujoco.load_model_from_path(filename)
        self.sim = mujoco.MjSim(self.model)

    #################
    # visualization #
    #################

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
            view_matrix (np.array[4,4]): 4x4 view matrix, see `compute_view_matrix`
            projection_matrix (np.array[4,4]): 4x4 projection matrix, see `compute_projection`
            light_direction (np.array[3]): `light_direction` specifies the world position of the light source,
                the direction is from the light source position to the origin of the world frame.
            light_color (np.array[3]): directional light color in [RED,GREEN,BLUE] in range 0..1
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
            np.array[width, heigth]: Depth buffer.
            np.int[width, height]: Segmentation mask buffer. For each pixels the visible object unique id.
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
            view_matrix (np.array[4,4]): 4x4 view matrix, see `compute_view_matrix`
            projection_matrix (np.array[4,4]): 4x4 projection matrix, see `compute_projection`
            light_direction (np.array[3]): `light_direction` specifies the world position of the light source,
                the direction is from the light source position to the origin of the world frame.
            light_color (np.array[3]): directional light color in [RED,GREEN,BLUE] in range 0..1
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
            view_matrix (np.array[4,4]): 4x4 view matrix, see `compute_view_matrix`
            projection_matrix (np.array[4,4]): 4x4 projection matrix, see `compute_projection`
            light_direction (np.array[3]): `light_direction` specifies the world position of the light source,
                the direction is from the light source position to the origin of the world frame.
            light_color (np.array[3]): directional light color in [RED,GREEN,BLUE] in range 0..1
            light_distance (float): distance of the light along the normalized `light_direction`
            shadow (bool): True for shadows, False for no shadows
            light_ambient_coeff (float): light ambient coefficient
            light_diffuse_coeff (float): light diffuse coefficient
            light_specular_coeff (float): light specular coefficient
            renderer (int): renderer.
            flags (int): flags.

        Returns:
            np.array[width, height]: Depth buffer.
        """
        # based on the arguments, check the camera name
        camera_name = None

        # return the depth image
        rgb, depth = self.sim.render(width, height, camera_name, depth=True)
        return depth

    ##############
    # Collisions #
    ##############

    def ray_test(self, from_position, to_position):
        """
        Performs a single raycast to find the intersection information of the first object hit.

        Args:
            from_position (np.array[3]): start of the ray in world coordinates
            to_position (np.array[3]): end of the ray in world coordinates

        Returns:
            list:
                int: object unique id of the hit object
                int: link index of the hit object, or -1 if none/parent
                float: hit fraction along the ray in range [0,1] along the ray.
                np.array[3]: hit position in Cartesian world coordinates
                np.array[3]: hit normal in Cartesian world coordinates
        """
        vec = to_position - from_position
        return self.sim.ray(pnt=from_position, vec=vec)  # this return the distance and id of the geom


# Test
if __name__ == '__main__':
    from itertools import count

    sim = Mujoco(render=True)

    for t in count():
        sim.step(sim.dt)
