#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Define the V-REP Simulator API (using PyRep).

This is the main interface that communicates with the V-REP simulator [1] through its Python wrapper PyRep [2].
By defining this interface, it allows to decouple the PyRoboLearn framework from the simulator. It also converts some
data types to the ones required by PyRep.

Warnings:
    - You have to install V-REP beforehand.
    - This only works with Python 3

- Supported Python versions: Python 3.*
- Python wrappers: cffi [3]

Dependencies in PRL:
* `pyrobolearn.simulators.simulator.Simulator`

References:
    - [1] V-REP: http://www.coppeliarobotics.com/
    - [2] PyRep: https://github.com/stepjam/PyRep
    - [3] CFFI: https://cffi.readthedocs.io/en/latest/
        - Cython, pybind11, cffi – which tool should you choose?:
          http://blog.behnel.de/posts/cython-pybind11-cffi-which-tool-to-choose.html
"""

# TODO: finish to implement this interface (use the pyrep / vrep interface)

import os
import time
import numpy as np

try:
    import pyrep
    from pyrep.objects.shape import Shape, PrimitiveShape
    from pyrep.backend import vrep      # this backend contains all the interesting methods
except ImportError as e:
    raise ImportError("PyRep seems to not be installed on this machine; try to follow the installation instructions "
                      "in: \nhttps://github.com/stepjam/PyRep\n" + "Original error: " + str(e))
except SyntaxError as e:
    raise SyntaxError("PyRep only works with Python 3!! \n" + "Original error: " + str(e))


from pyrobolearn.simulators.simulator import Simulator
from pyrobolearn.utils.transformation import get_rpy_from_quaternion


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2019, PyRoboLearn"
__credits__ = ["V-REP (Coppelia Robotics)", "PyRep (James et al.)", "Brian Delhaisse (PyRoboLearn interface)"]
__license__ = "Apache License 2.0"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class VREP(Simulator):
    r"""V-REP Simulator interface (using PyRep)

    This is the main interface that communicates with the V-REP simulator [1] through its Python wrapper PyRep [2].
    By defining this interface, it allows to decouple the PyRoboLearn framework from the simulator. It also converts
    some data types to the ones required by PyRep.

    Warnings:
        - You have to install V-REP beforehand.
        - This only works with Python 3

    References:
        - [1] V-REP: http://www.coppeliarobotics.com/
        - [2] PyRep: https://github.com/stepjam/PyRep
    """

    def __init__(self, render=True, num_instances=1, middleware=None, **kwargs):
        """
        Initialize the VREP simulator.

        Args:
            render (bool): if True, it will open the GUI, otherwise, it will just run the server.
            num_instances (int): number of simulator instances.
            middleware (MiddleWare, None): middleware instance.
            **kwargs (dict): optional arguments (this is not used here).
        """
        super(VREP, self).__init__(render=render, num_instances=num_instances, middleware=middleware)

        # create simulator
        self.sim = pyrep.PyRep()

        # launch simulator
        scene = "scene.ttt"  # ""
        self.sim.launch(scene_file=scene, headless=not render)
        self.sim.start()

        # visual and collision shape ids
        self.visual_shapes = dict()
        self.collision_shapes = dict()
        self._bodies = dict()

        # primitive shape mapping
        self.primitive_shape_map = {self.GEOM_BOX: PrimitiveShape.CUBOID, self.GEOM_SPHERE: PrimitiveShape.SPHERE,
                                    self.GEOM_CYLINDER: PrimitiveShape.CYLINDER}  # self.GEOM_CONE: PrimitiveShape.CONE}

        # keep track of the number of ids
        self._body_cnt = 0  # 0 is for the world

    ###########
    # Methods #
    ###########

    ##############
    # Simulators #
    ##############

    def close(self):
        """Close the simulator."""
        self.sim.stop()      # stop the simulation
        self.sim.shutdown()  # close the application

    def step(self, sleep_time=0):
        """Perform a step in the simulator, and sleep the specified amount of time.

        Args:
            sleep_time (float): amount of time to sleep after performing one step in the simulation.
        """
        self.sim.step()
        time.sleep(sleep_time)

    def get_time_step(self):
        """Get the time step in the simulator.

        Returns:
            float: time step in the simulator
        """
        return vrep.simGetFloatParameter(vrep.sim_floatparam_simulation_time_step)

    def set_time_step(self, time_step):
        """Set the time step in the simulator.

        Args:
            time_step (float): Each time you call 'step' the time step will proceed with 'time_step'.
        """
        # vrep.simSetFloatParameter(vrep.sim_floatparam_simulation_time_step, dt)
        self.sim.set_simulation_timestep(time_step)

    def get_gravity(self):
        """Return the gravity set in the simulator."""
        # TODO
        # return vrep.simGetGravity()
        pass

    def set_gravity(self, gravity=(0, 0, -9.81)):
        """Set the gravity in the simulator with the given acceleration.

        Args:
            gravity (list, tuple of 3 floats): acceleration in the x, y, z directions.
        """
        # TODO
        pass

    ######################################
    # loading URDFs, SDFs, MJCFs, meshes #
    ######################################

    def load_urdf(self, filename, position=None, orientation=(0., 0., 0., 1.), use_fixed_base=0, scale=1.0, *args,
                  **kwargs):
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
        # check given parameters
        color = rgba_color[:3] if rgba_color is not None else None
        orientation = get_rpy_from_quaternion(orientation).tolist() if orientation is not None else None
        static = True if mass == 0 else False
        mass = mass if mass <= 0 else mass
        position = list(position)

        # shape
        if shape_type == self.GEOM_BOX:
            size = [2*half_extents[0], 2*half_extents[1], 2*half_extents[2]]
            shape = Shape.create(PrimitiveShape.CUBOID, size=size, mass=mass, color=color, position=position,
                                 orientation=orientation, static=static)
        elif shape_type == self.GEOM_SPHERE:
            shape = Shape.create(PrimitiveShape.SPHERE, size=[1.], mass=mass, color=[1, 0, 0], position=position,
                                 orientation=orientation, static=static)
        elif shape_type == self.GEOM_CYLINDER:
            shape = Shape.create(PrimitiveShape.CYLINDER, size=[radius, height], mass=mass, color=color,
                                 position=position, orientation=orientation, static=static)
        elif shape_type == self.GEOM_CONE:
            shape = Shape.create(PrimitiveShape.CONE, size=[radius, height], mass=color, position=position, 
                                 orientation=orientation, static=static)
        # elif shape_type == self.GEOM_MESH:
        #     # TODO: use trimesh library
        #     shape = Shape.create_mesh()
        else:
            raise NotImplementedError("Primitive object not defined for the given shape type.")

        # save shape and return unique id
        self._body_cnt += 1
        self._bodies[self._body_cnt] = shape
        return self._body_cnt

    def create_body(self, visual_shape_id=-1, collision_shape_id=-1, mass=0., position=(0., 0., 0.),
                    orientation=(0., 0., 0., 1.), *args, **kwargs):
        """Create a body."""

        # TODO: finish this

        if visual_shape_id in self.visual_shapes:
            shape = self.visual_shapes[visual_shape_id]
            shape = Shape.create(type=shape['type'], size=shape['size'], color=shape['color'])
            shape.set_detectable()
            shape.set_position(position)
            shape.set_quaternion(orientation)
            if collision_shape_id in self.collision_shapes:
                if mass == 0:
                    shape.set_respondable(False)
                else:
                    shape.set_mass(mass)
                shape.set_collidable(True)

            # add body
            self._body_cnt += 1
            self._bodies[self._body_cnt] = shape
            return self._body_cnt

        raise ValueError("Visual shape id not known...")

    def remove_body(self, body_id):
        """Remove a particular body in the simulator.

        Args:
            body_id (int): unique body id.
        """
        if body_id in self._bodies:
            self._bodies[body_id].remove()

    def num_bodies(self):
        """Return the number of bodies present in the simulator.

        Returns:
            int: number of bodies
        """
        return len(self._bodies)

    #################
    # Visualization #
    #################

    def create_visual_shape(self, shape_type, radius=0.5, half_extents=(1., 1., 1.), length=1., filename=None,
                            mesh_scale=(1., 1., 1.), plane_normal=(0., 0., 1.), flags=-1, rgba_color=None,
                            specular_color=None, visual_frame_position=None, vertices=None, indices=None, uvs=None,
                            normals=None, visual_frame_orientation=None):
        """Create a visual shape in the simulator."""

        if shape_type not in self.primitive_shape_map:
            raise NotImplementedError("The specified shape is currently not supported in this simulator")

        # compute size
        size = (0, 0, 0)
        if shape_type == self.GEOM_BOX:  # if box
            size = (half_extents[0]*2, half_extents[1]*2, half_extents[2]*2)
        elif shape_type == self.GEOM_SPHERE:
            size = (radius, radius, radius)
        elif shape_type == self.GEOM_CYLINDER:
            size = (radius, radius, length)

        # check color
        if rgba_color is not None:
            rgba_color = rgba_color[:3]

        # save the visual shape
        self._body_cnt += 1
        self.visual_shapes[self._body_cnt] = {'type': self.primitive_shape_map[shape_type], 'size': size,
                                             'color': rgba_color}
        return self._body_cnt

    ##############
    # Collisions #
    ##############

    def create_collision_shape(self, shape_type, radius=0.5, half_extents=(1., 1., 1.), height=1., filename=None,
                               mesh_scale=(1., 1., 1.), plane_normal=(0., 0., 1.), flags=-1,
                               collision_frame_position=None, collision_frame_orientation=None):
        """Create a collision shape in the simulator."""

        if shape_type not in self.primitive_shape_map:
            raise NotImplementedError("The specified shape is currently not supported in this simulator")

        self._body_cnt += 1
        self.collision_shapes[self._body_cnt] = self.primitive_shape_map[shape_type]
        return self._body_cnt


# Tests
if __name__ == '__main__':

    # create simulator
    sim = VREP(render=True)

    # create sphere like in pybullet
    # visual = sim.create_visual_shape(sim.GEOM_SPHERE)
    # collision = sim.create_collision_shape(sim.GEOM_SPHERE)
    # body = sim.create_body(visual_shape_id=visual, collision_shape_id=collision, mass=1, position=(0., 0., 0.),
    #                        orientation=(0., 0., 0., 1.))

    box = sim.create_primitive_object(sim.GEOM_BOX, position=(0, 0, 2), mass=1, rgba_color=(1, 0, 0, 1))

    for t in range(1000):
        sim.step()
