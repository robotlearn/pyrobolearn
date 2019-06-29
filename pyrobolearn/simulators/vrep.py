#!/usr/bin/env python
"""Define the V-REP Simulator API (using PyRep).

This is the main interface that communicates with the V-REP simulator [1] through its Python wrapper PyRep [2].
By defining this interface, it allows to decouple the PyRoboLearn framework from the simulator. It also converts some
data types to the ones required by PyRep.

Warnings:
    - You have to install V-REP beforehand.
    - This only works with Python 3

Dependencies in PRL:
* `pyrobolearn.simulators.simulator.Simulator`

References:
    - [1] V-REP: http://www.coppeliarobotics.com/
    - [2] PyRep: https://github.com/stepjam/PyRep
"""

# TODO: finish to implement this interface (use the pyrep / vrep interface)

import time

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


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["V-REP (Coppelia Robotics)", "PyRep (James et al.)", "Brian Delhaisse"]
__license__ = "GNU GPLv3"
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

    def __init__(self, render=True):
        super(VREP, self).__init__(render=render)

        # create simulator
        self.sim = pyrep.PyRep()

        # launch simulator
        scene = ""  # "scene.ttt"
        self.sim.launch(scene_file=scene, headless=self._render)

        # visual and collision shape ids
        self.visual_shapes = dict()
        self.collision_shapes = dict()
        self.bodies = dict()

        # keep track of the number of ids
        self.count_id = 0

        # primitive shape mapping
        self.primitive_shape_map = {self.GEOM_BOX: PrimitiveShape.CUBOID, self.GEOM_SPHERE: PrimitiveShape.SPHERE,
                                    self.GEOM_CYLINDER: PrimitiveShape.CYLINDER}  # self.GEOM_CONE: PrimitiveShape.CONE}

        raise NotImplementedError

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
        self.count_id += 1
        self.visual_shapes[self.count_id] = {'type': self.primitive_shape_map[shape_type], 'size': size,
                                             'color': rgba_color}
        return self.count_id

    def create_collision_shape(self, shape_type, radius=0.5, half_extents=(1., 1., 1.), height=1., filename=None,
                               mesh_scale=(1., 1., 1.), plane_normal=(0., 0., 1.), flags=-1,
                               collision_frame_position=None, collision_frame_orientation=None):
        """Create a collision shape in the simulator."""

        if shape_type not in self.primitive_shape_map:
            raise NotImplementedError("The specified shape is currently not supported in this simulator")

        self.count_id += 1
        self.collision_shapes[self.count_id] = self.primitive_shape_map[shape_type]
        return self.count_id

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
            self.count_id += 1
            self.bodies[self.count_id] = shape
            return self.count_id

        raise ValueError("Visual shape id not known...")

    def remove_body(self, body_id):
        """Remove a particular body in the simulator.

        Args:
            body_id (int): unique body id.
        """
        if body_id in self.bodies:
            self.bodies[body_id].remove()

    def num_bodies(self):
        """Return the number of bodies present in the simulator.

        Returns:
            int: number of bodies
        """
        return len(self.bodies)


# Tests
if __name__ == '__main__':

    # create simulator
    sim = VREP()

    # create sphere like in pybullet
    visual = sim.create_visual_shape(sim.GEOM_SPHERE)
    collision = sim.create_collision_shape(sim.GEOM_SPHERE)
    body = sim.create_body(visual_shape_id=visual, collision_shape_id=collision, mass=1, position=(0., 0., 0.),
                           orientation=(0., 0., 0., 1.))

    for t in range(1000):
        sim.step()
