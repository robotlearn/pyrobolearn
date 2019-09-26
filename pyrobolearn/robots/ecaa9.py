# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""Provide the ECA A9 autonomous underwater vehicle platform.
"""

import os
import numpy as np

from pyrobolearn.robots.uuv import UUVRobot

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


# TODO: fix inertia tags in the URDF file
# TODO: several methods need to be moved to the `uuv.py` file.
class ECAA9(UUVRobot):
    r"""Autonomous Unmanned Underwater Vehicle A9 from the ECA group

    WARNING: Currently, pybullet does not simulate fluids, so we simulate the thrust, drag, buoyancy, lift, and weight
    forces acting on the body. The gravity/weight force is simulated by pybullet.

    The various forces are given by [3,4]:

    1. The gravity/weight force is due to the attraction pull of the Earth is given by:

    .. math:: F_g = m g

    where :math:`m` is the mass of the object, and :math:`g` is the gravity constant (which is around
    :math:`9.81 m/s^2` on the earth).

    2. The buoyancy force is given by:

    .. math:: F_b = \rho g V

    where :math:`\rho` is the fluid density, :math:`g` is the gravity constant, and `V` is the volume of the
    body submerged in the fluid.

    3. The thrust force is due to the engine/motor/propeller of the object:

    .. math:: F_t = \dot{m}_e v_e - \dot{m}_0 v_0 + (p_e - p_0) A_e

    where :math:`\dot{m} = \rho A v` is the mass flow rate (i.e. mass/time), :math:`\rho` is the fluid density,
    :math:`v` is the velocity, :math:`A` is the area where its normal is parallel to fluid flow, :math:`p` is the
    pressure. The indices :math:`e` and :math:`0` stands for the exit and free stream (at the front of the submarine).

    4. The lift force is given by:

    .. math:: F_l = 1/2 C_l \rho A v^2

    where :math:`C_l` is the lift coefficient at the desired angle of attack, :math:`A` is the platform area, \rho is
    the fluid density, and :math:`v` is the velocity.

    5. The drag force (which is opposed to the movement) is given by:

    .. math:: F_d = 1/2 C_d \rho A v^2

    where :math:`C_d` is the drag coefficient which is depending on the shape of the object, friction and viscosity
    of the fluid, :math:`\rho` is the fluid density, :math:`A` is the platform area, and :math:`v` is the velocity.
    For a submarine, the coefficient is approximately around 0.04 (see [3]).

    References:
        - [1] https://www.ecagroup.com/en/solutions/a9-s-auv-autonomous-underwater-vehicle
        - [2] UUV Simulator: https://uuvsimulator.github.io/
        - [3] Aerodynamics (Nasa - check for equation): https://www.grc.nasa.gov/www/k-12/airplane/short.html
        - [4] https://s2.smu.edu/propulsion/Pages/navigation.htm
        - [5] Introduction to Ocean Waves: http://pordlabs.ucsd.edu/rsalmon/111.textbook.pdf
        - [6] https://fenicsproject.org/
    """

    def __init__(self, simulator, position=(0, 0, 1.), orientation=(0, 0, 0, 1), fixed_base=False, scale=1.,
                 urdf=os.path.dirname(__file__) + '/urdfs/ecaa9/eca_a9.urdf'):
        """
        Initialize the ECAA9 vehicle.

        Args:
            simulator (Simulator): simulator instance.
            position (np.array[float[3]]): Cartesian world position.
            orientation (np.array[float[4]]): Cartesian world orientation expressed as a quaternion [x,y,z,w].
            fixed_base (bool): if True, the vehicle base will be fixed in the world.
            scale (float): scaling factor that is used to scale the vehicle.
            urdf (str): path to the urdf. Do not change it unless you know what you are doing.
        """
        # check parameters
        if position is None:
            position = (0., 0., 1.)
        if len(position) == 2:  # assume x, y are given
            position = tuple(position) + (1.,)
        if orientation is None:
            orientation = (0, 0, 0, 1)
        if fixed_base is None:
            fixed_base = False

        super(ECAA9, self).__init__(simulator, urdf, position, orientation, fixed_base, scale)
        self.name = 'eca_a9'

        self.volume = 0.0679998770412 * scale ** 3  # from urdf
        self.sea_water_density = 1027
        self.center_buoyancy = np.array([0.000106, 0., 0.6])  # from urdf

    def calculate_buoyancy_force(self, fluid_density=None, g=9.81):
        r"""
        Calculate the buoyancy force.

        Args:
            fluid_density (float, None): density of the fluid [kg/m^3]
            g (gravity): gravity value in the z direction.

        Returns:
            np.array[float[3]]: buoyancy force
        """
        # currently, we assume that the whole body is submerged in the
        if fluid_density is None:
            fluid_density = self.sea_water_density
        return fluid_density * g * self.volume * np.array([0., 0., 1.])

    # TODO: add `fill_tank(volume)` and `empty_tank(volume)`
    def add_mass(self, mass=0., local_inertia_diagonal=(0., 0., 0.)):
        r"""
        Add mass to the submarine; a submarine has multiple ballast/trim tanks to control its buoyancy. This is
        only valid in the simulator.

        Args:
            mass (float): mass that will be added to the base link.
            local_inertia_diagonal (np.array[float[3]]): local inertia diagonal around the CoM of the base link.
        """
        info = self.sim.get_dynamics_info(body_id=self.id, link_id=-1)
        mass += info[0]
        local_inertia_diagonal = np.array(local_inertia_diagonal) + np.array(info[2])
        self.sim.change_dynamics(body_id=self.id, link_id=-1, mass=mass, local_inertia_diagonal=local_inertia_diagonal)


# Test
if __name__ == "__main__":
    from itertools import count
    from pyrobolearn.simulators import Bullet
    from pyrobolearn.worlds import BasicWorld

    # Create simulator
    sim = Bullet()

    # create world
    world = BasicWorld(sim)

    # create robot
    robot = ECAA9(sim)

    # print information about the robot
    robot.print_info()

    fb = robot.calculate_buoyancy_force()
    robot.add_mass(0.05)
    # robot.add_joint_slider(range(5))

    # robot.change_transparency(alpha=1.)

    # run simulation
    for i in count():
        pos = robot.get_base_position()
        # apply force in the simulation
        robot.apply_external_force(force=fb, link_id=-1, position=pos+robot.center_buoyancy,
                                   frame=Bullet.WORLD_FRAME)
        # robot.update_joint_slider()
        # robot.set_joint_velocities([10], [4])
        # step in simulation
        world.step(sleep_dt=1./240)
