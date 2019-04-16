#!/usr/bin/env python
r"""Model interface used in priority tasks.

This is based on the implementation in `https://github.com/ADVRHumanoids/ModelInterfaceRBDL`.

References:
    [1] "Quadratic Programming in Python" (https://scaron.info/blog/quadratic-programming-in-python.html), Caron, 2017
    [2] "OpenSoT: A whole-body control library for the compliant humanoid robot COMAN", Rocchi et al., 2015
    [3] "Robot Control for Dummies: Insights and Examples using OpenSoT", Hoffman et al., 2017
"""

import numpy as np
import rbdl


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["OpenSoT (Enrico Mingo Hoffman and Alessio Rocchi)", "Songyan Xin"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class ModelInterface(object):
    r"""Model interface.

    """

    def __init__(self, urdf):
        self.model = rbdl.loadModel(filename=urdf)
        self.model = rbdl.Model()
        self.q = np.zeros(self.model.q_size)
        self.dq = np.zeros(self.model.qdot_size)
        self.ddq = np.zeros(self.model.qdot_size)
        self.mass = 0
        self.com = np.zeros(3)
        self.com_vel = np.zeros(3)
        self.com_acc = np.zeros(3)
        self.angular_momentum_com = np.zeros(3)
        self.change_angular_momentum_com = np.zeros(3)

    @property
    def num_dof(self):
        return self.model.dof_count

    def get_com(self):
        return rbdl.CalcCenterOfMass(self.model, self.q, self.dq, self.ddq, self.com, self.com_vel, self.com_acc,
                                     self.angular_momentum_com, self.change_angular_momentum_com,
                                     update_kinematics=True)

    def get_com_jacobian(self):
        pass

    def get_com_velocity(self):
        pass

    def get_com_acceleration(self):
        pass

    def get_gravity(self):
        pass

    def get_jacobian(self):
        pass

    def get_pose(self):
        pass

    def get_acceleration_twist(self):
        pass

    def get_velocity_twist(self):
        pass

    def set_floating_base_pose(self):
        pass

    def set_floating_base_twist(self):
        pass

    def set_gravity(self):
        pass

    def compute_gravity_compensation(self):
        pass

    def get_centroidal_momentum(self):
        pass

    def compute_inverse_dynamics(self):
        pass

    def compute_non_linear_term(self):
        pass

    def get_inertia_matrix(self):
        pass

    def get_link_id(self, link_name):
        pass

    def update(self, q=None, dq=None, ddq=None):
        if q is None:
            q = self.q
        if dq is None:
            dq = self.dq
        if ddq is None:
            ddq = self.ddq
        rbdl.UpdateKinematics(self.model, q, dq, ddq)
