#!/usr/bin/env python
"""Define the biologically-inspired discrete dynamic movement primitive (as described in [1,2])

References:
    [1] "Biologically-inspired Dynamical Systems for Movement Generation: Automatic Real-time Goal Adaptation
             and Obstacle Avoidance", Hoffmann et al., 2009
    [2] "Learning and Generalization of Motor Skills by Learning from Demonstration", Pastor et al., 2009
"""

import numpy as np
import torch

from pyrobolearn.models.dmp.dmpytorch.discrete_dmp import DiscreteDMP

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class BioDiscreteDMP(DiscreteDMP):
    r"""Biologically-inspired Discrete DMPs

    One of the main problems with the initial DMP formulation is when some goal coordinates coincide with their
    corresponding initial position coordinates, it results in an inappropriate rescaling when displacing a little bit
    the goal.

    To deal with this problem, a new formulation of the transformation system was proposed in [2] and is given by:

    .. math:: \tau^2 \ddot{y} = K (g - y) - D \tau \dot{y} - K(g - y_0)s + K f(s)

    where :math:`\tau` is a scaling factor that allows to slow down or speed up the reproduced movement, :math:`K`
    is the stiffness coefficient, :math:`D` is the damping coefficient, :math:`y, \dot{y}, \ddot{y}` are the position,
    velocity, and acceleration of a DoF, and :math:`f(s)` is the non-linear forcing term.

    The forcing term is expressed as:

    .. math:: f(s) = \frac{\sum_i \psi_i(s) w_i}{ \sum_j \psi_j(s)} s

    Properties (from [2]):
    * Invariant under affine transformation
    * Movement generalization to new targets

    References:
        [1] "Dynamical movement primitives: Learning attractor models for motor behaviors", Ijspeert et al., 2013
        [2] "Biologically-inspired Dynamical Systems for Movement Generation: Automatic Real-time Goal Adaptation
             and Obstacle Avoidance", Hoffmann et al., 2009
        [3] "Learning and Generalization of Motor Skills by Learning from Demonstration", Pastor et al., 2009
    """

    def __init__(self, num_dmps, num_basis, dt=0.01, y0=0, goal=1,
                 forces=None, stiffness=None, damping=None):
        """Initialize the discrete DMP

        Args:
            num_dmps (int): number of DMPs
            num_basis (int): number of basis functions
            dt (float): step integration for Euler's method
            y0 (float, np.array): initial position(s)
            goal (float, np.array): goal(s)
            forces (list, ForcingTerm): the forcing terms (which can have different basis functions)
            stiffness (float): stiffness coefficient
            damping (float): damping coefficient
        """
        # if stiffness is None and damping is None:
        #     # from paper [2]
        #     stiffness = 150 * np.ones(num_dmps)
        #     damping = 2 * np.sqrt(stiffness)

        self.cst = 0.75     # this depends on the K and D value

        super(BioDiscreteDMP, self).__init__(num_dmps, num_basis, dt=dt, y0=y0, goal=goal,
                                             forces=forces, stiffness=stiffness, damping=damping)

    def step(self, s=None, tau=1.0, error=0.0, forcing_term=None, new_goal=None, external_force=None,
             rescale_force=True):
        """Run the DMP transformation system for a single time step.

        Args:
            s (None, float): the phase value. If None, it will use the canonical system.
            tau (float): Increase tau to make the system slower, and decrease it to make it faster
            error (float): optional system feedback
            forcing_term (np.ndarray): if given, it will replace the forcing term (shape [dmp,])
            new_goal (np.ndarray): new goal (of shape [num_dmps,])
        """

        # system feedback
        error_coupling = 1.0 / (1.0 + error)

        # get phase from canonical system
        if s is None:
            s = self.cs.step(tau=tau, error_coupling=error_coupling)
        elif not isinstance(s, (float, int)):
            raise TypeError("Expecting the phase 's' to be a float or integer. Instead, I got {}".format(type(s)))

        # check if same phase as before
        if s == self.prev_s:
            return self.y, self.dy, self.ddy

        if new_goal is None:
            new_goal = self.goal
        else:
            new_goal = new_goal + self.cst * (new_goal - self.goal)

        # save previous position and velocity
        prev_y, prev_dy = self.y.clone(), self.dy.clone()

        # for each DMP, solve transformation system equation using Euler's method
        for d in range(self.num_dmps):

            # compute forcing term
            if forcing_term is None:
                f = self.forces[d](s) + self.K[d] * s * (self.goal[d] - new_goal[d])
            else:
                f = forcing_term[d]

            # DMP acceleration
            self.ddy[d] = self.K[d]/(tau**2) * (new_goal[d] - self.y[d]) - self.D[d]/tau * self.dy[d] + f/(tau**2)
            if external_force is not None:
                self.ddy[d] += external_force[d]
            self.dy[d] += self.ddy[d] / tau * self.dt * error_coupling
            self.y[d] += self.dy[d] * self.dt * error_coupling

        # return self.y, self.dy, self.ddy
        return prev_y, prev_dy, self.ddy

    def _check_offset(self):
        """No need to check for an offset with this class"""
        pass

    def generate_goal(self, y0=None, dy0=None, ddy0=None, f0=None):
        """
        Generate the goal from the initial positions, velocities, accelerations, and forces.

        Args:
            y0 (float[M], None): initial positions. If None, it will take the default initial positions.
            dy0 (float[M], None): initial velocities. If None, it will take the default initial velocities.
            ddy0 (float[M], None): initial accelerations. If None, it will take the default initial accerelations.
            f0 (float[M], None): initial forcing terms. If None, it will compute it based on the learned weights.
                You can also give `dmp.f_target[:,0]` to get the correct goal.

        Returns:
            float[M]: goal position for each DMP.
        """
        if y0 is None:
            y0 = self.y0
        if dy0 is None:
            dy0 = self.dy0
        if ddy0 is None:
            ddy0 = self.ddy0
        if f0 is None:
            s0 = self.cs.init_phase
            f0 = self.get_forcing_term(s0)

        return 1/self.K * (ddy0 + self.D * dy0 + self.K * y0 - self.K * f0)


# Tests
if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # tests basis functions
    num_basis = 100

    # Test Biologically-inspired DMP
    t = torch.linspace(0., 1., 100)
    y_d = torch.sin(np.pi * t)
    new_goal = torch.tensor([[0.8, -0.25],
                             [0.8, 0.25],
                             [1.2, -0.25]])

    discrete_dmp = DiscreteDMP(num_dmps=2, num_basis=num_basis)
    discrete_dmp.imitate(torch.stack([t, y_d]))
    y, dy, ddy = discrete_dmp.rollout()
    init_points = torch.stack([discrete_dmp.y0, discrete_dmp.goal])
    # print(discrete_dmp.generate_goal())
    # print(discrete_dmp.generate_goal(f0=discrete_dmp.f_target[:,0]))
    y = y.detach().numpy()  # convert to numpy

    # check with standard discrete DMP when rescaling the goal
    plt.subplot(1, 3, 1)
    plt.title('Initial discrete DMP')
    plt.scatter(init_points[:, 0], init_points[:, 1], color='b')
    plt.scatter(new_goal[:, 0], new_goal[:, 1], color='r')
    plt.plot(y[0], y[1], 'b', label='original')

    plt.subplot(1, 3, 2)
    plt.title('Rescaled discrete DMP')
    plt.scatter(init_points[:, 0], init_points[:, 1], color='b')
    plt.scatter(new_goal[:, 0], new_goal[:, 1], color='r')
    plt.plot(y[0], y[1], 'b', label='original')
    for g in new_goal:
        y, dy, ddy = discrete_dmp.rollout(new_goal=g)
        plt.plot(y[0].detach().numpy(), y[1].detach().numpy(), 'g', label='scaled')
    plt.legend(['original', 'scaled'])

    # change goal with biologically-inspired DMP
    new_goal = torch.tensor([[0.8, -0.25],
                             [0.8, 0.25],
                             [0.4, 0.1],
                             [5., 0.15],
                             [1.2, -0.25],
                             [-0.8, 0.1],
                             [-0.8, -0.25],
                             [5., -0.25]])
    bio_dmp = BioDiscreteDMP(num_dmps=2, num_basis=num_basis)
    bio_dmp.imitate(torch.stack([t, y_d]))
    y, dy, ddy = bio_dmp.rollout()
    init_points = torch.stack([bio_dmp.y0, bio_dmp.goal])
    y = y.detach().numpy()  # convert to numpy

    plt.subplot(1, 3, 3)
    plt.title('Biologically-inspired DMP')
    plt.scatter(init_points[:, 0], init_points[:, 1], color='b')
    plt.scatter(new_goal[:, 0], new_goal[:, 1], color='r')
    plt.plot(y[0], y[1], 'b', label='original')
    for g in new_goal:
        y, dy, ddy = bio_dmp.rollout(new_goal=g)
        y = y.detach().numpy()  # convert to numpy
        plt.plot(y[0], y[1], 'g', label='scaled')
    plt.legend(['original', 'scaled'])
    plt.show()

    # changing goal at the middle
    y_list = []
    for g in new_goal:
        bio_dmp.reset()
        y_traj = torch.zeros(2, 100)
        for t in range(100):
            if t < 30:
                y, dy, ddy = bio_dmp.step()
            else:
                y, dy, ddy = bio_dmp.step(new_goal=g)
            y_traj[:, t] = y
        y_list.append(y_traj)
    for y in y_list:
        y = y.detach().numpy()  # convert to numpy
        plt.plot(y[0], y[1])
    plt.scatter(bio_dmp.y0[0], bio_dmp.y0[1], color='b')
    plt.scatter(new_goal[:, 0], new_goal[:, 1], color='r')
    plt.title('change goal at the middle')
    plt.show()

    # changing goal at the middle but with a moving goal
    g = torch.cat((torch.arange(1.0, 2.0, 0.1).reshape(10, -1),
                   torch.arange(0.0, 1.0, 0.1).reshape(10, -1)), dim=1)

    bio_dmp.reset()
    y_traj = np.zeros((2, 100))
    y_list = []
    for t in range(100):
        y, dy, ddy = bio_dmp.step(new_goal=g[int(t/10)])
        y = y.detach().numpy()  # convert to numpy
        y_traj[:, t] = y
        if (t % 10) == 0:
            y_list.append(y)
    y_list = np.array(y_list)

    plt.plot(y_traj[0], y_traj[1])
    plt.scatter(bio_dmp.y0[0], bio_dmp.y0[1], color='b')
    plt.scatter(g[:, 0], g[:, 1], color='r')
    plt.scatter(y_list[:, 0], y_list[:, 1], color='g')
    plt.title('moving goal')
    plt.show()
