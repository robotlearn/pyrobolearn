#!/usr/bin/env python
"""Provides the discrete ProMP.

References
    - [1] "Probabilistic Movement Primitives", Paraschos et al., 2013
    - [2] "Using Probabilistic Movement Primitives in Robotics", Paraschos et al., 2018
"""

import numpy as np

from pyrobolearn.models.promp.basis_functions import VonMisesBM, BlockDiagonalMatrix
from pyrobolearn.models.promp.promp import ProMP


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class RhythmicProMP(ProMP):
    r"""Rhythmic ProMP

    ProMP to be used for rhythmic movements.
    """

    def __init__(self, num_dofs, num_basis, weights=None, canonical_system=None, noise_covariance=1.,
                 basis_width=None):
        """
        Initialize the Rhythmic ProMP.

        Args:
            num_dofs (int): number of degrees of freedom (denoted by `D`)
            num_basis (int): number of basis functions (denoted by `M`)
            weights (np.array[DM], Gaussian, None): the weights that can be optimized. If None, it will create a
                custom weight array.
            canonical_system (CS, None): canonical system. If None, it will create a Linear canonical system that goes
                from `t0=0` to `tf=1`.
            noise_covariance (np.array[2D,2D]): covariance noise matrix
            basis_width (None, float): width of the basis. By default, it will be 1./(2*num_basis) such that the
                basis_width represents the standard deviation, and such that 2*std_dev = 1./num_basis.
        """
        super(RhythmicProMP, self).__init__(num_dofs=num_dofs, weight_size=num_dofs * num_basis, weights=weights,
                                            canonical_system=canonical_system, noise_covariance=noise_covariance)

        # define the basis width if not defined
        if basis_width is None:
            basis_width = 1. / (2 * num_basis)

        # create Von-Mises basis matrix with shape: DMx2D
        if num_dofs == 1:
            self.Phi = VonMisesBM(self.cs, num_basis, basis_width=basis_width)
        else:
            self.Phi = BlockDiagonalMatrix([VonMisesBM(self.cs, num_basis, basis_width=basis_width)
                                            for _ in range(num_dofs)])


# TESTS
if __name__ == "__main__":
    import matplotlib.pyplot as plt


    def plot_state(Y, title=None, linewidth=1.):
        y, dy = Y.T
        plt.figure()
        if title is not None:
            plt.suptitle(title)

        # plot position y(t)
        plt.subplot(1, 2, 1)
        plt.title('y(t)')
        plt.plot(y, linewidth=linewidth)  # TxN

        # plot velocity dy(t)
        plt.subplot(1, 2, 2)
        plt.title('dy(t)')
        plt.plot(dy, linewidth=linewidth)  # TxN


    def plot_weighted_basis(promp):
        phi_track = promp.weighted_basis(t)  # shape: DM,T,2D

        plt.subplot(1, 2, 1)
        plt.plot(phi_track[:, :, 0].T, linewidth=0.5)

        plt.subplot(1, 2, 2)
        plt.plot(phi_track[:, :, 1].T, linewidth=0.5)


    # create data and plot it
    N = 8
    t = np.linspace(0., 1., 100)
    eps = 0.1
    y = np.array([np.sin(2*np.pi*t) + eps * np.random.rand(len(t)) for _ in range(N)])      # shape: NxT
    dy = np.array([2*np.pi*np.cos(2*np.pi*t) + eps * np.random.rand(len(t)) for _ in range(N)])     # shape: NxT
    Y = np.dstack((y, dy))  # N,T,2D  --> why not N,2D,T
    plot_state(Y, title='Training data')
    plt.show()

    # create discrete and rhythmic ProMP
    promp = RhythmicProMP(num_dofs=1, num_basis=10, basis_width=1./20)

    # plot the basis function activations
    plt.plot(promp.Phi(t)[:, :, 0].T)
    plt.title('basis functions')
    plt.show()

    # plot ProMPs
    y_pred = promp.rollout()
    plot_state(y_pred[None], title='ProMP prediction before learning', linewidth=2.)    # shape: N,T,2D
    plot_weighted_basis(promp)
    plt.show()

    # learn from demonstrations
    promp.imitate(Y)
    y_pred = promp.rollout()
    plot_state(y_pred[None], title='ProMP prediction after learning', linewidth=2.)   # N,T,2D
    plot_weighted_basis(promp)
    plt.show()

    # modulation: final positions (goals)

    # modulation: final velocities

    # modulation: via-points

    # combination/co-activation/superposition

    # blending
