#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Provides the various basis functions and matrices used in ProMPs.

A basis matrix contains the basis functions, and the derivative of the basis functions (with respect to the phase),
and is callable (it accepts the phase as input).

References
    - [1] "Probabilistic Movement Primitives", Paraschos et al., 2013
    - [2] "Using Probabilistic Movement Primitives in Robotics", Paraschos et al., 2018
"""

from abc import ABCMeta, abstractmethod
from typing import Callable, List, Tuple, Union
import numpy as np
import numpy.typing as npt
from scipy.linalg import block_diag

from pyrobolearn.models.promp.canonical_systems import CS


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "Apache License 2.0"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class BasisFunction:
    r"""Basis Function

    The choice of basis function depends on the type of movement the user which to model; a discrete (aka stroke-based)
    or rhythmic movement.
    """
    __metaclass__ = ABCMeta

    def __init__(self) -> None:
        pass

    ###########
    # Methods #
    ###########

    def compute(self, phase: Union[float, npt.ArrayLike]) -> Union[float, np.ndarray]:
        """
        Predict the value of the basis function given the phase variable :math:`s`.

        Args:
            phase (float, np.ndarray): phase value(s).

        Returns:
            float, np.ndarray: value of the basis function evaluated at the given phase(s).
        """
        raise NotImplementedError

    # # aliases
    # predict = compute
    # forward = compute

    @abstractmethod
    def grad(self, phase: Union[float, npt.ArrayLike]) -> Union[float, np.ndarray]:
        """
        Compute the gradient of the basis function with respect to the phase variable :math:`s`, evaluated at
        the given phase.

        Args:
            phase (float, np.ndarray): phase value(s).

        Returns:
            float, np.ndarray: gradient evaluated at the given phase(s).
        """
        raise NotImplementedError

    # @abstractmethod
    # def grad_t(self, s):        # TODO: use automatic differentiation
    #     """
    #     Compute the gradient of the basis function with respect to the time variable :math:`t`, evaluated at
    #     the given phase :math:`s(t)`.
    #
    #     Args:
    #         s (float): phase value s(t).
    #
    #     Returns:
    #         float: gradient evaluated at the given phase s(t).
    #     """
    #     raise NotImplementedError

    #############
    # Operators #
    #############

    def __call__(self, phase: Union[float, npt.ArrayLike]) -> Union[float, np.ndarray]:
        """Predict value of basis function given phase(s)."""
        return self.compute(phase)


# alias
BF = BasisFunction


class GaussianBF(BF):
    r"""Gaussian Basis Function

    This basis function is given by the formula:

    .. math:: b(s) = \exp \left( - \frac{1}{2 h} (s - c)^2 \right)

    where :math:`c` is the center, and :math:`h` is the width of the basis.

    This is often used for discrete movement primitives.
    """

    def __init__(self, center: Union[float, npt.ArrayLike] = 0., width: Union[float, npt.ArrayLike] = 1.) -> None:
        """Initialize basis function.

        Args:
            center (float, np.ndarray): center of the distribution.
            width (float, np.ndarray): width of the distribution.
        """
        super(GaussianBF, self).__init__()

        if isinstance(center, np.ndarray): pass

        self.c = center
        if width <= 0:
            raise ValueError("Invalid `width` argument: the width of the basis has to be strictly positive")
        self.h = width

    def compute(self, phase: Union[float, npt.ArrayLike]) -> Union[float, np.ndarray]:
        r"""
        Predict the value of the basis function given the phase variable :math:`s`, given by:

        .. math:: b(s) = \exp \left( - \frac{1}{2 h} (s - c)^2 \right)

        where :math:`c` is the center, and :math:`h` is the width of the basis.

        Args:
            phase (float, np.ndarray): phase value(s).

        Returns:
            float, np.ndarray: value of the basis function evaluated at the given phase(s).
        """
        if isinstance(phase, np.ndarray):
            phase = phase[:, None]
        return np.exp(- 0.5 / self.h * (phase - self.c)**2)

    def grad(self, phase: Union[float, npt.ArrayLike]) -> Union[float, np.ndarray]:
        r"""
        Return the gradient of the basis function :math:`b(s)` with respect to the phase variable :math:`s`,
        evaluated at the given phase.

        For the Gaussian basis function, this results in:

        .. math::

            \frac{d b(s)}{ds} = - b(s) \frac{(s - c)}{h}

        Args:
            phase (float, np.ndarray): phase value(s).

        Returns:
            float, np.ndarray: gradient evaluated at the given phase(s).
        """
        phase_ = phase[:, None] if isinstance(phase, np.ndarray) else phase
        return - self(phase) * (phase_ - self.c) / self.h


# aliases
GBF = GaussianBF


class VonMisesBF(BF):
    r"""Von-Mises Basis Function

    This basis function is given by the formula:

    .. math:: b(s) = \exp \left( \frac{ \cos( 2\pi (s - c)) }{h} \right)

    where :math:`c` is the center, and :math:`h` is the width of the basis.

    This is often used for rhythmic movement primitives.
    """

    def __init__(self, center: Union[float, npt.ArrayLike] = 0., width: Union[float, npt.ArrayLike] = 1.) -> None:
        """Initialize basis function

        Args:
            center (float, np.ndarray): center of the basis fct.
            width (float, np.ndarray): width of the distribution.
        """
        super(VonMisesBF, self).__init__()
        self.c = center
        if width <= 0:
            raise ValueError("Invalid `width` argument: the width of the basis has to be strictly positive")
        self.h = width

    def compute(self, phase: Union[float, npt.ArrayLike]) -> Union[float, np.ndarray]:
        r"""
        Predict the value of the basis function given the phase variable :math:`s`, given by:

        .. math:: b(s) = \exp \left( \frac{ \cos( 2\pi (s - c)) }{h} \right)

        where :math:`c` is the center, and :math:`h` is the width of the basis.

        Args:
            phase (float, np.ndarray): phase value(s).

        Returns:
            float, np.ndarray: value of the basis function evaluated at the given phase(s).
        """
        if isinstance(phase, np.ndarray):
            phase = phase[:, None]
        return np.exp(np.cos(2*np.pi * (phase - self.c)) / self.h)

    def grad(self, phase: Union[float, npt.ArrayLike]) -> float:
        r"""
        Return the gradient of the basis function :math:`b(s)` with respect to the phase variable :math:`s`,
        evaluated at the given phase.

        For the Von-Mises basis function, this results in:

        .. math::

            \frac{d b(s)}{ds} = - b(s) 2\pi \frac{ \sin(2 \pi (s - c)) }{ h }

        Args:
            phase (float, np.ndarray): phase value(s).

        Returns:
            float, np.ndarray: gradient evaluated at the given phase(s).
        """
        phase_ = phase[:, None] if isinstance(phase, np.ndarray) else phase
        return - 2 * np.pi * self(phase) * np.sin(2 * np.pi * (phase_ - self.c)) / self.h


# aliases
CBF = VonMisesBF    # Circular Basis Function


class Matrix:
    """Callable matrix."""

    def __call__(self, phase: Union[float, npt.ArrayLike]) -> np.ndarray:
        raise NotImplementedError


class BasisMatrix(Matrix):
    r"""Basis matrix

    The basis matrix contains the basis functions, and the derivative of the basis functions.

    This is given by:

    .. math:: \Phi(s) = [\phi(s) \dot{\phi}(s)] \in \mathcal{R}^{M \times 2}

    where :math:`s` is the phase variable, and :math:`M` is the total number of components.
    """

    def __init__(self, matrix: npt.ArrayLike) -> None:
        """
        Initialize the basis matrix.

        Args:
            matrix (np.array[D]): 2D matrix containing callable functions. Each function given the phase
              returns an M-dimensional array where `M` is the total number of basis functions / weights.
              Evaluating it will return a matrix of shape (M,D).
        """
        self.matrix = matrix

        # get shape
        self._shape = self(0.).shape

    @property
    def shape(self) -> Tuple[int]:
        """Return the shape of the matrix."""
        return self._shape

    @property
    def num_basis(self) -> int:
        """Return the number of basis function."""
        return self._shape[0]

    def append(self, fct: Callable) -> None:
        """Concatenate the given function(s) to the basis matrix."""
        self.matrix = np.concatenate((self.matrix, [fct]))

    def evaluate(self, phase: Union[float, npt.ArrayLike]) -> np.ndarray:
        """
        Return matrix evaluated at the given phase.

        Args:
            phase (float, np.array[T]): phase value(s).

        Returns:
            np.array: array of shape Mx2, or MxTx2.
        """
        # matrix = np.array([[fct(s) for fct in row]
        #                    for row in self.matrix])
        matrix = np.array([fct(phase) for fct in self.matrix]).T
        return matrix

    def __getitem__(self, idx: int) -> Callable:
        """Return the associated basis function."""
        return self.matrix[idx]

    def __setitem__(self, idx: int, fct: Callable) -> None:
        """Set the given basis function."""
        self.matrix[idx] = fct

    def __call__(self, phase: Union[float, npt.ArrayLike]) -> np.ndarray:
        """
        Return matrix evaluated at the given phase.

        Args:
            phase (float, np.float[T]): phase value(s).

        Returns:
            np.array: array of shape 2xM, or Tx2xM.
        """
        return self.evaluate(phase)


class GaussianBM(BasisMatrix):
    r"""Gaussian Basis Matrix

    Matrix containing Gaussian basis functions.
    """

    def __init__(self, cs: CS, num_basis: int, basis_width: float = 1., compute_derivative: bool = True) -> None:
        """
        Initialize the Gaussian basis matrix.

        Args:
            cs (CS): canonical system.
            num_basis (int): number of basis functions (=M).
            basis_width (float): width of the basis functions.
            compute_derivative (bool): if we should compute the derivative basis function wrt time or not.
        """
        if not isinstance(cs, CS):
            raise TypeError("Expecting the given canonical system to be an instance of `CS`, but instead got:"
                            f" {type(cs)}")

        # distribute centers for the Gaussian basis functions
        # the centers are placed uniformly between [-2*width, 1+2*width]
        if num_basis == 1:
            centers = (1.+4*basis_width)/2.
        else:
            centers = np.linspace(-2*basis_width, 1+2*basis_width, num_basis)

        # create basis function and its derivative (if necessary)
        phi = GaussianBF(centers, basis_width)
        matrix = np.array([phi])  # shape (M, 1)

        if compute_derivative:
            def dphi_t(cs: CS, phi: BasisFunction) -> Callable:
                """Create derivative of basis function wrt time."""
                def step(phase):
                    return phi.grad(phase) * cs.grad()
                return step

            dphi = dphi_t(cs, phi)
            matrix = np.array([phi, dphi])  # shape (M, 2)

        # call superclass constructor
        super(GaussianBM, self).__init__(matrix)


class VonMisesBM(BasisMatrix):
    r"""Von-Mises Basis Matrix

    Matrix containing Von-Mises basis functions.
    """

    def __init__(self, cs: CS, num_basis: int, basis_width: float = 1., compute_derivative: bool = True) -> None:
        """
        Initialize the Von-Mises basis matrix.

        Args:
            cs (CS): canonical system.
            num_basis (int): number of basis functions.
            basis_width (float): width of the basis functions.
            compute_derivative (bool): if we should compute the derivative basis function wrt time or not.
        """
        # distribute centers for the Gaussian basis functions
        # the centers are placed uniformly between [-2*width, 1+2*width]
        if num_basis == 1:
            centers = (1. + 4 * basis_width) / 2.
        else:
            centers = np.linspace(-2 * basis_width, 1 + 2 * basis_width, num_basis)

        # create basis function and its derivative
        phi = VonMisesBF(centers, basis_width)
        matrix = np.array([phi])  # shape (M, 1)

        if compute_derivative:
            def dphi_t(cs: CS, phi: BasisFunction) -> Callable:
                """Create derivative of basis function wrt time."""
                def step(phase):
                    return phi.grad(phase) * cs.grad()
                return step

            dphi = dphi_t(cs, phi)
            matrix = np.array([phi, dphi])  # shape (M, 2)

        super(VonMisesBM, self).__init__(matrix)


class BlockDiagonalMatrix(Matrix):
    r"""Callable Block Diagonal matrix."""

    def __init__(self, matrices: List[BasisMatrix]) -> None:
        """
        Initialize the block diagonal matrix which contains callable matrices in its diagonal.

        Args:
            matrices (list[BasisMatrix]): list of callable matrices.
        """
        self.matrices = matrices

    ##############
    # Properties #
    ##############

    @property
    def shape(self) -> Tuple[int]:
        """Return the shape of the block diagonal matrix."""
        shape = 0
        for matrix in self.matrices:
            shape += np.array(matrix.shape)
        return tuple(shape)

    @property
    def num_basis(self) -> List[int]:
        """Return the number of basis per dimension."""
        return [matrix.num_basis for matrix in self.matrices]

    ###########
    # Methods #
    ###########

    def evaluate(self, phase: Union[float, npt.ArrayLike]) -> np.ndarray:
        """
        Evaluate the block diagonal matrix on the given input.

        Args:
            phase (float, np.array): input phase value(s).

        Returns:
            np.array: block diagonal matrix.
        """
        if isinstance(phase, np.ndarray):
            block = np.dstack([block_diag(*[matrix(phase_) for matrix in self.matrices]) for phase_ in phase])
            return np.swapaxes(block, 1, 2)  # shape: (DM, T, d)
        return block_diag(*[matrix(phase) for matrix in self.matrices])


    #############
    # Operators #
    #############

    def __call__(self, phase: Union[float, npt.ArrayLike]) -> np.ndarray:
        """
        Evaluate the block diagonal matrix on the given input.

        Args:
            phase (float, np.array): input phase value(s).

        Returns:
            np.array: block diagonal matrix.
        """
        return self.evaluate(phase)

    def __getitem__(self, idx: Union[int, slice]) -> "BlockDiagonalMatrix":
        """
        Return a desired chunk of the block diagonal matrix.

        Args:
            idx (int, slice): index of the basis matrix(ces) we wish to keep

        Returns:
            BlockDiagonalMatrix: return the desired chunk of the diagonal matrix
        """
        return BlockDiagonalMatrix(matrices=self.matrices[idx])


# TESTS
if __name__ == "__main__":
    from pyrobolearn.models.promp.canonical_systems import LinearCS

    num_basis, width = 10, 1.
    centers = np.linspace(-2 * width, 1 + 2 * width, num_basis)
    cs = LinearCS()

    # create basis functions

    phi = GaussianBF(centers, width)
    s = 0.5
    # s = np.array([0.5, 0.6, 0.7])
    print("Gaussian basis function - phi(s) shape: {}".format(phi(s).shape))     # shape: (T,M)
    print("Gaussian basis function - phi(s) = {}".format(phi(s)))

    # create dphi function
    def dphi_t(cs, phi):
        def step(s):
            return phi.grad(s) * cs.grad()
        return step

    dphi = dphi_t(cs, phi)
    print("dphi(s) shape: {}".format(dphi(s).shape))
    print("dphi(s) = {}".format(dphi(s)))

    # create basis matrices

    bm = GaussianBM(cs, num_basis, width)
    print("Gaussian basis matrix Phi(s) shape: {}".format(bm(s).shape))   # shape: (M,2)
    print("Gaussian basis matrix Phi(s) = {}".format(bm(s)))

    bm = VonMisesBM(cs, num_basis, width)
    print("Von-Mises basis matrix Phi(s) shape: {}".format(bm(s).shape))    # shape: (M,2)
    print("Von-Mises basis matrix Phi(s) = {}".format(bm(s)))
