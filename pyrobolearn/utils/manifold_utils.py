#!/usr/bin/env python
"""Util methods for operations on manifolds (e.g. Riemannian manifolds)
"""

import numpy as np
import scipy.linalg


__author__ = "Leonel Rozo"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Leonel Rozo"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


def tensor_matrix_product(tensor, matrix, mode):
    r"""Tensor-matrix product.

    Args:
        tensor (np.array): tensor
        matrix (np.array): matrix
        mode (int): mode of the product

    Returns:
        np.array: tensor x_mode matrix
    """
    # Mode-n tensor-matrix product
    N = len(tensor.shape)

    # Compute the complement of the set of modes
    modec = range(0, N)
    modec.remove(mode)

    # Permutation of the tensor
    perm = [mode] + modec
    S = np.transpose(tensor, perm)
    size_S = S.shape
    S = S.reshape((size_S[0], -1), order='F')

    # n-mode product
    S = np.dot(matrix, S)
    size_S = matrix.shape[0:1] + size_S[1:]
    S = S.reshape(size_S, order='F')

    # Inverse permutation
    inv_perm = [0]*N
    for i in range(0, N):
        inv_perm[perm[i]] = i

    S = np.transpose(S, inv_perm)

    return S


def symmetric_matrix_to_vector(M):
    r"""
    Symmetric matrix to vector using Mandel notation

    Args:
        M (np.array): symmetric matrix

    Returns:
        np.array: vector
    """
    N = M.shape[0]

    v = np.copy(M.diagonal())

    for i in range(1, N):
        v = np.concatenate((v, 2.0**0.5 * M.diagonal(i)))

    return v


def vector_to_symmetric_matrix(v):
    r"""
    Vector to symmetric matrix using Mandel notation.

    Args:
        v (np.array): vector

    Returns:
        np.array: symmetric matrix M
    """
    n = v.shape[0]
    N = int((-1.0 + (1.0+8.0*n)**0.5) / 2.0)

    M = np.copy(np.diag(v[0:N]))

    id = np.cumsum(range(N, 0, -1))

    for i in range(0, N-1):
        M += np.diag(v[range(id[i], id[i+1])], i+1) / 2.0**0.5 + np.diag(v[range(id[i], id[i+1])], -i-1) / 2.0**0.5

    return M


def exponential_map(U, S):
    r"""
    Exponential map.

    Args:
        U (list of np.array): list of symmetric matrices
        S (np.array): SPD matrix

    Returns:
        list of np.array: list of SPD matrices computed as Expmap_S(U)
    """
    X = []
    for n in range(len(U)):
        D, V = np.linalg.eig(np.linalg.solve(S, U[n]))
        X += [S.dot(V.dot(np.diag(np.exp(D))).dot(np.linalg.inv(V)))]

    return X


def logarithm_map(X, S):
    r"""
    Logarithm map.

    Args:
        X (list of np.array): list of SPD matrices
        S (np.array): SPD matrix

    Returns:
        list of np.array: list of symmetric matrices computed as Logmap_S(X)
    """
    U = []
    for n in range(len(X)):
        D, V = np.linalg.eig(np.linalg.solve(S, X[n]))
        U += [S.dot(V.dot(np.diag(np.log(D))).dot(np.linalg.inv(V)))]

    return U


def distance_spd(S1, S2):
    r"""
    SPD affine invariant distance.

    Args:
        S1 (np.array): SPD matrix
        S2 (np.array): SPD matrix

    Returns:
        float: affine invariant distance between S1 and S2
    """
    S1_pow = scipy.linalg.fractional_matrix_power(S1, -0.5)
    return np.linalg.norm(scipy.linalg.logm(np.dot(np.dot(S1_pow, S2), S1_pow)), 'fro')


def parallel_transport(S1, S2):
    """
    Parallel transport operation.

    Args:
        S1 (np.array): SPD matrix
        S2 (np.array): SPD matrix

    Returns:
        np.array: parallel transport operator
    """
    return scipy.linalg.fractional_matrix_power(np.dot(S2, np.linalg.inv(S1)), 0.5)
