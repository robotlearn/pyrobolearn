# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""Provide functions to process inertia matrices and moments of inertia.

References:
    - Moment of inertia (Wikipedia): https://en.wikipedia.org/wiki/Moment_of_inertia
    - List of moments of inertia (Wikipedia): https://en.wikipedia.org/wiki/List_of_moments_of_inertia
"""

import numpy as np

from pyrobolearn.utils.transformation import skew_matrix
from pyrobolearn.utils.mesh import get_mesh_body_inertia

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2019, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


def get_full_inertia(inertia):
    r"""
    Get the full inertia matrix from a partial one.

    Args:
        inertia (np.array[float[3,3]], np.array[float[9]], np.array[float[6]], np.array[float[3]]): body frame inertia
            matrix relative to the center of mass. If 9 elements are given, these are assumed to be [ixx, ixy, ixz,
            ixy, iyy, iyz, ixz, iyz, izz]. If 6 elements are given, they are assumed to be [ixx, ixy, ixz, iyy, iyz,
            izz]. Finally, if only 3 elements are given, these are assumed to be [ixx, iyy, izz] and are considered
            already to be the principal moments of inertia.

    Returns:
        np.array[float[3,3]]: full inertia matrix.
    """
    # make sure inertia is a numpy array
    inertia = np.asarray(inertia)

    # get the full inertia matrix
    if inertia.shape == (3,):
        inertia = np.diag(inertia)
    elif inertia.shape == (6,):
        ixx, ixy, ixz, iyy, iyz, izz = inertia
        inertia = np.array([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]])
    elif inertia.shape == (9,):
        inertia = inertia.reshape(3, 3)

    # check the shape
    if inertia.shape != (3, 3):
        raise ValueError("Expecting the inertia matrix to be of shape (3,), (6,), (9,) or (3,3), but got a shape of: "
                         "{}".format(inertia.shape))

    return inertia


def get_principal_moments_and_axes_of_inertia(inertia, full=False):
    r"""
    Get the principal moments of inertia.

    Given a body frame inertia matrix relative to the center of mass :math:`I^B_C`, it can be decomposed using
    eigendecomposition into:

    .. math:: I^B_C = Q \Sigma Q^T

    where :math:`\Sigma = \text{diag}[I_1, I_2, I_3]` is a diagonal matrix :math:`\in \mathbb{R^{3 \times 3}}` where
    the diagonal elements (:math:`I_1, I_2, I_3`) are called the principal moments of inertia, and the columns of
    :math:`Q` are called the principal axes of the body.

    Args:
        inertia (np.array[float[3,3]], np.array[float[9]], np.array[float[6]], np.array[float[3]]): body frame inertia
            matrix relative to the center of mass. If 9 elements are given, these are assumed to be [ixx, ixy, ixz,
            ixy, iyy, iyz, ixz, iyz, izz]. If 6 elements are given, they are assumed to be [ixx, ixy, ixz, iyy, iyz,
            izz]. Finally, if only 3 elements are given, these are assumed to be [ixx, iyy, izz] and are considered
            already to be the principal moments of inertia.
        full (bool): if we should return the principal moments of inertia as a full matrix or an array of 3 float.

    Returns:
        if full:
            np.array[float[3,3]]: diagonal inertia matrix where the diagonal elements are the principal moments of
                inertia.
        else:
            np.array[float[3]]: principal moments of inertia.
    """
    # get full inertia
    inertia = get_full_inertia(inertia)

    # eigendecomposition
    evals, evecs = np.linalg.eigh(inertia)

    if full:
        evals = np.diag(evals)

    return evals, evecs


def get_principal_moments_of_inertia(inertia, full=False):
    r"""
    Get the principal moments of inertia.

    Given a body frame inertia matrix relative to the center of mass :math:`I^B_C`, it can be decomposed using
    eigendecomposition into:

    .. math:: I^B_C = Q \Sigma Q^T

    where :math:`\Sigma = \text{diag}[I_1, I_2, I_3]` is a diagonal matrix :math:`\in \mathbb{R^{3 \times 3}}` where
    the diagonal elements (:math:`I_1, I_2, I_3`) are called the principal moments of inertia, and the columns of
    :math:`Q` are called the principal axes of the body.

    Args:
        inertia (np.array[float[3,3]], np.array[float[9]], np.array[float[6]], np.array[float[3]]): body frame inertia
            matrix relative to the center of mass. If 9 elements are given, these are assumed to be [ixx, ixy, ixz,
            ixy, iyy, iyz, ixz, iyz, izz]. If 6 elements are given, they are assumed to be [ixx, ixy, ixz, iyy, iyz,
            izz]. Finally, if only 3 elements are given, these are assumed to be [ixx, iyy, izz] and are considered
            already to be the principal moments of inertia.
        full (bool): if we should return the principal moments of inertia as a full matrix or an array of 3 float.

    Returns:
        if full:
            np.array[float[3,3]]: diagonal inertia matrix where the diagonal elements are the principal moments of
                inertia.
        else:
            np.array[float[3]]: principal moments of inertia.
    """
    return get_principal_moments_and_axes_of_inertia(inertia, full)[0]


def get_principal_axes_of_inertia(inertia):
    r"""
    Get the principal moments of inertia.

    Given a body frame inertia matrix relative to the center of mass :math:`I^B_C`, it can be decomposed using
    eigendecomposition into:

    .. math:: I^B_C = Q \Sigma Q^T

    where :math:`\Sigma = \text{diag}[I_1, I_2, I_3]` is a diagonal matrix :math:`\in \mathbb{R^{3 \times 3}}` where
    the diagonal elements (:math:`I_1, I_2, I_3`) are called the principal moments of inertia, and the columns of
    :math:`Q` are called the principal axes of the body.

    Args:
        inertia (np.array[float[3,3]], np.array[float[9]], np.array[float[6]], np.array[float[3]]): body frame inertia
            matrix relative to the center of mass. If 9 elements are given, these are assumed to be [ixx, ixy, ixz,
            ixy, iyy, iyz, ixz, iyz, izz]. If 6 elements are given, they are assumed to be [ixx, ixy, ixz, iyy, iyz,
            izz]. Finally, if only 3 elements are given, these are assumed to be [ixx, iyy, izz] and are considered
            already to be the principal moments of inertia.

    Returns:
        np.array[float[3,3]]: principal axes of the body inertia.
    """
    return get_principal_moments_and_axes_of_inertia(inertia, full=False)[1]


def translate_inertia_matrix(inertia, vector, mass):
    r"""
    "The inertia matrix of a body depends on the choice of the reference point. There is a useful relationship between
    the inertia matrix relative to the center of mass C and the inertia matrix relative to the another point. This
    relationship is called the parallel axis theorem" [1].

    The result is given by:

    .. math:: I_R = I_C - M [d]^2

    where :math:`I_R \in \mathbb{R}^{3 \times 3}` is the inertia matrix relative to the point :math:`R`,
    :math:`I_C \in \mathbb{R}^{3 \times 3}` is the inertia matrix relative to the center of mass :math:`C`,
    :math:`M \in \mathbb{R}` is the total mass of the system, :math:`d \in \mathbb{R}^3` is the vector from the
    center of mass :math:`C` to the reference point :math:`R`, and :math:`[\cdot]` is the operation which transforms
    a vector into a skew-symmetric matrix.

    Args:
        inertia (np.array[float[3,3]]): full inertia matrix of a body around its CoM.
        vector (np.array[float[3]]): translation vector from the CoM to another point.
        mass (float): the total mass of the body.

    Returns:
        np.array[float[3,3]]: full inertia matrix of a body relative to another point.

    References:
        - [1] Parallel axis theorem (Wikipedia): https://en.wikipedia.org/wiki/Moment_of_inertia#Parallel_axis_theorem
    """
    d = skew_matrix(vector)
    return inertia - mass * d**2


def rotate_inertia_matrix(inertia, rotation):
    r"""
    Rotate the inertia matrix.

    Assuming a rotation matrix :math:`R` that defines the body frame orientation with respect to an inertial frame,
    and thus maps a vector :math:`x` described in the body fixed coordinate frame to the coordinates in the inertial
    frame :math:`y = R x`, the inertia matrix in the inertial frame is given by:

    .. math:: I_C = R I^B_C R^\top

    where :math:`R \in \mathbb{R}^{3 \times 3}` is the rotation matrix that represents the orientation of the body
    frame relative to an inertial frame (it can depend on the time, i.e. :math:`R(t)`), and :math:`I^B_C` is the
    inertia matrix of a body around its center of mass (this is constant over time), and :math:`I_C` is the inertia
    matrix of the body measured in the inertial frame (this is dependent of the time, if the rotation is dependent of
    the time).

    Args:
        inertia (np.array[float[3,3]]): full inertia matrix.
        rotation (np.array[float[3,3]]): rotation matrix.

    Returns:
        np.array[float[3,3]]: rotated inertia matrix.
    """
    return rotation.dot(inertia).dot(rotation.T)


def scale_inertia(inertia, scale=1):
    r"""
    Scale the inertia matrix.

    The inertia is given by:

    .. math:: I \sim m r^2

    where :math:`I` is the inertia [kg m^2], :math:`m` is the mass [kg], and :math:`r` is distance [m].

    If you scale :math:`r` from 1m to let's say 1mm thus a scaling factor of :math:`10^{-3}`, it has the effect that
    the inertia will be scaled by a factor of :math:`10^{-6}` because of the squared operation.

    Args:
        inertia :
        scale (float): scaling factor.

    Returns:
        np.array[float[3,3]]: full scaled inertia matrix.
    """
    return inertia * scale**2


def get_inertia_of_sphere(mass, radius, full=False):
    r"""
    Return the principal moments of the inertia matrix of a sphere.

    Args:
        mass (float): mass of the sphere.
        radius (float): radius of the sphere.
        full (bool): if we should return the full inertia matrix, or just the diagonal elements.

    Returns:
        if full:
            np.array[float[3,3]]: diagonal inertia matrix where the diagonal elements are the principal moments of
                inertia.
        else:
            np.array[float[3]]: principal moments of inertia.

    References:
        - List of moments of inertia (Wikipedia): https://en.wikipedia.org/wiki/List_of_moments_of_inertia
    """
    inertia = 2./5 * mass * radius**2 * np.ones(3)
    if full:
        return np.diag(inertia)
    return inertia


def get_inertia_of_box(mass, size, full=False):
    r"""
    Return the principal moments of the inertia matrix of a box/cuboid.

    Args:
        mass (float): mass of the box.
        size (np.array[float[3]]): dimensions of the box/cuboid along the 3 axes (width, height, depth).
        full (bool): if we should return the full inertia matrix, or just the diagonal elements.

    Returns:
        if full:
            np.array[float[3,3]]: diagonal inertia matrix where the diagonal elements are the principal moments of
                inertia.
        else:
            np.array[float[3]]: principal moments of inertia.

    References:
        - List of moments of inertia (Wikipedia): https://en.wikipedia.org/wiki/List_of_moments_of_inertia
    """
    w, h, d = size  # width, height, depth
    inertia = 1./12 * mass * np.array([h**2 + d**2, w**2 + d**2, w**2 + h**2])
    if full:
        return np.diag(inertia)
    return inertia


def get_inertia_of_cylinder(mass, radius, height, full=False):
    r"""
    Return the principal moments of the inertia matrix of a cylinder.

    Args:
        mass (float): mass of the cylinder.
        radius (float): radius of the cylinder.
        height (float): height of the cylinder.
        full (bool): if we should return the full inertia matrix, or just the diagonal elements.

    Returns:
        if full:
            np.array[float[3,3]]: diagonal inertia matrix where the diagonal elements are the principal moments of
                inertia.
        else:
            np.array[float[3]]: principal moments of inertia.

    References:
        - List of moments of inertia (Wikipedia): https://en.wikipedia.org/wiki/List_of_moments_of_inertia
    """
    r, h = radius, height
    inertia = 1./12 * mass * np.array([3*r**2 + h**2, 3*r**2 + h**2, r**2])
    if full:
        return np.diag(inertia)
    return inertia


def get_inertia_of_capsule(mass, radius, height, full=False):
    r"""
    Return the principal moments of the inertia matrix of a capsule.

    Args:
        mass (float): mass of the capsule.
        radius (float): radius of the capsule (i.e. radius of the hemispheres).
        height (float): height of the capsule.
        full (bool): if we should return the full inertia matrix, or just the diagonal elements.

    Returns:
        if full:
            np.array[float[3,3]]: diagonal inertia matrix where the diagonal elements are the principal moments of
                inertia.
        else:
            np.array[float[3]]: principal moments of inertia.

    References:
        - https://www.gamedev.net/articles/programming/math-and-physics/capsule-inertia-tensor-r3856/
    """
    r, h = radius, height

    # get mass of cylinder and hemisphere
    sphere_volume = 4. / 3 * np.pi * r ** 3
    cylinder_volume = np.pi * r ** 2 * h
    volume = sphere_volume + cylinder_volume
    density = mass / volume
    m_s = density * sphere_volume  # sphere mass = 2 * hemisphere mass
    m_c = density * cylinder_volume  # cylinder mass

    # from: https://www.gamedev.net/articles/programming/math-and-physics/capsule-inertia-tensor-r3856/
    ixx = m_c * (h ** 2 / 12. + r ** 2 / 4.) + m_s * (2 * r ** 2 / 5. + h ** 2 / 2. + 3 * h * r / 8.)
    iyy = ixx
    izz = m_c * r ** 2 / 2. + m_s * 2 * r ** 2 / 5.
    inertia = np.array([ixx, iyy, izz])

    if full:
        return np.diag(inertia)
    return inertia


def get_inertia_of_ellipsoid(mass, a, b, c, full=False):
    r"""
    Return the principal moments of the inertia matrix of a ellipsoid.

    Args:
        mass (float): mass of the ellipsoid.
        a (float): first semi-axis of the ellipsoid.
        b (float): second semi-axis of the ellipsoid.
        c (float): third semi-axis of the ellipsoid.
        full (bool): if we should return the full inertia matrix, or just the diagonal elements.

    Returns:
        if full:
            np.array[float[3,3]]: diagonal inertia matrix where the diagonal elements are the principal moments of
                inertia.
        else:
            np.array[float[3]]: principal moments of inertia.

    References:
        - List of moments of inertia (Wikipedia): https://en.wikipedia.org/wiki/List_of_moments_of_inertia
    """
    inertia = 1. / 5 * mass * np.array([b ** 2 + c ** 2, a ** 2 + c ** 2, a ** 2 + b ** 2])
    if full:
        return np.diag(inertia)
    return inertia


def get_inertia_of_mesh(mesh, mass=None, scale=1., density=1000, full=False):
    r"""
    Return the principal moments of the inertia matrix of a mesh.

    Warnings: the mesh has to be watertight.

    Args:
        mesh (str, trimesh.Trimesh): path to the mesh file, or a Trimesh instance. Note that the mesh has to be
          watertight.
        mass (float, None): mass of the mesh (in kg). If None, it will use the density.
        scale (float): scaling factor. If you have a mesh in meter but you want to scale it into centimeters, you need
          to provide a scaling factor of 0.01.
        density (float): density of the mesh (in kg/m^3). By default, it uses the density of the water 1000kg / m^3.
        full (bool): if we should return the full inertia matrix, or just the diagonal elements.

    Returns:
        if full:
            np.array[float[3,3]]: diagonal inertia matrix where the diagonal elements are the principal moments of
                inertia.
        else:
            np.array[float[3]]: principal moments of inertia.
    """
    inertia = get_mesh_body_inertia(mesh, mass=mass, density=density, scale=scale)
    if full:
        return np.diag(inertia)
    return inertia


def combine_inertias(coms, masses, inertias, rotations=None):
    r"""
    This combines the inertia matrices together to form the combined body frame inertia matrix relative to the
    combined center of mass.
    
    Args:
        coms (list[np.array[float[3]]): list of center of masses.
        masses (list[float]): list of total body masses.
        inertias (list[np.array[float[3,3]]]): list of body frame inertia matrices relative to their center of mass.
        rotations (list[np.array[float[3,3]]]): list of rotation matrices where each rotation has to be applied on
          the inertia matrix before translating it.

    Returns:
        float: total mass.
        np.array[float[3]]: combined center of mass.
        np.array[float[3,3]]: combined inertia matrix.
    """
    if len(coms) != len(masses) or len(coms) != len(inertias):
        raise ValueError("The given lists do not have the same length: len(coms)={}, len(masses)={}, "
                         "len(inertias)={}".format(len(coms), len(masses), len(inertias)))
    if len(coms) == 0:
        raise ValueError("Expecting the length of the provided parameters to be bigger than 0")
    if rotations is not None and len(rotations) != len(coms):
        raise ValueError("The given list of rotations do not have the same length (={}) as the other arguments (={})"
                         ".".format(len(rotations), len(masses)))

    total_mass = np.sum(masses)
    new_com = np.sum([mass * com for mass, com in zip(masses, coms)], axis=0)
    new_com /= total_mass
    if rotations is None:
        inertia = np.sum([translate_inertia_matrix(inertia, vector=new_com-com, mass=mass)
                          for mass, com, inertia in zip(masses, coms, inertias)], axis=0)
    else:
        inertia = np.sum([translate_inertia_matrix(rotate_inertia_matrix(inertia, rot), vector=new_com-com, mass=mass)
                          for mass, com, inertia, rot in zip(masses, coms, inertias, rotations)], axis=0)
    return total_mass, new_com, inertia
