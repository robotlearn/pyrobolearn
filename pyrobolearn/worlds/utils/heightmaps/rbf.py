#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Provide the radial-basis function heightmap generator.

Generate a heightmap by interpolating the given initial points using RBF functions.
"""

import numpy as np
from scipy.interpolate import Rbf


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "Apache License 2.0"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


def rbf_heightmap(init_values, x, y, function='multiquadric', min_height=0, max_height=255, dtype=np.int):
    r"""
    Generate heightmap by interpolating the given initial points using RBF functions.

    Advantages: fast and easy to use, and the results are pretty good. Heightmaps can also be rectangular.

    Args:
        init_values (np.array[M,3]): list of `M` 3D points which corresponds to initial values that are used to fit
            the gaussian process.
        x (np.array[N], np.array[N,O]): If 1d array, it will compute the meshgrid. Otherwise, the resulting 2D array
            from the meshgrid is expected. This is used to predict the heightmap at the given points.
        y (np.array[0], np.array[N,O]): If 1d array, it will compute the meshgrid. Otherwise, the resulting 2D array
            from the meshgrid is expected. This is used to predict the heightmap at the given points.
        function (str, callable): "The radial basis function, based on the radius, r, given by the norm
            (default is Euclidean distance);
                'multiquadric': sqrt((r/self.epsilon)**2 + 1)
                'inverse': 1.0/sqrt((r/self.epsilon)**2 + 1)
                'gaussian': exp(-(r/self.epsilon)**2)
                'linear': r
                'cubic': r**3
                'quintic': r**5
                'thin_plate': r**2 * log(r)
            If callable, then it must take 2 arguments (self, r). The epsilon parameter will be available as
            self.epsilon. Other keyword arguments passed in will be available as well." [1]
        min_height (int,float): lower bound; each value in the heightmap will be higher than or equal to this bound
        max_height (int,float): upper bound; each value in the heightmap will be lower than or equal to this bound
        dtype (np.int, np.float): type of the returned array for the heightmap

    Returns:
        np.array[N,O]: resulting 2D heightmap

    Examples:
        >>> # generate heightmap using RBF interpolations
        >>> x = np.array(range(256))
        >>> y = np.array(range(256))  # range(128)
        >>> N_init = 20     # number of bumps
        >>> x_init = np.random.randint(low=x.min(), high=x.max(), size=N_init)
        >>> y_init = np.random.randint(low=y.min(), high=y.max(), size=N_init)
        >>> z_init = np.random.randint(low=0, high=20, size=N_init)
        >>> init_values = np.vstack((x_init, y_init, z_init)).T     # shape: Nx3
        >>> heightmap = rbf_heightmap(init_values, x, y, function='gaussian')  # 'linear', 'multiquadric'

    References:
        [1] https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.Rbf.html
    """
    # check given x and y
    if len(x.shape) == 1 and len(y.shape) == 1:
        x, y = np.meshgrid(x, y)
    if x.shape != y.shape:
        raise ValueError("Expecting x and y to have the same shape, which should be the case if it is a meshgrid")
    origin_shape = x.shape

    rbf = Rbf(init_values[:, 0], init_values[:, 1], init_values[:, 2], function=function)
    heightmap = rbf(x.reshape(-1), y.reshape(-1))
    heightmap = heightmap.reshape(origin_shape)

    # make sure the values of the heightmap are between the bounds (in-place), and is the correct type
    np.clip(heightmap, min_height, max_height, heightmap)
    heightmap.astype(dtype)

    return heightmap
