# -*- coding: utf-8 -*-
#!/usr/bin/env python
r"""Provide the equation heightmap generator.

Generate a heightmap from a 3D equation :math:`z = f(x, y)`.
"""

import numpy as np


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


def equation_heightmap(x, y, z, min_height=0, max_height=255, dtype=np.int):
    r"""
    Generate heightmap from 3D equation :math:`z = f(x,y)`.

    Args:
        x (np.array[N], np.array[N,O]): If 1d array, it will compute the meshgrid. Otherwise, the resulting 2D array
            from the meshgrid is expected. This is used to predict the heightmap at the given points.
        y (np.array[0], np.array[N,O]): If 1d array, it will compute the meshgrid. Otherwise, the resulting 2D array
            from the meshgrid is expected. This is used to predict the heightmap at the given points.
        z (callable): it must be a function that accepts two arguments `x` and `y` which will be the arrays from the
            meshgrid.
        min_height (int,float): lower bound; each value in the heightmap will be higher than or equal to this bound
        max_height (int,float): upper bound; each value in the heightmap will be lower than or equal to this bound
        dtype (np.int, np.float): type of the returned array for the heightmap

    Examples of 2D surfaces:
        z = lambda x,y: np.log(y)
        z = lambda x,y: np.sin(np.pi * x) * np.sin(np.pi * y)

    Returns:
        np.array[N,O]: resulting 2D heightmap
    """
    # check given x and y
    if len(x.shape) == 1 and len(y.shape) == 1:
        x, y = np.meshgrid(x, y)
    if x.shape != y.shape:
        raise ValueError("Expecting x and y to have the same shape, which should be the case if it is a meshgrid")
    origin_shape = x.shape

    # call z function: z=f(x,y)
    heightmap = z(x, y)

    # make sure the values of the heightmap are between the bounds (in-place), and is the correct type
    np.clip(heightmap, min_height, max_height, heightmap)
    heightmap.astype(dtype)

    return heightmap
