# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""Provide heightmap generators.

The various functions defined here generate
"""

import numpy as np
from scipy.interpolate import Rbf

try:
    import gdal
except ImportError as e:
    raise ImportError(repr(e) + '\nTry to install gdal: pip install gdal')


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


def gdal_heightmap(filename, subsample=None, interpolate_fct='multiquadric', min_height=0, max_height=255,
                   dtype=np.int):
    r"""
    Heightmap generated using the Geospatial Data Abstraction Library (GDAL), which allows to open Digital Elevation
    Models (DEM) or Geographic Information System (GIS). It can open a .tiff, .geotiff, ascii grid, or
     image (jpg, png,...) file.

    Args:
        filename (str): path to a DEM, GIS, or image file (e.g. png)
        subsample (int, None): if not None, it is the number of points to sub-sample (to smooth the heightmap using
            the specified function)
        interpolate_fct (str, callable): "The radial basis function, based on the radius, r, given by the norm
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
        np.array[H,W]: resulting 2D array of size width `W` and height `H`

    Examples:
        >>> # generate heightmap from an image or tif file
        >>> dem = gdal_heightmap('../pictures/canyon-geo.tif')
        >>> dem = gdal_heightmap('../pictures/dem.jpg')
        >>> dem = gdal_heightmap('../pictures/heightmap.png')

    References:
        [1] https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.Rbf.html
    """
    # load data (raster)
    data = gdal.Open(filename)
    band = data.GetRasterBand(1)
    heightmap = band.ReadAsArray()  # elevation values

    if isinstance(subsample, int) and subsample > 0:
        height, width = heightmap.shape
        idx_x = np.linspace(0, height-1, subsample, dtype=np.int)
        idx_y = np.linspace(0, width-1, subsample, dtype=np.int)
        idx_x, idx_y = np.meshgrid(idx_x, idx_y)
        x, y = np.arange(width), np.arange(height)
        x, y = np.meshgrid(x, y)
        rbf = Rbf(x[idx_x, idx_y], y[idx_x, idx_y], heightmap[idx_x, idx_y], function=interpolate_fct)
        # Nx, Ny = x.shape[0] / subsample, x.shape[1] / subsample
        # rbf = Rbf(x[::Nx, ::Ny], y[::Nx, ::Ny], heightmap[::Nx, ::Ny], function=interpolate_fct)
        heightmap = rbf(x, y)

    # make sure the values of the heightmap are between the bounds (in-place), and is the correct type
    if min_height and max_height:
        np.clip(heightmap, min_height, max_height, heightmap)
    elif min_height:
        np.clip(heightmap, min_height, heightmap.max(), heightmap)
    elif max_height:
        np.clip(heightmap, heightmap.min(), max_height, heightmap)
    if dtype:
        heightmap.astype(dtype)

    return heightmap
