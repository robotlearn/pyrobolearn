#!/usr/bin/env python
"""Provide the gaussian process regression heightmap generator.

Generate a heightmap using gaussian process regression.
"""

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


def gpr_heightmap(init_values, x, y, kernel=None, alpha=1e-10, min_height=0, max_height=255, dtype=np.int):
    r"""
    Generate a heightmap using gaussian process regression. The advantages of using this method over others to
    generate terrains lies in the capacity of adding prior knowledge through the kernel and the given initial values.
    For instance, using a RBF kernel means that we want a smooth terrain instead of a bumpy one.
    Furthermore, it allows to generate heightmaps which are not necessary square; i.e. they can be rectangular. 

    Warnings: this is pretty difficult to exploit if the given data is not consistent. See `heigthmap_rbf` for
        a better way to generate heightmap.

    Args:
        init_values (np.array[M,3]): list of `M` 3D points which corresponds to initial values that are used to fit
            the gaussian process.
        x (np.array[N], np.array[N,O]): If 1d array, it will compute the meshgrid. Otherwise, the resulting 2D array
            from the meshgrid is expected. This is used to predict the heightmap at the given points.
        y (np.array[0], np.array[N,O]): If 1d array, it will compute the meshgrid. Otherwise, the resulting 2D array
            from the meshgrid is expected. This is used to predict the heightmap at the given points.
        kernel (None, sklearn.gaussian_process.kernels.Kernel): "The kernel specifying the covariance function of
            the GP. If None is passed, the kernel '1.0 * RBF(1.0)' is used as default. Note that the kernel's
            hyperparameters are optimized during fitting" [2]
        alpha (float, array_like): "Value added to the diagonal of the kernel matrix during fitting. Larger values
            correspond to increased noise level in the observations. This can also prevent a potential numerical issue
            during fitting, by ensuring that the calculated values form a positive definite matrix. If an array is
            passed, it must have the same number of entries as the data used for fitting and is used as
            datapoint-dependent noise level. Note that this is equivalent to adding a WhiteKernel with c=alpha.
            Allowing to specify the noise level directly as a parameter is mainly for convenience and for consistency
            with Ridge." [2]
        min_height (int,float): lower bound; each value in the heightmap will be higher than or equal to this bound
        max_height (int,float): upper bound; each value in the heightmap will be lower than or equal to this bound
        dtype (np.int, np.float): type of the returned array for the heightmap

    Returns:
        np.array[N,O]: resulting 2D heightmap

    Examples:
        >>> # generate heightmap using gaussian process regression
        >>> x = np.array(range(256))
        >>> y = np.array(range(256))
        >>> N_init = 20
        >>> x_init = np.random.randint(low=x.min(), high=x.max(), size=N_init)
        >>> y_init = np.random.randint(low=y.min(), high=y.max(), size=N_init)
        >>> z_init = np.random.randint(low=0, high=20, size=N_init)
        >>> init_values = np.vstack((x_init, y_init, z_init)).T     # shape: Nx3
        >>> heightmap = gpr_heightmap(init_values, x, y)

    References:
        - [1] "Gaussian Processes for Machine Learning", Rasmussen and Williams, 2006
        - [2] Sklearn: https://scikit-learn.org/stable/modules/gaussian_process.html
    """
    # check given x and y
    if len(x.shape) == 1 and len(y.shape) == 1:
        x, y = np.meshgrid(x, y)
    if x.shape != y.shape:
        raise ValueError("Expecting x and y to have the same shape, which should be the case if it is a meshgrid")

    # compute the minimum distance between points
    N = len(init_values)
    min_dist = np.inf
    for i in range(N):
        for j in range(i+1, N):
            dist = np.linalg.norm(init_values[i, :2] - init_values[j, :2])
            if dist < min_dist:
                min_dist = dist
    print("Min dist: {}".format(min_dist))

    # check initial values
    if not isinstance(init_values, np.ndarray):
        raise TypeError("Expecting init_values to be a numpy array")
    if init_values.shape[1] != 3:
        raise ValueError("Expecting a numpy array of 3D points for init_values")

    # create gaussian process and fit on the given initial values
    kernel = RBF(length_scale=np.sqrt(min_dist))
    gpr = GaussianProcessRegressor(kernel=kernel, alpha=alpha, normalize_y=True)
    gpr.fit(init_values[:, :2], init_values[:, 2])

    # predict the heightmap using GPR
    X = np.dstack((x, y)).reshape(-1, 2)
    heightmap = gpr.predict(X)
    heightmap = heightmap.reshape(x.shape)

    print("Params: {}".format(gpr.get_params()))

    # make sure the values of the heightmap are between the bounds (in-place), and is the correct type
    np.clip(heightmap, min_height, max_height, heightmap)
    heightmap.astype(dtype)

    return heightmap
