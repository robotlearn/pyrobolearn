#!/usr/bin/env python
"""Provide heightmap generators.

The various functions defined here generate
"""

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from scipy.interpolate import Rbf

try:
    import gdal
except ImportError as e:
    raise ImportError(repr(e) + '\nTry to install gdal: pip install gdal')


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


def diamond_square_algorithm(n=8, init_values=None, noise=0, lower_bound=0, upper_bound=255, dtype=np.int, seed=None):
    r"""Diamond-Square Algorithm

    This function implements the diamond-square algorithm [1], to generate random terrains given an initial value
    for each corner.

    Warnings: the diamond-square algo assumes that the heightmap is a 2D square array.

    Args:
        n (int): number of points (must be a power of 2). From this, the width and the height will automatically be
            computed, such that width = height = 2**n + 1.
        init_values (np.array[4], None): the four initial values for the corners. If None, it will generate 4 values
            randomly such that they are between the lower_bound and upper_bound.
        noise (int,float): noise level to add. This corresponds to the standard deviation of the normal distribution.
        lower_bound (int,float): lower bound; each value in the heightmap will be higher than or equal to this bound
        upper_bound (int,float): upper bound; each value in the heightmap will be lower than or equal to this bound
        dtype (np.int, np.float): type of the returned array for the heightmap
        seed (int, None): random seed

    Returns:
        np.array[2**n+1, 2**n+1]: resulting 2D square heightmap

    References:
        [1] Wikipedia: https://en.wikipedia.org/wiki/Diamond-square_algorithm
        [2] https://blog.habrador.com/2013/02/how-to-generate-random-terrain.html
    """
    # set the seed if given
    if seed:
        np.random.seed(seed)

    # create initial heightmap
    width, height = 2**n + 1, 2**n + 1
    heightmap = -1 * np.ones((height, width), dtype=dtype)
    if not init_values:
        if dtype == np.int:
            init_values = np.random.randint(low=lower_bound, high=upper_bound+1, size=4)
        else:
            init_values = np.random.uniform(low=lower_bound, high=upper_bound, size=4)
    heightmap[0, 0], heightmap[0, width - 1], heightmap[height - 1, 0], heightmap[height - 1, width - 1] = init_values

    # define diamond-square step function
    def diamond_square_step(heightmap, square=None, noise=0, lower_bound=0, upper_bound=255):
        """
        Diamond-square step which  which performs a diamond step followed by a square step.

        Args:
            heightmap (np.array[2*N+1,2*N+1]): heightmap (initial square)
            square (np.array[M,M]): the current square we focus on.
        """
        # if no square given
        if square is None:
            height, width = heightmap.shape
            square = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])

        # check size of square
        xmin, xmax, ymin, ymax = square[:, 0].min(), square[:, 0].max(), square[:, 1].min(), square[:, 1].max()
        dx, dy = (xmax - xmin), (ymax - ymin)
        if dx == 0 or dx == 1 or dy == 0 or dy == 1:
            return

        # DIAMOND STEP
        center = np.array([xmin + dx / 2, ymin + dy / 2])
        yc, xc = center
        heightmap[xc, yc] = np.mean([heightmap[x, y] for (y, x) in square])  # + np.random.normal(scale=noise)
        heightmap[xc, yc] = min(max(lower_bound, heightmap[xc, yc]), upper_bound)  # lower and upper bound

        # SQUARE STEP
        # triangles: a triangle is defined by 3 points
        triangles = np.array([[c1, c2, center] for c1, c2 in zip(square, list(square[1:]) + [square[0]])])

        squares = []
        for i, triangle in enumerate(triangles):
            xmin, xmax, ymin, ymax = triangle[:, 0].min(), triangle[:, 0].max(), triangle[:, 1].min(), triangle[:,
                                                                                                       1].max()

            if i == 0:  # upper triangle
                center = np.array([xmin + (xmax - xmin) / 2, ymin])
                square = np.array([[xmin, ymin], center, [center[0], ymax], [xmin, ymax]])  # left upper square
            elif i == 1:  # right triangle
                center = np.array([xmax, ymin + (ymax - ymin) / 2])
                square = np.array([[xmin, ymin], [xmax, ymin], center, [xmin, center[1]]])  # right upper square
            elif i == 2:  # lower triangle
                center = np.array([xmin + (xmax - xmin) / 2, ymax])
                square = np.array([[center[0], ymin], [xmax, ymin], [xmax, ymax], center])  # right lower square
            else:  # left triangle
                center = np.array([xmin, ymin + (ymax - ymin) / 2])
                square = np.array([center, [xmax, center[1]], [xmax, ymax], [xmin, ymax]])  # left lower square

            yc, xc = center
            heightmap[xc, yc] = np.mean([heightmap[x, y] for (y, x) in triangle])  # + np.random.normal(scale=noise)
            heightmap[xc, yc] = min(max(lower_bound, heightmap[xc, yc]), upper_bound)  # lower and upper bound

            # a square is defined by 4 points
            squares.append(square)

        # for each subsquare in the original square, compute the heightmap recursively
        for square in squares:
            diamond_square_step(heightmap, square, noise, lower_bound, upper_bound)

    # start diamond-square algorithm (recursively)
    diamond_square_step(heightmap, noise=noise, lower_bound=lower_bound, upper_bound=upper_bound)
    return heightmap


def heightmap_gpr(init_values, x, y, kernel=None, alpha=1e-10, lower_bound=0, upper_bound=255, dtype=np.int):
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
        lower_bound (int,float): lower bound; each value in the heightmap will be higher than or equal to this bound
        upper_bound (int,float): upper bound; each value in the heightmap will be lower than or equal to this bound
        dtype (np.int, np.float): type of the returned array for the heightmap

    Returns:
        np.array[N,O]: resulting 2D heightmap

    References:
        [1] "Gaussian Processes for Machine Learning", Rasmussen and Williams, 2006
        [2] Sklearn: https://scikit-learn.org/stable/modules/gaussian_process.html
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
    np.clip(heightmap, lower_bound, upper_bound, heightmap)
    heightmap.astype(dtype)

    return heightmap


def heightmap_rbf(init_values, x, y, function='multiquadric', lower_bound=0, upper_bound=255, dtype=np.int):
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
        lower_bound (int,float): lower bound; each value in the heightmap will be higher than or equal to this bound
        upper_bound (int,float): upper bound; each value in the heightmap will be lower than or equal to this bound
        dtype (np.int, np.float): type of the returned array for the heightmap

    Returns:
        np.array[N,O]: resulting 2D heightmap

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
    np.clip(heightmap, lower_bound, upper_bound, heightmap)
    heightmap.astype(dtype)

    return heightmap


def heighmap_equation(x, y, z, lower_bound=0, upper_bound=255, dtype=np.int):
    r"""
    Generate heightmap from 3D equation :math:`z = f(x,y)`.

    Args:
        x (np.array[N], np.array[N,O]): If 1d array, it will compute the meshgrid. Otherwise, the resulting 2D array
            from the meshgrid is expected. This is used to predict the heightmap at the given points.
        y (np.array[0], np.array[N,O]): If 1d array, it will compute the meshgrid. Otherwise, the resulting 2D array
            from the meshgrid is expected. This is used to predict the heightmap at the given points.
        z (callable): it must be a function that accepts two arguments `x` and `y` which will be the arrays from the
            meshgrid.
        lower_bound (int,float): lower bound; each value in the heightmap will be higher than or equal to this bound
        upper_bound (int,float): upper bound; each value in the heightmap will be lower than or equal to this bound
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
    np.clip(heightmap, lower_bound, upper_bound, heightmap)
    heightmap.astype(dtype)

    return heightmap


def heightmap_gdal(filename, subsample=None, interpolate_fct='multiquadric', lower_bound=0, upper_bound=255,
                   dtype=np.int):
    r"""
    Heightmap generated using the Geospatial Data Abstraction Library (GDAL), which allows to open Digital Elevation
    Models (DEM) or Geographic Information System (GIS). It can open a .tiff, .geotiff, ascii grid, or
     image (jpg, png,...) file.

    Args:
        filename (str): path to a DEM, GIS, or image file
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
        lower_bound (int,float): lower bound; each value in the heightmap will be higher than or equal to this bound
        upper_bound (int,float): upper bound; each value in the heightmap will be lower than or equal to this bound
        dtype (np.int, np.float): type of the returned array for the heightmap

    Returns:
        np.array[H,W]: resulting 2D array of size width `W` and height `H`

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
    if lower_bound and upper_bound:
        np.clip(heightmap, lower_bound, upper_bound, heightmap)
    elif lower_bound:
        np.clip(heightmap, lower_bound, heightmap.max(), heightmap)
    elif upper_bound:
        np.clip(heightmap, heightmap.min(), upper_bound, heightmap)
    if dtype:
        heightmap.astype(dtype)

    return heightmap


# alias
heigtmap_from_image = heightmap_gdal


# Tests
# Conclusion: use `heightmap_rbf` or `heightmap_gdal` as it is pretty good
if __name__ == '__main__':
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt

    # define plot figure for heightmap
    def plot_figure(heightmap, title='', block=True, z_upper_lim=256):
        fig = plt.figure()
        fig.suptitle(title)

        # 1st subplot: 2D heightmap
        ax = fig.add_subplot(1, 2, 1)
        ax.set_title('2D heightmap')
        ax.imshow(heightmap, cmap='gray')

        # 2nd subplot: associated 3D terrain
        ax = fig.add_subplot(1, 2, 2, projection='3d')
        ax.set_title('3D terrain')
        x = np.linspace(0, 1, heightmap.shape[0])
        y = np.linspace(0, 1, heightmap.shape[1])
        x, y = np.meshgrid(y, x)
        ax.plot_surface(x, y, heightmap)
        ax.set_zlim(0, z_upper_lim)
        print(x.shape)

        plt.show(block=block)


    # # generate heightmap using the diamond-square algorithm
    # N = 8       # shape of map: 2**N+1, 2**N+1
    # heightmap = diamond_square_algorithm(N)
    # plot_figure(heightmap, title='Diamond-Square algorithm')

    # # generate heightmap using gaussian process regression
    # x = np.array(range(256))
    # y = np.array(range(256))
    # N_init = 20
    # x_init = np.random.randint(low=x.min(), high=x.max(), size=N_init)
    # y_init = np.random.randint(low=y.min(), high=y.max(), size=N_init)
    # z_init = np.random.randint(low=0, high=20, size=N_init)
    # init_values = np.vstack((x_init, y_init, z_init)).T     # shape: Nx3
    # #init_values = np.array([[163, 73, 0], [13, 15, 1],[69, 102, 2]])
    # #init_values = np.array([[182, 48, 89], [182, 20, 150], [167, 247, 131]])
    # heightmap = heightmap_gpr(init_values=init_values, x=x, y=y)
    # plot_figure(heightmap, title='Gaussian Process Regression')

    # generate heightmap using RBF interpolations
    x = np.array(range(256))
    y = np.array(range(256)) # range(128)
    N_init = 20     # number of bumps
    x_init = np.random.randint(low=x.min(), high=x.max(), size=N_init)
    y_init = np.random.randint(low=y.min(), high=y.max(), size=N_init)
    z_init = np.random.randint(low=0, high=20, size=N_init)
    init_values = np.vstack((x_init, y_init, z_init)).T     # shape: Nx3
    # init_values = np.array([[211, 184, 3], [97, 59, 4], [37, 179, 8], [168, 32, 8], [198, 74, 13],
    #                         [44, 10, 2], [175, 102, 6], [6, 22, 1], [35, 165, 6], [169, 211, 16],
    #                         [158, 119, 18], [228, 63, 13], [40, 62, 15], [76, 221, 10], [1, 113, 10],
    #                         [178, 194, 2], [23, 176,10], [231,  88, 7], [247, 209, 6], [72, 94, 2]])
    heightmap = heightmap_rbf(init_values=init_values, x=x, y=y, function='gaussian') # 'linear', 'multiquadric'
    plot_figure(heightmap, title='RBF interpolation')

    # # generate heigthmap from an image or tif file
    # dem = heightmap_gdal('../tests/canyon-geo.tif')
    # dem = heightmap_gdal('../tests/dem.jpg')
    dem = heightmap_gdal('../tests/heightmap.png')
    plot_figure(dem, block=True)
