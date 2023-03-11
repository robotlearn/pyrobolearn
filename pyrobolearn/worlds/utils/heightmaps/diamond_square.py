#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Provide the diamond-square heightmap generator.

Generate the heightmap using the diamond-square algorithm [1].

References:
    [1] Wikipedia: https://en.wikipedia.org/wiki/Diamond-square_algorithm
"""

import numpy as np


__author__ = ["Brian Delhaisse", "Jamie Scollay"]
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse", "Jamie Scollay"]
__license__ = "Apache License 2.0"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"

# the `diamond_square_heightmap_2` was originally written by Jamie Scollay
# it was then reviewed by Brian Delhaisse, notably with respect to the original code:
# - it has been cleaned; removed all the ";"
# - it has been optimized:
#     - it now uses numpy instead of math and lists, it uses `list.append` instead of adding strings, and then `join`
#     - it has been simplified.
# - comments have been added and a better documentation is provided


def diamond_square_heightmap(n=8, min_height=0, max_height=255, noise=0, noise_factor=1, init_values=None,
                             dtype=np.int, seed=None):
    r"""Diamond-Square Algorithm

    This function implements the diamond-square algorithm [1], to generate random terrains given an initial value
    for each corner.

    Warnings: the diamond-square algo assumes that the heightmap is a 2D square array.

    Args:
        n (int): used to create a square array of width and height of 2**n + 1. It also specifies the number of
            diamond and square steps.
        min_height (int,float): lower bound; each value in the heightmap will be higher than or equal to this bound
        max_height (int,float): upper bound; each value in the heightmap will be lower than or equal to this bound
        noise (int, float): noise level to add. This corresponds to the standard deviation of the normal distribution.
        noise_factor (int, float): after each step, the noise is divided by the given factor.
        init_values (np.array[4], None): the four initial values for the corners. If None, it will generate 4 values
            randomly such that they are between the min_height and max_height.
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
            init_values = np.random.randint(low=min_height, high=max_height + 1, size=4)
        else:
            init_values = np.random.uniform(low=min_height, high=max_height, size=4)
    heightmap[0, 0], heightmap[0, width - 1], heightmap[height - 1, 0], heightmap[height - 1, width - 1] = init_values

    # define diamond-square step function
    def diamond_square_step(heightmap, square=None, noise=0, min_height=0, max_height=255):
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
        heightmap[xc, yc] = min(max(min_height, heightmap[xc, yc]), max_height)  # lower and upper bound

        # SQUARE STEP
        # triangles: a triangle is defined by 3 points
        triangles = np.array([[c1, c2, center] for c1, c2 in zip(square, list(square[1:]) + [square[0]])])

        squares = []
        for i, triangle in enumerate(triangles):
            xmin, xmax, ymin, ymax = triangle[:, 0].min(), triangle[:, 0].max(), \
                                     triangle[:, 1].min(), triangle[:, 1].max()

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
            heightmap[xc, yc] = min(max(min_height, heightmap[xc, yc]), max_height)  # lower and upper bound

            # a square is defined by 4 points
            squares.append(square)

        # for each subsquare in the original square, compute the heightmap recursively
        for square in squares:
            diamond_square_step(heightmap, square, noise / noise_factor, min_height, max_height)

    # start diamond-square algorithm (recursively)
    diamond_square_step(heightmap, noise=noise, min_height=min_height, max_height=max_height)
    return heightmap


def diamond_square_heightmap_2(n=8, min_height=0, max_height=255, noise=0, noise_factor=1):
    """
    Create a 2D square heightmap using the diamond square algorithm [1]

    Args:
        n (int): used to create a square array of width and height of 2**n + 1. It also specifies the number of
            diamond and square steps.
        min_height (float): minimum height
        max_height (float): maximum height
        noise (float): magnitude of the noise added to the computed height.
        noise_factor (float): after each step, the jitter is divided by the given factor.

    Returns:
        np.array[size, size]: 2D square heightmap

    References:
        [1] Wikipedia: https://en.wikipedia.org/wiki/Diamond-square_algorithm
    """
    # compute the width and height size (i.e. size of the 2D square array/heightmap)
    size = int(2 ** n + 1)

    # create initial heightmap of width and height size
    heightmap = np.zeros((size, size))

    # assign a random height at each corner of the heightmap
    heightmap[0, 0] = np.random.rand() * max_height
    heightmap[0, size - 1] = np.random.rand() * max_height
    heightmap[size - 1, 0] = np.random.rand() * max_height
    heightmap[size - 1, size - 1] = np.random.rand() * max_height

    # for each diamond and square step
    for i in range(n):
        stride = int((size - 1) / 2 ** (i + 1))
        radius = int((size - 1) / 2 ** i)

        for j in range(2 ** i):
            for k in range(2 ** i):
                height = (heightmap[j * radius, k * radius] + heightmap[2 * stride + j * radius, k * radius] +
                          heightmap[j * radius, 2 * stride + k * radius] +
                          heightmap[2 * stride + j * radius, 2 * stride + k * radius]) / 4. + (
                                     np.random.rand() - 0.5) * noise
                heightmap[stride + j * radius, stride + k * radius] = height

        for j in range(2 ** (i + 1) + 1):
            for k in range(2 ** i + j % 2):
                cnt = 0
                if j == 0:
                    height1 = 0
                else:
                    height1 = heightmap[(j - 1) * stride, stride * ((j + 1) % 2) + k * radius]
                    cnt += 1

                if k == 0 and j % 2 == 1:
                    height4 = 0
                else:
                    height4 = heightmap[j * stride, stride * (((j + 1) % 2) - 1) + k * radius]
                    cnt += 1

                if j == 2 ** (i + 1):
                    height3 = 0
                else:
                    height3 = heightmap[(j + 1) * stride, stride * ((j + 1) % 2) + k * radius]
                    cnt += 1

                if k == (2 ** i + j % 2 - 1) and j % 2 == 1:
                    height2 = 0
                else:
                    height2 = heightmap[j * stride, stride * (((j + 1) % 2) + 1) + k * radius]
                    cnt += 1

                height = float(height1 + height2 + height3 + height4) / cnt + (np.random.rand() - 0.5) * noise
                heightmap[j * stride, ((j + 1) % 2) * stride + k * radius] = height

        noise /= noise_factor

    lowest_point = heightmap.min()
    highest_point = heightmap.max()

    if n > 1:
        for i in range(size):
            for j in range(4):
                heightmap[j, i] = lowest_point + j * 0.25 * (heightmap[j, i] - lowest_point)
                heightmap[size - j - 1, i] = lowest_point + j * 0.25 * (heightmap[size - j - 1, i] - lowest_point)
                heightmap[i, j] = lowest_point + j * 0.25 * (heightmap[i, j] - lowest_point)
                heightmap[i, size - j - 1] = lowest_point + j * 0.25 * (heightmap[i, size - j - 1] - lowest_point)

    return heightmap
