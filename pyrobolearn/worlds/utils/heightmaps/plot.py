#!/usr/bin/env python
"""Plot 2D and 3D heightmap.
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


def plot_heightmap(heightmap, title='', block=True, max_height=None):
    """
    Plot the given heightmap.

    Args:
        heightmap (np.array[height, width]): 2D heightmap.
        title (str): title of the plot.
        block (bool): if we should block when showing the plot.
        max_height (None, int, float): max height (z-limit).
    """
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
    if max_height is not None:
        ax.set_zlim(0, max_height)

    plt.show(block=block)


def save_heightmap(heightmap, filename='heightmap.bmp'):
    """
    Save the heightmap as an image.

    Args:
        heightmap (np.array[height, width]): 2D heightmap.
        filename (str): filename to save the image.
    """
    min_height = heightmap.min()
    max_height = heightmap.max()
    height, width = heightmap.shape

    # create heightmap image
    img = Image.new('RGB', (height, width), "black")
    pixels = img.load()

    dist = (max_height - min_height)
    middle_point = min_height + dist / 2.
    for i in range(height):
        for j in range(width):
            if heightmap[i, j] > middle_point:
                pixels[i, j] = (0, int(255 * (heightmap[i, j] - middle_point) / (dist / 2.)), 0)
            else:
                pixels[i, j] = (10, 10, 200)

    # save image
    img.save(filename)
