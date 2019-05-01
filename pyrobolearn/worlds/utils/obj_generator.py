#!/usr/bin/env python
"""OBJ generator.

Create an OBJ file from a heightmap (2D np.array).

References:
    [1] https://github.com/deltabrot/random-terrain-generator
"""

import numpy as np
import time


__author__ = ["Jamie Scollay", "Brian Delhaisse"]
# the code was originally written by Jamie Scollay
# it was then reviewed by Brian Delhaisse, notably with respect to the original code:
# - it has been cleaned; removed all the ";"
# - it has been optimized:
#     - it now uses numpy instead of math and lists, it uses `list.append` instead of adding strings, and then `join`
#     - it has been simplified.
# - comments have been added and a better documentation is provided
__credits__ = ["Jamie Scollay"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


def unit_vector(v):
    """Normalize the given vector.

    Args:
        v (np.array[3]): vector to normalize.

    Returns:
        np.array[3]: unit vector
    """
    magnitude = np.linalg.norm(v)
    if magnitude == 0:
        return v
    return v / magnitude


def unit_normal(v1, v2, v3):
    """Compute the unit normal between two vectors: (v2-v1) and (v3 - v1).

    Args:
        v1 (np.array[3]): 3d common point between the two vectors.
        v2 (np.array[3]): 3d point for 1st vector.
        v3 (np.array[3]): 3d point for 2nd vector.

    Returns:
        np.array[3]: unit vector
    """
    return unit_vector(np.cross(v2 - v1, v3 - v1))


def create_hexagonal_terrain(heightmap, scale=1., tile=True, min_height=0., smooth=True, verbose=True, verbose_rate=1):
    """
    Create the terrain with hexagonal tiles from the heightmap, and return the vertices, textures, normals and faces
    to create an OBJ file.

    Args:
        heightmap (np.array[height, width]): 2D square heightmap
        scale (float): scaling factor.
        tile (bool): if True, it will create a tile texture.
        min_height (float): minimum height.
        smooth (bool): if the normals should be smooth.
        verbose (bool): if True, it will output information about the creation of the terrain.
        verbose_rate (int): if :attr:`verbose` is True, it will output

    Returns:
        list: vertices (for OBJ)
        list: textures (for OBJ)
        list: (smooth) normals (for OBJ)
        list: faces (for OBJ)

    References:
        [1] Wavefront .obj file (Wikipedia): https://en.wikipedia.org/wiki/Wavefront_.obj_file
    """
    # The heightmap contains the height for each point on the map. If you have 3 points, then you have two
    # tiles/segments. Number of segments = number of square tiles in rows or columns.
    height, width = heightmap.shape
    num_y_tiles, num_x_tiles = height - 1, width - 1

    scale = float(scale)
    vertices = []
    vertices_obj = []
    textures_obj = []
    normals_obj = []
    faces_obj = []

    prevent_output = False

    if tile:
        textures_obj = [[0, 0], [0, 1], [1, 0], [1, 1]]

    for i in range(num_y_tiles):

        if i == num_y_tiles - 1:
            tmp_vertices_obj = []

        for j in range(num_x_tiles):

            if not smooth:
                tmp = 2 * (i * num_x_tiles + j)
                # add 2 faces each one composed of 3 vertices/textures/normals (with the format: vertex/texture/normal)
                faces_obj.append(str(i * width + j + 1) + '/' + str(1) + '/' + str(tmp + 1) + ' ' +
                                 str(i * width + j + 2) + '/' + str(2) + '/' + str(tmp + 1) + ' ' +
                                 str((i + 1) * width + j + 1) + '/' + str(3) + '/' + str(tmp + 1))
                faces_obj.append(str((i + 1) * width + j + 1) + '/' + str(3) + '/' + str(tmp + 2) + ' ' +
                                 str(i * width + j + 2) + '/' + str(2) + '/' + str(tmp + 2) + ' ' +
                                 str((i + 1) * width + j + 2) + '/' + str(4) + '/' + str(tmp + 2))

            else:
                # add 2 faces each one composed of 3 vertices/textures/normals (with the format: vertex/texture/normal)
                faces_obj.append(str(i * width + j + 1) + '/' + str(1) + '/' + str(i * width + j + 1) + ' ' +
                                 str(i * width + j + 2) + '/' + str(2) + '/' + str(i * width + j + 2) + ' ' +
                                 str((i + 1) * width + j + 1) + '/' + str(3) + '/' + str((i + 1) * width + j + 1))
                faces_obj.append(str((i + 1) * width + j + 1) + '/' + str(3) + '/' + str((i + 1)*width + j + 1) + ' ' +
                                 str(i * width + j + 2) + '/' + str(2) + '/' + str(i * width + j + 2) + ' ' +
                                 str((i + 1) * width + j + 2) + '/' + str(4) + '/' + str((i + 1) * width + j + 2))

            # T1
            half_scale = scale / 2.
            scale_tile = scale / num_x_tiles

            # add vertex [x,y,z] for the obj file
            vertices_obj.append([-half_scale + i * scale_tile,
                                 heightmap[i, j],
                                 -half_scale + j * scale_tile])

            if j == num_x_tiles - 1:
                vertices_obj.append([-half_scale + i * scale_tile,
                                     heightmap[i, j+1],
                                     -half_scale + (j+1) * scale_tile])

            if i == num_y_tiles - 1:
                tmp_vertices_obj.append([-half_scale + (i+1) * scale_tile,
                                         heightmap[i+1, j],
                                         -half_scale + j * scale_tile])
                if j == num_x_tiles - 1:
                    tmp_vertices_obj.append([-half_scale + (i+1) * scale_tile,
                                             heightmap[i+1, j+1],
                                             -half_scale + (j+1) * scale_tile])

            # T1: add 3 vertices [x,y,z] (that are used to compute the vertex normal)
            vertices.append(np.array([-half_scale + i * scale_tile,
                                      heightmap[i, j],
                                      -half_scale + j * scale_tile]))
            vertices.append(np.array([-half_scale + i * scale_tile,
                                      heightmap[i, j + 1],
                                      -half_scale + (j+1) * scale_tile]))
            vertices.append(np.array([-half_scale + (i+1) * scale_tile,
                                      heightmap[i + 1, j],
                                      -half_scale + j * scale_tile]))
            #            else:
            #                textures.append([i/segment, j/segment])
            #                textures.append([i/segment, (j+1)/segment])
            #                textures.append([(i+1)/segment, j/segment])

            # compute vertex normal [x,y,z] based on last 3 vertices
            normal = unit_normal(vertices[-3], vertices[-2], vertices[-1])
            normals_obj.append(normal)

            # T2: add 3 vertices [x,y,z] (that are used to compute the vertex normal)
            vertices.append(np.array([-half_scale + (i+1) * scale_tile,
                                      heightmap[i+1, j],
                                      -half_scale + j * scale_tile]))
            vertices.append(np.array([-half_scale + i * scale_tile,
                                      heightmap[i, j+1],
                                      -half_scale + (j+1) * scale_tile]))
            vertices.append(np.array([-half_scale + (i+1) * scale_tile,
                                      heightmap[i+1, j+1],
                                      -half_scale + (j+1) * scale_tile]))

            #            else:
            #                textures.append([(i+1)/segment, j/segment])
            #                textures.append([i/segment, (j+1)/segment])
            #                textures.append([(i+1)/segment, (j+1)/segment])

            # compute vertex normal [x,y,z] based on last 3 vertices
            normal = unit_normal(vertices[-3], vertices[-2], vertices[-1])
            normals_obj.append(normal)

            # print information if specified
            if verbose:
                if (time.time() % verbose_rate) < 0.05 and not prevent_output:
                    print("{}/{} tiles".format(i * num_x_tiles + j, num_x_tiles * num_y_tiles))
                    prevent_output = True
                elif time.time() % verbose_rate > 0.05:
                    prevent_output = False

    vertices_obj += tmp_vertices_obj

    # if we don't have to smooth the normals
    if not smooth:
        return [vertices_obj, textures_obj, normals_obj, faces_obj]

    # smooth the normals using 6 normals
    smooth_normals = []

    for i in range(num_y_tiles):
        tmp_smooth_normals = []

        for j in range(num_x_tiles):
            if j > 0:
                norm0 = normals_obj[(i * num_x_tiles + (j - 1) * 2)]
                norm1 = normals_obj[(i * num_x_tiles + (j - 1) * 2) + 1]
            else:
                norm0 = np.zeros(3)
                norm1 = np.zeros(3)

            norm2 = normals_obj[((i * num_x_tiles + j) * 2)]

            if i > 0 and j > 0:
                norm3 = normals_obj[(((i - 1) * num_x_tiles + (j - 1)) * 2) + 1]
            else:
                norm3 = np.zeros(3)

            if i > 0:
                norm4 = normals_obj[(((i - 1) * num_x_tiles + j) * 2)]
                norm5 = normals_obj[(((i - 1) * num_x_tiles + j) * 2) + 1]
            else:
                norm4 = np.zeros(3)
                norm5 = np.zeros(3)

            smooth_normals.append(unit_vector(norm0 + norm1 + norm2 + norm3 + norm4 + norm5))

            if j == num_x_tiles - 1:
                norm0 = normals_obj[((i * num_x_tiles + (j - 1)) * 2)]
                norm1 = normals_obj[((i * num_x_tiles + (j - 1)) * 2) + 1]
                if i > 0:
                    norm2 = normals_obj[(((i - 1) * num_x_tiles + (j - 1)) * 2) + 1]
                else:
                    norm2 = np.zeros(3)
                smooth_normals.append(unit_vector(norm0 + norm1 + norm2))

            if i == num_y_tiles - 1:
                if j > 0:
                    norm0 = normals_obj[(((i - 1) * num_x_tiles + (j - 1)) * 2) + 1]
                else:
                    norm0 = np.zeros(3)

                norm1 = normals_obj[(((i - 1) * num_x_tiles + j) * 2)]
                norm2 = normals_obj[(((i - 1) * num_x_tiles + j) * 2) + 1]
                tmp_smooth_normals.append(unit_vector(norm0 + norm1 + norm2))

                if j == num_x_tiles - 1:
                    norm0 = normals_obj[(((i - 1) * num_x_tiles + (j - 1)) * 2) + 1]
                    tmp_smooth_normals.append(norm0)

    smooth_normals += tmp_smooth_normals

    return [vertices_obj, textures_obj, smooth_normals, faces_obj]


def create_obj(vertices, textures, normals, faces, filename=None):
    """
    Create content of the OBJ file given the vertices, textures, normals, and faces.

    Args:
        vertices (list): list of vertices (for each corner of a triangular mesh)
        textures (list): list of textures
        normals (list): list of normals
        faces (list): list of faces
        filename (None, str): if a string is provided, it will save the OBJ file in the given file path.

    Returns:
        str: content of the OBJ file.

    References:
        [1] Wavefront .obj file (Wikipedia): https://en.wikipedia.org/wiki/Wavefront_.obj_file
    """

    # create obj list
    obj = []

    # add vertices
    for v in vertices:
        obj.append("v " + str(v[0]) + " " + str(v[1]) + " " + str(v[2]))

    # add textures
    for t in textures:
        obj.append("vt " + str(t[0]) + " " + str(t[1]))

    # add normals
    for n in normals:
        obj.append("vn " + str(n[0]) + " " + str(n[1]) + " " + str(n[2]))

    # add faces
    for f in faces:
        obj.append("f " + f)

    # create document
    obj = '\n'.join(obj)

    # create file if specified
    if filename is not None:
        with open(filename, "w+") as f:
            f.write(obj)
    return obj


def create_obj_from_heightmap(heightmap, scale=1., tile=True, min_height=0., smooth=True, verbose=True, verbose_rate=1,
                              filename=None):
    """
    Create an OBJ file from the given 2D heightmap.

    Args:
        heightmap (np.array[height, width]): 2D square heightmap
        scale (float): scaling factor.
        tile (bool): if True, it will create a tile texture.
        min_height (float): minimum height.
        smooth (bool): if the normals should be smooth.
        verbose (bool): if True, it will output information about the creation of the terrain.
        verbose_rate (int): if :attr:`verbose` is True, it will output
        filename (None, str): if a string is provided, it will save the OBJ file in the given file path.

    Returns:
        str: content of the OBJ file.

    References:
        [1] Wavefront .obj file (Wikipedia): https://en.wikipedia.org/wiki/Wavefront_.obj_file
    """
    # create vertices, textures, normals, and faces
    terrain = create_hexagonal_terrain(heightmap, scale, tile, min_height, smooth, verbose, verbose_rate)

    # create obj based on above information
    obj = create_obj(vertices=terrain[0], textures=terrain[1], normals=terrain[2], faces=terrain[3],
                     filename=filename)

    return obj
