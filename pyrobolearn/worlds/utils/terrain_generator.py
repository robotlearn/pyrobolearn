#!/usr/bin/env python
"""Random terrain generator using the Diamond square algorithm.

It generates a random heightmap / terrain using the diamond square algorithm, and outputs an OBJ file.

The code comes from [1], and has been optimized.

References:
    [1] https://github.com/deltabrot/random-terrain-generator
"""

import sys
import os
import time
from PIL import Image
import numpy as np


__author__ = ["Jamie Scollay", "Brian Delhaisse"]
# the code was originally written by Jamie Scollay
# it was then reviewed by Brian Delhaisse, notably with respect to the original code:
# - it has been cleaned; removed all the ";"
# - it has been optimized: it now uses numpy instead of math and lists, it uses list.append instead of adding strings
# - it is now better documented.
__credits__ = ["Jamie Scollay"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


def unit_vector(v):
    """Normalize the given vector.
    
    Args:
        v (np.float[3]): vector to normalize.

    Returns:
        np.float[3]: unit vector
    """
    magnitude = np.linalg.norm(v)
    if magnitude == 0:
        return v
    return v / magnitude


def unit_normal(v1, v2, v3):
    """Compute the unit normal between two vectors: (v2-v1) and (v3 - v1).

    Args:
        v1 (np.float[3]): 3d common point between the two vectors.
        v2 (np.float[3]): 3d point for 1st vector.
        v3 (np.float[3]): 3d point for 2nd vector.

    Returns:
        np.float[3]: unit vector
    """
    return unit_vector(np.cross(v2 - v1, v3 - v1))


def display_loading(count, finish, message):
    """
    Display loading message.

    Args:
        count (int): each row of the surface.
        finish (int): surface.
        message (str): message to print.
    """
    print(message + ": " + str(count) + "/" + str(finish))


def diamond_square_heightmap(n, max_height, jitter, jitter_factor):
    """
    Create a 2D square heightmap using the diamond square algorithm [1]

    Args:
        n (int): used to create a square array of width and height of 2**n + 1. It also specifies the number of
            diamond and square steps.
        max_height (float): maximum height
        jitter (float): magnitude of the noise added to the computed height.
        jitter_factor (float): after each step, the jitter is divided by the given factor.

    Returns:
        np.float[size, size]: 2D square heightmap

    References:
        [1] Wikipedia: https://en.wikipedia.org/wiki/Diamond-square_algorithm
    """
    # compute the width and height size (i.e. size of the 2D square array/heightmap)
    size = int(2**n + 1)

    # create initial heightmap of width and height size
    heightmap = np.zeros((size, size))
    
    # assign a random height at each corner of the heightmap
    heightmap[0, 0] = np.random.rand() * max_height
    heightmap[0, size-1] = np.random.rand() * max_height
    heightmap[size-1, 0] = np.random.rand() * max_height
    heightmap[size-1, size-1] = np.random.rand() * max_height

    # for each diamond and square step
    for i in range(n):
        stride = int((size-1) / 2**(i+1))
        radius = int((size-1) / 2**i)

        #
        for j in range(2**i):
            for k in range(2**i):
                height = (heightmap[j*radius, k*radius] + heightmap[2*stride + j*radius, k*radius] +
                          heightmap[j*radius, 2*stride + k*radius] +
                          heightmap[2*stride + j*radius, 2*stride + k*radius]) / 4. + (np.random.rand()-0.5) * jitter
                heightmap[stride + j*radius, stride + k*radius] = height

        for j in range(2**(i+1) + 1):
            for k in range(2**i + j%2):
                cnt = 0
                if j == 0:
                    height1 = 0
                else: 
                    height1 = heightmap[(j-1) * stride, stride*((j+1)%2) + k*radius]
                    cnt += 1
                
                if k == 0 and j%2 == 1:
                    height4 = 0
                else:
                    height4 = heightmap[j * stride, stride*(((j+1)%2)-1) + k*radius]
                    cnt += 1

                if  j == 2**(i+1):
                    height3 = 0
                else:
                    height3 = heightmap[(j+1) * stride, stride*((j+1)%2) + k*radius]
                    cnt += 1

                if k == (2**i + j%2 - 1) and j%2 == 1:
                    height2 = 0
                else:
                    height2 = heightmap[j*stride, stride*(((j+1)%2)+1) + k*radius]
                    cnt += 1

                height = float(height1 + height2 + height3 + height4) / cnt + (np.random.rand()-0.5) * jitter
                heightmap[j*stride, ((j+1)%2) * stride + k*radius] = height
        
        jitter /= jitter_factor

    lowest_point = heightmap.min()
    highest_point = heightmap.max()
            
    if n > 1:
        for i in range(size):
            for j in range(4):
                heightmap[j, i] = lowest_point + j * 0.25 * (heightmap[j, i] - lowest_point) 
                heightmap[size-j-1, i] = lowest_point + j * 0.25 * (heightmap[size-j-1, i] - lowest_point)
                heightmap[i, j] = lowest_point + j * 0.25 * (heightmap[i, j] - lowest_point)
                heightmap[i, size-j-1] = lowest_point + j * 0.25 * (heightmap[i, size-j-1] - lowest_point)
    
    # create heightmap image
    img = Image.new('RGB', (size, size), "black")
    pixels = img.load()
    
    dist = (highest_point - lowest_point)
    middle_point = lowest_point + dist/2.
    for i in range(size):
        for j in range(size):
            if heightmap[i, j] > middle_point:
                pixels[i,j] = (0, int(255 * (heightmap[i, j] - middle_point) / (dist/2.) ), 0)
            else:
                pixels[i,j] = (10, 10, 200)

    # save image
    img.save("map.bmp")

    return heightmap


def create_hexagonal_terrain(segment, scale, tile, max_height, min_height, heightmap, smooth=True, verbose=True,
                             output_rate=1):
    vertices = []
    vertices_obj = []
    textures_obj = []
    normals_obj = []
    facesOBJ = []
    
    prevent_output = False

    if tile:
        textures_obj.append([0, 0])
        textures_obj.append([0, 1])
        textures_obj.append([1, 0])
        textures_obj.append([1, 1])
    
    for i in range(segment):
        if i == segment-1:
            tmp_vertices_obj = []
        
        for j in range(segment):
            if not smooth:
                tmp = 2 * (i*segment + j)
                facesOBJ.append(str(i*(segment+1) + j + 1) + '/' + str(1) + '/' + str(tmp + 1) + ' ' +\
                                str(i*(segment+1) + j + 2) + '/' + str(2) + '/' + str(tmp + 1) + ' ' +\
                                str((i+1)*(segment+1) + j + 1) + '/' + str(3) + '/' + str(tmp + 1) )
                facesOBJ.append(str((i+1)*(segment+1) + j + 1) + '/' + str(3) + '/' + str(tmp + 2) + ' ' +\
                                str(i*(segment+1) + j + 2) + '/' + str(2) + '/' + str(tmp + 2) + ' ' +\
                                str((i+1)*(segment+1) + j + 2) + '/' + str(4) + '/' + str(tmp + 2) )

            else:
                facesOBJ.append(str(i*(segment+1) + j + 1) + '/' + str(1) + '/' + str(i*(segment+1) + j + 1) + ' ' +
                                str(i*(segment+1) + j + 2) + '/' + str(2) + '/' + str(i*(segment+1) + j + 2) + ' ' +
                                str((i+1)*(segment+1) + j + 1) + '/' + str(3) + '/' + str((i+1)*(segment+1) + j + 1))
                facesOBJ.append(str((i+1)*(segment+1) + j + 1) + '/' + str(3) + '/' + str((i+1)*(segment+1) + j + 1) +
                                ' ' + str(i*(segment+1) + j + 2) + '/' + str(2) + '/' + str(i*(segment+1) + j + 2) +
                                ' ' + str((i+1)*(segment+1) + j + 2) + '/' + str(4) + '/' +
                                str((i+1)*(segment+1) + j + 2))
            
            #T1
            half_scale = scale/2
            scale_seg = scale/segment
            vertices_obj.append([-half_scale + i*scale_seg, heightmap[i, j], -half_scale + j*scale_seg])
            if j == segment-1:
                vertices_obj.append([-half_scale + i*scale_seg, heightmap[i, j+1], -half_scale + (j+1)*scale_seg])
            if i == segment-1:
                tmp_vertices_obj.append([-half_scale + (i+1)*scale_seg, heightmap[i+1, j], -half_scale + j*scale_seg])
                if j == segment-1:
                    tmp_vertices_obj.append([-half_scale + (i+1)*scale_seg, heightmap[i+1, j+1],
                                             -half_scale + (j+1)*scale_seg])

            vertices.append(np.array([-half_scale + i*scale_seg,
                                      heightmap[i, j],
                                      -half_scale + j*scale_seg]))
            vertices.append(np.array([-half_scale + i*scale_seg,
                                      heightmap[i, j+1],
                                      -half_scale + (j+1)*scale_seg]))
            vertices.append(np.array([-half_scale + (i+1)*scale_seg,
                                      heightmap[i+1, j],
                                      -half_scale + j*scale_seg]))
#            else:
#                textures.append([i/segment, j/segment])
#                textures.append([i/segment, (j+1)/segment])
#                textures.append([(i+1)/segment, j/segment])

            num_vertices = len(vertices)
            normal = unit_normal(vertices[num_vertices-3], vertices[num_vertices-2], vertices[num_vertices-1])
            normals_obj.append(normal)

            # T2
            vertices.append(np.array([-half_scale + (i+1)*scale_seg, heightmap[i+1, j], -half_scale + j*scale_seg]))
            vertices.append(np.array([-half_scale + i*scale_seg, heightmap[i, j+1], -half_scale + (j+1)*scale_seg]))
            vertices.append(np.array([-half_scale + (i+1)*scale_seg, heightmap[i+1, j+1], -half_scale + (j+1)*scale_seg]))

#            else:
#                textures.append([(i+1)/segment, j/segment])
#                textures.append([i/segment, (j+1)/segment])
#                textures.append([(i+1)/segment, (j+1)/segment])

            num_vertices = len(vertices)
            normal = unit_normal(vertices[num_vertices-3], vertices[num_vertices-2], vertices[num_vertices-1])
            normals_obj.append(normal)
            
            if verbose:
                if (time.time() % output_rate) < 0.05 and not prevent_output:
                    display_loading(i*segment + j, segment*segment, "TER | Segm")
                    prevent_output = True
                elif time.time() % output_rate > 0.05:
                    prevent_output = False

    # smooth the normals
    smooth_normals = []
    for i in range(segment):
        tmp_smooth_normals = []
        for j in range(segment):
            if j > 0:
                norm0 = normals_obj[(i*segment + (j-1)*2)]
                norm1 = normals_obj[(i*segment + (j-1)*2) + 1]
            else:
                norm0 = np.zeros(3)
                norm1 = np.zeros(3)
            
            norm2 = normals_obj[((i*segment + j)*2)]

            if i > 0 and j > 0:
                norm3 = normals_obj[(((i-1)*segment + (j-1))*2) + 1]
            else:
                norm3 = np.zeros(3)

            if i > 0:
                norm4 = normals_obj[(((i-1)*segment + j)*2)]
                norm5 = normals_obj[(((i-1)*segment + j)*2) + 1]
            else:
                norm4 = np.zeros(3)
                norm5 = np.zeros(3)
            
            smooth_normals.append(unit_vector(norm0 + norm1 + norm2 + norm3 + norm4 + norm5))

            if j == segment-1:
                norm0 = normals_obj[((i*segment + (j-1))*2)]
                norm1 = normals_obj[((i*segment + (j-1))*2) + 1]
                if i > 0:
                    norm2 = normals_obj[(((i-1)*segment + (j-1))*2) + 1]
                else:
                    norm2 = np.zeros(3)
                smooth_normals.append(unit_vector(norm0 + norm1 + norm2))

            if i == segment-1:
                if j > 0:
                    norm0 = normals_obj[(((i-1)*segment + (j-1))*2) + 1]
                else:
                    norm0 = np.zeros(3)

                norm1 = normals_obj[(((i-1)*segment + j)*2)]
                norm2 = normals_obj[(((i-1)*segment + j)*2) + 1]
                tmp_smooth_normals.append(unit_vector(norm0 + norm1 + norm2))

                if j == segment-1:
                    norm0 = normals_obj[(((i-1)*segment + (j-1))*2) + 1]
                    tmp_smooth_normals.append(norm0)
    
    smooth_normals += tmp_smooth_normals

    vertices_obj += tmp_vertices_obj
    if smooth:
        return [vertices_obj, textures_obj, smooth_normals, facesOBJ]
    return [vertices_obj, textures_obj, normals_obj, facesOBJ]


def create_obj(vertices, textures, normals, faces):
    """
    Create content of the OBJ file.

    Args:
        vertices (list): list of vertices (for each corner of a triangular mesh)
        textures (list): list of textures
        normals (list): list of normals
        faces (list): list of faces

    Returns:
        str: content of the OBJ file.
    """
    obj = []
    for v in vertices:
        obj.append("v " + str(v[0]) + " " + str(v[1]) + " " + str(v[2]))
    for t in textures:
        obj.append("vt " + str(t[0]) + " " + str(t[1]))
    for n in normals:
        obj.append("vn " + str(n[0]) + " " + str(n[1]) + " " + str(n[2]))
    for f in faces:
        obj.append("f " + f)
    return '\n'.join(obj)
                
                
segments = 8
scale = 600
tile = True
max_height = 75
min_height = 0
verbose = True
output_rate = 1
jitter = 40
jitter_factor = 1.5
smooth = True

for i in range(len(sys.argv)):
    if sys.argv[i] == '-s' or sys.argv[i] == '--segment':
        segments = int(sys.argv[i+1])
    if sys.argv[i] == '-z' or sys.argv[i] == '--scale':
        scale = float(sys.argv[i+1])
    if sys.argv[i] == '-m' or sys.argv[i] == '--min':
        min_height = float(sys.argv[i+1])
    if sys.argv[i] == '-x' or sys.argv[i] == '--max':
        max_height = float(sys.argv[i+1])
    if sys.argv[i] == '-v' or sys.argv[i] == '--verbose':
        verbose = bool(sys.argv[i+1])
    if sys.argv[i] == '-r' or sys.argv[i] == '--rate':
        output_rate = float(sys.argv[i+1])
    if sys.argv[i] == '-j' or sys.argv[i] == '--jitter':
        jitter = float(sys.argv[i+1])
    if sys.argv[i] == '-f' or sys.argv[i] == '--factor':
        jitter_factor = float(sys.argv[i+1])
    if sys.argv[i] == '-e' or sys.argv[i] == '--edges':
        smooth = bool(sys.argv[i+1])

# create heightmap, generate terrain from it, and obj mesh
print("Generating terrain")
start = time.time()

heightmap = diamond_square_heightmap(segments, max_height, jitter, jitter_factor)
terrain = create_hexagonal_terrain(2**segments, scale, tile, max_height, min_height, heightmap, smooth, verbose,
                                   output_rate)
OBJ = create_obj(terrain[0], terrain[1], terrain[2], terrain[3])

end = time.time()
print("Terrain generated in {:.2f} seconds.".format(end - start))

# save terrain
file = open("terrain.obj", "w+")
file.write(OBJ)
file.close()
