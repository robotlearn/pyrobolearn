
import numpy as np
try:
    from mayavi import mlab
except ImportError as e:
    raise ImportError(repr(e) + '\nTry to install Mayavi: pip install mayavi')
try:
    import gdal
except ImportError as e:
    raise ImportError(repr(e) + '\nTry to install gdal: pip install gdal')

import subprocess
import fileinput
import sys
import os
import scipy.interpolate


def recenter(coords):
    """
    Recenter the data.

    Args:
        coords (list of np.array[N], np.array[N]): coordinate(s) to recenter

    Returns:
        list of np.array[N], np.array[N]: recentered coordinate(s)
    """
    if isinstance(coords, (list, tuple)) or len(coords.shape) > 1:
        centered_coords = []
        for coord in coords:
            c_min, c_max = coord.min(), coord.max()
            c_center = c_min + (c_max - c_min) / 2.
            centered_coord = coord - c_center
            centered_coords.append(centered_coord)
        return np.array(centered_coords)

    c_min, c_max = coords.min(), coords.max()
    c_center = c_min + (c_max - c_min) / 2.
    return (coords - c_center)


def createMesh(x, y, z, filename=None, show=False, center=True):
    """
    Create mesh from x,y,z arrays, and save it in the obj format.

    Args:
        x (float[N,M]): 2D array representing the x coordinates for the mesh
        y (float[N,M]): 2D array representing the y coordinates for the mesh
        z (float[N,M]): 2D array representing the z coordinates for the mesh
        filename (str, None): filename to save the mesh. If None, it won't save it.
        show (bool): if True, it will show the mesh using `mayavi.mlab`.
        center (bool): if True, it will center the mesh

    Examples:
        # create ellipsoid
        import numpy as np

        a,b,c,n = 2., 1., 1., 100
        theta, phi = np.meshgrid(np.linspace(-np.pi/2, np.pi/2, n), np.linspace(-np.pi, np.pi, n))

        x, y, z = a * np.cos(theta) * np.cos(phi), b * np.cos(theta) * np.sin(phi), c * np.sin(theta)

        createMesh(x, y, z, show=True)
    """
    #if not (isinstance(x, np.ndarray) and isinstance(y, np.ndarray) and isinstance(z, np.ndarray)):
    #    raise TypeError("Expecting x, y, and z to be numpy arrays")

    if isinstance(x, list) and isinstance(y, list) and isinstance(z, list):
        # create several 3D mesh
        for i,j,k in zip(x,y,z):
            # if we need to recenter
            if center:
                i,j,k = recenter([i,j,k])
            mlab.mesh(i,j,k)
    else:
        # if we need to recenter the data
        if center:
            x,y,z = recenter([x,y,z])

        # create 3D mesh
        mlab.mesh(x,y,z)

    # save mesh
    if filename is not None:
        if filename[-4:] == '.obj': # This is because the .obj saved by Mayavi is not correct (see in Meshlab)
            x3dfile = filename[:-4] + '.x3d'
            mlab.savefig(x3dfile)
            convertX3dToObj(x3dfile, removeX3d=True)
        else:
            mlab.savefig(filename)

    # show / close
    if show:
        mlab.show()
    else:
        mlab.close()


def createSurfMesh(surface, filename=None, show=False, subsample=None, interpolate_fct='multiquadric',
                   lower_bound=None, upper_bound=None, dtype=None):
    """
    Create surface (heightmap) mesh, and save it in the obj format.

    Args:
        surface (float[M,N], str): 2D array where each value represents the height. If it is a string, it is assumed
            that is the path to a file .tif, .geotiff or an image (.png, .jpg, etc). It will be opened using the
            `gdal` library.
        filename (str, None): filename to save the mesh. If None, it won't save it.
        show (bool): if True, it will show the mesh using `mayavi.mlab`.
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
        lower_bound (int, float, None): lower bound; each value in the heightmap will be higher than or equal to
            this bound
        upper_bound (int, float, None): upper bound; each value in the heightmap will be lower than or equal to
            this bound
        dtype (np.int, np.float, None): type of the returned array for the heightmap

    Examples:
        # create heightmap
        import numpy as np

        height = np.random.rand(100,100) # in meters
        createSurfMesh(height, show=True)
    """
    if isinstance(surface, str):
        from utils.heightmap_generator import heightmap_gdal
        surface = heightmap_gdal(surface, subsample=subsample, interpolate_fct=interpolate_fct,
                                 lower_bound=lower_bound, upper_bound=upper_bound, dtype=dtype)

    if not isinstance(surface, np.ndarray):
        raise TypeError("Expecting a 2D numpy array")
    if len(surface.shape) != 2:
        raise ValueError("Expecting a 2D numpy array")

    # create surface mesh
    mlab.surf(surface)

    # save mesh
    if filename is not None:
        if filename[-4:] == '.obj':  # This is because the .obj saved by Mayavi is not correct (see in Meshlab)
            x3dfile = filename[:-4] + '.x3d'
            mlab.savefig(x3dfile)
            convertX3dToObj(x3dfile, removeX3d=True)
        else:
            mlab.savefig(filename)

    # show / close
    if show:
        mlab.show()
    else:
        mlab.close()


def create3DMesh(heightmap, x=None, y=None, depth_level=1., filename=None, show=False, subsample=None,
                 interpolate_fct='multiquadric', lower_bound=None, upper_bound=None, dtype=None, center=True):
    """
    Create 3D mesh from heightmap (which can be a 2D array or an image (.tif, .png, .jpg, etc), and save it in
    the obj format.

    Args:
        heightmap (float[M,N], str): 2D array where each value represents the height. If it is a string, it is assumed
            that is the path to a file .tif, .geotiff or an image (.png, .jpg, etc). It will be opened using the
            `gdal` library.
        x (float[M,N], None): 2D array where each value represents the x position (array from meshgrid). If None, it
            will generate it automatically from the heightmap. If `heightmap` is a string, this `x` won't be taken
            into account.
        y (float[M,N], None): 2D array where each value represents the y position (array from meshgrid). If None, it
            will generate it automatically from the heightmap. If `heightmap` is a string, this `y` won't be taken
            into account.
        depth_level (float): the depth will be the minimum depth of the heightmap minus the given depth_level.
        filename (str, None): filename to save the mesh. If None, it won't save it.
        show (bool): if True, it will show the mesh using `mayavi.mlab`.
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
        lower_bound (int, float, None): lower bound; each value in the heightmap will be higher than or equal to
            this bound
        upper_bound (int, float, None): upper bound; each value in the heightmap will be lower than or equal to
            this bound
        dtype (np.int, np.float, None): type of the returned array for the heightmap
        center (bool): if True, it will center the mesh

    Examples:
        import numpy as np

        height = np.random.rand(100,100) # in meters
        create3DMesh(height, show=True)
    """
    if isinstance(heightmap, str):
        # load data (raster)
        data = gdal.Open(heightmap)

        gt = data.GetGeoTransform()
        # gt is an array with:
        # 0 = x-coordinate of the upper-left corner of the upper-left pixel
        # 1 = width of a pixel
        # 2 = row rotation (typically zero)
        # 3 = y-coordinate of the of the upper-left corner of the upper-left pixel
        # 4 = column rotation (typically zero)
        # 5 = height of a pixel (typically negative)

        # numpy array of shape: (channel, height, width)
        #dem = data.ReadAsArray()

        # get elevation values (i.e. height values) with shape (height, width)
        band = data.GetRasterBand(1)
        band = band.ReadAsArray()

        # generate coordinates (x,y,z)
        xres, yres = gt[1], gt[5]
        width, height = data.RasterXSize * xres, data.RasterYSize * yres
        xmin = gt[0] + xres * 0.5
        xmax = xmin + width - xres * 0.5
        ymin = gt[3] + yres * 0.5
        ymax = ymin + height - yres * 0.5

        x, y = np.arange(xmin, xmax, xres), np.arange(ymin, ymax, yres)
        x, y = np.meshgrid(x, y)
        z = band

        # if we need to subsample, it will smooth the heightmap
        if isinstance(subsample, int) and subsample > 0:
            height, width = z.shape
            idx_x = np.linspace(0, height - 1, subsample, dtype=np.int)
            idx_y = np.linspace(0, width - 1, subsample, dtype=np.int)
            idx_x, idx_y = np.meshgrid(idx_x, idx_y)
            rbf = scipy.interpolate.Rbf(x[idx_x, idx_y], y[idx_x, idx_y], z[idx_x, idx_y], function=interpolate_fct)
            # Nx, Ny = x.shape[0] / subsample, x.shape[1] / subsample
            # rbf = Rbf(x[::Nx, ::Ny], y[::Nx, ::Ny], z[::Nx, ::Ny], function=interpolate_fct)
            z = rbf(x, y)

        # make sure the values of the heightmap are between the bounds (in-place), and is the correct type
        if lower_bound and upper_bound:
            np.clip(z, lower_bound, upper_bound, z)
        elif lower_bound:
            np.clip(z, lower_bound, z.max(), z)
        elif upper_bound:
            np.clip(z, z.min(), upper_bound, z)
        if dtype:
            z.astype(dtype)

    else:
        # check the heightmap is a 2D array
        if not isinstance(heightmap, np.ndarray):
            raise TypeError("Expecting a 2D numpy array")
        if len(heightmap.shape) != 2:
            raise ValueError("Expecting a 2D numpy array")

        z = heightmap
        if x is None or y is None:
            height, width = z.shape
            x, y = np.meshgrid(np.arange(width), np.arange(height))

    # center the coordinates if specified
    if center:
        x,y = recenter([x,y])

    # create lower plane
    z0 = np.min(z) * np.ones(z.shape) - depth_level

    # create left, right, front, and back planes
    c1 = (np.vstack((x[0], x[0])), np.vstack((y[0], y[0])), np.vstack((z0[0], z[0])))
    c2 = (np.vstack((x[-1], x[-1])), np.vstack((y[-1], y[-1])), np.vstack((z0[-1], z[-1])))
    c3 = (np.vstack((x[:, 0], x[:, 0])), np.vstack((y[:, 0], y[:, 0])), np.vstack((z0[:, 0], z[:, 0])))
    c4 = (np.vstack((x[:, -1], x[:, -1])), np.vstack((y[:, -1], y[:, -1])), np.vstack((z0[:, -1], z[:, -1])))
    c = [c1, c2, c3, c4]

    # createMesh([x, x] + [i[0] for i in c], [y, y] + [i[1] for i in c], [z, z0] + [i[2] for i in c],
    #            filename=filename, show=show, center=False)
    createMesh([x, x] + [i[0] for i in c], [y, y] + [i[1] for i in c], [z, z0] + [i[2] for i in c],
               filename=filename, show=show, center=False)


def createURDFFromMesh(meshfile, filename, position=(0.,0.,0.), orientation=(0.,0.,0.), scale=(1.,1.,1.),
                       color=(1,1,1,1), texture=None, mass=0., inertia=(0.,0.,0.,0.,0.,0.),
                       lateral_friction=0.5, rolling_friction=0., spinning_friction=0., restitution=0.,
                       kp=None, kd=None): #, cfm=0., erf=0.):
    """
    Create a URDF file and insert the specified mesh inside.

    Args:
        meshfile (str): path to the mesh file
        filename (str): filename of the urdf
        position (float[3]): position of the mesh
        orientation (float[3]): orientation (roll, pitch, yaw) of the mesh
        scale (float[3]): scale factor in the x, y, z directions
        color (float[4]): RGBA color where rgb=(0,0,0) is for black, rgb=(1,1,1) is for white, and a=1 means opaque.
        texture (str, None): path to the texture to be applied to the object. If None, provided it will use the
            given color.
        mass (float): mass in kg
        inertia (float[6]): upper/lower triangle of the inertia matrix (read from left to right, top to bottom)
        lateral_friction (float): friction coefficient
        rolling_friction (float): rolling friction coefficient orthogonal to contact normal
        spinning_friction (float): spinning friction coefficient around contact normal
        kp (float, None): contact stiffness (useful to make surfaces soft). Set it to None/-1 if not using it.
        kd (float, None): contact damping (useful to make surfaces soft). Set it to None/-1 if not using it.
        #cfm: constraint force mixing
        #erp: error reduction parameter

    Returns:
        None

    References:
        - "ROS URDF Tutorial": http://wiki.ros.org/urdf/Tutorials
        - "URDF: Link": http://wiki.ros.org/urdf/XML/link
        - "Tutorial: Using a URDF in Gazebo": http://gazebosim.org/tutorials/?tut=ros_urdf
        - SDF format: http://sdformat.org/spec
    """
    def getStr(lst):
        return ' '.join([str(i) for i in lst])

    position = getStr(position)
    orientation = getStr(orientation)
    color = getStr(color)
    scale = getStr(scale)
    name = meshfile.split('/')[-1][:-4]
    ixx, ixy, ixz, iyy, iyz, izz = [str(i) for i in inertia]

    with open(filename, 'w') as f:
        f.write('<?xml version="0.0" ?>')
        f.write('<robot name="'+name+'">')
        f.write('\t<link name="base">')

        f.write('\t\t<contact>')
        f.write('\t\t\t<lateral_friction value="' + str(lateral_friction) + '"/>')
        f.write('\t\t\t<rolling_friction value="' + str(rolling_friction) + '"/>')
        f.write('\t\t\t<spinning_friction value="' + str(spinning_friction) + '"/>')
        f.write('\t\t\t<restitution value="' + str(restitution) + '"/>')
        if kp is not None:
            f.write('\t\t\t<stiffness value="' + str(kp) + '"/>')
        if kd is not None:
            f.write('\t\t\t<damping value="' + str(kd) + '"/>')
        # f.write('\t\t\t<contact_cfm value="' + str(cfm) + '"/>')
        # f.write('\t\t\t<contact_erp value="' + str(erp) + '"/>')
        # f.write('\t\t\t<inertia_scaling value="' + str(inertia_scaling) + '"/>')
        f.write('\t\t</contact>')

        f.write('\t\t<inertial>')
        f.write('\t\t\t<origin rpy="' + orientation + '" xyz="' + position + '"/>')
        f.write('\t\t\t<mass value="' + str(mass) + '"/>')
        f.write('\t\t\t<inertia ixx="'+str(ixx)+'" ixy="'+str(ixy)+'" ixz="'+str(ixz)+'" iyy="'+str(iyy)+'" iyz="'+
                str(iyz)+'" izz="'+str(izz)+'"/>')
        f.write('\t\t</inertial>')

        f.write('\t\t<visual>')
        f.write('\t\t\t<origin rpy="' + orientation + '" xyz="' + position + '"/>')
        f.write('\t\t\t<geometry>')
        f.write('\t\t\t\t<mesh filename="' + meshfile + '" scale="' + scale + '"/>')
        f.write('\t\t\t</geometry>')
        f.write('\t\t\t<material name="color">')
        if texture is not None:
            f.write('\t\t\t\t<texture filename="' + texture + '"/>')
        else:
            f.write('\t\t\t\t<color rgba="' + color + '"/>')
        f.write('\t\t\t</material>')
        f.write('\t\t</visual>')

        f.write('\t\t<collision>')
        f.write('\t\t\t<origin rpy="' + orientation + '" xyz="' + position + '"/>')
        f.write('\t\t\t<geometry>')
        f.write('\t\t\t\t<mesh filename="' + meshfile + '" scale="' + scale + '"/>')
        f.write('\t\t\t</geometry>')
        f.write('\t\t</collision>')

        f.write('\t</link>')
        f.write('</robot>')



def convertX3dToObj(filename, removeX3d=True):
    """
    Convert a .x3d into an .obj file.

    Warnings: This method use the `meshlabserver` bash command. Be sure that `meshlab` is installed on the computer.

    Args:
        filename (str): path to the .x3d file
        removeX3d (bool): True if it should remove the old .x3d file.

    Returns:
        None
    """
    obj_filename = filename[:-4] + '.obj'

    try:
        # convert mesh (check `meshlabserver` command for more info)
        subprocess.call(['meshlabserver', '-i', filename, '-o', obj_filename])  # same as calling Popen(...).wait()

        # replace all commas by dots
        for line in fileinput.input(obj_filename, inplace=True):
            line = line.replace(',', '.')
            sys.stdout.write(line)

        # remove the old .x3d file if specified
        if removeX3d:
            subprocess.call(['rm', filename])
    except OSError as e:
        if e.errno == os.errno.ENOENT:
            raise OSError(
                "The command `meshlabserver` is not installed on this system. Verify that meshlab is installed.")
        else:
            raise OSError("Error while running the command `meshlabserver`: {}".format(e))


def convertMesh(fromFilename, toFilename, removeFile=True):
    """
    Convert the given file containing the original mesh to the other specified format.
    The available formats are the ones supported by `meshlab`.

    Args:
        fromFilename (str): filename of the mesh to convert
        toFilename (str): filename of the converted mesh
        removeFile (bool): True if the previous file should be deleted

    Returns:
        None
    """
    try:
        # convert mesh (check `meshlabserver` command for more info)
        subprocess.call(['meshlabserver', '-i', fromFilename, '-o', toFilename])  # same as calling Popen(...).wait()

        # replace all commas by dots
        for line in fileinput.input(toFilename, inplace=True):
            line = line.replace(',', '.')
            sys.stdout.write(line)

        # remove the old .x3d file if specified
        if removeFile:
            subprocess.call(['rm', fromFilename])
    except OSError as e:
        if e.errno == os.errno.ENOENT:
            raise OSError(
                "The command `meshlabserver` is not installed on this system. Verify that meshlab is installed.")
        else:
            raise OSError("Error while running the command `meshlabserver`: {}".format(e))


def readObjFile(filename):
    r"""
    Read an .obj file and returns the whole file, as well as the list of vertices, and faces.

    Args:
        filename (str): path to the obj file

    Returns:
        list[str]: each line in the file
        np.array[N,3]: list of vertices, where each vertex is a 3D position
        list[list[M]]: list of faces, where each face is a list of vertex ids which composed the face. Note that the
            first vertex id starts from 0 and not 1 like in the file.
    """
    data, vertices, faces = [], [], []

    with open(filename) as f:
        for i, line in enumerate(f):
            data.append(line)
            words = line.split()
            if len(words) > 0:
                if words[0] == 'v':  # vertex
                    if len(words) > 3:
                        x, y, z = words[1:4]
                        vertices.append(np.array([float(x), float(y), float(z)]))
                elif words[0] == 'f':  # face
                    face = []
                    for word in words[1:]:
                        numbers = word.split('//')
                        if len(numbers) > 0:
                            face.append(int(numbers[0]) - 1)
                    faces.append(face)

    vertices = np.array(vertices)
    return data, vertices, faces


def flipFaceNormalsInObj(filename):
    """
    Flip all the face normals in .obj file.

    Args:
        filename (str): path to the obj file
    """
    # read (load) all the file
    with open(filename) as f:
        data = f.readlines()

    # flip the faces
    for i in range(len(data)):
        words = data[i].split()
        if len(words) > 0:
            if words[0] == 'f':  # face
                data[i] = words[0] + ' ' + words[-1] + ' ' + words[-2] + ' ' + words[-3] + '\n'

    # rewrite the obj file
    with open(filename, 'w') as f:
        f.writelines(data)


def flipFaceNormalsForConvexObj(filename, outward=True):
    """
    Flip the face normals for convex objects, and rewrite the obj file

    Args:
        filename (str): the path to the obj file
        outward (bool): if the face normals should point outward. If False, they will be flipped such that they point
            inward the object.
    """
    # read the obj file
    data, vertices, faces = readObjFile(filename)

    # compute the center of the object
    center = np.mean(vertices, axis=0)
    print('Center of object: {}'.format(center))

    # flip the faces that points inward or outward
    v = vertices
    face_id = 0
    for i in range(len(data)):
        words = data[i].split()
        if len(words) > 0:
            if words[0] == 'f':  # face
                # compute the center of the face
                face = faces[face_id]
                face_center = np.mean([v[face[i]] for i in range(len(face))], axis=0)
                print('Face id: {}'.format(face_id))
                print('Face center: {}'.format(face_center))

                # compute the surface vector that goes from the center of the object to the face center
                vector = face_center - center

                # compute the normal vector of the face
                normal = np.cross( (v[face[2]] - v[face[1]]), (v[face[0]] - v[face[1]]) )

                # compute the dot product between the normal and the surface vector
                direction = np.dot(vector, normal)

                print('direction: {}'.format(direction))

                # flip the faces that need to be flipped
                if (direction > 0 and not outward) or (direction < 0 and outward):
                    data[i] = words[0] + ' ' + words[-1] + ' ' + words[-2] + ' ' + words[-3] + '\n'

                # increment face id
                face_id +=1

    # rewrite the obj file
    with open(filename, 'w') as f:
        f.writelines(data)


def flipFaceNormalsForExpandedObj(filename, expanded_filename, outward=True, remove_expanded_file=False):
    r"""
    By comparing the expanded object with the original object, we can compute efficiently the normal vector to each
    face such that it points outward. Then comparing the direction of these obtained normal vectors with the ones
    computed for the original faces, we can correct them.

    Args:
        filename (str): the path to the original obj file
        expanded_filename (str): the path to the expanded obj file; the file that contains the same object but which
            has been expanded in every dimension.
        outward (bool): if the face normals should point outward. If False, they will be flipped such that they point
            inward the object.
    """
    # read the obj files
    d1, v1, f1 = readObjFile(filename)
    d2, v2, f2 = readObjFile(expanded_filename)

    # check the size of the obj files (they have to match)
    if len(v1) != len(v2) or len(f1) != len(f2):
        raise ValueError("Expecting to have the same number of vertices and faces in each file: "
                         "v1={}, v2={}, f1={}, f2={}".format(len(v1), len(v2), len(f1), len(f2)))
    if len(d1) != len(d2):
        raise ValueError("Expecting the files to have the same size, but instead we have {} and {}".format(len(d1),
                                                                                                           len(d2)))

    # flip the faces that points inward or outward
    face_id = 0
    for i in range(len(d1)):
        words = d1[i].split()
        if len(words) > 0:
            if words[0] == 'f':  # face
                # compute the center of the faces
                face1, face2 = f1[face_id], f2[face_id]
                face1_center = np.mean([v1[face1[i]] for i in range(len(face1))], axis=0)
                face2_center = np.mean([v2[face2[i]] for i in range(len(face2))], axis=0)

                # compute the surface vector that goes from the original face to the expanded one
                vector = face2_center - face1_center

                # compute the normal vector of the face
                normal = np.cross((v1[face1[2]] - v1[face1[1]]), (v1[face1[0]] - v1[face1[1]]))

                # compute the dot product between the normal and the surface vector
                direction = np.dot(vector, normal)

                # flip the faces that need to be flipped
                if (direction < 0 and not outward) or (direction > 0 and outward):
                    d1[i] = words[0] + ' ' + words[-1] + ' ' + words[-2] + ' ' + words[-3] + '\n'

                # increment face id
                face_id += 1

    # rewrite the obj file
    with open(filename, 'w') as f:
        f.writelines(d1)

    # remove the expanded file
    if remove_expanded_file:
        os.remove(expanded_filename)


# Test
if __name__ == '__main__':

    # 1. create 3D ellipsoid mesh (see `https://en.wikipedia.org/wiki/Ellipsoid` for more info)
    a,b,c,n = 1., 0.5, 0.5, 50
    #a,b,c,n = .5, .5, .5, 37
    theta, phi = np.meshgrid(np.linspace(-np.pi/2, np.pi/2, n), np.linspace(-np.pi, np.pi, n))

    x = a * np.cos(theta) * np.cos(phi)
    y = b * np.cos(theta) * np.sin(phi)
    z = c * np.sin(theta)

    createMesh(x, y, z, show=True)
    #createMesh(x, y, z, filename='ellipsoid.obj', show=True)

    # 2. create heightmap mesh
    height = np.random.rand(100,100) # in meters
    createSurfMesh(height, show=True)

    # 3. create right triangular prism
    x = np.array([[-0.5,-0.5],
                  [0.5, 0.5],
                  [-0.5,-0.5],
                  [-0.5,-0.5],
                  [-0.5,0.5],
                  [0.5,-0.5],
                  [-0.5, 0.5],
                  [0.5, -0.5]])
    y = np.array([[-0.5,0.5],
                  [-0.5,0.5],
                  [-0.5,0.5],
                  [-0.5,0.5],
                  [-0.5,-0.5],
                  [-0.5,-0.5],
                  [0.5, 0.5],
                  [0.5, 0.5]])
    z = np.array([[0.,0.],
                  [0.,0.],
                  [1.,1.],
                  [0.,0.],
                  [0.,0.],
                  [0.,1.],
                  [0., 0.],
                  [0., 1.]])

    #createMesh(x, y, z, show=True)
    createMesh(x, y, z, filename='right_triangular_prism.obj', show=True)
    flipFaceNormalsForConvexObj('right_triangular_prism.obj', outward=True)

    exit()

    # 4. create cone
    radius, height, n = 0.5, 1., 50
    [r, theta] = np.meshgrid((radius, 0.), np.linspace(0, 2*np.pi, n))
    [h, theta] = np.meshgrid((0., height), np.linspace(0, 2*np.pi, n))
    x, y, z = r * np.cos(theta), r * np.sin(theta), h
    # close the cone at the bottom
    [r, theta] =  np.meshgrid((0., radius), np.linspace(0, 2*np.pi, n))
    x = np.vstack((x, r * np.cos(theta)))
    y = np.vstack((y, r * np.sin(theta)))
    z = np.vstack((z, np.zeros(r.shape)))

    createMesh(x, y, z, show=True)
    #createMesh(x, y, z, filename='cone.obj', show=True)

    # 5. create 3D heightmap
    dx, dy, dz = 5., 5., 0.01
    x,y = np.meshgrid(np.linspace(-dx, dx, int(2*dx)), np.linspace(-dy, dy, int(2*dy)))
    z = np.random.rand(*x.shape) + dz

    # z0 = np.zeros(x.shape)
    #
    # w = np.dstack((x,y,z0,z)) # 2DX x 2DY x 4
    #
    # c1 = (np.vstack((x[0], x[0])), np.vstack((y[0],y[0])), np.vstack((z0[0],z[0])))
    # c2 = (np.vstack((x[-1], x[-1])), np.vstack((y[-1],y[-1])), np.vstack((z0[-1],z[-1])))
    # c3 = (np.vstack((x[:,0], x[:,0])), np.vstack((y[:,0], y[:,0])), np.vstack((z0[:,0], z[:,0])))
    # c4 = (np.vstack((x[:,-1], x[:,-1])), np.vstack((y[:,-1], y[:,-1])), np.vstack((z0[:,-1], z[:,-1])))
    # c = [c1,c2,c3,c4]
    #
    # createMesh([x,x]+[i[0] for i in c], [y,y]+[i[1] for i in c], [z,z0]+[i[2] for i in c], show=True)

    create3DMesh(z, x, y, dz, show=True)
