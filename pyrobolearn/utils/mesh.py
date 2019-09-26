# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""Provide functions to create, convert, and get information from meshes, using the `trimesh` and `pyassimp` libraries.

Note that some methods just wrap the methods / attributes provided by the `trimesh` library.

Warnings: the meshes have to be watertight.

References:
    - Pyassimp:
        - doc: http://www.assimp.org/index.php
        - github: https://github.com/assimp/assimp
    - Trimesh: https://github.com/mikedh/trimesh
    - Pymesh: https://pymesh.readthedocs.io/en/latest/user_guide.html
"""

import numpy as np

# import XML parser
import xml.etree.ElementTree as ET
from xml.dom import minidom  # to print in a pretty way the XML file

# import mesh related libraries
try:
    import trimesh  # processing triangular meshes
    from trimesh.exchange.export import export_mesh
    # import pymesh    # rapid prototyping platform focused on geometry processing
    import pyassimp  # library to import and export various 3d-model-formats
except ImportError as e:
    raise ImportError(str(e) + "\nTry to install `pymesh` and `pyassimp`: `pip install pymesh pyassimp`")


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2019, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


def convert_mesh(from_filename, to_filename, library='pyassimp', binary=False):
    """
    Convert the given file containing the original mesh to the other specified format using the `pyassimp` library.

    Args:
        from_filename (str): filename of the mesh to convert.
        to_filename (str): filename of the converted mesh.
        library (str): library to use to convert the meshes. Select between 'pyassimp' and 'trimesh'.
        binary (bool): if True, it will be in a binary format. This is only valid for some formats such as STL where
          you have the ASCII version 'stl' and the binary version 'stlb'.
    """
    if library == 'pyassimp':
        scene = pyassimp.load(from_filename)
        extension = to_filename.split('.')[-1].lower()
        if binary:  # for binary add 'b' as a suffix. Ex: '<file>.stlb'
            pyassimp.export(scene, to_filename, file_type=extension + 'b')
        else:
            pyassimp.export(scene, to_filename, file_type=extension)
        pyassimp.release(scene)
    elif library == 'trimesh':
        export_mesh(trimesh.load(from_filename), to_filename)
    else:
        raise NotImplementedError("The given library '{}' is currently not supported, select between 'pyassimp' and "
                                  "'trimesh'".format(library))


def mesh_to_urdf(filename, name=None, mass=None, inertia=None, density=1000, visual=True, collision=True, scale=1.,
                 position=None, orientation=None, color=None, texture=None, urdf_filename=None):
    """
    Write the given mesh to the URDF; it creates the following XML structure:

    <link name="...">
        <inertial>
            ...
        </inertial>
        <visual>
            ...
        </visual>
        <collision>
            ...
        </collision>
    </link>

    The XML element <link> is returned by this function. The inertial elements are computed given

    Args:
        filename (str): path to the mesh file.
        name (str, None): name of the mesh. If None, it will use the name of the filename.
        mass (float, None): mass of the mesh (in kg). If None, it will use the density.
        inertia (np.array[float[3,3]], np.array[float[9]], np.array[float[6]], np.array[float[3]], None): body frame
            inertia matrix relative to the center of mass. If 9 elements are given, these are assumed to be [ixx, ixy,
            ixz, ixy, iyy, iyz, ixz, iyz, izz]. If 6 elements are given, they are assumed to be [ixx, ixy, ixz, iyy,
            iyz, izz]. Finally, if only 3 elements are given, these are assumed to be [ixx, iyy, izz] and are
            considered already to be the principal moments of inertia.
        density (float): density of the mesh (in kg/m^3). By default, it uses the density of the water 1000kg / m^3.
        visual (bool): if we should have a <visual> tag or not.
        collision (bool): if we should have a <collision> tag or not.
        scale (float): scaling factor. If you have a mesh in meter but you want to scale into centimeters, you need
            to provide a scaling factor of 0.01.
        position (np.array[float[3]]): position of the visual and collision meshes.
        orientation (np.array[float[3]]): orientation (represented as roll-pitch-yaw angles) of the visual and
            collision meshes.
        color (list/tuple[float[4]], None): RGBA color.
        texture (str, None): path to the texture to apply to the mesh.
        urdf_filename (str, None): path to the urdf file we wish to write in.

    Returns:
        ET.Element: root element <robot> containing the information about the mesh.
    """
    # get name if filename
    if name is None:
        name = filename.split('/')[-1]

    # get mesh
    mesh = get_mesh(filename)

    def set_origin(tag, position=None, orientation=None):
        origin = {}
        if position is not None:
            origin['xyz'] = str(np.asarray(position))[1:-1]
        if orientation is not None:
            origin['rpy'] = str(np.asarray(orientation))[1:-1]
        if len(origin) > 0:
            ET.SubElement(tag, 'origin', attrib=origin)

    def set_geometry(tag):
        geometry_tag = ET.SubElement(tag, 'geometry')
        attrib = {'filename': filename, 'scale': str(np.asarray([scale, scale, scale]))[1:-1]}
        ET.SubElement(geometry_tag, 'mesh', attrib=attrib)

    # create root element
    root = ET.Element('robot', attrib={'name': name})

    # create <link> tag
    link_tag = ET.SubElement(root, 'link', attrib={'name': name + '_link'})

    # create <inertial> tag
    inertial_tag = ET.SubElement(link_tag, 'inertial')

    # <origin>
    set_origin(inertial_tag, position=mesh.moment_inertia)

    # <mass>
    if mass is None:
        mass = get_mesh_mass(mesh, density=density, scale=scale)
    ET.SubElement(inertial_tag, 'mass', attrib={'value': str(mass)})

    # <inertia>
    if inertia is None:
        inertia = get_mesh_body_inertia(mesh, mass=mass, density=density, scale=scale)
    inertia = {'ixx': inertia[0, 0], 'ixy': inertia[0, 1], 'ixz': inertia[0, 2], 'iyy': inertia[1, 1],
               'iyz': inertia[1, 2], 'izz': inertia[2, 2]}
    ET.SubElement(inertial_tag, 'inertia', attrib=inertia)

    # create <visual> tag
    if visual:
        visual_tag = ET.SubElement(link_tag, 'visual')

        # <origin>
        set_origin(visual_tag, position=position, orientation=orientation)

        # <geometry>
        set_geometry(visual_tag)

        # <material>
        if color is not None or texture is not None:
            material_tag = ET.SubElement(visual_tag, 'material')
            if color is not None:
                ET.SubElement(material_tag, 'color', attrib={'rgba': str(np.asarray(color))[1:-1]})
            if texture is not None:
                ET.SubElement(material_tag, 'texture', attrib={'filename': texture})

    # create <collision> tag
    if collision:
        collision_tag = ET.SubElement(link_tag, 'collision')

        # <origin>
        set_origin(collision_tag, position=position, orientation=orientation)

        # <geometry>
        set_geometry(collision_tag)

    # save to urdf_filename
    if urdf_filename is not None:
        xml_str = minidom.parseString(ET.tostring(root)).toprettyxml(indent="   ")
        with open(urdf_filename, "w") as f:
            f.write(xml_str)  # .encode('utf-8'))

    # return root element
    return root


def get_mesh(filename):
    r"""
    Return the mesh instance returned by the `trimesh` library.

    Args:
        filename (str): path to the mesh file. Note that `trimesh` supports several formats such as STL, PLY, OBJ, DAE,
            GLTF, and others.

    Returns:
        trimesh.base.Trimesh: trimesh instance.

    References:
        - To load with trimesh: https://github.com/mikedh/trimesh/blob/master/trimesh/exchange/load.py
        - To export with trimesh: https://github.com/mikedh/trimesh/blob/master/trimesh/exchange/export.py
    """
    mesh = filename
    if isinstance(filename, str):
        mesh = trimesh.load(filename)
    elif not isinstance(filename, trimesh.base.Trimesh):
        raise TypeError("Expecting the given 'filename' to be a string, or an instance of `trimesh.base.Trimesh`, but "
                        "instead got: {}".format(filename))
    return mesh


def get_mesh_volume(mesh, scale=1.):
    """
    Get the volume of the mesh.

    Args:
        mesh (trimesh.base.Trimesh, str): trimesh instance, or path to the mesh file.
        scale (float): scaling factor. If you have a mesh in meter but you want to scale into centimeters, you need
            to provide a scaling factor of 0.01.

    Returns:
        float: volume of the mesh.
    """
    mesh = get_mesh(mesh)
    return mesh.volume * scale**3  # the scale is for each dimension


def get_mesh_convex_volume(mesh, scale=1.):
    """
    Get the convex hull volume of the mesh.

    Args:
        mesh (trimesh.base.Trimesh, str): trimesh instance, or path to the mesh file.
        scale (float): scaling factor. If you have a mesh in meter but you want to scale into centimeters, you need
            to provide a scaling factor of 0.01.

    Returns:
        float: convex hull volume of the mesh.
    """
    mesh = get_mesh(mesh)
    return mesh.convex_hull.volume * scale**3  # the scale is for each dimension


def get_mesh_com(mesh, scale=1.):
    """
    Get the mesh's center of mass.

    Args:
        mesh (trimesh.base.Trimesh, str): trimesh instance, or path to the mesh file.
        scale (float): scaling factor. If you have a mesh in meter but you want to scale it into centimeters, you need
            to provide a scaling factor of 0.01.

    Returns:
        np.array[float[3]]: center of mass of the mesh.
    """
    mesh = get_mesh(mesh)
    return mesh.center_mass * scale


def get_mesh_mass(mesh, density=1000, scale=1.):
    """
    Get the mass of the mesh using the given density, and assuming a uniform density.

    Args:
        mesh (trimesh.base.Trimesh, str): trimesh instance, or path to the mesh file.
        density (float): density of the mesh. By default, it is the density of the water 1000 kg / m^3.
        scale (float): scaling factor. If you have a mesh in meter but you want to scale it into centimeters, you need
            to provide a scaling factor of 0.01.

    Returns:
        float: mass of the mesh.
    """
    volume = get_mesh_volume(mesh, scale=scale)
    return density * volume


def get_mesh_body_inertia(mesh, mass=None, density=1000, scale=1.):
    """
    Get the full inertia matrix of the mesh relative to its center of mass.

    Args:
        mesh (trimesh.base.Trimesh, str): trimesh instance, or path to the mesh file.
        mass (float, None): mass of the mesh (in kg). If None, it will use the density.
        density (float): density of the mesh. By default, it is the density of the water 1000 kg / m^3.
        scale (float): scaling factor. If you have a mesh in meter but you want to scale it into centimeters, you need
            to provide a scaling factor of 0.01.

    Returns:
        np.array[float[3,3]]: full inertia matrix of the mesh relative to its center of mass.
    """
    mesh = get_mesh(mesh)

    # volume = mesh.volume  # in m^3  (in trimesh: mash.mass = mash.volume, i.e. density = 1)
    # volume *= scale ** 3  # the scale is for each dimension
    # inertia = mesh.moment_inertia * scale ** 2  # I ~ mr^2
    #
    # # the previous inertia is based on the assumption that mesh.mass = mesh.volume
    # density = mass / volume  # density = new_mass / old_mass
    # inertia *= density
    #
    # # com = mesh.center_mass * scale  # uniform density assumption
    # # (mesh.center_mass is a bit different from mesh.centroid)

    mesh.apply_scale(scale)  # note: this is an inplace operation
    default_density = mesh.density

    # compute density
    volume = mesh.volume
    if mass is not None:
        density = mass / volume
    mesh.density = density

    # compute inertia
    inertia = mesh.moment_inertia

    # because of the inplace operation, put back default values
    mesh.density = default_density
    mesh.apply_scale(1. / scale)

    return inertia
