#!/usr/bin/env python
"""Define the MuJoCo parser/generator.

Notes:
    - MuJoCo only accepts STL meshes
    - MuJoCo can load PNG files for textures and heightmap.

References:
    - MuJoCo overview: http://www.mujoco.org/book/index.html
    - MuJoCo XML format: http://www.mujoco.org/book/XMLreference.html
"""

# import XML parser
import xml.etree.ElementTree as ET

# import mesh converter (from .obj to .stl)
try:
    import trimesh  # https://pypi.org/project/trimesh/

    # import pymesh    # rapid prototyping platform focused on geometry processing
    # doc: https://pymesh.readthedocs.io/en/latest/user_guide.html

    import pyassimp  # library to import and export various 3d-model-formats
    # doc: http://www.assimp.org/index.php
    # github: https://github.com/assimp/assimp
except ImportError as e:
    raise ImportError(str(e) + "\nTry to install pymesh pyassimp: `pip install pymesh pyassimp`")

from pyrobolearn.utils.parsers.robots.world_parser import WorldParser
from pyrobolearn.utils.parsers.robots.data_structures import *
from pyrobolearn.utils.transformation import rotation_matrix_x, rotation_matrix_y, rotation_matrix_z


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2019, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class MuJoCoParser(WorldParser):
    r"""MuJoCo Parser and Generator"""

    def __init__(self, filename=None):
        """
        Initialize the MuJoCo parser.

        Args:
            filename (str, None): path to the MuJoCo XML file.
        """
        super().__init__(filename)
        self.simulator = None
        self.compiler = dict()  # set options for the built-in parser and compiler
        self.options = dict()   # simulation options
        self.defaults = dict()  # default values for the attributes when they are not specified
        self.assets = dict()    # assets (textures, meshes, etc)

    def parse(self, filename):
        """
        Load and parse the given MuJoCo XML file.

        Args:
            filename (str): path to the MuJoCo XML file.
        """
        # load and parse the XML file
        tree_xml = ET.parse(filename)

        # get the root
        root = tree_xml.getroot()

        # check that the root is <mujoco>
        if root.tag != 'mujoco':
            raise RuntimeError("Expecting the first XML tag to be 'mujoco' but found instead: {}".format(root.tag))

        # build the world
        world = World(name=root.attrib.get('model', 'world'))

        # check compiler
        compiler_tag = root.find('compiler')  # TODO: check other
        if compiler_tag is not None:

            def update_compiler(attributes):
                for attribute in attributes:
                    attrib = compiler_tag.attrib.get(attribute)
                    if attrib is not None:
                        self.compiler[attribute] = attrib

            coordinate = compiler_tag.attrib.get('coordinate')
            if coordinate == 'global':
                raise NotImplementedError("Currently, we only support local coordinate frames.")

            update_compiler(['coordinate', 'angle', 'meshdir', 'texturedir', 'eulerseq', 'discardvisual',
                             'convexhull', 'inertiafromgeom', 'fitaabb', 'fusestatic'])

        # check default (this is the default configuration when they are not specified)
        default_tag = root.find('default')
        if default_tag is not None:

            def update_default(tag, attributes):
                tag = default_tag.find(tag)
                if tag is not None:
                    self.defaults[tag] = {}
                    for attribute in attributes:
                        item = tag.attrib.get(attribute)
                        if item is not None:
                            self.defaults['tag'][attribute] = item

            update_default('mesh', ['scale'])
            update_default('material', ['texture', 'emission', 'specular', 'shininess', 'reflectance', 'rgba'])
            update_default('joint', ['type', 'pos', 'axis', 'limited', 'range', 'springdamper', 'stiffness',
                                     'damping', 'frictionloss', 'armature', 'margin', 'ref', 'springref'])
            update_default('geom', ['type', 'contype', 'conaffinity', 'condim', 'size', 'material', 'rgba',
                                    'friction', 'mass', 'density', 'margin', 'fromto', 'pos', 'quat', 'axisangle',
                                    'xyaxes', 'zaxis', 'euler', 'hfield', 'mesh'])
            update_default('site', ['type', 'material', 'rgba', 'size', 'fromto', 'pos', 'quat', 'axisangle',
                                    'xyaxes', 'zaxis', 'euler'])
            update_default('camera', ['mode', 'target', 'fovy', 'ipd', 'pos', 'quat', 'axisangle', 'xyaxes', 'zaxis',
                                      'euler'])
            update_default('light', ['mode', 'target', 'directional', 'castshadow', 'active', 'pos', 'dir',
                                     'attenuation', 'cutoff', 'exponent', 'ambient', 'diffuse', 'specular'])
            update_default('pair', ['condim', 'friction', 'margin', 'gap'])
            update_default('equality', ['active'])
            # update_default('tendon', [''])
            update_default('general', ['ctrllimited', 'ctrlrange', 'forcelimited', 'forcerange', 'lengthrange',
                                       'gear', 'cranklength', 'dyntype', 'gaintype', 'biastype', 'dynprm', 'gainprm',
                                       'biasprm'])
            # name, class, joint, jointinparent, site, tendon, slidersite, cranksite
            update_default('motor', ['ctrllimited', 'ctrlrange', 'forcelimited', 'forcerange', 'lengthrange', 'gear',
                                     'cranklength'])
            update_default('position', ['ctrllimited', 'ctrlrange', 'forcelimited', 'forcerange', 'lengthrange',
                                        'gear', 'cranklength', 'kp'])
            update_default('velocity', ['ctrllimited', 'ctrlrange', 'forcelimited', 'forcerange', 'lengthrange',
                                        'gear', 'cranklength', 'kv'])
            update_default('cylinder', ['ctrllimited', 'ctrlrange', 'forcelimited', 'forcerange', 'lengthrange',
                                        'gear', 'cranklength', 'timeconst', 'area', 'diameter', 'bias'])
            update_default('muscle', ['ctrllimited', 'ctrlrange', 'forcelimited', 'forcerange', 'lengthrange', 'gear',
                                      'cranklength', 'timeconst', 'range', 'force', 'scale', 'lmin', 'lmax', 'vmax',
                                      'fpmax', 'fvmax'])

        # check options
        option_tag = root.find('option')
        if option_tag is not None:
            self.options = option_tag.attrib
            # TODO: check flag

        # check assets
        asset_tag = root.find('asset')
        if asset_tag is not None:
            # check texture, material, and mesh
            for tag in ['texture', 'material', 'mesh']:
                for i, inner_tag in asset_tag.findall(tag):
                    attrib = inner_tag.attrib
                    if len(attrib) > 0:
                        self.assets.setdefault(tag, dict())[attrib.get('name')] = attrib

        # check world body
        worldbody_tag = root.find('worldbody')
        if worldbody_tag is not None:

            # light
            for i, light_tag in enumerate(worldbody_tag.findall('light')):
                attrib = light_tag.attrib
                light = Light(name=attrib.get('name', 'light_' + str(i)), cast_shadows=attrib.get('castshadow'),
                              position=attrib.get('pos'), direction=attrib.get('dir'), ambient=attrib.get('ambient'),
                              diffuse=attrib.get('diffuse'), specular=attrib.get('specular'))

                world.lights[light.name] = light

            # check each (multi-)body
            for i, body_tag in enumerate(worldbody_tag.findall('body')):
                # create tree
                tree = Tree(name=body_tag.attrib.get('name', 'prl_multibody_' + str(i)))

                # check recursively body
                self._check_body(tree, body_tag, body_idx=i)
                world.trees[tree.name] = tree

        # check contact

        # check equality constraint

        # check actuator

        # check sensor

        # set the world
        self.world = world

    def _check_orientation(self, attrib):
        orientation = None

        # quaternion
        if attrib.get('quat') is not None:
            orientation = attrib.get('quat')

        # euler orientation
        elif attrib.get('euler') is not None:
            orientation = attrib.get('euler')
            if self.compiler.get('eulerseq') != "xyz":  # xyz = rpy
                eulerseq = self.compiler['eulerseq'].lower()
                orientation = [float(c) for c in orientation.split()]

                rot = np.identity(3)
                for letter, angle in zip(eulerseq, orientation):
                    if letter == 'x':
                        rot *= rotation_matrix_x(angle)
                    elif letter == 'y':
                        rot *= rotation_matrix_y(angle)
                    elif letter == 'z':
                        rot *= rotation_matrix_z(angle)
                    else:
                        raise ValueError("Expecting the letter in 'eulerseq' to be {'x', 'y', 'z', 'X', 'Y', 'Z'}.")
                orientation = rot

        # axis-angle
        elif attrib.get('axisangle') is not None:
            axisangle = attrib.get('axisangle').split()
            orientation = (axisangle[:3], axisangle[3])

        # XY axes
        elif attrib.get('xyaxes') is not None:
            xyaxes = attrib.get('xyaxes')
            xyaxes = np.array([float(c) for c in xyaxes.split()])
            x = xyaxes[:3] / np.linalg.norm(xyaxes[:3])
            y = xyaxes[3:] / np.linalg.norm(xyaxes[3:])
            z = np.cross(x, y)
            orientation = np.array([x, y, z]).T

        # Z axis
        elif attrib.get('zaxis') is not None:
            z = np.array([float(c) for c in attrib.get('zaxis').split()])
            z = z / np.linalg.norm(z)
            z_ = np.array([0., 0., 1.])  # old z axis
            x = np.cross(z, z_)
            y = np.cross(z, x)
            orientation = np.array([x, y, z]).T

        return orientation

    def _check_body(self, tree, body_tag, body_idx, parent_body=None, joint_idx=0):  # TODO: check with self.defaults
        """
        Construct recursively the given tree, and return Body instance from a <body>.

        Args:
            tree (Tree): tree data structure containing the model.
            body_tag (ET.Element): body XML element.
            body_idx (int): link index.
            parent_body (Body, None): the parent body instance.
            joint_idx (int): joint index.

        Returns:
            Body: body data structure.
        """
        # create body
        body = Body(body_id=body_idx, name=body_tag.attrib.get('name', 'prl_body_' + str(body_idx)))

        # add body to the tree
        tree.bodies[body.name] = body

        # check inertial
        inertial = Inertial()
        inertial_tag = body_tag.find('inertial')
        if inertial_tag is not None:
            # position and orientation
            position = inertial_tag.attrib.get('pos')
            if position is not None:
                inertial.position = position
            orientation = self._check_orientation(inertial_tag.attrib)
            if orientation is not None:
                inertial.orientation = orientation

            # mass and inertia
            inertial.mass = inertial_tag.attrib.get('mass')
            inertia = inertial_tag.attrib.get('diaginertia')
            if inertia is not None:
                inertial.inertia = inertia
            else:
                inertial.inertia = inertial_tag.attrib.get('fullinertia')

        # check geoms
        for i, geom_tag in enumerate(body_tag.findall('geom')):

            ##########
            # visual #
            ##########

            attrib = geom_tag.attrib
            dtype = attrib.get('type')
            visual = Visual(name=attrib.get('name'), dtype=dtype, color=attrib.get('rgba'))

            # get position and orientation
            visual.pos = attrib.get('pos')
            visual.orientation = self._check_orientation(attrib)

            # compute size (rescale them)
            if dtype == 'plane':
                size = attrib.get('size')

            if dtype in {'capsule', 'cylinder', 'ellipsoid', 'box'}:
                fromto = attrib.get('fromto')
                if fromto is not None:
                    fromto = np.array([float(n) for n in fromto.split()])
                    from_pos, to_pos = fromto[:3], fromto[3:]
                    v = to_pos - from_pos
                    pos = from_pos + v / 2.
                    length = np.linalg.norm(v)
                    z = v / length
                    z_ = np.zeros([0., 0., 1.])  # old z axis
                    x = np.cross(z, z_)
                    y = np.cross(z, x)
                    rot = np.array([x, y, z]).T

                    # set new position and orientation
                    visual.pos = pos
                    visual.orientation = rot

                    # set size
                    if dtype == 'capsule':
                        size = None  # TODO

            # check texture
            material = attrib.get('material')
            if material in self.assets:
                pass

            # check mesh
            mesh = attrib.get('mesh')
            if mesh is not None:
                # get the mesh from the assets
                mesh_dict = self.assets.get('mesh')
                if mesh_dict is not None:
                    mesh = mesh_dict.get(mesh)

                    # check mesh format

                    # get the texture for the mesh

            # set visual to body
            body.visual = visual

            #############
            # collision #
            #############

            if not (attrib.get('contype') == "0" and attrib.get('conaffinity') == "0"):
                # copy collision shape information from visual shape
                collision = Collision()
                collision.name = visual.name
                collision.frame = visual.frame
                collision.geometry = visual.geometry
                body.collision = collision

            ############
            # inertial #  just the mass
            ############

            # if the <inertial> tag was not given, compute based on information in geom
            if inertial.mass is None or inertial.inertia is None:

                # if mesh, load it in memory
                if dtype == 'mesh':
                    mesh = trimesh.load(mesh_path)
                    if not mesh.is_watertight:  # mesh.is_convex
                        raise ValueError("Could not compute the volume because the mesh is not watertight...")

                # get mass
                mass = inertial.mass
                if mass is None:
                    if attrib.get('mass') is not None:
                        inertial.mass = attrib.get('mass')
                        mass = inertial.mass  # this makes the conversion to float
                    else:
                        # get density
                        density = float(attrib.get('density', 1000.))

                        # get mass by computing the volume
                        volume = 1
                        if dtype == 'box':
                            w, h, d = visual.size  # width, height, depth
                            volume = w*h*d
                        elif dtype == 'capsule':
                            r, h = visual.size  # radius, height
                            sphere_volume = 4. / 3 * np.pi * r ** 3
                            cylinder_volume = np.pi * r ** 2 * h
                            volume = sphere_volume + cylinder_volume
                        elif dtype == 'cylinder':
                            r, h = visual.size  # radius, height
                            volume = np.pi * r ** 2 * h
                        elif dtype == 'ellipsoid':
                            a, b, c = visual.size
                            volume = 4./3 * np.pi * a * b * c
                        elif dtype == 'mesh':
                            scale = float(attrib.get('fitscale', 1))
                            volume = mesh.volume  # in m^3  (in trimesh: mash.mass = mash.volume, i.e. density = 1)
                            volume *= scale ** 3  # the scale is for each dimension
                        elif dtype == 'sphere':
                            r = visual.size  # radius
                            volume = 4. / 3 * np.pi * r ** 3
                        mass = density * volume
                        inertial.mass = mass

                # compute inertia if not given
                if inertial.inertia is None:
                    inertia = None
                    if dtype == 'box':
                        w, h, d = visual.size  # width, height, depth
                        inertia = 1./12 * mass * np.array([h**2 + d**2, w**2 + d**2, w**2 + h**2])
                    elif dtype == 'capsule':
                        r, h = visual.size  # radius, height

                        # get mass of cylinder and hemisphere
                        sphere_volume = 4./3 * np.pi * r**3
                        cylinder_volume = np.pi * r**2 * h
                        volume = sphere_volume + cylinder_volume
                        density = mass / volume
                        m_s = density * sphere_volume    # sphere mass = 2 * hemisphere mass
                        m_c = density * cylinder_volume  # cylinder mass

                        # from: https://www.gamedev.net/articles/programming/math-and-physics/capsule-inertia-\
                        # tensor-r3856/
                        ixx = m_c * (h**2/12. + r**2/4.) + m_s * (2*r**2/5. + h**2/2. + 3*h*r/8.)
                        iyy = ixx
                        izz = m_c * r**2/2. + m_s * 2 * r**2 / 5.
                        inertia = np.array([ixx, iyy, izz])
                    elif dtype == 'cylinder':
                        r, h = visual.size  # radius, height
                        inertia = 1./12 * mass * np.array([3*r**2 + h**2, 3*r**2 + h**2, r**2])
                    elif dtype == 'ellipsoid':
                        a, b, c = visual.size
                        inertia = 1./5 * mass * np.array([b**2 + c**2, a**2 + c**2, a**2 + b**2])
                    elif dtype == 'mesh':
                        scale = float(attrib.get('fitscale', 1))

                        volume = mesh.volume  # in m^3  (in trimesh: mash.mass = mash.volume, i.e. density = 1)
                        volume *= scale**3  # the scale is for each dimension
                        inertia = mesh.moment_inertia * scale**2  # I ~ mr^2

                        # the previous inertia is based on the assumption that mesh.mass = mesh.volume
                        density = mass / volume  # density = new_mass / old_mass
                        inertia *= density

                        # com = mesh.center_mass * scale  # uniform density assumption
                        # (mesh.center_mass is a bit different from mesh.centroid)
                    elif dtype == 'sphere':
                        radius = visual.size
                        inertia = 2./5 * mass * radius**2 * np.ones(3)

                    inertial.inertia = inertia

        # check sites
        for i, site_tag in enumerate(body_tag.findall('site')):
            pass

        # check joints that connect the current body with its parent
        joints = []
        for i, joint_tag in enumerate(body_tag.findall('joint')):

            # create joint with the corresponding attributes
            attrib = joint_tag.attrib
            joint = Joint(joint_id=joint_idx, name=attrib.get('name', 'prl_joint_' + str(joint_idx)),
                          dtype=attrib.get('type'), position=attrib.get('pos'), axis=attrib.get('axis'),
                          friction=attrib.get('frictionloss'), damping=attrib.get('damping'),
                          parent=parent_body, child=body)

            limited = attrib.get('limited')
            if limited is not None:
                limited = limited.lower().strip()
                if limited == 'true':
                    joint.limits = attrib.get('range')

            # add joint in tree and parent body
            tree.joints[joint.name] = joint
            if parent_body is not None:
                parent_body.joints[joint.name] = joint

            # increment joint counter
            joint_idx += 1
            joints.append(joint)

        # check bodies
        for i, new_body_tag in enumerate(body_tag.findall('body')):
            body_idx += 1
            self._check_body(tree, new_body_tag, body_idx=body_idx, parent_body=body, joint_idx=joint_idx)

        # check include
        for i, include_tag in enumerate(body_tag.findall('include')):
            # create MuJoCoParser
            parser = MuJoCoParser(filename=include_tag.attrib.get('include'))

            # get the tree
            raise NotImplementedError("We can not parse the <include> tag yet...")

    def generate(self, world=None):  # TODO: check texture and mesh format
        """
        Generate the XML world from the `World` data structure.

        Args:
            world (World): world data structure.

        Returns:
            ET.Element: root element in the XML file.
        """
        if world is None:
            world = self.world

        # create root element
        root = ET.Element('mujoco', attrib={'model': world.name})

        # create compiler

        # create asset

        # create world
        worldbody_tag = ET.SubElement(root, 'worldbody')  # name = 'world'

        # create models
        for tree in world.trees:

            # create bodies
            for i, body in enumerate(tree.bodies):
                # create <body>
                attrib = {}
                if body.name is not None:
                    attrib['name'] = body.name
                if body.visual is not None:
                    visual = body.visual
                    if visual.position is not None:
                        attrib['pos'] = str(np.asarray(visual.position))[1:-1]
                    if visual.orientation is not None:
                        attrib['quat'] = str(np.asarray(visual.quaternion))[1:-1]
                body_tag = ET.SubElement(worldbody_tag, 'body', attrib=attrib)

                # create <inertial>
                if body.inertial is not None:
                    inertial = body.inertial
                    if inertial.inertia is not None:
                        ET.SubElement(body_tag, 'inertial')

                # create <geom>
                if body.visual is not None:
                    visual = body.visual
                    ET.SubElement(body_tag, 'geom')

                    # if mesh, make sure it is a STL file
                    if visual.dtype == 'mesh':
                        filename = visual.filename
                        extension = filename.split('.')[-1]
                        if extension.lower() != 'stl':
                            scene = pyassimp.load(filename)
                            filename_without_extension = ''.join(filename.split('.')[:-1])
                            new_filename = filename_without_extension + '.stl'
                            pyassimp.export(scene, new_filename, file_type='stl')
                            pyassimp.release(scene)

                # create <joint>
