#!/usr/bin/env python
"""Define the MuJoCo parser/generator.

Notes:
    - MuJoCo only accepts STL meshes
    - MuJoCo can load PNG files for textures and heightmap.

References:
    - MuJoCo overview: http://www.mujoco.org/book/index.html
    - MuJoCo XML format: http://www.mujoco.org/book/XMLreference.html
"""

import os
import copy
import numpy as np
import xml.etree.ElementTree as ET

# import mesh converter (from .obj to .stl)
try:
    import trimesh  # https://pypi.org/project/trimesh/
    from trimesh.exchange.export import export_mesh

    # import pymesh    # rapid prototyping platform focused on geometry processing
    # doc: https://pymesh.readthedocs.io/en/latest/user_guide.html

    import pyassimp  # library to import and export various 3d-model-formats
    # doc: http://www.assimp.org/index.php
    # github: https://github.com/assimp/assimp
except ImportError as e:
    raise ImportError(str(e) + "\nTry to install trimesh pyassimp: `pip install trimesh pyassimp`")

from pyrobolearn.utils.parsers.robots.world_parser import WorldParser
from pyrobolearn.utils.parsers.robots.data_structures import Simulator, World, Tree, Body, Joint, Inertial, \
    Visual, Collision, Light, Floor
from pyrobolearn.utils.transformation import rotation_matrix_x, rotation_matrix_y, rotation_matrix_z, \
    get_inverse_homogeneous


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2019, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class MuJoCoParser(WorldParser):
    r"""MuJoCo Parser and Generator

    The MuJoCo parser and generator keeps track of two data structures:

    1. The XML tree describing the world (such that we can generate an XML file or string from it).
    2. The World data structure that we can pass to other generators to generate their corresponding world file.

    Using this class, you can `parse` an XML file or XML string that will automatically generate the XML tree and
    the `World` data structure. You can also build the tree from scratch by yourself using the provided methods.
    However, note that with the latter approach, you will have to generate the `World` data structure by yourself if
    you need it, by calling the `parse` method.


    Example of a simple MuJoCo file from [1]:

        <mujoco>
           <worldbody>
              <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>
              <geom type="plane" size="1 1 0.1" rgba=".9 0 0 1"/>
              <body pos="0 0 1">
                 <joint type="free"/>
                 <geom type="box" size=".1 .2 .3" rgba="0 .9 0 1"/>
              </body>
           </worldbody>
        </mujoco>

    Notes:
    - MuJoCo only accepts STL meshes
    - MuJoCo can load PNG files for textures and heightmap.

    Parts of the documentation has been copied-pasted from [1, 2] for completeness purpose.

    References:
        - [1] MuJoCo overview: http://www.mujoco.org/book/index.html
        - [2] MuJoCo XML format: http://www.mujoco.org/book/XMLreference.html
    """

    def __init__(self, filename=None):
        """
        Initialize the MuJoCo parser and generator.

        Args:
            filename (str, None): path to the MuJoCo XML file.
        """
        super().__init__(filename)
        self.simulator = Simulator()
        self.compiler = dict()  # set options for the built-in parser and compiler
        self.options = dict()   # simulation options

        # default values for the attributes when they are not specified
        # {default_type (str): {attribute_name (str): attribute_value (str)}}
        self.defaults = dict()

        # assets (textures, meshes, etc)
        # {asset_type (str): {attribute_name (str): attribute_value (str)}}
        self.assets = dict()

        # create root XML element
        self.create_root("mujoco")
        # add compiler
        self.add_element("compiler", self._root, attributes={'coordinate': 'local', 'angle': 'radian'})
        # add size
        self.nconmax = 200  # increase this if necessary (this depends on how many models are loaded)
        self.add_element("size", self._root, attributes={"nconmax": str(self.nconmax)})
        # add worldbody
        self.worldbody = self.add_element(name="worldbody", parent_element=self.root)

        # set some counters
        self._world_cnt = 0
        self._light_cnt = 0
        self._tree_cnt = 0
        self._body_cnt = 0
        self._joint_cnt = 0
        self._material_cnt = 0  # material/asset counter
        # self._geom_cnt = 0
        # self._site_cnt = 0

        # save homogenous matrices (for later)
        self._h_bodies, self._h_joints, self._h_visuals, self._h_collisions, self._h_inertials = {}, {}, {}, {}, {}
        self._assets_tmp = set([])
        self._mesh_dirname = ''

    #################
    # Utils methods #
    #################

    @staticmethod
    def _convert_wxyz_to_xyzw(q):
        """Convert a quaternion in the (w,x,y,z) format to (x,y,z,w).

        - Mujoco: (w,x,y,z)
        - Bullet: (x,y,z,w)
        """
        return np.roll(q, shift=-1)

    @staticmethod
    def _convert_xyzw_to_wxyz(q):
        """Convert a quaternion in the (x,y,z,w) format to (w,x,y,z).

        - Mujoco: (w,x,y,z)
        - Bullet: (x,y,z,w)
        """
        return np.roll(q, shift=1)

    ##########
    # Parser #
    ##########

    def _get_orientation(self, attrib):
        """
        Get the orientation based on the XML attributes.

        Args:
            attrib (dict): dictionary which contains 'quat', 'euler' (with possibly 'eulereq'), 'axiangle', 'xyaxes',
              'zaxis'}.

        Returns:
            np.array[float[3,3]], np.array[float[3]], str, None: orientation.
        """
        orientation = None

        # quaternion
        if attrib.get('quat') is not None:
            quat = attrib.get('quat')
            orientation = self._convert_wxyz_to_xyzw([float(q) for q in quat.split()])

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

    @staticmethod
    def _set_attribute(element, name, value):
        """
        Set an attribute value to the given attribute name belonging to the given element.
        """
        if hasattr(element, name):
            setattr(element, name, value)

    def parse(self, filename):
        """
        Load and parse the given MuJoCo XML file.

        Args:
            filename (str, ET.Element): path to the MuJoCo XML file, or XML root element.
        """
        if isinstance(filename, str):
            # load and parse the XML file
            tree_xml = ET.parse(filename)
            # get the root
            root = tree_xml.getroot()
        elif isinstance(filename, ET.Element):
            root = filename
        else:
            raise TypeError("Expecting the given 'filename' to be a string or an ET.Element, but got instead: "
                            "{}".format(type(filename)))

        # check that the root is <mujoco>
        if root.tag != 'mujoco':
            raise RuntimeError("Expecting the first XML tag to be 'mujoco' but found instead: {}".format(root.tag))

        # build the simulator data structure
        self.simulator.name = root.attrib.get('model', 'simulator')

        # parse compiler: This element is used to set options for the built-in parser and compiler. After parsing and
        # compilation it no longer has any effect.
        # compiler attributes: boundmass, boundinertia, settotalmass, balanceinertia, strippath, coordinate, angle,
        #                      fitaabb, eulerseq, meshdir, texturedir, discardvisual, convexhull, userthread,
        #                      fusestatic, inertiafromgeom, inertiagrouprange
        self._parse_compiler(parent_tag=root)

        # parse default: this is the default configuration when they are not specified
        # default attribute: mesh, material, joint, geom, site, camera, light, pair, equality, tendon, general, motor,
        #                    position, velocity, cylinder, muscle, custom
        self._parse_default(parent_tag=root)

        # parse options
        self._parse_option(parent_tag=root)

        # parse assets: This is a grouping element for defining assets. Assets are created in the model so that they
        # can be referenced from other model elements
        # asset attributes: texture, hfield, mesh, skin, material
        self._parse_asset(parent_tag=root)

        # parse (world) body: This element is used to construct the kinematic tree via nesting. The element worldbody
        # is used for the top-level body, while the element body is used for all other bodies.
        # body attributes: name, childclass, mocap, pos, quat, axisangle, xyaxes, zaxis, euler
        self._parse_worldbody(parent_tag=root, update_world_attribute=True)

        # parse contact
        # contact attributes: pair,
        self._parse_contact(parent_tag=root)

        # parse equality constraint
        # equality attributes: connect, weld, joint, tendon, distance
        self._parse_equality_constraint(parent_tag=root)

        # parse actuator
        # actuator attributes: general, motor, position, velocity, cylinder, muscle
        self._parse_actuator(parent_tag=root)

        # parse sensor
        # sensor attributes: touch, accelerometer, velocimeter, gyro, force, torque, magnetometer, rangefinder,
        #                    jointpos, jointvel, tendonpos, tendonvel, actuatorpos, actuatorvel, actuatorfrce,
        #                    ballquat, ballangvel, jointlimitpos, jointlimitvel, jointlimitfrc, tendonlimitpos,
        #                    tendonlimitvel, tendonlimitfrc, framepos, framequat, framexaxis, frameyaxis, framezaxis,
        #                    framelinvel, frameangvel, framelinacc, frameangacc, subtreecom, subtreelinvel,
        #                    subtreeangmom
        self._parse_sensor(parent_tag=root)

    def _parse_compiler(self, parent_tag):
        """
        Parse the compiler tag if present, and update the `compiler` attribute of this class.

        From the main documentation [2]: "This element is used to set options for the built-in parser and compiler.
        After parsing and compilation it no longer has any effect. The settings here are global and apply to the
        entire model.

        Attributes:
            - boundmass (real, "0"): This attribute imposes a lower bound on the mass of each body except for the
              world body. It can be used as a quick fix for poorly designed models that contain massless moving
              bodies, such as the dummy bodies often used in URDF models to attach sensors. Note that in MuJoCo
              there is no need to create dummy bodies.
            - boundinertia (real, "0"): This attribute imposes a lower bound on the diagonal inertia components of
              each body except for the world body.
            - settotalmass (real, "-1"): If this value is positive, the compiler will scale the masses and inertias of
              all bodies in the model, so that the total mass equals the value specified here. The world body has mass
              0 and does not participate in any mass-related computations. This scaling is performed last, after all
              other operations affecting the body mass and inertia.
            - balanceinertia ([false, true], "false"): A valid diagonal inertia matrix must satisfy A+B>=C for all
              permutations of the three diagonal elements. Some poorly designed models violate this constraint, which
              will normally result in compile error. If this attribute is set to "true", the compiler will silently
              set all three diagonal elements to their average value whenever the above condition is violated.
            - etc
        "

        Args:
            parent_tag (ET.Element): parent XML element to check if it has a 'compiler' tag.
        """
        compiler_tag = parent_tag.find('compiler')  # TODO: check other
        if compiler_tag is not None:

            def update_compiler(attributes):
                """
                Update the `compiler` attribute of this class.

                Args:
                    attributes (list[str]): list of attributes to check.
                """
                for attribute in attributes:
                    attrib = compiler_tag.attrib.get(attribute)
                    if attrib is not None:
                        self.compiler[attribute] = attrib

            coordinate = compiler_tag.attrib.get('coordinate')
            if coordinate == 'global':
                raise NotImplementedError("Currently, we only support local coordinate frames.")

            update_compiler(['coordinate', 'angle', 'meshdir', 'texturedir', 'eulerseq', 'discardvisual',
                             'convexhull', 'inertiafromgeom', 'fitaabb', 'fusestatic'])

    def _parse_option(self, parent_tag):
        """
        Parse the option tag if present, and update the `options` attribute of this class.

        From the main documentation [2]: "This element is is one-to-one correspondence with the low level structure
        mjOption contained in the field mjModel.opt of mjModel. Options can be modified during runtime by the user."

        Option attributes: timestep, apirate, impratio, gravity, wind, magnetic, density, viscosity, o_margin,
                           o_solref, o_solimp, integrator, collision, cone, jacobian, solver, iterations, tolerance,
                           noslip_iterations, noslip_tolerance, mpr_iterations, mpr_tolerance

        Args:
            parent_tag (ET.Element): parent XML element to check if it has a 'compiler' tag.
        """
        option_tag = parent_tag.find('option')
        if option_tag is not None:
            self.options = option_tag.attrib
            # TODO: check flag

    def _parse_default(self, parent_tag):
        """
        Parse the default tag if present, and update the `defaults` attribute of this class.

        Args:
            parent_tag (ET.Element): parent XML element to check if it has a 'compiler' tag.
        """
        default_tag = parent_tag.find('default')
        if default_tag is not None:

            def update_default(tag, attributes):
                """
                Update the default dictionary which will be useful later.

                Args:
                    tag (str): tag to check under the default XML element.
                    attributes (list[str]): list of attributes to check in the specified tag XML element (if found).
                """
                # find tag
                tag = default_tag.find(tag)

                # if tag was found
                if tag is not None:
                    # go through each attribute and update the default dict
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

    def _parse_asset(self, parent_tag):
        """
        Parse the asset tag if present, and update the 'assets' attribute of this class.

        Args:
            parent_tag (ET.Element): parent XML element to check if it has a 'asset' tag.
        """
        asset_tag = parent_tag.find('asset')
        if asset_tag is not None:
            # check texture, material, and mesh
            for tag in ['texture', 'material', 'mesh']:
                for i, inner_tag in asset_tag.findall(tag):
                    attrib = inner_tag.attrib
                    if len(attrib) > 0:
                        self.assets.setdefault(tag, dict())[attrib.get('name')] = attrib

    def _parse_worldbody(self, parent_tag, update_world_attribute=False):
        """
        Parse the worldbody tag if present, and instantiate the `World` data structure, update the `world` attribute
        of this class, and add the world to the simulator.

        Args:
            parent_tag (ET.Element): parent XML element to check if it has a 'worldbody' tag.
            update_world_attribute (bool): if we should update the `world` attribute of this class.
        """
        worldbody_tag = parent_tag.find('worldbody')
        if worldbody_tag is not None:

            # instantiate the world data structure
            name = parent_tag.attrib.get('model')
            if name is None:
                name = '__prl_world_' + str(self._world_cnt)
                self._world_cnt += 1
            world = World(name=name)

            # light: This element creates a light, which moves with the body where it is defined.
            # light attributes: name, class, mode, target, directional, castshadow, active, pos, dir, attenuation,
            #                   cutoff, exponent, ambient, diffuse, specular
            for i, light_tag in enumerate(worldbody_tag.findall('light')):
                light = self._parse_light(light_tag)
                world.lights[light.name] = light

            # check each (multi-)body
            for i, body_tag in enumerate(worldbody_tag.findall('body')):
                # create tree
                name = body_tag.attrib.get('name')
                if name is None:
                    name = '__prl_multibody_' + str(self._tree_cnt)
                    self._tree_cnt += 1
                tree = Tree(name=name)

                # check recursively body
                self._parse_body(tree, body_tag=body_tag)

                # save tree
                world.trees[tree.name] = tree

            # update the world attribute of this class if specified.
            if update_world_attribute:
                self.world = world

    def _parse_light(self, light_tag):
        """
        Parse the given light tag and return the instance.

        Args:
            light_tag (ET.Element): light XML tag.

        Returns:
            Light: light data structure.
        """
        attrib = light_tag.attrib
        name = attrib.get('name')
        if name is None:
            name = '__prl_light_' + str(self._light_cnt)
            self._light_cnt += 1
        light = Light(name=name, cast_shadows=attrib.get('castshadow'), position=attrib.get('pos'),
                      direction=attrib.get('dir'), ambient=attrib.get('ambient'), diffuse=attrib.get('diffuse'),
                      specular=attrib.get('specular'))
        return light

    def _parse_body(self, tree, body_tag, parent_body=None):  # TODO: check with self.defaults
        """
        Parse the given body tag, and construct recursively the given tree, and return Body instance from a <body>.

        Args:
            tree (Tree): tree data structure containing the model.
            body_tag (ET.Element): body XML element.
            parent_body (Body, None): the parent body instance.

        Returns:
            Body: body data structure.
        """
        # create body
        name = body_tag.attrib.get('name')
        if name is None:
            name = '__prl_body_' + str(self._body_cnt)
            self._body_cnt += 1
        body = Body(body_id=self._body_cnt, name=name)

        # add body to the tree
        tree.bodies[body.name] = body

        # check inertial
        inertial_tag = body_tag.find('inertial')
        self._parse_inertial(body, inertial_tag)

        # check geoms: geoms (short for geometric primitive) are used to specify appearance and collision geometry.
        # geom attributes: name, class, type, contype, conaffinity, condim, group, priority, size, material, rgba,
        #                  friction, mass, density, solmix, solref, solimp, margin, gap, fromto, pos, quat, axisangle,
        #                  xyaxes, zaxis, euler, hfield, mesh, fitscale, user
        for i, geom_tag in enumerate(body_tag.findall('geom')):
            self._parse_geom(body=body, geom_tag=geom_tag)

        # check sites: Sites are light geoms. They have the same appearance properties but cannot participate in
        # collisions and cannot be used to infer body masses.
        for i, site_tag in enumerate(body_tag.findall('site')):
            self._parse_site(body=body, site_tag=site_tag, site_idx=i)

        # check joints that connect the current body with its parent
        joints = []
        for i, joint_tag in enumerate(body_tag.findall('joint')):
            # create joint with the corresponding attributes
            joint = self._parse_joint(tree, body, joint_tag, parent_body=parent_body)
            joints.append(joint)

        # check bodies
        for i, new_body_tag in enumerate(body_tag.findall('body')):
            self._parse_body(tree, new_body_tag, parent_body=body)

        # check include
        for i, include_tag in enumerate(body_tag.findall('include')):
            # create MuJoCoParser
            parser = MuJoCoParser(filename=include_tag.attrib.get('include'))

            # get the tree
            raise NotImplementedError("We can not parse the <include> tag yet...")

    def _parse_joint(self, tree, body, joint_tag, parent_body=None):
        """
        Parse the joint tag if present and return the joint data structure.

        Args:
            tree (Tree): tree data structure containing the model.
            body (Body): body data structure instance.
            joint_tag (ET.Element): joint XML tag.
            parent_body (Body, None): the parent body instance.

        Returns:
            Joint: joint data structure instance.
        """
        # create joint with the corresponding attributes
        attrib = joint_tag.attrib

        # get joint name
        name = attrib.get('name')
        if name is None:
            name = '__prl_joint_' + str(self._joint_cnt)
            self._joint_cnt += 1

        # create joint data structure
        joint = Joint(joint_id=self._joint_cnt, name=name, dtype=attrib.get('type'), position=attrib.get('pos'),
                      axis=attrib.get('axis'), friction=attrib.get('frictionloss'), damping=attrib.get('damping'),
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

        return joint

    def _parse_inertial(self, body, inertial_tag):  # DONE
        """
        Parse the inertial tag if present, and set the inertial data structure to the given body.

        From the main documentation [2]: "This element specifies the mass and inertial properties of the body. If this
        element is not included in a given body, the inertial properties are inferred from the geoms attached to the
        body. When a compiled MJCF model is saved, the XML writer saves the inertial properties explicitly using this
        element, even if they were inferred from geoms. The inertial frame is such that its center coincides with the
        center of mass of the body, and its axes coincide with the principal axes of inertia of the body. Thus the
        inertia matrix is diagonal in this frame.

        Attributes:
            - pos (real[3], required): position of the inertial frame.
            - quat, axisangle, xyaxes, zaxis, euler: orientation of the inertial frame.
            - mass (real, required): mass of the body.
            - diaginertia (real[3], optional): diagonal inertia matrix, expressing the body inertia relative to the
              inertial frame.
            - fulldiagonal (real[6], optional): Full inertia matrix M (Ixx, Iyy, Izz, Ixy, Ixz, Iyz)."

        Args:
            body (Body): body data structure instance.
            inertial_tag (ET.Element, None): XML element.
        """
        if inertial_tag is not None:
            # instantiate inertial data structure
            inertial = Inertial()

            # position and orientation
            position = inertial_tag.attrib.get('pos')
            if position is not None:
                inertial.position = position
            orientation = self._get_orientation(inertial_tag.attrib)
            if orientation is not None:
                inertial.orientation = orientation

            # mass and inertia
            inertial.mass = inertial_tag.attrib.get('mass')
            inertia = inertial_tag.attrib.get('diaginertia')
            if inertia is not None:
                inertial.inertia = inertia
            else:
                inertial.inertia = inertial_tag.attrib.get('fullinertia')

            # add inertial element to body
            body.add_inertial(inertial)

    def _parse_geom(self, body, geom_tag):
        """
        Parse the geom tag if present, and set the visuals, and collisions to the given body. It can also set the
        inertial elements if it was not defined previously.

        From the main documentation [2]: "This element creates a geom, and attaches it rigidly to the body within
        which the geom is defined. Multiple geoms can be attached to the same body. At runtime they determine the
        appearance and collision properties of the body. At compile time they can also determine the inertial
        properties of the body, depending on the presence of the inertial element and the setting of the
        inertiafromgeom attribute of compiler. This is done by summing the masses and inertias of all geoms attached
        to the body with geom group in the range specified by the inertiagrouprange attribute of compiler. The geom
        masses and inertias are computed using the geom shape, a specified density or a geom mass which implies a
        density, and the assumption of uniform density.

        Attributes:
            - name (string, optional): Name of the geom.
            - class (string, optional): Defaults class for setting unspecified attributes.
            - type (string, [plane, hfield, sphere, capsule, ellipsoid, cylinder, box, mesh], "sphere"): Type of
              geometric shape
            - contype (int, "1"): This attribute and the next specify 32-bit integer bitmasks used for contact
              filtering of dynamically generated contact pairs. Two geoms can collide if the contype of one geom is
              compatible with the conaffinity of the other geom or vice versa. Compatible means that the two bitmasks
              have a common bit set to 1.

        Args:
            body (Body): body data structure instance.
            geom_tag (ET.Element): geom XML field.
        """
        # visual #

        attrib = geom_tag.attrib
        dtype = attrib.get('type')
        visual = Visual(name=attrib.get('name'), dtype=dtype, color=attrib.get('rgba'))

        # set position and orientation
        visual.position = attrib.get('pos')
        visual.orientation = self._get_orientation(attrib)

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
        material_name = attrib.get('material')  # TODO
        if 'material' in self.assets:  # and material in :
            # material =
            # visual.material =
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
        body.add_visual(visual)

        # collision #

        if not (attrib.get('contype') == "0" and attrib.get('conaffinity') == "0"):
            # copy collision shape information from visual shape
            collision = Collision()
            collision.name = visual.name
            collision.frame = visual.frame
            collision.geometry = visual.geometry
            body.add_collision(collision)

        # inertial #  just the mass

        # if the <inertial> tag was not given, compute based on information in geom
        if body.inertial is None:
            inertial = Inertial()

            # if mesh, load it in memory
            if dtype == 'mesh':
                mesh = trimesh.load(mesh)
                if not mesh.is_watertight:  # mesh.is_convex
                    raise ValueError("Could not compute the volume because the mesh is not watertight...")

            # get mass
            mass = inertial.mass
            if mass is None:

                # if the mass is defined
                if attrib.get('mass') is not None:
                    inertial.mass = attrib.get('mass')
                    mass = inertial.mass  # this makes the conversion to float

                # if the mass is not defined, compute it from the density
                else:
                    density = float(attrib.get('density', 1000.))
                    dimensions = float(attrib.get('fitscale', 1)) if dtype == 'mesh' else visual.size
                    mass = Inertial.compute_mass_from_density(shape=dtype, dimensions=dimensions, density=density,
                                                              mesh=mesh)
                    inertial.mass = mass

            # compute inertia if not given
            if inertial.inertia is None:
                dimensions = float(attrib.get('fitscale', 1)) if dtype == 'mesh' else visual.size
                inertia = Inertial.compute_inertia(shape=dtype, dimensions=dimensions, mass=mass, mesh=mesh)
                inertial.inertia = inertia

            # add inertial in body
            body.add_inertial(inertial)

    def _parse_site(self, body, site_tag, site_idx):
        """
        Parse the site XML field.

        From the main documentation [1]: "Sites are light geoms. They have the same appearance properties but cannot
        participate in collisions and cannot be used to infer body masses. On the other hand sites can do things that
        geoms cannot do: they can specify the volumes of touch sensors, the attachment of IMU sensors, the routing of
        spatial tendons, the end-points of slider-crank actuators. These are all spatial quantities, and yet they do
        not correspond to entities that should have mass or collide other entities - which is why the site element
        was created. Sites can also be used to specify points (or rather frames) of interest to the user."


        Args:
            body (Body): body data structure instance.
            site_tag (ET.Element): site XML field.
            site_idx (int): site index.
        """
        # visual #
        pass

    def _parse_contact(self, parent_tag):
        """
        Parse contact XML field.

        Args:
            parent_tag (ET.Element): parent XML element to check if it has a 'compiler' tag.
        """
        pass

    def _parse_equality_constraint(self, parent_tag):
        """
        Parse the equality XML field.

        Args:
            parent_tag (ET.Element): parent XML element to check if it has a 'compiler' tag.
        """
        pass

    def _parse_actuator(self, parent_tag):
        """
        Parse the actuator XML field.

        Args:
            parent_tag (ET.Element): parent XML element to check if it has a 'compiler' tag.
        """
        pass

    def _parse_sensor(self, parent_tag):
        """
        Parse the sensor XML field.

        Args:
            parent_tag (ET.Element): parent XML element to check if it has a 'compiler' tag.
        """
        pass

    #############
    # Generator #
    #############

    def _update_attribute_dict(self, dictionary, element, name, key=None, slice=None, fct=None, required=False,
                               *args, **kwargs):
        """
        Update the attribute dictionary given to an XML element.

        Args:
            dictionary (dict): attribute dictionary to update.
            element (object): data structure to check if it has the given attribute name.
            name (str): name of the attribute.
            key (str): name of the key to add in the dictionary. If None, it will take be the same as the given
              :attr:`name`.
            slice (slice): slice to apply if the attribute is a list/tuple/np.array.
            fct (callable): optional callable function to be applied on the given attribute before processing it.
              This function has to return the processed attribute.
            required (bool): if the attribute is required but was not present (i.e. it returned None), it will raise
              a `RuntimeError`.
            *args: list of arguments given to the provided function.
            **kwargs: dictionary of arguments given to the provided function.
        """
        attribute = getattr(element, name, None)
        if attribute is not None:
            # if key is None, set it to given name
            if key is None:
                key = name

            # if callable function is provided, call it on the attribute beforehand to process it
            if fct is not None:
                attribute = fct(attribute, *args, **kwargs)

            if attribute is not None:  # this is for the returned attribute by the fct
                # check attribute type and convert it to a string
                if isinstance(attribute, str):
                    dictionary[key] = attribute
                elif isinstance(attribute, bool):
                    dictionary[key] = 'true' if attribute else 'false'
                elif isinstance(attribute, (list, tuple, np.ndarray)):
                    attribute = np.asarray(attribute)
                    if slice is not None:
                        attribute = attribute[slice]
                    dictionary[key] = str(attribute)[1:-1]
                elif isinstance(attribute, (int, float)):
                    dictionary[key] = str(attribute)
                else:
                    raise NotImplementedError("The given attribute type is not supported: {}".format(type(attribute)))

        if attribute is None and required:
            raise RuntimeError("The given attribute '{}' was required but it was not defined in the given element: "
                               "{}".format(name, element))

    def generate(self, world=None):
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
        root = self.root
        if root is None:
            root = ET.Element('mujoco', attrib={'model': world.name})

        # create <compiler>
        self.generate_compiler(parent_tag=root)

        # create <option>
        self.generate_options(parent_tag=root, options=self.options)

        # create <default>
        self.generate_default(parent_tag=root, default=self.defaults)

        # create asset
        self.generate_assets(parent_tag=root, assets=self.assets)

        # create world
        self.generate_world(root, world)

        return root

    def generate_compiler(self, parent_tag, compiler=None):
        """
        Generate the compiler tag based on the `compiler` attribute of this class.

        From the main documentation [2]: "This element is used to set options for the built-in parser and compiler.
        After parsing and compilation it no longer has any effect. The settings here are global and apply to the
        entire model.

        Attributes:
            - boundmass (real, "0"): This attribute imposes a lower bound on the mass of each body except for the
              world body. It can be used as a quick fix for poorly designed models that contain massless moving
              bodies, such as the dummy bodies often used in URDF models to attach sensors. Note that in MuJoCo
              there is no need to create dummy bodies.
            - boundinertia (real, "0"): This attribute imposes a lower bound on the diagonal inertia components of
              each body except for the world body.
            - settotalmass (real, "-1"): If this value is positive, the compiler will scale the masses and inertias of
              all bodies in the model, so that the total mass equals the value specified here. The world body has mass
              0 and does not participate in any mass-related computations. This scaling is performed last, after all
              other operations affecting the body mass and inertia.
            - balanceinertia ([false, true], "false"): A valid diagonal inertia matrix must satisfy A+B>=C for all
              permutations of the three diagonal elements. Some poorly designed models violate this constraint, which
              will normally result in compile error. If this attribute is set to "true", the compiler will silently
              set all three diagonal elements to their average value whenever the above condition is violated.
            - etc
        "

        Args:
            parent_tag (ET.Element): parent XML element.
            compiler (dict): compiler dictionary. If None, it will take the one from this class.

        Returns:
            ET.Element: compiler XML tag.
        """
        # if compiler, create tag
        if compiler:
            return ET.SubElement(parent_tag, 'compiler', attrib=self.compiler)

    def generate_options(self, parent_tag, options=None):  # TODO
        """
        Generate the option tag.

        Args:
            parent_tag (ET.Element): parent XML element.
            options (dict): options dictionary.

        Returns:
            ET.Element: option XML tag.
        """
        if options:
            attrib = {}
            ET.SubElement(parent_tag, 'option', attrib=attrib)

    def generate_default(self, parent_tag, default=None):  # TODO
        """
        Generate the default tag.

        Args:
            parent_tag (ET.Element): parent XML element.
            default (dict): default dictionary. If None, it will take the one from this class.

        Returns:
            ET.Element: default tag
        """
        if default:
            attrib = {}
            ET.SubElement(parent_tag, 'asset', attrib=attrib)

    def generate_assets(self, parent_tag, assets=None):  # TODO
        """
        Generate the asset tag based on the 'asset' attribute of this class.

        Args:
            parent_tag (ET.Element): parent XML element.
            assets (dict): assets dictionary.
        """
        if assets:
            attrib = {}
            ET.SubElement(parent_tag, 'asset', attrib=attrib)

    def generate_world(self, parent_tag, world=None):
        """
        Generate the worldbody tag based on the given `World` data structure.

        Args:
            parent_tag (ET.Element): parent XML element (which normally should be the root XML tag)
            world (World): world data structure.
        """
        # check world argument
        if world is None:
            world = self.world
        if world is None:
            return
        if not isinstance(world, World):
            raise TypeError("Expecting the given 'world' to be an instance of `World`, but got instead: "
                            "{}".format(type(world)))

        # generate world #

        # create worldbody
        attrib = {}
        self._update_attribute_dict(attrib, world, 'name')
        self._update_attribute_dict(attrib, world, 'position', key='pos')
        self._update_attribute_dict(attrib, world, 'quaternion', key='quat', fct=self._convert_xyzw_to_wxyz)
        world_tag = parent_tag.find('worldbody')
        if world_tag is None:
            world_tag = ET.SubElement(parent_tag, 'worldbody', attrib=attrib)

        # create lights
        for light in world.lights:
            self.generate_light(world_tag, light)

        # create floor
        if world.floor is not None:
            self.generate_floor(world_tag, world.floor)

        # create multi-bodies
        for tree in world.trees:
            self.generate_tree(world_tag, tree, root=parent_tag)

        return world_tag

    # alias
    generate_worldbody = generate_world

    def generate_light(self, parent_tag, light):
        """
        Generate the light tag based on the `Light` data structure.

        Args:
            parent_tag (ET.Element): parent XML element.
            light (Light): light data structure structure.

        Returns:
            ET.Element: light XML element.
        """
        # check type
        if not isinstance(light, Light):
            raise TypeError("Expecting the given 'light' to be an instance of `Light`, but got instead: "
                            "{}".format(type(light)))

        # create <light>
        attrib = {}
        for name in ['name', 'castshadow', 'active']:
            self._update_attribute_dict(attrib, light, name)
        for name, key in zip(['position', 'direction'], ['pos', 'dir']):
            self._update_attribute_dict(attrib, light, name, key=key)
        for name in ['ambient', 'diffuse', 'specular']:
            self._update_attribute_dict(attrib, light, name, slice=slice(3))

        return ET.SubElement(parent_tag, 'light', attrib=attrib)

    def generate_floor(self, parent_tag, floor, assets=None):
        """
        Generate the floor 'geom' tag based on the `Floor` data structure.

        Args:
            parent_tag (ET.Element): parent XML element.
            floor (Floor): floor data structure structure.
            assets (dict): asset dictionary. If None, it will take the assets defined in this class.

        Returns:
            ET.Element: floor XML element.
        """
        # check type
        if not isinstance(floor, Floor):
            raise TypeError("Expecting the given 'floor' to be an instance of `Floor`, but got instead: "
                            "{}".format(type(floor)))

        if assets is None:
            assets = self.assets

        # create <geom>
        attrib = {'type': 'plane'}
        for name in ['name', 'rgba']:
            self._update_attribute_dict(attrib, floor, name)

        # check size / dimensions
        def check_size(size):
            if size is not None:
                if len(size) == 2:
                    # concatenate the 3rd element for spacing between square grid lines
                    size = np.concatenate((size, np.array([1.])))
            return size

        self._update_attribute_dict(attrib, floor, name='dimensions', key='size', fct=check_size)

        # check material / texture
        def check_texture(texture, material_name):
            if texture is not None:
                # check if material name in assets, if not, add it
                if material_name is None:
                    material_name = '__prl_material_' + str(self._material_cnt)
                    self._material_cnt += 1
                if material_name not in assets:
                    assets[material_name] = {'texture': texture}

                    # create new XML tag for that asset

                    # check if asset tag already created, if not create one
                    asset_tag = parent_tag.find('asset')
                    if asset_tag is None:
                        asset_tag = ET.SubElement(parent_tag, 'asset')

                    # create texture tag
                    texture_name = material_name + '_texture'
                    ET.SubElement(asset_tag, 'texture', attrib={'name': texture_name, 'file': texture})

                    # create material tag
                    ET.SubElement(asset_tag, 'material', attrib={'name': material_name, 'texture': texture_name})

            return texture

        self._update_attribute_dict(attrib, floor, name='texture', key='material', fct=check_texture,
                                    material_name=floor.material.name)

        # create <geom> tag
        return ET.SubElement(parent_tag, 'geom', attrib=attrib)

    def generate_tree(self, parent_tag, tree, root=None):
        """
        Generate tree / multibody tag.

        Args:
            parent_tag (ET.Element): parent XML element.
            tree (Tree): Tree / MultiBody data structure.
            root (ET.Element, None): root XML element.

        Returns:
            ET.Element: body XML element.
        """
        # check arguments
        if not isinstance(parent_tag, ET.Element):
            raise TypeError("Expecting the given 'parent_tag' to be an instance of `ET.Element`, but got instead: "
                            "{}".format(type(parent_tag)))
        if not isinstance(tree, Tree):
            raise TypeError("Expecting the given 'tree' to be an instance of `Tree`, but got instead: "
                            "{}".format(type(tree)))

        # update the body and joint positions because in MuJoCo "all elements in defined in the kinematic tree are
        # expressed in local coordinates, relative to the parent body for bodies, and relative to the body that owns
        # the element for geoms, joints, sites, cameras and lights", and a joint defined in a body connects that body
        # with its parent body.

        h_bodies, h_joints, h_visuals, h_collisions, h_inertials = {}, {}, {}, {}, {}
        for i, body in enumerate(tree.bodies.values()):
            print(body.name, body.homogeneous)
            h_inv = get_inverse_homogeneous(body.homogeneous)  # get link/joint frame
            for visual in body.visuals:  # because geom is described wrt body frame and not link/joint frame
                h_visuals[visual] = h_inv.dot(visual.homogeneous)
            for collision in body.collisions:  # because geom is described wrt body frame and not link/joint frame
                h_collisions[collision] = h_inv.dot(collision.homogeneous)
            for inertial in body.inertials:  # because inertial is described wrt body frame and not link/joint frame
                h_inertials[inertial] = h_inv.dot(inertial.homogeneous)
            for joint in body.joints.values():
                if joint.child is not None:
                    h_child = joint.child.homogeneous
                    h = h_inv.dot(joint.homogeneous).dot(h_child)
                    h_bodies[joint.child] = h  # because child body is described wrt parent body frame in MJC
                    h_joints[joint] = get_inverse_homogeneous(h_child)  # because joint are wrt child body frame in MJC

        self._h_bodies, self._h_joints = h_bodies, h_joints
        self._h_visuals, self._h_collisions, self._h_inertials = h_visuals, h_collisions, h_inertials

        # TODO: WARNING - this modify the given Tree!!!
        for body, homogeneous in h_bodies.items():
            body.homogeneous = homogeneous
        for joint, homogeneous in h_joints.items():
            joint.homogeneous = homogeneous
        for visual, homogeneous in h_visuals.items():
            visual.homogeneous = homogeneous
        for collision, homogeneous in h_collisions.items():
            collision.homogeneous = homogeneous
        for inertial, homogeneous in h_inertials.items():
            inertial.homogeneous = homogeneous

        # generate bodies (inertial, visual, collision) and joints in a recursive manner
        body_tag = self.generate_body(parent_tag, body=tree.root, root=root)

        # empty the temporary assets
        self._assets_tmp = set([])

        return body_tag

    def _convert_mesh(self, filename):
        """
        Convert mesh (from any format) to an STL mesh format. This is because MuJoCo only accepts STL meshes.

        Warnings: note that this method might create new STL files on your system as these files are not deleted here.

        Args:
            filename (str): path to the mesh file.

        Returns:
            str: filename with the correct extension (.stl)
        """
        # if mesh, make sure it is a STL file
        extension = filename.split('.')[-1]
        if extension.lower() != 'stl':
            # create filename with the correction extension (STL)
            dirname = os.path.dirname(filename)
            basename = os.path.basename(filename)
            basename_without_extension = ''.join(basename.split('.')[:-1])
            # filename_without_extension = dirname + basename_without_extension
            # new_filename = filename_without_extension + '.stl'
            new_filename = self._mesh_dirname + '/' + basename_without_extension + '.stl'

            # if file does not already exists, convert it
            if not os.path.isfile(new_filename):

                # # Arf, pyassimp export an ASCII STL, but Mujoco requires a binary STL --> use trimesh
                # scene = pyassimp.load(filename)
                # pyassimp.export(scene, new_filename, file_type='stl')
                # pyassimp.release(scene)

                export_mesh(trimesh.load(filename), new_filename)

            return new_filename
        return filename

    @staticmethod
    def _generate_name(name, cnt):
        """
        Generate a new name. Even if a link, joint, and other components have a name. If we need to load them multiple
        times, they will have the same and thus conflicts will arise. In order to avoid this, we will generate a new
        unique name by prefixing 'prl_' and suffixing '_str(cnt)'.

        Args:
            name (str): original name.
            cnt (int): counter number.

        Returns:
            str: new name.
        """
        if name is not None:
            name = 'prl_' + name + '_' + str(cnt)
        return name

    def generate_body(self, parent_tag, body, root=None):
        r"""
        Generate the body.

        Args:
            parent_tag (ET.Element): parent body XML element to which the given Body data structure will be added.
            body (Body): Body data structure.
            root (ET.Element): root element. If None, it will take the root element given in this class.

        Returns:
            ET.Element: body XML element.
        """
        # check arguments
        if not isinstance(parent_tag, ET.Element):
            raise TypeError("Expecting the given 'parent_tag' to be an instance of `ET.Element`, but got instead: "
                            "{}".format(type(parent_tag)))
        if not isinstance(body, Body):
            raise TypeError("Expecting the given 'body' to be an instance of `Body`, but got instead: "
                            "{}".format(type(body)))
        if root is None:
            root = self.root

        # create <body> tag
        attrib = {}
        self._body_cnt += 1
        self._update_attribute_dict(attrib, body, 'name', fct=self._generate_name, cnt=self._body_cnt)
        self._update_attribute_dict(attrib, body, 'position', key='pos')
        self._update_attribute_dict(attrib, body, 'quaternion', key='quat', fct=self._convert_xyzw_to_wxyz)
        body_tag = ET.SubElement(parent_tag, "body", attrib=attrib)

        # create <inertial> tag
        inertial = body.inertial
        inertial_tag = None
        if inertial is not None:
            attrib = {}

            def check_inertia(inertia):
                if inertia is not None:
                    inertia = inertia.inertia
                return inertia

            self._update_attribute_dict(attrib, inertial, 'inertia', key='fullinertia', fct=check_inertia)

            # create the <inertial> tag only if inertia is given, otherwise specify mass in <geom>
            if 'fullinertia' in attrib:

                # check that the inertia is not 0 0 0 0 0 0
                if not np.allclose(inertial.inertia.inertia, np.zeros(6)):
                    self._update_attribute_dict(attrib, inertial, 'position', key='pos')
                    self._update_attribute_dict(attrib, inertial, 'quaternion', key='quat',
                                                fct=self._convert_xyzw_to_wxyz)
                    self._update_attribute_dict(attrib, inertial, 'mass', required=True)
                    inertial_tag = ET.SubElement(body_tag, 'inertial', attrib=attrib)

        # create <geom> tags
        # TODO: consider when multiple visual and collision shapes
        visual = body.visual
        collision = body.collision
        if visual is None:  # if no visual, use collision shape
            if collision is not None:  # if collision shape is defined
                attrib = {"rgba": "0 0 0 0"}  # transparent
                self._update_attribute_dict(attrib, collision, 'name', fct=self._generate_name, cnt=self._body_cnt)
                self._update_attribute_dict(attrib, collision, 'position', key='pos')
                self._update_attribute_dict(attrib, collision, 'quaternion', key='quat', fct=self._convert_xyzw_to_wxyz)
                self._update_attribute_dict(attrib, collision, 'dtype', key='type')
                self._update_attribute_dict(attrib, collision, 'size')

                # check type and change size
                if 'type' in attrib and 'size' in attrib:
                    if collision.dtype == 'box':  # divide by 2 the dimensions
                        attrib['size'] = str(np.asarray(collision.size) / 2.)[1:-1]
                    elif collision.dtype == 'cylinder' or collision.dtype == 'capsule':  # divide by 2 the height
                        radius, length = collision.size
                        attrib['size'] = str(np.asarray([radius, length/2]))[1:-1]
                    elif collision.dtype == 'mesh':  # set the scale
                        attrib['fitscale'] = str(collision.size)

                if inertial is not None and inertial_tag is None:
                    self._update_attribute_dict(attrib, inertial, 'mass', required=True)

                # create geom tag
                geom = ET.SubElement(body_tag, "geom", attrib=attrib)

                # check if mesh in attrib
                if collision.dtype == 'mesh':
                    # check <asset> tag in xml
                    asset_tag = root.find("asset")

                    # if no <asset> tag, create one
                    if asset_tag is None:
                        asset_tag = ET.SubElement(root, "asset")

                    # create <mesh> tag in <asset>
                    mesh_path = self._convert_mesh(collision.filename)  # convert to STL if necessary
                    mesh_name = os.path.basename(mesh_path).split('.')[0]
                    if mesh_name not in self._assets_tmp:  # if the <mesh> doesn't already exists
                        self._assets_tmp.add(mesh_name)
                        attrib = {'name': mesh_name, 'file': mesh_path}
                        self._update_attribute_dict(attrib, collision, 'size', key='scale')
                        ET.SubElement(asset_tag, "mesh", attrib=attrib)

                    # set the mesh asset name
                    geom.attrib["mesh"] = mesh_name

        else:
            # if visual is given, use this one instead
            attrib = {}

            self._update_attribute_dict(attrib, visual, 'name', fct=self._generate_name, cnt=self._body_cnt)
            self._update_attribute_dict(attrib, visual, 'position', key='pos')
            self._update_attribute_dict(attrib, visual, 'quaternion', key='quat', fct=self._convert_xyzw_to_wxyz)
            self._update_attribute_dict(attrib, visual, 'dtype', key='type')
            self._update_attribute_dict(attrib, visual, 'size')
            self._update_attribute_dict(attrib, visual, 'rgba')
            if inertial is not None and inertial_tag is None:
                self._update_attribute_dict(attrib, inertial, 'mass', required=True)

            # check type and change size
            if 'type' in attrib and 'size' in attrib:
                if visual.dtype == 'box':  # divide by 2 the dimensions
                    attrib['size'] = str(np.asarray(visual.size) / 2.)[1:-1]
                elif visual.dtype == 'cylinder' or visual.dtype == 'capsule':  # divide by 2 the height
                    radius, length = visual.size
                    attrib['size'] = str(np.asarray([radius, length / 2]))[1:-1]
                elif visual.dtype == 'mesh':  # set the scale
                    attrib['fitscale'] = str(visual.size)

            # create <geom> tag
            geom = ET.SubElement(body_tag, "geom", attrib=attrib)

            # if primitive shape type is a mesh
            if visual.dtype == "mesh":
                # check <asset> tag in xml
                asset_tag = root.find("asset")

                # if no <asset> tag, create one
                if asset_tag is None:
                    asset_tag = ET.SubElement(root, "asset")

                # create mesh tag
                mesh_path = self._convert_mesh(visual.filename)  # convert to STL if necessary
                mesh_name = os.path.basename(mesh_path).split('.')[0]
                if mesh_name not in self._assets_tmp:  # if the mesh doesn't already exists
                    self._assets_tmp.add(mesh_name)
                    attrib = {'name': mesh_name, 'file': mesh_path}
                    self._update_attribute_dict(attrib, visual, 'size', key='scale')
                    ET.SubElement(asset_tag, "mesh", attrib=attrib)

                # set the mesh asset name
                geom.attrib["mesh"] = mesh_name

            # if no collision shape
            if collision is None:
                geom.attrib["contype"] = "0"
                geom.attrib["conaffinity"] = "0"

        # create <joint>
        for joint in body.parent_joints.values():  # parent_joints
            self.generate_joint(body_tag, joint)

        # create inner <body>
        for body in body.bodies:
            self.generate_body(body_tag, body, root=root)

        return body_tag

    def generate_joint(self, parent_tag, joint):
        """
        Generate the joint.

        Args:
            parent_tag (ET.Element): parent XML element.
            joint (Joint): Joint data structure.

        Returns:
            ET.Element, None: joint XML element. None if it couldn't generate the joint, this can happen if for
              instance, the joint type is not known.
        """
        # check type
        if not isinstance(joint, Joint):
            raise TypeError("Expecting the given 'joint' to be an instance of `Joint`, but got instead: "
                            "{}".format(type(joint)))

        # create <joint>
        attrib = {}
        self._joint_cnt += 1
        self._update_attribute_dict(attrib, joint, 'name', fct=self._generate_name, cnt=self._joint_cnt)
        for name in ['axis', 'damping']:
            self._update_attribute_dict(attrib, joint, name)
        for name, key in zip(['limits', 'friction', 'position'],
                             ['range', 'frictionloss', 'pos']):
            self._update_attribute_dict(attrib, joint, name, key=key)
        if 'range' in attrib:
            attrib['limited'] = 'true'

        # check the joint type
        def check_joint_type(dtype):
            if dtype == 'revolute' or dtype == 'continuous':
                return 'hinge'
            if dtype == 'prismatic':
                return 'slide'
            if dtype == 'floating':
                return 'free'
            if dtype in {'ball', 'hinge', 'slide', 'ball'}:
                return dtype
            if dtype == 'fixed':
                return None
            print("WARNING: Unknown joint type during generation: {}".format(dtype))
            return None

        self._update_attribute_dict(attrib, joint, name='dtype', key='type', fct=check_joint_type)

        # return if the joint type is known
        if 'type' in attrib:
            return ET.SubElement(parent_tag, "joint", attrib=attrib)

    def add_multibody(self, tree, mesh_directory_path=''):
        r"""
        Add the given tree / multi-body data structure.

        Args:
            tree (Tree, Body): multi-body data structure. If it is a body instance, it will automatically be
              wrapped in a Tree instance.
            mesh_directory_path (str): path to the mesh directory to add the converted meshes.

        Returns:
            ET.Element: body XML element
        """
        if not isinstance(tree, Tree):
            if isinstance(tree, Body):
                tree = Tree(name=tree.name, root=tree, position=tree.position, orientation=tree.orientation)
            else:
                raise TypeError("Expecting the given 'tree' to be an instance of `Tree` or `Body` but got "
                                "instead: {}".format(type(tree)))

        self._mesh_dirname = mesh_directory_path if isinstance(mesh_directory_path, str) else ''

        # generate tree
        return self.generate_tree(parent_tag=self.worldbody, tree=tree, root=self.root)

    # alias
    add_tree = add_multibody
