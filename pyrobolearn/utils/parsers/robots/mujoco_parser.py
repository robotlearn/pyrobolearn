#!/usr/bin/env python
"""Define the MuJoCo parser/generator.

Notes:
    - MuJoCo only accepts STL meshes
    - MuJoCo can load PNG files for textures and heightmap.

References:
    - MuJoCo overview: http://www.mujoco.org/book/index.html
    - MuJoCo XML format: http://www.mujoco.org/book/XMLreference.html
"""

import numpy as np
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
    raise ImportError(str(e) + "\nTry to install trimesh pyassimp: `pip install trimesh pyassimp`")

from pyrobolearn.utils.parsers.robots.world_parser import WorldParser
from pyrobolearn.utils.parsers.robots.data_structures import Simulator, World, Tree, Body, Joint, Inertial, \
    Visual, Collision, Light
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
        self.defaults = dict()  # default values for the attributes when they are not specified
        self.assets = dict()    # assets (textures, meshes, etc)

        # create root XML element
        self.create_root("mujoco")
        self.worldbody = self.add_element(name="worldbody", parent_element=self.root)

        # set some counters
        self._world_cnt = 0
        self._tree_cnt = 0
        self._body_cnt = 0
        self._joint_cnt = 0
        # self._geom_cnt = 0
        # self._site_cnt = 0

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
            parent_tag (ET.Element): parent XML element to check if it has a 'compiler' tag.
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
            parent_tag (ET.Element): parent XML element to check if it has a 'compiler' tag.
            update_world_attribute (bool): if we should
        """
        worldbody_tag = parent_tag.find('worldbody')
        if worldbody_tag is not None:

            # instantiate the world data structure
            name = parent_tag.attrib.get('model')
            if name is None:
                name = 'prl_world_' + str(self._world_cnt)
                self._world_cnt += 1
            world = World(name=name)

            # light: This element creates a light, which moves with the body where it is defined.
            # light attributes: name, class, mode, target, directional, castshadow, active, pos, dir, attenuation,
            #                   cutoff, exponent, ambient, diffuse, specular
            for i, light_tag in enumerate(worldbody_tag.findall('light')):
                attrib = light_tag.attrib
                light = Light(name=attrib.get('name', 'light_' + str(i)), cast_shadows=attrib.get('castshadow'),
                              position=attrib.get('pos'), direction=attrib.get('dir'), ambient=attrib.get('ambient'),
                              diffuse=attrib.get('diffuse'), specular=attrib.get('specular'))

                world.lights[light.name] = light

            # check each (multi-)body
            for i, body_tag in enumerate(worldbody_tag.findall('body')):
                # create tree
                name = body_tag.attrib.get('name')
                if name is None:
                    name = 'prl_multibody_' + str(self._tree_cnt)
                    self._tree_cnt += 1
                tree = Tree(name=name)

                # check recursively body
                self._parse_body(tree, body_tag=body_tag)

                # save tree
                world.trees[tree.name] = tree

            # update the world attribute of this class if specified.
            if update_world_attribute:
                self.world = world

    def _parse_body(self, tree, body_tag, parent_body=None):  # TODO: check with self.defaults
        """
        Construct recursively the given tree, and return Body instance from a <body>.

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
            name = 'prl_body_' + str(self._body_cnt)
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
            self._parse_geom(body=body, geom_tag=geom_tag, geom_idx=i)

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
            name = 'prl_joint_' + str(self._joint_cnt)
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

    def _parse_geom(self, body, geom_tag, geom_idx):
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
            geom_idx (int): geom index.
        """
        ##########
        # visual #
        ##########

        attrib = geom_tag.attrib
        dtype = attrib.get('type')
        visual = Visual(name=attrib.get('name'), dtype=dtype, color=attrib.get('rgba'))

        # get position and orientation
        visual.pos = attrib.get('pos')
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
        body.visuals = visual

        #############
        # collision #
        #############

        if not (attrib.get('contype') == "0" and attrib.get('conaffinity') == "0"):
            # copy collision shape information from visual shape
            collision = Collision()
            collision.name = visual.name
            collision.frame = visual.frame
            collision.geometry = visual.geometry
            body.collisions = collision

        ############
        # inertial #  just the mass
        ############

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
        ##########
        # visual #
        ##########
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

    def generate_body(self, parent_body, body):
        r"""
        Generate the body.

        Args:
            parent_body (ET.Element): parent body XML element to which the given Body data structure will be added.
            body (Body): Body data structure.

        Returns:
            ET.Element: body XML element.
        """
        if not isinstance(parent_body, ET.Element):
            raise TypeError("Expecting the given 'parent_body' to be an instance of `ET.Element`, but got instead: "
                            "{}".format(type(parent_body)))
        if not isinstance(body, Body):
            raise TypeError("Expecting the given 'body' to be an instance of `Body`, but got instead: "
                            "{}".format(type(body)))

        body_tag = ET.SubElement(parent_body, "body", attrib={"name": "body_" + str(self._body_cnt),
                                                              "pos": pos, "quat": quat})

        # create <body/joint> tag in xml

        joint = ET.SubElement(body, "joint", attrib={"type": "free"})

        # create <body/geom> tag in xml
        if visual is None:  # if no visual, use collision shape
            geom = ET.SubElement(body, "geom", attrib={"type": collision.dtype,
                                                       "size": str(np.asarray(collision.size).reshape(-1))[1:-1],
                                                       "rgba": "0 0 0 0", "mass": str(mass)})  # transparent

            # if primitive shape type is a mesh
            if collision.dtype == "mesh":
                # check <asset> tag in xml
                asset = self._root.find("asset")

                # if no <asset> tag, create one
                if asset is None:
                    asset = ET.SubElement(self._root, "asset")

                # create mesh tag
                mesh_name = "mesh_" + str(collision.id)
                mesh = ET.SubElement(asset, "mesh", attrib={"name": mesh_name,
                                                            "file": collision.mesh,
                                                            "scale": str(np.asarray(collision.size)[1:-1])})

                # set the mesh asset name
                geom.attrib["mesh"] = mesh_name
        else:
            # if visual is given, use this one instead
            geom = ET.SubElement(body, "geom", attrib={"type": visual.dtype,
                                                       "size": str(np.asarray(visual.size).reshape(-1))[1:-1],
                                                       "rgba": str(np.asarray(visual.color))[1:-1],
                                                       "mass": str(mass)})

            # if primitive shape type is a mesh
            if visual.dtype == "mesh":
                # check <asset> tag in xml
                asset = self._root.find("asset")

                # if no <asset> tag, create one
                if asset is None:
                    asset = ET.SubElement(self._root, "asset")

                # create mesh tag
                mesh_name = "mesh_" + str(visual.id)
                mesh = ET.SubElement(asset, "mesh", attrib={"name": mesh_name,
                                                            "file": visual.mesh,
                                                            "scale": str(np.asarray(visual.size)[1:-1])})

                # set the mesh asset name
                geom.attrib["mesh"] = mesh_name

            # if no collision shape
            if collision is None:
                geom.attrib["contype"] = "0"
                geom.attrib["conaffinity"] = "0"

    def add_tree(self, tree):
        r"""
        Add the given tree / multi-body data structure.

        Args:
            tree (Tree, Body): multi-body data structure. If it is a body instance, it will automatically be
              wrapped in a Tree instance.

        Returns:
            ET.Element: body XML element
        """
        if not isinstance(tree, Tree):
            if isinstance(tree, Body):
                tree = Tree(name=tree.name, root=tree, position=tree.position, orientation=tree.orientation)
            else:
                raise TypeError("Expecting the given 'tree' to be an instance of `Tree` or `Body` but got "
                                "instead: {}".format(type(tree)))

        pass
