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
    import pymesh    # rapid prototyping platform focused on geometry processing
    # doc: https://pymesh.readthedocs.io/en/latest/user_guide.html

    import pyassimp  # library to import and export various 3d-model-formats
    # doc: http://www.assimp.org/index.php
    # github: https://github.com/assimp/assimp
except ImportError as e:
    raise ImportError(str(e) + "\nTry to install pymesh pyassimp: `pip install pymesh pyassimp`")

from pyrobolearn.utils.parsers.robots.world_parser import WorldParser
from pyrobolearn.utils.parsers.robots.data_structures import *


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
        self.assets = OrderedDict()
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

        # check assets
        asset_tag = root.find('asset')
        if asset_tag is not None:
            pass

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

            # check each multi-body
            for i, body_tag in enumerate(worldbody_tag.findall('body')):
                # create tree
                tree = Tree(name=body_tag.attrib.get('name', 'prl_multibody_' + str(i)))

                # check recursively body
                body = self._check_body(tree, body_tag, idx=i)
                world.trees[tree.name] = tree

        # check contact

        # check equality constraint

        # check actuator

        # check sensor

        # set the world
        self.world = world

    def _check_model(self, model_tag, idx):
        """
        Return the Tree instance from a <model>.

        Args:
            model_tag (ET.Element): model XML element
            idx (int): model index.

        Returns:
            Tree: tree data structure containing the model.
        """
        # create tree
        tree = Tree(name=model_tag.attrib.get('name'))

        # check bodies/links
        for i, link_tag in enumerate(model_tag.findall('link')):
            body = self._check_body(link_tag, idx=i)
            # add body to tree
            tree.bodies[body.name] = body

        # check joints
        for i, joint_tag in enumerate(root.findall('joint')):
            # get joint instance from tag
            joint = self._check_joint(joint_tag, idx=i)

            # add joint in trees
            tree.joints[joint.name] = joint

            # add joint in parent body
            parent_body = tree.bodies[joint.parent]
            parent_body.joints[joint.name] = joint

        return tree

    def _check_body(self, tree, body_tag, idx):
        """
        Construct recursively the given tree, and return Body instance from a <body>.

        Args:
            tree (Tree): tree data structure containing the model.
            body_tag (ET.Element): body XML element.
            idx (int): link index.

        Returns:
            Body: body data structure.
        """
        # create body
        body = Body(body_id=idx, name=body_tag.attrib.get('name', 'prl_body_' + str(idx)))

        # check geom

        # check joints

        # check bodies

        # check include

    def _check_joint(self, joint_tag, idx):
        """
        Return Joint instance from a <joint> tag.

        Args:
            joint_tag (ET.Element): joint XML element.
            idx (int): joint index.

        Returns:
            Joint: joint data structure.
        """
        pass

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
        root = ET.Element('mujoco', attrib={'model': world.name})

        # create compiler

        # create asset

        # create world
        worldbody_tag = ET.SubElement(root, 'worldbody')  # name = 'world'

        # create models
        for tree in world.trees:

            # create bodies
            for i, body in enumerate(tree.bodies):
                pass
