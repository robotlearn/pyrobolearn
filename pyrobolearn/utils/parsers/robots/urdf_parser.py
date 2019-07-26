#!/usr/bin/env python
"""Define the URDF parser.

URDF files are notably used in ROS, Gazebo, Bullet, Dart, and MuJoCo.
"""

# import XML parser
import xml.etree.ElementTree as ET

from pyrobolearn.utils.parsers.robots.robot_parser import RobotParser
from pyrobolearn.utils.parsers.robots.data_structures import *


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2019, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class URDFParser(RobotParser):
    r"""URDF Parser"""

    def __init__(self, filename=None):
        """
        Initialize the URDF parser.

        Args:
            filename (str, None): path to the MuJoCo XML file.
        """
        super().__init__(filename)

    def parse(self, filename):
        """
        Load and parse the given URDF file.

        Args:
            filename (str): path to the MuJoCo XML file.
        """
        # load and parse the XML file
        tree_xml = ET.parse(filename)

        # get the root
        root = tree_xml.getroot()

        # check that the root is <robot>
        if root.tag != 'robot':
            raise RuntimeError("Expecting the first XML tag to be 'robot' but found instead: {}".format(root.tag))

        # build the tree
        tree = Tree(name=root.attrib.get('name'))

        # check materials
        for i, material in enumerate(root.findall('material')):
            attrib = material.attrib
            mat = Material(name=attrib.get('name', 'material_' + str(i)), color=attrib.get('color'))
            tree.materials[mat.name] = mat

        # check bodies / links
        for i, body in enumerate(root.findall('link')):
            attrib = body.attrib
            b = Body(body_id=i, name=attrib.get('name', 'body_' + str(i)))

            # check <inertial> tag
            inertial = body.find('inertial')
            if inertial is not None:
                i = Inertial()

                # origin
                origin = inertial.find('origin')
                if origin is not None:
                    i.position = origin.attrib.get('xyz')
                    i.orientation = origin.attrib.get('rpy')

                # mass
                mass = inertial.find('mass')
                if mass is not None:
                    i.mass = mass.attrib.get('value')

                # inertia
                inertia = inertial.find('inertia')
                if inertia is not None:
                    i.inertia = {name: inertia.attrib.get(name) for name in ['ixx', 'ixy', 'ixz', 'iyy', 'iyz', 'izz']}

                # set inertial to body
                b.inertial = i

            # check <visual> tag
            visual = body.find('visual')
            if visual is not None:
                v = Visual()

                # name
                v.name = visual.attrib.get('name')

                # origin
                origin = visual.find('origin')
                if origin is not None:
                    v.position = origin.attrib.get('xyz')
                    v.orientation = origin.attrib.get('rpy')

                # geometry
                geometry = visual.find('geometry')
                if geometry is not None:
                    for geometry_type in ['box', 'mesh', 'cylinder', 'sphere']:
                        geom = geometry.find(geometry_type)
                        if geom is not None:
                            dtype = geometry_type
                            v.dtype = dtype
                            if dtype == 'box':
                                v.size = geom.attrib['size']
                            elif dtype == 'sphere':
                                v.size = geom.attrib['radius']
                            elif dtype == 'cylinder':
                                v.size = (geom.attrib['radius'], geom.attrib['length'])
                            elif dtype == 'mesh':
                                v.filename = geom.attrib['filename']
                                v.size = geom.attrib.get('scale')

                # material
                material = visual.find('material')
                if material is not None:
                    name = material.attrib.get('name')
                    color = material.find('color')
                    if color is not None:
                        v.color = color.attrib['rgba']
                    else:
                        mat = tree.materials.get(name)
                        if mat is not None:
                            v.material = mat

                # set visual to body
                b.visual = v

            # check <collision> tag
            collision = body.find('collision')
            if collision is not None:
                c = Collision()

                # name
                c.name = collision.attrib.get('name')

                # origin
                origin = collision.find('origin')
                if origin is not None:
                    c.position = origin.attrib.get('xyz')
                    c.orientation = origin.attrib.get('rpy')

                # geometry
                geometry = collision.find('geometry')
                if geometry is not None:
                    for geometry_type in ['box', 'mesh', 'cylinder', 'sphere']:
                        geom = geometry.find(geometry_type)
                        if geom is not None:
                            dtype = geometry_type
                            c.dtype = dtype
                            if dtype == 'box':
                                c.size = geom.attrib['size']
                            elif dtype == 'sphere':
                                c.size = geom.attrib['radius']
                            elif dtype == 'cylinder':
                                c.size = (geom.attrib['radius'], geom.attrib['length'])
                            elif dtype == 'mesh':
                                c.filename = geom.attrib['filename']
                                c.size = geom.attrib.get('scale')

                # set collision to body
                b.collision = c

            # add body to tree
            tree.bodies[b.name] = b

        # check joints
        for i, joint in enumerate(root.findall('joint')):
            attrib = joint.attrib
            j = Joint(joint_id=i, name=attrib.get('name', 'joint_' + str(i)), dtype=attrib['type'])

            # add parent and child body/link
            parent = joint.find('parent')
            if parent is None:
                raise RuntimeError("Expecting the joint '" + j.name + "' to have a parent link/body")
            j.parent = parent.attrib['link']

            child = joint.find('child')
            if child is None:
                raise RuntimeError("Expecting the joint '" + j.name + "' to have a child link/body")
            j.child = child.attrib['link']

            # origin
            origin = joint.find('origin')
            if origin is not None:
                j.position = origin.attrib.get('xyz')
                j.orientation = origin.attrib.get('rpy')

            # axis
            axis = joint.find('axis')
            if axis is not None:
                j.axis = axis.attrib.get('xyz')

            # dynamics
            dynamics = joint.find('dynamics')
            if dynamics is not None:
                j.damping = dynamics.attrib.get('damping')
                j.friction = dynamics.attrib.get('friction')

            # limits
            limits = joint.find('limits')
            if limits is not None:
                j.effort = limits.attrib.get('effort')
                j.velocity = limits.attrib.get('velocity')
                lower_limit = limits.attrib.get('lower')
                upper_limit = limits.attrib.get('upper')
                if lower_limit is not None and upper_limit is not None:  # TODO: check if we can have one limit
                    j.limits = [lower_limit, upper_limit]

            # add joint in trees
            tree.joints[j.name] = j

            # add joint in parent body
            tree.bodies[j.parent] = j

        # set the tree
        self.tree = tree

    def generate(self, tree=None):
        """
        Generate the XML tree from the `Tree` data structure.

        Args:
            tree (Tree): Tree data structure.

        Returns:
            ET.Element: root element in the XML file.
        """
        if tree is None:
            tree = self.tree

        pass
