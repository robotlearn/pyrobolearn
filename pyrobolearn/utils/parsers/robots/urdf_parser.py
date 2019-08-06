#!/usr/bin/env python
"""Define the URDF parser/generator.

URDF files are notably used in ROS, Gazebo, Bullet, Dart, and MuJoCo.

References:
    - URDF XML specifications: http://wiki.ros.org/urdf/XML
    - Tutorial: Using a URDF in Gazebo: http://gazebosim.org/tutorials/?tut=ros_urdf
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
    r"""URDF Parser and Generator"""

    def __init__(self, filename=None):
        """
        Initialize the URDF parser.

        Args:
            filename (str, None): path to the URDF XML file.
        """
        super().__init__(filename)

    def parse(self, filename):
        """
        Load and parse the given URDF file.

        Args:
            filename (str): path to the URDF XML file.
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
        for i, material_tag in enumerate(root.findall('material')):
            attrib = material_tag.attrib
            material = Material(name=attrib.get('name', 'material_' + str(i)))
            color_tag = material_tag.find('color')
            if color_tag is not None:
                material.color = color_tag.attrib.get('rgba')
            texture_tag = material_tag.find('texture')
            if texture_tag is not None:
                material.texture = texture_tag.attrib.get('filename')
            tree.materials[material.name] = material

        # check bodies / links
        for i, body_tag in enumerate(root.findall('link')):
            # get body instance from tag
            body = self._check_body(tree, body_tag, idx=i)

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

        # TODO: check sensor, plugins, transmission, etc

        # set the tree
        self.tree = tree

    @staticmethod
    def _check_body(tree, body_tag, idx):
        """
        Return Body instance from a <link> tag.

        Args:
            tree (Tree): Tree data structure.
            body_tag (ET.Element): link XML element.
            idx (int): link index.

        Returns:
            Body: body data structure.
        """
        attrib = body_tag.attrib
        body = Body(body_id=idx, name=attrib.get('name', 'body_' + str(idx)))

        # check <inertial> tag
        inertial_tag = body_tag.find('inertial')
        if inertial_tag is not None:
            inertial = Inertial()

            # origin
            origin_tag = inertial_tag.find('origin')
            if origin_tag is not None:
                inertial.position = origin_tag.attrib.get('xyz')
                inertial.orientation = origin_tag.attrib.get('rpy')

            # mass
            mass_tag = inertial_tag.find('mass')
            if mass_tag is not None:
                inertial.mass = mass_tag.attrib.get('value')

            # inertia
            inertia_tag = inertial_tag.find('inertia')
            if inertia_tag is not None:
                inertial.inertia = {name: inertia_tag.attrib.get(name)
                                    for name in ['ixx', 'ixy', 'ixz', 'iyy', 'iyz', 'izz']}

            # set inertial to body
            body.inertial = inertial

        # check <visual> tag
        visual_tag = body_tag.find('visual')
        if visual_tag is not None:
            visual = Visual()

            # name
            visual.name = visual_tag.attrib.get('name')

            # origin
            origin_tag = visual_tag.find('origin')
            if origin_tag is not None:
                visual.position = origin_tag.attrib.get('xyz')
                visual.orientation = origin_tag.attrib.get('rpy')

            # geometry
            geometry_tag = visual_tag.find('geometry')
            if geometry_tag is not None:
                for geometry_type in {'box', 'mesh', 'cylinder', 'sphere'}:
                    geometry_type_tag = geometry_tag.find(geometry_type)
                    if geometry_type_tag is not None:
                        dtype = geometry_type
                        visual.dtype = dtype
                        if dtype == 'box':
                            visual.size = geometry_type_tag.attrib['size']
                        elif dtype == 'sphere':
                            visual.size = geometry_type_tag.attrib['radius']
                        elif dtype == 'cylinder':
                            visual.size = (geometry_type_tag.attrib['radius'], geometry_type_tag.attrib['length'])
                        elif dtype == 'mesh':
                            visual.filename = geometry_type_tag.attrib['filename']
                            visual.size = geometry_type_tag.attrib.get('scale')

            # material
            material_tag = visual_tag.find('material')
            if material_tag is not None:
                material = Material()
                name = material_tag.attrib.get('name')
                color = material_tag.find('color')
                texture = material_tag.find('texture')
                if color is not None or texture is not None:
                    material.name = name
                    if color is not None:
                        material.color = color.attrib['rgba']
                    elif texture is not None:
                        material.texture = texture.attrib['filename']
                else:
                    material = tree.materials.get(name)
                visual.material = material

            # set visual to body
            body.visual = visual

        # check <collision> tag
        collision_tag = body_tag.find('collision')
        if collision_tag is not None:
            collision = Collision()

            # name
            collision.name = collision_tag.attrib.get('name')

            # origin
            origin_tag = collision_tag.find('origin')
            if origin_tag is not None:
                collision.position = origin_tag.attrib.get('xyz')
                collision.orientation = origin_tag.attrib.get('rpy')

            # geometry
            geometry_tag = collision_tag.find('geometry')
            if geometry_tag is not None:
                for geometry_type in {'box', 'mesh', 'cylinder', 'sphere'}:
                    geometry_type_tag = geometry_tag.find(geometry_type)
                    if geometry_type_tag is not None:
                        dtype = geometry_type
                        collision.dtype = dtype
                        if dtype == 'box':
                            collision.size = geometry_type_tag.attrib['size']
                        elif dtype == 'sphere':
                            collision.size = geometry_type_tag.attrib['radius']
                        elif dtype == 'cylinder':
                            collision.size = (geometry_type_tag.attrib['radius'], geometry_type_tag.attrib['length'])
                        elif dtype == 'mesh':
                            collision.filename = geometry_type_tag.attrib['filename']
                            collision.size = geometry_type_tag.attrib.get('scale')

            # set collision to body
            body.collision = collision

        return body

    @staticmethod
    def _check_joint(joint_tag, idx):
        """
        Return Joint instance from a <joint> tag.

        Args:
            joint_tag (ET.Element): joint XML element.
            idx (int): joint index.

        Returns:
            Joint: joint data structure.
        """
        attrib = joint_tag.attrib
        joint = Joint(joint_id=idx, name=attrib.get('name', 'joint_' + str(idx)), dtype=attrib['type'])

        # add parent and child body/link
        parent_tag = joint_tag.find('parent')
        if parent_tag is None:
            raise RuntimeError("Expecting the joint '" + joint.name + "' to have a parent link/body")
        joint.parent = parent_tag.attrib['link']

        child_tag = joint_tag.find('child')
        if child_tag is None:
            raise RuntimeError("Expecting the joint '" + joint.name + "' to have a child link/body")
        joint.child = child_tag.attrib['link']

        # origin
        origin_tag = joint_tag.find('origin')
        if origin_tag is not None:
            joint.position = origin_tag.attrib.get('xyz')
            joint.orientation = origin_tag.attrib.get('rpy')

        # axis
        axis_tag = joint_tag.find('axis')
        if axis_tag is not None:
            joint.axis = axis_tag.attrib.get('xyz')

        # dynamics
        dynamics_tag = joint_tag.find('dynamics')
        if dynamics_tag is not None:
            joint.damping = dynamics_tag.attrib.get('damping')
            joint.friction = dynamics_tag.attrib.get('friction')

        # limits
        limits_tag = joint_tag.find('limits')
        if limits_tag is not None:
            joint.effort = limits_tag.attrib.get('effort')
            joint.velocity = limits_tag.attrib.get('velocity')
            lower_limit = limits_tag.attrib.get('lower')
            upper_limit = limits_tag.attrib.get('upper')
            if lower_limit is not None and upper_limit is not None:  # TODO: check if we can have one limit
                joint.limits = [lower_limit, upper_limit]

        return joint

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

        # create root element
        root = ET.Element('robot')

        # generate material tags
        for material in tree.materials:
            material_tag = ET.SubElement(root, 'material', attrib={'name': material.name})
            if material.color is not None:
                ET.SubElement(material_tag, 'color', attrib={'rgba': str(np.asarray(material.rgba))[1:-1]})
            if material.texture is not None:
                ET.SubElement(material_tag, 'texture', attrib={'filename': material.texture})

        # define some common functions
        def set_name(parent_tag, tag, item):
            attrib = {}
            if item.name is not None:
                attrib['name'] = item.name
            new_tag = ET.SubElement(parent_tag, tag, attrib=attrib)
            return new_tag

        def set_origin(tag, item):
            origin = {}
            if item.position is not None:
                origin['xyz'] = str(np.asarray(item.position))[1:-1]
            if item.orientation is not None:
                origin['rpy'] = str(np.asarray(item.orientation))[1:-1]
            if len(origin) > 0:
                ET.SubElement(tag, 'origin', attrib=origin)

        def set_geometry(tag, item):
            if item.geometry is not None:
                geometry_tag = ET.SubElement(tag, 'geometry')
                geometry = item.geometry
                dtype = geometry.dtype

                if dtype in {'box', 'sphere', 'cylinder', 'mesh'}:
                    attrib = {}
                    if dtype == 'box':
                        attrib['size'] = str(np.asarray(geometry.size))[1:-1]
                    elif dtype == 'sphere':
                        attrib['radius'] = str(geometry.size)
                    elif dtype == 'cylinder':
                        attrib['radius'] = str(geometry.size[0])
                        attrib['length'] = str(geometry.size[1])
                    else:  # mesh
                        attrib['filename'] = geometry.filename
                        attrib['scale'] = str(np.asarray(geometry.size))[1:-1]

                    ET.SubElement(geometry_tag, dtype, attrib=attrib)

        # generate <link>
        for link in tree.bodies:
            link_tag = ET.SubElement(root, 'link', attrib={'name': link.name})

            # create <inertial> tag
            inertial = link.inertial
            if inertial is not None:
                inertial_tag = ET.SubElement(link_tag, 'inertial')

                # <origin>
                set_origin(inertial_tag, inertial)

                # <mass>
                if inertial.mass is not None:
                    ET.SubElement(inertial_tag, 'mass', attrib={'value': str(inertial.mass)})

                # <inertia>
                if inertial.inertia is not None:
                    I = inertial.inertia
                    inertia = {}
                    if I.ixx is not None:
                        inertia['ixx'] = str(I.ixx)
                    if I.iyy is not None:
                        inertia['iyy'] = str(I.iyy)
                    if I.izz is not None:
                        inertia['izz'] = str(I.izz)
                    if I.ixy is not None:
                        inertia['ixy'] = str(I.ixy)
                    if I.ixz is not None:
                        inertia['ixz'] = str(I.ixz)
                    if I.iyz is not None:
                        inertia['iyz'] = str(I.iyz)
                    ET.SubElement(inertial_tag, 'inertia', attrib=inertia)

            # create <visual> tag
            visual = link.visual
            if visual is not None:
                # create visual tag with name
                visual_tag = set_name(link_tag, 'visual', visual)

                # <origin>
                set_origin(visual_tag, visual)

                # <geometry>
                set_geometry(visual_tag, visual)

                # <material>
                if visual.material is not None:
                    material = visual.material
                    material_tag = ET.SubElement(visual_tag, 'material', attrib={'name': material.name})
                    if material.color is not None:
                        ET.SubElement(material_tag, 'color', attrib={'rgba': str(np.asarray(material.rgba))[1:-1]})
                    if material.texture is not None:
                        ET.SubElement(material_tag, 'texture', attrib={'filename': material.texture})

            # create <collision> tag
            collision = link.collision
            if collision is not None:
                # create collision tag with name
                collision_tag = set_name(link_tag, 'collision', collision)

                # <origin>
                set_origin(collision_tag, collision)

                # <geometry>
                set_geometry(collision_tag, collision)

        def set_name_and_type(parent_tag, tag, item):
            kwargs = {}
            if item.name is not None:
                kwargs['name'] = item.name
            if item.dtype is not None:
                kwargs['type'] = item.dtype
            return ET.SubElement(parent_tag, tag, attrib=kwargs)

        # generate <joint>
        for joint in tree.joints:
            # set joint name and type
            joint_tag = set_name_and_type(root, 'joint', joint)

            # <origin>
            set_origin(joint_tag, joint)

            # <parent>
            if joint.parent is not None:
                ET.SubElement(joint_tag, 'parent', attrib={'link': joint.parent})

            # <child>
            if joint.child is not None:
                ET.SubElement(joint_tag, 'child', attrib={'link': joint.child})

            # <axis>
            if joint.axis is not None:
                ET.SubElement(joint_tag, 'axis', attrib={'xyz': str(np.asarray(joint.axis))[1:-1]})

            # <limit>
            if joint.limits is not None or joint.effort is not None or joint.velocity is not None:
                kwargs = {}
                if joint.effort is not None:
                    kwargs['effort'] = str(joint.effort)
                if joint.velocity is not None:
                    kwargs['velocity'] = str(joint.velocity)
                if joint.limits is not None:
                    kwargs['lower'] = str(joint.limits[0])
                    kwargs['upper'] = str(joint.limits[1])
                ET.SubElement(joint_tag, 'limit', attrib=kwargs)

        # return root XML element
        return root
