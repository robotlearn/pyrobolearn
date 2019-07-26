#!/usr/bin/env python
"""Define the SDF parser.

SDF files are notably used in Gazebo, and Bullet.
"""

# import XML parser
import xml.etree.ElementTree as ET

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


class SDFParser(WorldParser):
    r"""SDF Parser and Generator"""

    def __init__(self, filename=None):
        """
        Initialize the SDF parser.

        Args:
            filename (str, None): path to the SDF file.
        """
        super().__init__(filename)
        self.worlds = []

    def parse(self, filename):
        """
        Load and parse the given SDF file.

        Args:
            filename (str): path to the SDF file.
        """
        # load and parse the XML file
        tree_xml = ET.parse(filename)

        # get the root
        root = tree_xml.getroot()

        # check that the root is <sdf>
        if root.tag != 'sdf':
            raise RuntimeError("Expecting the first XML tag to be 'sdf' but found instead: {}".format(root.tag))

        # check world(s)
        for i, world_tag in enumerate(root.findall('world')):
            # build the world
            world = World(name=world_tag.attrib.get('name', 'world_' + str(i)))

            # check model
            for idx, model_tag in enumerate(root.findall('model')):
                tree = self._check_model(model_tag, idx=idx)
                world.trees[tree.name] = tree

            # check physics

            # append the world to the list of worlds
            self.worlds.append(world)

        # check model
        models = root.findall('model')
        if len(models) > 0:
            world = World()
        for i, model_tag in enumerate(models):
            tree = self._check_model(model_tag, idx=i)
            world.trees[tree.name] = tree
        if len(models) > 0:
            self.worlds.append(world)

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

    @staticmethod
    def _check_body(body_tag, idx):
        """
        Return Body instance from a <link>.

        Args:
            body_tag (ET.Element): link XML element.
            idx (int): link index.

        Returns:
            Body: body data structure.
        """
        # create body/link
        body = Body(body_id=idx, name=body_tag.attrib.get('name', 'body_' + str(idx)))

        # check <inertial> tag
        inertial_tag = body_tag.find('inertial')
        if inertial_tag is not None:
            inertial = Inertial()

            # pose
            pose_tag = inertial_tag.find('pose')
            if pose_tag is not None:
                inertial.pose = pose_tag.text

            # mass
            mass_tag = inertial_tag.find('mass')
            if mass_tag is not None:
                inertial.mass = mass_tag.text

            # inertia
            inertia_tag = inertial_tag.find('inertia')
            if inertia_tag is not None:
                ixx = inertia_tag.find('ixx')
                if ixx is not None:
                    ixx = ixx.text
                ixy = inertia_tag.find('ixy')
                if ixy is not None:
                    ixy = ixy.text
                ixz = inertia_tag.find('ixz')
                if ixz is not None:
                    ixz = ixz.text
                iyy = inertia_tag.find('iyy')
                if iyy is not None:
                    iyy = iyy.text
                iyz = inertia_tag.find('iyz')
                if iyz is not None:
                    iyz = iyz.text
                izz = inertia_tag.find('izz')
                if izz is not None:
                    izz = izz.text
                inertial.inertia = {'ixx': ixx, 'ixy': ixy, 'ixz': ixz, 'iyy': iyy, 'iyz': iyz, 'izz': izz}

            # set inertial to body
            body.inertial = inertial

        # check <visual> tag
        visual_tag = body_tag.find('visual')
        if visual_tag is not None:
            visual = Visual()

            # name
            visual.name = visual_tag.attrib.get('name')

            # pose
            pose_tag = visual_tag.find('pose')
            if pose_tag is not None:
                visual.pose = pose_tag.text

            # geometry
            geometry_tag = visual_tag.find('geometry')
            if geometry_tag is not None:
                for geometry_type in ['box', 'mesh', 'cylinder', 'sphere', 'plane', 'heightmap']:  # polyline, image
                    geometry_type_tag = geometry_tag.find(geometry_type)
                    if geometry_type_tag is not None:
                        dtype = geometry_type
                        visual.dtype = dtype
                        if dtype == 'box':
                            size_tag = geometry_type_tag.find('size')
                            visual.size = size_tag.text
                        elif dtype == 'sphere':
                            radius_tag = geometry_type_tag.find('radius')
                            visual.size = radius_tag.text
                        elif dtype == 'cylinder':
                            radius_tag = geometry_type_tag.find('radius')
                            length_tag = geometry_type_tag.find('length')
                            visual.size = (radius_tag.text, length_tag.text)
                        elif dtype == 'mesh':
                            uri_tag = geometry_type_tag.find('uri')
                            scale_tag = geometry_type_tag.find('scale')
                            visual.filename = uri_tag.text
                            visual.size = scale_tag.text

            # material
            # material = visual.find('material')
            # if material is not None:
            #     name = material.attrib.get('name')
            #     color = material.find('color')
            #     if color is not None:
            #         v.color = color.attrib['rgba']
            #     else:
            #         mat = tree.materials.get(name)
            #         if mat is not None:
            #             v.material = mat

            # set visual to body
            body.visual = visual

        # check <collision> tag
        collision_tag = body_tag.find('collision')
        if collision_tag is not None:
            collision = Collision()

            # name
            collision.name = collision_tag.attrib.get('name')

            # origin
            pose_tag = collision_tag.find('pose')
            if pose_tag is not None:
                collision.pose = pose_tag.text

            # geometry
            geometry_tag = collision_tag.find('geometry')
            if geometry_tag is not None:
                for geometry_type in ['box', 'mesh', 'cylinder', 'sphere', 'plane', 'heightmap']:  # polyline, image
                    geometry_type_tag = geometry_tag.find(geometry_type)
                    if geometry_type_tag is not None:
                        dtype = geometry_type
                        collision.dtype = dtype
                        if dtype == 'box':
                            size_tag = geometry_type_tag.find('size')
                            collision.size = size_tag.text
                        elif dtype == 'sphere':
                            radius_tag = geometry_type_tag.find('radius')
                            collision.size = radius_tag.text
                        elif dtype == 'cylinder':
                            radius_tag = geometry_type_tag.find('radius')
                            length_tag = geometry_type_tag.find('length')
                            collision.size = (radius_tag.text, length_tag.text)
                        elif dtype == 'mesh':
                            uri_tag = geometry_type_tag.find('uri')
                            scale_tag = geometry_type_tag.find('scale')
                            collision.filename = uri_tag.text
                            collision.size = scale_tag.text

            # set collision to body
            body.collision = collision

        # return the body instance
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
        joint.parent = parent_tag.text

        child_tag = joint_tag.find('child')
        if child_tag is None:
            raise RuntimeError("Expecting the joint '" + joint.name + "' to have a child link/body")
        joint.child = child_tag.text

        # pose
        pose_tag = joint_tag.find('pose')
        if pose_tag is not None:
            joint.pose = pose_tag.text

        # axis
        axis_tag = joint_tag.find('axis')
        if axis_tag is not None:
            axis_xyz_tag = axis_tag.find('xyz')
            if axis_xyz_tag is not None:
                joint.axis = axis_xyz_tag.text

            # dynamics
            dynamics_tag = axis_tag.find('dynamics')
            if dynamics_tag is not None:
                damping_tag = dynamics_tag.find('damping')

                # damping
                if damping_tag is not None:
                    joint.damping = damping_tag.text

                # friction
                friction_tag = dynamics_tag.find('friction')
                if friction_tag is not None:
                    joint.friction = friction_tag.text

            # limits
            limits_tag = axis_tag.find('limits')
            if limits_tag is not None:
                effort_tag = limits_tag.find('effort')
                if effort_tag is not None:
                    joint.effort = effort_tag.text

                velocity_tag = limits_tag.find('velocity')
                if velocity_tag is not None:
                    joint.velocity = velocity_tag.text

                lower_limit_tag = limits_tag.find('lower')
                upper_limit_tag = limits_tag.find('upper')
                if lower_limit_tag is not None and upper_limit_tag is not None:  # TODO: check if we can have one limit
                    joint.limits = [lower_limit_tag.text, upper_limit_tag.text]

        return joint

    def generate(self, world=None):
        """
        Generate the XML world from the `World` data structure.

        Args:
            world (World, Tree): world / tree data structure.

        Returns:
            ET.Element: root element in the XML file.
        """
        if world is None:
            world = self.worlds[0]

        # create root element
        root = ET.Element('sdf', attrib={'version': '1.6'})

        # create world tag
        name = world.name if world.name is not None else 'default'
        world_tag = ET.SubElement(root, 'world', attrib={'name': name})

        # create models
        for tree in world.trees:
            model_tag = ET.SubElement(world_tag, 'model', attrib={'name': tree.name})
            if tree.position is not None or tree.orientation is not None:
                pose_tag = ET.SubElement(model_tag, 'pose')
                pose_tag.text = str(np.asarray(tree.pose))[1:-1]

            # create links
            for body in tree.bodies:  # TODO
                pass

            # create joints
            for joint in tree.joints:  # TODO
                pass

        # return root XML element
        return root
