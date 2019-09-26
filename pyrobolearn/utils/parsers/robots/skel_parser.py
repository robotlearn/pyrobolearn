#!/usr/bin/env python
"""Define the Skel parser/generator.

Skel files are notably used in Dart.

References:
    - SKEL file format: https://dartsim.github.io/skel_file_format.html
        - Note: the specification is incomplete on the above website, check the examples in the next link.
    - Examples of Skeleton: https://github.com/dartsim/dart/tree/master/data/skel
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


class SkelParser(WorldParser):
    r"""Skel Parser and Generator"""

    def __init__(self, filename=None):
        """
        Initialize the Skel parser.

        Args:
            filename (str, None): path to the Skel file.
        """
        super().__init__(filename)
        self.worlds = []

    def parse(self, filename):
        """
        Load and parse the given Skel file.

        Args:
            filename (str): path to the Skel file.
        """
        # load and parse the XML file
        tree_xml = ET.parse(filename)

        # get the root
        root = tree_xml.getroot()

        # check that the root is <skel>
        if root.tag != 'skel':
            raise RuntimeError("Expecting the first XML tag to be 'skel' but found instead: {}".format(root.tag))

        # check world(s)
        for i, world_tag in enumerate(root.findall('world')):
            # build the world
            world = World(name=world_tag.attrib.get('name', 'world_' + str(i)))

            # check physics
            physics_tag = world_tag.find('physics')
            physics = None
            if physics_tag is not None:
                physics = Physics()
                timestep_tag = physics_tag.find('time_step')
                if timestep_tag is not None:
                    physics.timestep = timestep_tag.text
                gravity_tag = physics_tag.find('gravity')
                if gravity_tag is not None:
                    physics.gravity = gravity_tag.text  # TODO: check the convention that DART uses; g = (0, -9.81, 0)?
                # TODO: collision collector

            if physics is not None:
                world.physics = physics

            # check skeleton
            for idx, skeleton_tag in enumerate(world_tag.findall('skeleton')):
                tree = self._check_skeleton(skeleton_tag, idx=idx)
                world.trees[tree.name] = tree

            # append the world to the list of worlds
            self.worlds.append(world)

        # set the first world
        self.world = self.worlds[0]

    def _check_skeleton(self, skeleton_tag, idx):
        """
        Return the Tree instance from a <skeleton>.

        Args:
            skeleton_tag (ET.Element): skeleton XML element
            idx (int): skeleton index.

        Returns:
            MultiBody: tree data structure containing the skeleton.
        """
        # create tree
        tree = MultiBody(name=skeleton_tag.attrib.get('name', 'skeleton_' + str(idx)))

        # check bodies/links
        for i, body_tag in enumerate(skeleton_tag.findall('body')):
            body = self._check_body(body_tag, idx=i)
            # add body to tree
            tree.bodies[body.name] = body

        # check joints
        for i, joint_tag in enumerate(skeleton_tag.findall('joint')):
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
        Return Body instance from a <body>.

        Args:
            body_tag (ET.Element): body XML element.
            idx (int): body index.

        Returns:
            Body: body data structure.
        """
        # create body/link
        body = Body(body_id=idx, name=body_tag.attrib.get('name', 'body_' + str(idx)))

        # check <inertia> tag
        inertial_tag = body_tag.find('inertia')
        if inertial_tag is not None:
            inertial = Inertial()

            # transformation
            pose_tag = inertial_tag.find('transformation')
            if pose_tag is not None:
                inertial.pose = pose_tag.text

            # offset
            offset_tag = inertial_tag.find('offset')
            if offset_tag is not None:
                inertial.position = offset_tag.text

            # mass
            mass_tag = inertial_tag.find('mass')
            if mass_tag is not None:
                inertial.mass = mass_tag.text

            # moment_of_inertia
            inertia_tag = inertial_tag.find('moment_of_inertia')
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

        # check <visualization_shape> tag
        visual_tag = body_tag.find('visualization_shape')
        if visual_tag is not None:
            visual = Visual()

            # name
            visual.name = visual_tag.attrib.get('name')

            # transformation
            pose_tag = visual_tag.find('transformation')
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

            # set visual to body
            body.visual = visual

        # check <collision_shape> tag
        collision_tag = body_tag.find('collision_shape')
        if collision_tag is not None:
            collision = Collision()

            # name
            # collision.name = collision_tag.attrib.get('name')

            # origin
            pose_tag = collision_tag.find('transformation')
            if pose_tag is not None:
                collision.pose = pose_tag.text

            # geometry
            geometry_tag = collision_tag.find('geometry')
            if geometry_tag is not None:
                geometry_types = ['box', 'capsule', 'cone', 'cylinder', 'ellipsoid', 'mesh', 'sphere']  # multi_sphere
                for geometry_type in geometry_types:
                    geometry_type_tag = geometry_tag.find(geometry_type)
                    if geometry_type_tag is not None:
                        dtype = geometry_type
                        collision.dtype = dtype
                        if dtype == 'box' or dtype == 'ellipsoid':
                            size_tag = geometry_type_tag.find('size')
                            collision.size = size_tag.text
                        elif dtype == 'sphere':
                            radius_tag = geometry_type_tag.find('radius')
                            collision.size = radius_tag.text
                        elif dtype == 'cylinder' or dtype == 'capsule' or dtype == 'cone':
                            radius_tag = geometry_type_tag.find('radius')
                            height_tag = geometry_type_tag.find('height')
                            collision.size = (radius_tag.text, height_tag.text)
                        elif dtype == 'mesh':
                            filename_tag = geometry_type_tag.find('file_name')
                            scale_tag = geometry_type_tag.find('scale')
                            collision.filename = filename_tag.text
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

        # transformation
        pose_tag = joint_tag.find('transformation')
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

        # init_pos
        init_pos_tag = joint_tag.find('init_pos')
        if init_pos_tag is not None:
            joint.init_position = init_pos_tag.text

        # init_vel
        init_vel_tag = joint_tag.find('init_vel')
        if init_vel_tag is not None:
            joint.init_velocity = init_vel_tag.text

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
        root = ET.Element('skel', attrib={'version': '1.0'})

        # create world tag
        name = world.name if world.name is not None else 'world'
        world_tag = ET.SubElement(root, 'world', attrib={'name': name})

        def set_transformation(parent_tag, item):
            if item.position is None and item.orientation is None:
                return None
            position = [0., 0., 0.] if item.position is None else item.position
            orientation = [0., 0., 0.] if item.orientation is None else item.orientation
            pose = np.concatenate((position, orientation))
            pose_tag = ET.SubElement(parent_tag, 'transformation')
            pose_tag.text = str(pose)[1:-1]

        def set_geometry(parent_tag, item):
            if item.geometry.dtype is not None:
                geometry = item.geometry
                dtype = geometry.dtype
                geometry_tag = ET.SubElement(parent_tag, 'geometry')

                if dtype in {'box', 'capsule', 'cone', 'cylinder', 'ellipsoid', 'mesh', 'sphere'}:
                    dtype_tag = ET.SubElement(geometry_tag, dtype)
                    if dtype == 'box' or dtype == 'ellipsoid':
                        size_tag = ET.SubElement(dtype_tag, 'size')
                        size_tag.text = str(np.asarray(geometry.size))[1:-1]
                    elif dtype == 'sphere':
                        radius_tag = ET.SubElement(dtype_tag, 'radius')
                        radius_tag.text = str(geometry.size)
                    elif dtype == 'cylinder' or dtype == 'capsule' or dtype == 'cone':
                        radius_tag = ET.SubElement(dtype_tag, 'radius')
                        height_tag = ET.SubElement(dtype_tag, 'height')
                        radius_tag.text = str(geometry.size[0])
                        height_tag.text = str(geometry.size[1])
                    elif dtype == 'mesh':
                        filename_tag = ET.SubElement(dtype_tag, 'file_name')
                        filename_tag.text = geometry.filename
                        if geometry.size is not None:
                            scale_tag = ET.SubElement(dtype_tag, 'scale')
                            scale_tag.text = str(np.asarray())[1:-1]
                    # elif dtype == 'plane':
                    #     pass
                    # elif dtype == 'heightmap':
                    #     pass

        # create skeletons
        for tree in world.trees:
            skeleton_tag = ET.SubElement(world_tag, 'skeleton', attrib={'name': tree.name})

            # transformation
            set_transformation(skeleton_tag, tree)

            # create bodies/links
            for i, body in enumerate(tree.bodies):
                body_tag = ET.SubElement(skeleton_tag, 'body', attrib={'name': body.name})

                # create inertial
                if body.inertial is not None:
                    inertial = body.inertial
                    inertial_tag = ET.SubElement(body_tag, 'inertia')

                    # offset
                    if inertial.position is not None:
                        if inertial.orientation is not None:
                            set_transformation(inertial_tag, inertial)
                        else:
                            offset_tag = ET.SubElement(inertial_tag, 'offset')
                            offset_tag.text = str(np.asarray(inertial.position))[1:-1]

                    # mass
                    if inertial.mass is not None:
                        mass_tag = ET.SubElement(inertial_tag, 'mass')
                        mass_tag.text = str(inertial.mass)

                    # inertia
                    if inertial.inertia is not None:
                        inertia = inertial.inertia
                        inertia_tag = ET.SubElement(inertial_tag, 'moment_of_inertia')

                        if inertia.ixx is not None:
                            ixx_tag = ET.SubElement(inertia_tag, 'ixx')
                            ixx_tag.text = str(inertia.ixx)
                        if inertia.ixy is not None:
                            ixy_tag = ET.SubElement(inertia_tag, 'ixy')
                            ixy_tag.text = str(inertia.ixy)
                        if inertia.ixz is not None:
                            ixz_tag = ET.SubElement(inertia_tag, 'ixz')
                            ixz_tag.text = str(inertia.ixz)
                        if inertia.iyy is not None:
                            iyy_tag = ET.SubElement(inertia_tag, 'iyy')
                            iyy_tag.text = str(inertia.iyy)
                        if inertia.iyz is not None:
                            iyz_tag = ET.SubElement(inertia_tag, 'iyz')
                            iyz_tag.text = str(inertia.iyz)
                        if inertia.izz is not None:
                            izz_tag = ET.SubElement(inertia_tag, 'izz')
                            izz_tag.text = str(inertia.izz)

                # create visual
                if body.visual is not None:
                    visual = body.visual
                    # name = 'visual_' + str(i) if visual.name is None else visual.name
                    visual_tag = ET.SubElement(body_tag, 'visualization_shape')  # attrib={'name': name})

                    # transformation
                    set_transformation(visual_tag, visual)

                    # geometry
                    set_geometry(visual_tag, visual)

                # create collision
                if body.collision is not None:
                    collision = body.collision
                    # name = 'collision_' + str(i) if collision.name is None else collision.name
                    collision_tag = ET.SubElement(body_tag, 'collision_shape')  # , attrib={'name': name})

                    # transformation
                    set_transformation(collision_tag, collision)

                    # geometry
                    set_geometry(collision_tag, collision)

            # create joints
            for i, joint in enumerate(tree.joints):
                joint_tag = ET.SubElement(skeleton_tag, 'joint', attrib={'name': joint.name, 'type': joint.dtype})
                parent_tag = ET.SubElement(joint_tag, 'parent')
                parent_tag.text = joint.parent
                child_tag = ET.SubElement(joint_tag, 'child')
                child_tag.text = joint.child

                # transformation
                set_transformation(joint_tag, joint)

                # axis
                axis_tag = None
                if joint.axis is not None:
                    axis_tag = ET.SubElement(joint_tag, 'axis')
                    xyz_tag = ET.SubElement(axis_tag, 'xyz')
                    xyz_tag.text = str(np.asarray(joint.axis))[1:-1]

                # dynamics
                if joint.damping is not None or joint.friction is not None:
                    if axis_tag is None:
                        axis_tag = ET.SubElement(joint_tag, 'axis')
                    dynamics_tag = ET.SubElement(axis_tag, 'dynamics')

                    if joint.friction is not None:
                        friction_tag = ET.SubElement(dynamics_tag, 'friction')
                        friction_tag.text = str(joint.friction)
                    if joint.damping is not None:
                        damping_tag = ET.SubElement(dynamics_tag, 'damping')
                        damping_tag.text = str(joint.damping)

                # limits
                if joint.limits is not None or joint.effort is not None or joint.velocity is not None:
                    if axis_tag is None:
                        axis_tag = ET.SubElement(joint_tag, 'axis')
                    limit_tag = ET.SubElement(axis_tag, 'limit')

                    if joint.limits is not None:
                        lower_tag = ET.SubElement(limit_tag, 'lower')
                        upper_tag = ET.SubElement(limit_tag, 'upper')
                        lower_tag.text = str(joint.limits[0])
                        upper_tag.text = str(joint.limits[1])

                    if joint.velocity is not None:
                        velocity_tag = ET.SubElement(limit_tag, 'velocity')
                        velocity_tag.text = str(joint.velocity)

                    if joint.effort is not None:
                        effort_tag = ET.SubElement(limit_tag, 'effort')
                        effort_tag.text = str(joint.effort)

                # init_pos
                if joint.init_position is not None:
                    init_pos_tag = ET.SubElement(joint_tag, 'init_pos')
                    if isinstance(joint.init_position, (float, int)):
                        init_pos_tag.text = str(joint.init_position)
                    else:
                        init_pos_tag.text = str(np.asarray(joint.init_position))[1:-1]

                # init_vel
                if joint.init_velocity is not None:
                    init_vel_tag = ET.SubElement(joint_tag, 'init_vel')
                    if isinstance(joint.init_velocity, (float, int)):
                        init_vel_tag.text = str(joint.init_velocity)
                    else:
                        init_vel_tag.text = str(np.asarray(joint.init_velocity))[1:-1]

        # return root XML element
        return root
