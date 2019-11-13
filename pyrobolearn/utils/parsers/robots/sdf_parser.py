#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Define the SDF parser/generator.

SDF files are notably used in Gazebo, and Bullet.

References:
    - SDF file format: http://sdformat.org/
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

            # check light
            for l, light_tag in enumerate(world_tag.findall('light')):
                attrib = light_tag.attrib
                light = Light(name=attrib.get('name', 'light_' + str(l)), dtype=attrib.get('type'))

                for tag in ['cast_shadows', 'diffuse', 'specular', 'direction']:
                    item_tag = light_tag.find(tag)
                    if item_tag is not None:
                        if hasattr(light, tag):
                            setattr(light, tag, item_tag.text)

                pose_tag = light_tag.find('pose')
                if pose_tag is not None:
                    light.pose = pose_tag.text

                # add the light in the world
                world.lights[light.name] = light

            # check model
            for idx, model_tag in enumerate(world_tag.findall('model')):
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

        # set the first world
        self.world = self.worlds[0]

    def _check_model(self, model_tag, idx):
        """
        Return the Tree instance from a <model>.

        Args:
            model_tag (ET.Element): model XML element
            idx (int): model index.

        Returns:
            MultiBody: tree data structure containing the model.
        """
        # create tree
        tree = MultiBody(name=model_tag.attrib.get('name', 'model_' + str(idx)))

        # check bodies/links
        for i, link_tag in enumerate(model_tag.findall('link')):
            body = self._check_body(link_tag, idx=i)
            # add body to tree
            tree.bodies[body.name] = body

        # check joints
        for i, joint_tag in enumerate(model_tag.findall('joint')):
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

            # material  # TODO
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

            # init_position
            init_pos_tag = axis_tag.find('init_position')
            if init_pos_tag is not None:
                joint.init_position = init_pos_tag.text

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

        def set_pose(parent_tag, item):
            if item.position is None and item.orientation is None:
                return None
            position = [0., 0., 0.] if item.position is None else item.position
            orientation = [0., 0., 0.] if item.orientation is None else item.orientation
            pose = np.concatenate((position, orientation))
            pose_tag = ET.SubElement(parent_tag, 'pose')
            pose_tag.text = str(pose)[1:-1]

        def set_geometry(parent_tag, item):
            if item.geometry.dtype is not None:
                geometry = item.geometry
                dtype = geometry.dtype
                geometry_tag = ET.SubElement(parent_tag, 'geometry')

                if dtype in {'box', 'sphere', 'cylinder', 'mesh'}:
                    dtype_tag = ET.SubElement(geometry_tag, dtype)
                    if dtype == 'box':
                        size_tag = ET.SubElement(dtype_tag, 'size')
                        size_tag.text = str(np.asarray(geometry.size))[1:-1]
                    elif dtype == 'sphere':
                        radius_tag = ET.SubElement(dtype_tag, 'radius')
                        radius_tag.text = str(geometry.size)
                    elif dtype == 'cylinder':
                        radius_tag = ET.SubElement(dtype_tag, 'radius')
                        length_tag = ET.SubElement(dtype_tag, 'length')
                        radius_tag.text = str(geometry.size[0])
                        length_tag.text = str(geometry.size[1])
                    elif dtype == 'mesh':
                        uri_tag = ET.SubElement(dtype_tag, 'uri')
                        uri_tag.text = geometry.filename
                        if geometry.size is not None:
                            scale_tag = ET.SubElement(dtype_tag, 'scale')
                            scale_tag.text = str(np.asarray())[1:-1]
                    # elif dtype == 'plane':
                    #     pass
                    # elif dtype == 'heightmap':
                    #     pass

        # create models
        for tree in world.trees:
            model_tag = ET.SubElement(world_tag, 'model', attrib={'name': tree.name})

            # pose
            set_pose(model_tag, tree)

            # create links
            for i, body in enumerate(tree.bodies):
                link_tag = ET.SubElement(model_tag, 'link', attrib={'name': body.name})

                # create inertial
                if body.inertial is not None:
                    inertial = body.inertial
                    inertial_tag = ET.SubElement(link_tag, 'inertial')

                    # pose
                    set_pose(inertial_tag, inertial)

                    # mass
                    if inertial.mass is not None:
                        mass_tag = ET.SubElement(inertial_tag, 'mass')
                        mass_tag.text = str(inertial.mass)

                    # inertia
                    if inertial.inertia is not None:
                        inertia = inertial.inertia
                        inertia_tag = ET.SubElement(inertial_tag, 'inertia')

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
                    name = 'visual_' + str(i) if visual.name is None else visual.name
                    visual_tag = ET.SubElement(link_tag, 'visual', attrib={'name': name})

                    # pose
                    set_pose(visual_tag, visual)

                    # material
                    if visual.material is not None:
                        material = visual.material
                        material_tag = ET.SubElement(visual_tag, 'material')

                        if material.color is not None:
                            ambient_tag = ET.SubElement(material_tag, 'ambient')
                            ambient_tag.text = str(np.asarray(material.rgba))[1:-1]

                        if material.diffuse is not None:
                            diffuse_tag = ET.SubElement(material_tag, 'diffuse')
                            diffuse_tag.text = str(np.asarray(material.diffuse))[1:-1]

                        if material.specular is not None:
                            specular_tag = ET.SubElement(material_tag, 'specular')
                            specular_tag.text = str(np.asarray(material.specular))[1:-1]

                        if material.emissive is not None:
                            emissive_tag = ET.SubElement(material_tag, 'emissive')
                            emissive_tag.text = str(np.asarray(material.emissive))[1:-1]

                        if material.texture is not None:
                            script_tag = ET.SubElement(material_tag, 'script')
                            name = 'material_' + str(i) if material.name is None else material.name
                            name_tag = ET.SubElement(script_tag, 'name')
                            name_tag.text = name

                            # create Ogre script
                            with open(name + '.material', "w") as f:
                                s = "material {}" \
                                    "\n{" \
                                    "\n\ttechnique" \
                                    "\n\t{" \
                                    "\n\t\tpass" \
                                    "\n\t\t{" \
                                    "\n\t\t\ttexture_unit" \
                                    "\n\t\t\t{" \
                                    "\n\t\t\t\ttexture {}" \
                                    "\n\t\t\t}" \
                                    "\n\t\t}" \
                                    "\n\t}" \
                                    "\n}".format(name, material.texture)
                                f.write(s)

                            uri_tag = ET.SubElement(script_tag, 'uri')
                            uri_tag.text = "file://" + name + '.material'

                    # geometry
                    set_geometry(visual_tag, visual)

                # create collision
                if body.collision is not None:
                    collision = body.collision
                    name = 'collision_' + str(i) if collision.name is None else collision.name
                    collision_tag = ET.SubElement(link_tag, 'collision', attrib={'name': name})

                    # pose
                    set_pose(collision_tag, collision)

                    # geometry
                    set_geometry(collision_tag, collision)

            # create joints
            for i, joint in enumerate(tree.joints):
                joint_tag = ET.SubElement(model_tag, 'joint', attrib={'name': joint.name, 'type': joint.dtype})
                parent_tag = ET.SubElement(joint_tag, 'parent')
                parent_tag.text = joint.parent
                child_tag = ET.SubElement(joint_tag, 'child')
                child_tag.text = joint.child

                # pose
                set_pose(joint_tag, joint)

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

                # init_position
                if isinstance(joint.init_position, (float, int)):
                    if axis_tag is None:
                        axis_tag = ET.SubElement(joint_tag, 'axis')
                    initial_pos_tag = ET.SubElement(axis_tag, 'initial_position')
                    initial_pos_tag.text = str(joint.init_position)

        # return root XML element
        return root
