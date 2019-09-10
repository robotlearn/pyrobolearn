#!/usr/bin/env python
"""Define the SRDF parser.

The Semantic Robot Description Format (SRDF) format allows to represent semantic information about the robot
structure [1]. Specifically, it allows to "specify joint groups, default robot configurations, additional collision
checking information, and additional transforms that may be needed to completely specify the robot’s pose." [2]

References:
    - [1] SRDF: http://wiki.ros.org/srdf
    - [2] URDF and SRDF: http://docs.ros.org/kinetic/api/moveit_tutorials/html/doc/urdf_srdf/urdf_srdf_tutorial.html
"""

import numpy as np
import xml.etree.ElementTree as ET


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2019, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class SRDFParser(object):
    r"""SRDF Parser

    The Semantic Robot Description Format (SRDF) format allows to represent semantic information about the robot
    structure [1]. Specifically, it allows to "specify joint groups, default robot configurations, additional collision
    checking information, and additional transforms that may be needed to completely specify the robot’s pose." [2]

    Warnings: this currently only parse the groups and the group states!! Note that every name has been lower cased.

    Example of a simple SRDF file from [1]:

    <robot name="robot_name">
        <group name="group_link_name1">
            <chain base_link="base_link_name" tip_link="tip_link_name"/>
        </group>
        <group name="group_link_name2">
            <group name="group_link_name1"/>
            <link name="link_name"/>
        </group>

        <group_state name="home" group="group_name_to_use">
            <joint name="joint_name1" value="joint_value1"/>
            <joint name="joint_name2" value="joint_value2"/>
            ...
        </group_state>

    </robot>

    References:
        - [1] SRDF: http://wiki.ros.org/srdf
        - [2] URDF and SRDF: http://docs.ros.org/kinetic/api/moveit_tutorials/html/doc/urdf_srdf/urdf_srdf_tutorial.html
    """

    def __init__(self, filename=None):
        """
        Initialize the SRDF parser.

        Args:
            filename (str, None): path to the file to parse.
        """
        self.filename = filename
        self.dirname = ''
        if filename is not None:
            self.dirname = os.path.dirname(filename) + '/'
            self.parse(filename)

    @staticmethod
    def parse(filename):
        """
        Load and parse a given file.

        Args:
            filename (str): path to the file to parse.

        Returns:
            dict: dictionary containing the tags and their values (which are converted to their correct data type if
              necessary).
        """
        # load and parse the XML file
        tree_xml = ET.parse(filename)

        # get the root
        root = tree_xml.getroot()

        results = {}

        # parse <group> tags
        for group_tag in root.findall('group'):
            name = group_tag.attrib['name']
            results.setdefault('group', {}).setdefault(name, {})
            group = results['group'][name]

            # parse each <link>
            for link_tag in group_tag.findall('link'):
                group.setdefault('link', []).append(link_tag.attrib['name'])

            # parse each <chain>
            for chain_tag in group_tag.findall('chain'):
                group.setdefault('chain', []).append([chain_tag.attrib['base_link'], chain_tag.attrib['tip_link']])

            # parse each <joint>
            for joint_tag in group_tag.findall('joint'):
                values = [float(c) for c in joint_tag.attrib['value'].split()]
                values = values[0] if len(values) == 1 else np.array(values)
                group.setdefault('joint', {})[joint_tag.attrib['name']] = values

            # parse each <group>
            for group_subtag in group_tag.findall('group'):
                group.setdefault('group', []).append(group_subtag.attrib['name'])

        # parse <group_state> tags
        for group_state_tag in root.findall('group_state'):
            name = group_state_tag.attrib['name'].lower()
            group_name = group_state_tag.attrib['group']
            results.setdefault('group_state', {}).setdefault(name, {})
            group_state = results['group_state'][name]
            group_state['group'] = group_name

            # parse each <joint>
            for joint_tag in group_state_tag.findall('joint'):
                values = [float(c) for c in joint_tag.attrib['value'].split()]
                values = values[0] if len(values) == 1 else np.array(values)
                group_state[joint_tag.attrib['name']] = values

        return results

    # def parse_string(self, string):
    #     """
    #     Parse the given string.
    #
    #     Args:
    #         string (str): string to parse.
    #
    #     Returns:
    #         dict: dictionary containing the tags and their values.
    #     """
    #     return

    # def _parse(self, root):
    #     """
    #     Parse the given root element.
    #
    #     Args:
    #         root (ET.Element): root XML tag element.
    #
    #     Returns:
    #         dict: dictionary containing the tags and their values.
    #     """
    #     pass
