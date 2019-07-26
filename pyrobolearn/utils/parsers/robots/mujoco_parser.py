#!/usr/bin/env python
"""Define the MuJoCo parser.
"""

# import XML parser
import xml.etree.ElementTree as ET

from pyrobolearn.utils.parsers.robots.world_parser import WorldParser
from pyrobolearn.utils.parsers.robots.data_structures import Tree, World


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
        world = World()

        # check default (this is the default configuration when they are not specified)
        default_tag = root.find('default')
        if default_tag is not None:
            pass

        # check physics

        # check assets
        asset_tag = root.find('asset')
        if asset_tag is not None:
            pass

        # check world body
        worldbody_tag = root.find('worldbody')
        if worldbody_tag is not None:
            pass

        # check contact

        # check equality constraint

        # check actuator

        # check sensor

        # set the world
        self.world = world

    def _check_body(self, body_tag, idx):
        """
        Return Body instance from a <body>.

        Args:
            body_tag (ET.Element): body XML element.
            idx (int): link index.

        Returns:
            Body: body data structure.
        """
        pass

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

    def generate(self, tree=None):
        """
        Generate the XML tree from the `Tree` data structure.

        Args:
            tree (Tree): Tree data structure.

        Returns:
            ET.Element: root element in the XML file.
        """
        pass

