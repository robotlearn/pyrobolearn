#!/usr/bin/env python
"""Define the SDF parser.

SDF files are notably used in Gazebo, and Bullet.
"""

# import XML parser
import xml.etree.ElementTree as ET

from pyrobolearn.utils.parsers.robots.robot_parser import RobotParser
from pyrobolearn.utils.parsers.robots.data_structures import Tree


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2019, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class SDFParser(RobotParser):
    r"""SDF Parser"""

    def __init__(self, filename=None):
        """
        Initialize the SDF parser.

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
        tree = ET.parse(filename)

        # get the root
        root = tree.getroot()

        # check that the root is <robot>
        if root.tag != 'sdf':
            raise RuntimeError("Expecting the first XML tag to be 'sdf' but found instead: {}".format(root.tag))

        # build the tree

    def get_tree(self):
        """
        Return the Tree containing all the elements.

        Returns:
            Tree: tree data structure.
        """
        pass

    def get_world(self):
        """
        Return the world (which is basically a list of Tree).
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

