#!/usr/bin/env python
"""Define the Proto parser/generator.

Proto files are notably used in Webots.
"""

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


class ProtoParser(RobotParser):
    r"""Proto Parser and Generator."""

    def __init__(self, filename=None):
        """
        Initialize the Proto parser.

        Args:
            filename (str, None): path to the proto file.
        """
        super().__init__(filename)

    def parse(self, filename):
        """
        Load and parse the given proto file.

        Args:
            filename (str): path to the proto file.
        """
        pass

    def get_tree(self):
        """
        Return the Tree containing all the elements.

        Returns:
            Tree: tree data structure.
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
