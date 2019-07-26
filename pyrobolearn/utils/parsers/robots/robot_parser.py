#!/usr/bin/env python
"""Define the abstract Robot parser.
"""

# import XML parser
import xml.etree.ElementTree as ET
from xml.dom import minidom  # to print in a pretty way the XML file

from pyrobolearn.utils.parsers.robots.data_structures import Tree

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2019, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class RobotParser(object):
    r"""Robot Parser"""

    def __init__(self, filename=None):
        """
        Initialize the robot parser.

        Args:
            filename (str, None): path to the MuJoCo XML file.
        """
        self.root = None
        self.tree = None
        self.filename = filename
        if filename is not None:
            self.parse(filename)

    def parse(self, filename):
        """
        Load and parse a given MuJoCo XML filename.

        Args:
            filename (str): path to the MuJoCo XML file.
        """
        pass

    def get_tree(self):
        """
        Return the Tree containing all the elements.

        Returns:
            Tree: tree data structure.
        """
        return self.tree

    def generate(self, tree=None):
        """
        Generate the XML tree from the `Tree` data structure.

        Args:
            tree (Tree): Tree data structure.

        Returns:
            ET.Element: root element in the XML file.
        """
        pass

    def get_string(self, root=None):
        """
        Return the XML string from the root element.

        Args:
            root (ET.Element): root element in the XML file.

        Returns:
            str: string representing the XML file.
        """
        if root is None:
            root = self.root
        if not isinstance(root, ET.Element):
            raise ValueError("Expecting the root to be an instance of `ET.Element`, but got instead: "
                             "{}".format(type(root)))
        return minidom.parseString(ET.tostring(root)).toprettyxml(indent="   ")

    def write(self, filename, root=None):
        """
        Write the XML tree in the specified XML file.

        Args:
            filename (str): path to the file to write the XML in.
            root (ET.Element): root element in the XML file.
        """
        xml_str = self.get_string(root)
        with open(filename, "w") as f:
            f.write(xml_str)  # .encode('utf-8'))
