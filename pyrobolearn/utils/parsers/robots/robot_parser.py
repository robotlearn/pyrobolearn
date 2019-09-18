#!/usr/bin/env python
"""Define the abstract Robot parser.
"""

import os

# import XML parser
import xml.etree.ElementTree as ET
from xml.dom import minidom  # to print in a pretty way the XML file

from pyrobolearn.utils.parsers.robots.data_structures import MultiBody

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2019, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class RobotParser(object):
    r"""Robot Parser and Generator."""

    def __init__(self, filename=None):
        """
        Initialize the robot parser.

        Args:
            filename (str, None): path to the file to parse.
        """
        self.root = None
        self.tree = None
        self.filename = filename
        self.dirname = ''
        if filename is not None:
            self.dirname = os.path.dirname(filename) + '/'
            self.parse(filename)

    ##############
    # Properties #
    ##############

    @property
    def root(self):
        """Return the root XML tag element."""
        return self._root

    @root.setter
    def root(self, root):
        """Set the root XML tag element."""
        if root is not None and not isinstance(root, ET.Element):
            raise TypeError("Expecting the root to be an instance of `ET.Element`, but got instead: "
                            "{}".format(type(root)))
        self._root = root

    @property
    def tree(self):
        """Return the tree / multi-body data structure."""
        return self._tree

    @tree.setter
    def tree(self, tree):
        """Set the tree / multi-body data structure."""
        if tree is not None and not isinstance(tree, MultiBody):
            raise TypeError("Expecting the given tree to be an instance of `Tree`, but got instead: "
                            "{}".format(type(tree)))
        self._tree = tree

    ###########
    # Methods #
    ###########

    def parse(self, filename):
        """
        Load and parse a given file.

        Args:
            filename (str): path to the file to parse.

        Returns:
            object: data structure that can be passed to other generators.
        """
        pass

    # def parse_string(self, string):
    #     """
    #     Parse the given string.
    #
    #     Args:
    #         string (str): string to parse.
    #
    #     Returns:
    #         object: data structure that can be passed to other generators.
    #     """
    #     pass

    def get_tree(self):
        """
        Return the Tree containing all the elements.

        Returns:
            MultiBody: tree data structure.
        """
        return self.tree

    def generate(self, tree=None):
        """
        Generate the XML tree from the `Tree` data structure.

        Args:
            tree (MultiBody): Tree data structure.

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
        print(xml_str)
        with open(filename, "w") as f:
            f.write(xml_str)  # .encode('utf-8'))
