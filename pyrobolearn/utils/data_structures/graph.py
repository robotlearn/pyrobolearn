

class Graph(object):
    r"""Graph.

    This class described the graph data structure.
    """
    pass


class DirectedGraph(Graph):
    r"""Directed Graph.

    This class described the directed graph data structure.

    The graph is described using a dictionary.
    graph = {node: [[parent nodes], [child nodes]]}
    """
    class Root(object):
        pass

    def __init__(self):
        self._root = self.Root()
        self._graph = {self._root: []}

    def addNode(self, parent, node):
        """
        Add a node to the graph.
        """
        pass

    def getParents(self, node):
        """
        Return the parent nodes of the given node.
        """
        pass

    def getChildren(self, node):
        """
        Return the child nodes of the given node.
        """
        pass


class DirectedAcyclicGraph(DirectedGraph):
    """Directed Acyclic Graph.

    In this data structure, cycles are not allowed.
    """
    def __init__(self):
        super(DirectedAcyclicGraph, self).__init__()
