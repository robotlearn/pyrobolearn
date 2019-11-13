#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Define the queue data structures.

These inherit from the various queues in the `queue` Python library, to which we provide few more functionalities.
"""

import queue
import heapq

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


# # add functionalities to the original queue.Queue (from which queue.PriorityQueue and LifoQueue inherit from)
#
# def append(self, item):
#     """
#     Append an item in at the end of the FIFO queue. If the queue is already full it will remove the first item.
#     """
#     # if full remove the first item
#     if self.full():
#         self.get()
#
#     # add the new item
#     self.put(item)
#
#
# def tolist(self):
#     """
#     Return a list of the queue.
#     """
#     return list(self.queue)
#
#
# def __repr__(self):
#     """Return a representation string."""
#     return str(self.queue)
#
#
# def __len__(self):
#     """Return the current length of the FIFO queue."""
#     return self.qsize()
#
#
# def __getitem__(self, idx):
#     """Return the item corresponding to the given index.
#
#     Args:
#         idx (int): index.
#     """
#     return self.queue[idx]
#
#
# def __setitem__(self, idx, item):
#     """Set the given item to the given index."""
#     self.queue[idx] = item
#
#
# def __iter__(self):
#     """Return an iterator over the queue."""
#     return iter(self.queue)
#
#
# # add functionalities
# queue.Queue.append = append
# queue.Queue.tolist = tolist
# queue.Queue.__repr__ = __repr__
# queue.Queue.__len__ = __len__
# queue.Queue.__getitem__ = __getitem__
# queue.Queue.__setitem__ = __setitem__
# queue.Queue.__iter__ = __iter__
#
#
# # provide aliases
# FIFOQueue = queue.Queue
# LIFOQueue = queue.LifoQueue


class FIFOQueue(queue.Queue):
    r"""FIFO Queue

    We provide few more functionalities on top of the original `queue.Queue` class.
    """

    def __init__(self, maxsize=0):
        """
        Initialize the FIFO queue.

        Args:
            maxsize (int): maximum size of the queue.
        """
        queue.Queue.__init__(self, maxsize)

    def append(self, item):
        """
        Append an item in at the end of the FIFO queue. If the queue is already full it will remove the first item.
        """
        # if full remove the first item
        if self.full():
            self.get()

        # add the new item
        self.put(item)

    def tolist(self):
        """
        Return a list of the queue.
        """
        return list(self.queue)

    def __repr__(self):
        """Return a representation string."""
        return str(self.queue)

    def __len__(self):
        """Return the current length of the FIFO queue."""
        return self.qsize()

    def __getitem__(self, idx):
        """Return the item corresponding to the given index.

        Args:
            idx (int): index.
        """
        return self.queue[idx]

    def __setitem__(self, idx, item):
        """Set the given item to the given index."""
        self.queue[idx] = item

    def __iter__(self):
        """Return an iterator over the queue."""
        return iter(self.queue)


class LIFOQueue(queue.LifoQueue):
    r"""LIFO Queue.

    We provide few more functionalities on top of the original `queue.LifoQueue` class.
    """

    def __init__(self, maxsize=0):
        """
        Initialize the FIFO queue.

        Args:
            maxsize (int): maximum size of the queue.
        """
        queue.Queue.__init__(self, maxsize)

    def append(self, item):
        """
        Append an item in at the end of the FIFO queue. If the queue is already full it will remove the first item.
        """
        # if full remove the first item
        if self.full():
            self.get()

        # add the new item
        self.put(item)

    def tolist(self):
        """
        Return a list of the queue.
        """
        return list(self.queue)

    def __repr__(self):
        """Return a representation string."""
        return str(self.queue)

    def __len__(self):
        """Return the current length of the FIFO queue."""
        return self.qsize()

    def __getitem__(self, idx):
        """Return the item corresponding to the given index.

        Args:
            idx (int): index.
        """
        return self.queue[idx]

    def __setitem__(self, idx, item):
        """Set the given item to the given index."""
        self.queue[idx] = item

    def __iter__(self):
        """Return an iterator over the queue."""
        return reversed(self.queue)


class PriorityQueue(queue.PriorityQueue):
    r"""Priority Queue.

    We provide few more functionalities on top of the original `queue.PriorityQueue` class.
    """

    def __init__(self, maxsize=0, ascending=True):
        """
        Initialize the FIFO queue.

        Args:
            maxsize (int): maximum size of the queue.
            ascending (bool): if True, the item with the lowest priority will be the first one to be retrieved.
        """
        queue.Queue.__init__(self, maxsize)
        self.ascending = bool(ascending)

    def append(self, item):
        """
        Append an item in at the end of the FIFO queue. If the queue is already full it will remove the first item.
        """
        # if full remove the first item
        if self.full():
            self.get()

        # add the new item
        self.put(item)

    def tolist(self):
        """
        Return a list of the queue.
        """
        return list(self.queue)

    def _put(self, item, heappush=heapq.heappush):
        if not self.ascending:
            item = (-item[0], item[1])
        queue.PriorityQueue._put(self, item, heappush=heappush)

    def _get(self, heappop=heapq.heappop):
        item = queue.PriorityQueue._get(self, heappop=heappop)
        if not self.ascending:
            item = (-item[0], item[1])
        return item

    def get_lowest(self):
        """
        Get the lowest priority item.
        """
        if len(self) == 0:
            raise ValueError("The priority queue is empty.")

        if self.ascending:
            return self[0][1]
        else:
            return self[-1][1]

    def get_highest(self):
        """
        Get the highest priority item.
        """
        if len(self) == 0:
            raise ValueError("The priority queue is empty.")
        if self.ascending:
            return self[-1][1]
        else:
            return self[0][1]

    def __repr__(self):
        """Return a representation string."""
        return str(self.queue)

    def __len__(self):
        """Return the current length of the FIFO queue."""
        return self.qsize()

    def __getitem__(self, idx):
        """Return the item corresponding to the given index.

        Args:
            idx (int): index.
        """
        return self.queue[idx]

    def __setitem__(self, idx, item):
        """Set the given item to the given index."""
        self.queue[idx] = item

    def __iter__(self):
        """Return an iterator over the queue."""
        if self.ascending:
            return iter(self.queue)
        else:
            return reversed(self.queue)


# Tests
if __name__ == '__main__':
    # create queue
    q = FIFOQueue(2)
    print("Initial queue (maxsize={}): {}".format(q.maxsize, q))

    # add two elements in the queue
    q.append(1)
    q.append(2)
    print("After adding '1' and '2' in queue: {}".format(q))

    # add third element in the queue
    q.append(3)
    print("After adding '3' in queue: {}".format(q))

    # iterate over queue
    for i, item in enumerate(q):
        print("Item {}: {}".format(i, item))

    # print last item put in the queue
    print("Last item in queue: {}".format(q[-1]))

    # create stack
    stack = LIFOQueue(maxsize=2)
    print("\nInitial stack (maxsize={}): {}".format(stack.maxsize, stack))

    # add two elements in the stack
    stack.append(1)
    stack.append(2)
    print("After adding '1' and '2' in the stack: {}".format(stack))

    # add third element
    stack.append(3)
    print("After adding '3' in queue: {}".format(stack))

    # iterate over stack
    for i, item in enumerate(stack):
        print("Item {}: {}".format(i, item))

    # create priority queue
    pq = PriorityQueue(maxsize=2, ascending=True)
    print("\nInitial priority queue (maxsize={}): {}".format(pq.maxsize, pq))

    # add two elements in the queue
    pq.append((1, 'hello'))
    pq.append((2, 'world'))
    print("After adding (1, 'hello') and (2, 'world') in priority queue: {}".format(pq))
    print("Lowest priority item: {}".format(pq.get_lowest()))
    print("Highest priority item: {}".format(pq.get_highest()))

    print("Remove the 1st item: {} --> resulting priority queue: {}".format(pq.get(), pq))

    # add two elements in the queue
    pq.append((3, 'hello'))
    pq.append((4, 'sir'))
    print("After adding (3, 'hello') and (4, 'sir') in priority queue: {}".format(pq))
