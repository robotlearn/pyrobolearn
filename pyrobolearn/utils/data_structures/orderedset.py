#!/usr/bin/env python
"""Define the OrderedSet data structure class.
"""

import collections

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class OrderedSet(collections.MutableSet):
    r"""Ordered Set

    This is my own implementation of an ordered set, and was inspired a bit from [1] and [2].
    In this class, we internally use a set and an (ordered) list to describe an ordered set.

    In this implementation, the `delete/remove/discard item`, `move item`, `insert item`, `pop item` (except last
    item) operations are pretty expensive with a time complexity of O(N). However, operations such as get item` and
    `set item` have a time complexity of O(1).

    If you need to easily add items and get access to them, and don't need to remove, insert, and move items,
    use this class.

    Here are the time complexities for the average (and worst) case scenario (more info on [3,4]):
    * Iterate: O(N)
    * Copy: O(N)
    * Get Length: O(1)
    * Item in set: O(1) (worst: O(N))
    * Subset in set (without respecting the order): O(K) (worst: O(N))
    * Add/append item: O(1)
    * Delete/remove/discard item: O(N)
    * Pop last: O(1)
    * Pop first: O(N)
    * Pop given index: O(N)
    * Insert item: O(N)
    * Move item: O(N)
    * Get item from key: O(1)
    * Get items from slice: O(K)
    * Set item from key: O(1) if it doesn't have to move the data, otherwise O(N)
    * Set items from slice: O(K+N) (N because it might have to move some data to accomodate for the new items, see [4])
    * Delete item from key: O(N)

    * Is superset/subset (while respecting the order): O(N)
    * Union:
    * Intersection:
    * Difference:

    References:
        [1] http://code.activestate.com/recipes/576694-orderedset/
        [2] https://stackoverflow.com/questions/1653970/does-python-have-an-ordered-set
        [3] http://bigocheatsheet.com/
        [4] https://wiki.python.org/moin/TimeComplexity?
    """

    def __init__(self, iterator=None):
        """
        Initialize the ordered set, which basically contains a set and a list.

        Args:
            iterator: an iterator
        """
        self._set = set()
        self._list = []

        if isinstance(iterator, collections.Iterable):
            for item in iterator:
                self.add(item)

    ###########
    # Methods #
    ###########

    def add(self, item):
        """
        Add/Append an item to the ordered set.
        Time complexity: O(1)
        """
        if item not in self._set:
            self._set.add(item)
            self._list.append(item)

    # alias
    append = add

    def extend(self, iterator):
        """
        Extend the ordered set by adding/appending elements from the given iterator.
        Time complexity: O(K) where K is the size of the iterator
        """
        for item in iterator:
            self.add(item)

    def insert(self, idx, item):
        """
        Insert an item into the set at the specified index. If the item is already in the set, it moves it
        to the specified index.
        Time complexity: O(N)
        """
        # check idx
        idx = self._check_index(idx)
        if item in self._set:
            # move the item at the specified location
            self.move(idx, item)  # O(N)
        else:
            # add it
            self._list.insert(idx, item)  # O(N)
            self._set.add(item)  # O(1)

    def move(self, idx, item):
        """
        Move an item to the specified index. If the item is not in the set, it raises a KeyError.
        Time complexity: O(N)
        """
        # remove the item from the list/set
        self.remove(item)  # O(N)

        # insert item
        self.insert(idx, item)  # O(N)

    def discard(self, item):
        """
        Remove an item from the ordered set if it is a member. If the item is not a member do nothing.
        Time complexity: O(N)
        """
        if item in self._set:
            self._list.remove(item)  # O(N)
            self._set.remove(item)  # O(1)

    def remove(self, item):
        """
        Remove an item from the ordered set. If the item is not a member, it raises a KeyError.
        Time complexity: O(N)
        """
        if item not in self._set:
            raise KeyError(item)
        self.discard(item)

    def pop(self, index=None):
        """
        Remove and return an item of the ordered set at the specified index (default last).
        Time complexity: O(1) if last, O(N) if first.
        Args:
            index: index in the ordered set.
        """
        if index is None: index = len(self._list)
        index = self._check_index(index)  # to be sure the index is valid
        self._set.remove(self._list[index])  # O(1)
        item = self._list.pop(index)  # O(1) if last, O(N) if first
        return item

    def copy(self):
        """
        Return a shallow copy of an ordered set.
        Time complexity: O(N)
        """
        return self.__class__(self)

    def _check_index(self, idx):
        """
        Check the given index; if it is in the range of the ordered set, and if it is negative return the
        corresponding positive index.
        """
        if not isinstance(idx, int):
            raise TypeError("idx should be an integer.")
        if idx > len(self._list) or idx < -len(self._list):
            return KeyError(idx)
        if idx < 0:
            idx = len(self._list) + idx
        return idx

    def union(self, *others):
        """
        Return the union of sets as a new set.
        """
        s = self.copy()
        s.update(*others)
        return s

    def update(self, *others):
        """
        Update a set with the union of itself and others.
        """
        for other in others:
            self |= other

    def intersection(self, *others):
        """
        Return the intersection of two or more sets as a new set.
        """
        s = self.copy()
        s.intersection_update(*others)
        return s

    def intersection_update(self, *others):
        """
        Update a set with the intersection of itself and another.
        """
        for other in others:
            self &= other

    def difference(self, *others):
        """
        Return the difference of two or more sets as a new set; i.e. all elements that are in this set but not
        the others.
        """
        s = self.copy()
        s.difference_update(*others)
        return s

    def difference_update(self, *others):
        """
        Remove all elements of another set from this set.
        """
        for other in others:
            self -= other

    def symmetric_difference(self, *others):
        """
        Return the symmetric difference of several sets as a new set; i.e. union of sets - intersection of sets.
        """
        s = self.copy()
        s.symmetric_difference_update(*others)
        return s

    def symmetric_difference_update(self, *others):
        """
        Update a set with the symmetric difference of itself and others; i.e. set = union(sets) - intersection(sets)
        """
        intersection = self.copy()
        self.update(*others) # compute union
        intersection.intersection_update(*others) # compute intersection
        self -= intersection

    def issuperset(self, other, order=True):
        """
        Return True if the other set is a subset of this set. If 'order' is True, then the other set has to be
        a subset of this set, and have the same order as this one.
        Time complexity: O(N) if order, O(K) otherwise where N is the size of this set, and K is the size of the
                         other set.
        """
        if not isinstance(other, (OrderedSet, set)):
            raise TypeError("The 'other' argument should be a set, or an ordered set.")
        if len(other) == 0: # the empty set is always a subset of a set
            return True
        if len(other) > len(self): # the other set is bigger than this set, and thus is not a subset of that one
            return False

        # take into account the order if specified
        if order:
            if not isinstance(other, OrderedSet):
                raise TypeError("The 'other' argument should be an ordered set.")

            # traverse the subset and check each element appeared in the same order in the set
            iterator = iter(self._list)
            for item in other:
                # check if item of subset is in the set
                if item not in self:
                    return False

                # traverse the ordered set until we find this item
                while True:
                    try:
                        curr = next(iterator)
                        if curr == item: break
                    except StopIteration:
                        return False

            # return True as we checked that all the items in the subset are in the set
            return True

        # the order is not important
        else:
            return all([(item in self) for item in other])

    # alias
    contains = issuperset

    def issubset(self, other, order=True):
        """
        Return True if the other set is a superset of this set; i.e. return true if this set is a subset of the
        other set.
        Time complexity: same as `issuperset()`.
        """
        return other.issuperset(self, order=order)

    #############
    # Operators #
    #############

    def __repr__(self):
        """Return a representation string."""
        return '%s(%r)' % (self.__class__.__name__, list(self))

    def __contains__(self, item):
        """
        Check if the given item/subset is in the set. The subset doesn't have the same order.
        If the order is important, see the `issubset()` method.
        Time complexity: O(1) if single item, O(K) if subset (where K is the size of the subset)
        """
        if isinstance(item, OrderedSet):
            return self.issuperset(item, order=False)
        return item in self._set

    def __len__(self):
        """
        Return the length of the ordered set.
        Time complexity: O(1)
        """
        return len(self._list)

    def __iter__(self):
        """
        Iterate over the ordered set in the order the items have been added.
        Time complexity: O(N)
        """
        for item in self._list:
            yield item

    def __reversed__(self):
        """
        Iterate over the ordered set in the reverse order the items have been added.
        Time complexity: O(N)
        """
        for item in reversed(self._list):
            yield item

    def __getitem__(self, idx):
        """
        Return the item (ordered set) associated to the given index (indices).
        Time complexity: O(1) if index, O(K) if slice (where K is the size of slice)
        """
        # checks
        if len(self) == 0:
            raise KeyError("Trying to get an item from an empty set.")
        if isinstance(idx, int):
            return self._list[idx]
        else: # slice
            return OrderedSet(self._list[idx])

    def __setitem__(self, idx, item):
        """
        Replace the item at the specified index. If the item is already in the ordered set, it will move it to the
        specified index.
        Time complexity: O(1) if index is an int and it doesn't have to move an item, O(N) if it has to move it,
                         and O(K+N) if slice
        """
        if isinstance(idx, int):
            if item in self._set:
                self.move(idx, item)
            else:
                self._list[idx] = item
                self._set.add(item)
        elif isinstance(idx, slice):  # slice
            # replace in list
            items_to_remove = self._list[idx]  # O(K)
            self._list[idx] = item  # O(K+N)

            # remove previous items from the set
            for elem in items_to_remove:
                self._set.remove(elem)  # O(1)

            # add new items in the set
            for elem in item:
                if elem not in self._set:
                    self._set.add(elem)
        else:
            raise TypeError("Expecting idx to be an int or slice.")

    def __add__(self, other):
        """
        Union between two ordered sets.
        """
        return self | other

    def __iadd__(self, other):
        """
        Update a set with the union of itself and the other set.
        """
        self |= other

    def __and__(self, other):
        """
        Intersection between two ordered sets.
        """
        return super(OrderedSet, other).__and__(self)

    def __iand__(self, other):
        """
        Update a set with the intersection of itself and the other set.
        """
        super(OrderedSet, self).__iand__(other)

    def __mul__(self, other):
        """
        Intersection between two ordered sets.
        """
        return self & other

    def __imul__(self, other):
        """
        Update a set with the intersection of itself and the other set.
        """
        self &= other


################################################################################################


class OrderedSet2(collections.MutableSet):
    r"""Ordered Set

    This is my own implementation of an ordered set, and was inspired a bit from [1] and [2].
    In this class, we internally use a dictionary where keys are the items of the ordered set,
    and each associated value is a tuple containing the pointer to the previous and next items.
    It can thus be seen as a double-linked list with fast access.

    In this implementation, the `get item`, `set item`, `move item`, and `insert item` operations are pretty
    expensive with a time complexity of O(N) compared to a list (which has O(1)). However, operations such as
    `delete/remove/discard item`, and `pop first/last items` have a time complexity of O(1).

    If you need to easily remove and append items, and you don't need to access (get/set) the items in the set,
    use this class.

    Here are the time complexities for the average (and worst) case scenario (more info on [3,4]):
    * Iterate: O(N)
    * Copy: O(N)
    * Get Length: O(1)
    * Item in set: O(1) (worst: O(N))
    * Subset in Set (without respecting the order): O(K) (worst: O(N))
    * Add/append item: O(1)
    * Delete/remove/discard item: O(1)
    * Pop last: O(1)
    * Pop first: O(1)
    * Pop given index: O(N)
    * Insert item: O(N)
    * Move item: O(N)
    * Get item from key: O(N)
    * Get items from slice: O(N+K) (+K because we build a new ordered set)
    * Set item from key: O(N)
    * Set items from slice: Not Implemented
    * Delete item from key: O(N)

    * Is superset/subset (while respecting the order): O(N)
    * Union:
    * Intersection:
    * Difference:

    References:
        [1] http://code.activestate.com/recipes/576694-orderedset/
        [2] https://stackoverflow.com/questions/1653970/does-python-have-an-ordered-set
        [3] http://bigocheatsheet.com/
        [4] https://wiki.python.org/moin/TimeComplexity?
    """

    class _NonePtr(object): pass
    NonePtr = _NonePtr()

    def __init__(self, iterator=None):
        """
        Initialize the ordered set.

        Args:
            iterator: An iterator
        """
        self._start, self._end = self.NonePtr, self.NonePtr
        self._map = {}

        if isinstance(iterator, collections.Iterable):
            for item in iterator:
                self.add(item)

    ###########
    # Methods #
    ###########

    def add(self, item):
        """
        Add/Append an item to the ordered set.
        Time complexity: O(1)
        """
        if item not in self._map:
            if self._end == self.NonePtr: # first item
                self._map[item] = [self.NonePtr, self.NonePtr]
                self._start = item
                self._end = item
            else: # subsequent item
                # update previous item to point to new item
                self._map[self._end][1] = item
                # append new item at the end
                self._map[item] = [self._end, self.NonePtr]
                # update end pointer
                self._end = item

    # alias
    append = add

    def extend(self, iterator):
        """
        Extend the ordered set by adding/appending elements from the given iterator.
        Time complexity: O(K) where K is the size of the iterator
        """
        for item in iterator:
            self.add(item)

    def insert(self, idx, item):
        """
        Insert an item into the set at the specified index. If the item is already in the set, it moves it
        to the specified index.
        Time complexity: O(N)
        """
        # check idx
        idx = self._check_index(idx)

        # if the set is initially empty or index is the size of the set, just add the item (at the end)
        if len(self._map) == 0 or idx == len(self._map):
            self.append(item)
        else:
            if item in self._map:
                self.move(idx, item)
            else:
                # get current item at the specified index, update the items nearby, and insert the new item
                curr = self[idx]
                prev_item, next_item = self._map[curr]
                if prev_item == self.NonePtr:  # beginning of the ordered set (idx=0)
                    self._map[curr][0] = item
                    self._map[item] = [self.NonePtr, curr]
                else:  # somewhere between the start and the end (not included)
                    self._map[item] = [prev_item, next_item]
                    self._map[prev_item][1] = item
                    self._map[next_item][0] = item

    def move(self, idx, item):
        """
        Move an item to the specified index. If the item is not in the set, it raises a ValueError.
        Time complexity: O(N)
        """
        if item not in self._map:
            return ValueError("The given item is not in the set.")

        # remove the item from the list/set (time complexity: O(1))
        self.remove(item)

        # insert item
        self.insert(idx, item)

    def discard(self, item):
        """
        Remove an item from the ordered set if it is a member. If the item is not a member do nothing.
        Time complexity: O(1)
        """
        if item in self._map:
            prev_item, next_item = self._map[item]

            # update previous item
            if prev_item != self.NonePtr:
                self._map[prev_item][1] = next_item
            else:  # we are removing the first item
                self._start = next_item

            # update next item
            if next_item != self.NonePtr:
                self._map[next_item][0] = prev_item
            else:  # we are removing the last item
                self._end = prev_item

            # remove item
            self._map.pop(item)

    def remove(self, item):
        """
        Remove an item from the ordered set. If the item is not a member, it raises a KeyError.
        Time complexity: O(1)
        """
        if item not in self._map:
            raise KeyError(item)
        self.discard(item)

    def pop(self, last=True):
        """
        Remove and return the first or last element of the ordered set depending on the provided argument.
        Time complexity: O(1)
        Args:
            last: if True, remove and return the last item added to the set. If False, remove and return the first one.
        """
        if last:
            item = self._end
        else:
            item = self._start
        self.remove(item)
        return item

    def copy(self):
        """
        Return a shallow copy of an ordered set.
        Time complexity: O(N)
        """
        return self.__class__(self)

    def _check_index(self, idx):
        """
        Check the given index; if it is in the range of the ordered set, and if it is negative return the
        corresponding positive index.
        """
        if not isinstance(idx, int):
            raise TypeError("idx should be an integer.")
        if idx > len(self._map) or idx < -len(self._map):
            return KeyError(idx)
        if idx < 0:
            idx = len(self._map) + idx
        return idx

    def union(self, *others):
        """
        Return the union of sets as a new set.
        """
        s = self.copy()
        s.update(*others)
        return s

    def update(self, *others):
        """
        Update a set with the union of itself and others.
        """
        for other in others:
            self |= other

    def intersection(self, *others):
        """
        Return the intersection of two or more sets as a new set.
        """
        s = self.copy()
        s.intersection_update(*others)
        return s

    def intersection_update(self, *others):
        """
        Update a set with the intersection of itself and another.
        """
        for other in others:
            self &= other

    def difference(self, *others):
        """
        Return the difference of two or more sets as a new set; i.e. all elements that are in this set but not
        the others.
        """
        s = self.copy()
        s.difference_update(*others)
        return s

    def difference_update(self, *others):
        """
        Remove all elements of another set from this set.
        """
        for other in others:
            self -= other

    def symmetric_difference(self, *others):
        """
        Return the symmetric difference of several sets as a new set; i.e. union of sets - intersection of sets.
        """
        s = self.copy()
        s.symmetric_difference_update(*others)
        return s

    def symmetric_difference_update(self, *others):
        """
        Update a set with the symmetric difference of itself and others; i.e. set = union(sets) - intersection(sets)
        """
        intersection = self.copy()
        self.update(*others) # compute union
        intersection.intersection_update(*others) # compute intersection
        self -= intersection

    def issuperset(self, other, order=True):
        """
        Return True if the other set is a subset of this set. If 'order' is True, then the other set has to be
        a subset of this set, and have the same order as this one.
        Time complexity: O(N) if order, O(K) otherwise where N is the size of this set, and K is the size of the
                         other set.
        """
        if not isinstance(other, (OrderedSet, set)):
            raise TypeError("The 'other' argument should be a set or an ordered set.")
        if len(other) == 0: # the empty set is always a subset of a set
            return True
        if len(other) > len(self): # the other set is bigger than this set, and thus is not a subset of that one
            return False

        # take into account the order if specified
        if order:
            if not isinstance(other, OrderedSet):
                raise TypeError("The 'other' argument should be an ordered set.")

            # check first item in the subset
            curr_other = other._start
            # check if inside the set
            if curr_other not in self._map: return False
            # same start pointer in the set
            curr = curr_other

            # traverse the subset and check each element appeared in the same order in the set
            for item in other:
                while True:
                    # return False if we are at the end
                    if curr == self.NonePtr: return False
                    # if item in the set, go to the next item in the subset
                    if curr == item: break
                    # go to the next item in set
                    curr = self._map[curr][1]

            # return True as we checked that all the items in the subset are in the set
            return True

        # the order is not important
        else:
            return all([(item in self) for item in other])

    # alias
    contains = issuperset

    def issubset(self, other, order=True):
        """
        Return True if the other set is a superset of this set; i.e. return true if this set is a subset of the
        other set.
        Time complexity: same as `issuperset()`.
        """
        return other.issuperset(self, order=order)

    #############
    # Operators #
    #############

    def __repr__(self):
        """Return a representation string."""
        return '%s(%r)' % (self.__class__.__name__, list(self))

    def __contains__(self, item):
        """
        Check if the given item/subset is in the set. The subset doesn't have the same order.
        If the order is important, see the `issubset()` method.
        Time complexity: O(1) if single item, O(K) if subset (where K is the size of the subset)
        """
        if isinstance(item, OrderedSet):
            return self.issuperset(item, order=False)
        return item in self._map

    def __len__(self):
        """
        Return the length of the ordered set.
        Time complexity: O(1)
        """
        return len(self._map)

    def __iter__(self):
        """
        Iterate over the ordered set in the order the items have been added.
        Time complexity: O(N)
        """
        curr = self._start
        while curr != self.NonePtr:
            yield curr
            curr = self._map[curr][1]

    def __reversed__(self):
        """
        Iterate over the ordered set in the reverse order the items have been added.
        Time complexity: O(N)
        """
        curr = self._end
        while curr != self.NonePtr:
            yield curr
            curr = self._map[curr][0]

    def __getitem__(self, idx):
        """
        Return the item (ordered set) associated to the given index (indices).
        Time complexity: O(N)
        """
        # checks
        if not isinstance(idx, (int, slice)):
            raise KeyError("Expecting an int or slice for the index.")
        if len(self._map) == 0:
            raise KeyError("Trying to get an item from an empty set.")

        if isinstance(idx, int): # index is an integer

            # check index
            idx = self._check_index(idx)

            # traverse the set in a specific order based on how close the index is wrt the start/end of the set
            curr = self.NonePtr
            if 0 <= idx <= len(self._map) / 2:  # traverse from the beginning
                count = 0
                for curr in self:
                    if idx == count: break
                    count += 1
            else:  # traverse from the end
                count = len(self._map)-1
                for curr in reversed(self):
                    if idx == count: break
                    count -= 1

            # return corresponding item
            return curr

        else: # multiple indices
            # check arguments from slice
            lst = []
            start, stop, step = idx.start, idx.stop, idx.step
            iterator = self
            if step is None: step = 1
            if step > 0:
                if start is None: start = 0
                if stop is None: stop = len(self)
            else:
                iterator = reversed(self)
                if start is None: start = len(self) - 1
                if stop is None: stop = -1
                start = len(self) - 1 - start
                stop = len(self) - 1 - stop
                step = abs(step)

            # traverse the ordered set, and add the requested items into the list
            count = 0
            for item in iterator:
                if count >= stop: break
                if count < start: pass
                else:
                    if ((count - start) % step) == 0:
                        lst.append(item)
                count += 1

            # return a new ordered set
            return OrderedSet(lst)

    def __setitem__(self, idx, item):
        """
        Replace the item at the specified index. If the item is already in the ordered set, it will move it to the
        specified index.
        Time complexity: O(N)
        """
        # check idx
        idx = self._check_index(idx)

        # check if item already in the set
        if item in self._map:
            # move the item at the specified index
            self.move(idx, item)
        else: # item is not in the set, thus replace the item at the specified index
            count = 0
            for curr in self: # go through the ordered set
                if count == idx:
                    # add new item
                    prev_item, next_item = self._map[curr]
                    self._map[item] = (prev_item, next_item)

                    # set the start/end pointers if needed
                    if count == 0:
                        self._start = item
                    if count == len(self):
                        self._end = item

                    # remove the item we have to replaced
                    self._map.pop(curr)
                    break
                count += 1

    def __add__(self, other):
        """
        Union between two ordered sets.
        """
        return self | other

    def __iadd__(self, other):
        """
        Update a set with the union of itself and the other set.
        """
        self |= other

    def __and__(self, other):
        """
        Intersection between two ordered sets.
        """
        return super(OrderedSet, other).__and__(self)

    def __iand__(self, other):
        """
        Update a set with the intersection of itself and the other set.
        """
        super(OrderedSet, self).__iand__(other)

    def __mul__(self, other):
        """
        Intersection between two ordered sets.
        """
        return self & other

    def __imul__(self, other):
        """
        Update a set with the intersection of itself and the other set.
        """
        self &= other


# OrderedSet = OrderedSet2


# Tests
if __name__ == '__main__':
    # Test the first order set
    s0 = OrderedSet()
    l = [9,82,10,-5,-6,4]
    s1 = OrderedSet(l + [82,10])
    s2 = OrderedSet([10,9,48,56])
    s3 = s2 & s1

    print("\nOrdered sets:")
    print("s0: {}".format(s0))
    print("s1: {}".format(s1))
    print("s2: {}".format(s2))
    print("s3: {}".format(s3))

    print("\nSubsets:")
    print("82 in s1? {}".format(82 in s1))
    print("s0 in s0? {}".format(s0 in s0))
    print("s0 in s1? {}".format(s0 in s1))
    print("s2 in s1? {}".format(s2 in s1))
    print("s3 in s1? {}".format(s3 in s1))
    print("s3 in s2? {}".format(s3 in s2))
    print("s1.issuperset(s3, order=True) = {}".format(s1.issuperset(s3)))
    print("s3.issubset(s1) = {}".format(s3.issubset(s1)))
    print("s1.contains(s3) = {}".format(s1.contains(s3)))
    print("s2.issuperset(s3, order=True) = {}".format(s2.issuperset(s3)))
    print("s3.issubset(s2) = {}".format(s3.issubset(s2)))
    print("s2.contains(s3) = {}".format(s2.contains(s3)))

    print("\nIndexing:")
    print("s1[0] = {}".format(s1[0]))
    print("s1[2] = {}".format(s1[2]))
    print("s1[-1] = {}".format(s1[-1]))
    print("s1[:4] = {}  and  l[:4] = {}".format(s1[:4], l[:4]))
    print("s1[2:4] = {}  and  l[2:4] = {}".format(s1[2:4], l[2:4]))
    print("s1[2:] = {}  and  l[2:] = {}".format(s1[2:], l[2:]))
    print("s1[::2] = {}  and  l[::2] = {}".format(s1[::2], l[::2]))
    print("s1[1:4:2] = {}  and  l[1:4:2] = {}".format(s1[1:4:2], l[1:4:2]))
    print("s1[::-1] = {}  and  l[::-1] = {}".format(s1[::-1], l[::-1]))
    print("s1[4::-1] = {}  and  l[4::-1] = {}".format(s1[4::-1], l[4::-1]))
    print("s1[:1:-1] = {}  and  l[:1:-1] = {}".format(s1[:1:-1], l[:1:-1]))
    print("s1[:2:-1] = {}  and  l[:2:-1] = {}".format(s1[:2:-1], l[:2:-1]))
    print("s1[4:1:-1] = {}  and  l[4:1:-1] = {}".format(s1[4:1:-1], l[4:1:-1]))
    print("s1[4:1:-2] = {}  and  l[4:1:-2] = {}".format(s1[4:1:-2], l[4:1:-2]))

    print("\nDisjoint:")
    print("s1.isdisjoint(s0) = {}".format(s1.isdisjoint(s0)))
    print("s1.isdisjoint(s2) = {}".format(s1.isdisjoint(s2)))

    print("\nUnion:")
    print("s1 | s2 = {}".format(s1 | s2))
    print("s1 + s2 = {}".format(s1 + s2))
    print("s1.union(s2) = {}".format(s1.union(s2)))
    print("s2 | s1 = {}".format(s2 | s1))
    print("s2 + s1 = {}".format(s2 + s1))
    print("s2.union(s1) = {}".format(s2.union(s1)))

    print("\nIntersection:")
    print("s1 & s2 = {}".format(s1 & s2))
    print("s1 * s2 = {}".format(s1 * s2))
    print("s1.intersection(s2) = {}".format(s1.intersection(s2)))
    print("s2 & s1 = {}".format(s2 & s1))
    print("s2 * s1 = {}".format(s2 * s1))
    print("s2.intersection(s1) = {}".format(s2.intersection(s1)))

    print("\nDifference:")
    print("s1 - s3 = {}".format(s1 - s3))
    print("s1.difference(s3) = {}".format(s1.difference(s3)))
    print("s3 - s1 = {}".format(s3 - s1))
    print("s3.difference(s1) = {}".format(s3.difference(s1)))
    print("s1 - s2 = {}".format(s1 - s2))
    print("s1 - s0 = {}".format(s1 - s0))
