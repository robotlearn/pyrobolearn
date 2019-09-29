# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""Test different states
"""

import pyrobolearn.states as states

s1 = states.CumulativeTimeState()
s2 = states.AbsoluteTimeState()
s = s1 + s2

print("\ns = s1 + s2 = {}".format(s))

# update s1
print("\nUpdating state s1 by calling s1():")
s1.read()  # or s1()
print("s after calling s1() = {}".format(s))
print("s1 after calling s1() = {}".format(s1))
print("s2 after calling s1() = {}".format(s2))

# update s1 and s2 by calling s
print("\nUpdating the combined state s=(s1+s2) by calling s():")
s()
print("s after calling s() = {}".format(s))
print("s1 after calling s() = {}".format(s1))
print("s2 after calling s() = {}".format(s2))

# get the data
print("\nGetting the state data:")

# this will return a list of 2 arrays; each one of shape (1,). The size of the list 
# is equal to the number of states that it contains
print("s.data = {}".format(s.data))

# this will return a list with one array of shape (1,).
print("s1.data = {}".format(s1.data))

# this will return a merged state; it will merge the states that have the same dimensions 
# together and return a list of arrays which has a size equal to the number of different 
# dimensions. The arrays inside that list are ordered by their dimensionality in an 
# ascending way.
print("s.merged_data = {}".format(s.merged_data))

# this will return the same as s1.data because it just have one state
print("s1.merged_data = {}\n".format(s1.merged_data))
