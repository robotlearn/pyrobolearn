#!/usr/bin/env python
# -*- coding: utf-8 -*-


class Task:
    def __init__(self, A, b, w=1):
        self.A = A  # matrix
        self.b = b  # vector
        self.w = w  # float

    def set_weight(self, w):
        self.w = w

    def show(self):
        print "A", self.A.shape, ":", "\n", self.A
        print "b", self.b.shape, ":", "\n", self.b
        print "w:", self.w
