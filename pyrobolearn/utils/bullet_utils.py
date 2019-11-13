#!/usr/bin/env python
# -*- coding: utf-8 -*-
# This file provides some utilities with the pybullet interface.


class RGBColor(object):
    red = (1, 0, 0)
    green = (0, 1, 0)
    blue = (0, 0, 1)
    black = (0, 0, 0)
    white = (1, 1, 1)
    orange = (1, 0.647, 0)
    dark_orange = (1, 0.549, 0)
    yellow = (1, 1, 0)
    pink = (1, 0.753, 0.796)
    light_pink = (1, 0.714, 0.757)
    deep_pink = (1, 0.078, 0.576)
    grey = (0.502, 0.502, 0.502)


class RGBAColor(object):
    alpha = 1  # 0 = transparent, 1 = opaque
    red = (1, 0, 0, alpha)
    green = (0, 1, 0, alpha)
    blue = (0, 0, 1, alpha)
    black = (0, 0, 0, alpha)
    white = (1, 1, 1, alpha)
    orange = (1, 0.647, 0, alpha)
    dark_orange = (1, 0.549, 0, alpha)
    yellow = (1, 1, 0, alpha)
    pink = (1, 0.753, 0.796, alpha)
    light_pink = (1, 0.714, 0.757, alpha)
    deep_pink = (1, 0.078, 0.576, alpha)
    grey = (0.502, 0.502, 0.502, alpha)


class Key(object):  # BulletKeys
    """Map keys to ascii and bullet id"""
    a = 97
    b = 98
    c = 99
    d = 100
    e = 101
    f = 102
    g = 103
    h = 104
    i = 105
    j = 106
    k = 107
    l = 108
    m = 109
    n = 110
    o = 111
    p = 112
    q = 113
    r = 114
    s = 115
    t = 116
    u = 117
    v = 118
    w = 119
    x = 120
    y = 121
    z = 122
    n0 = 48
    n1 = 49
    n2 = 50
    n3 = 51
    n4 = 52
    n5 = 53
    n6 = 54
    n7 = 55
    n8 = 56
    n9 = 57
    space = 32
    shift = 65306
    ctrl = 65307
    alt = 65308
    enter = 65309
    left_arrow = 65295
    right_arrow = 65296
    top_arrow = 65297
    bottom_arrow = 65298

    # state
    nothing = 0
    down = 1
    triggered = 2
    pressed = 3
    released = 4

    # def __init__(self):
    #     # add symbols (<,>,[,',...) and numbers (0,1,2,...)
    #     keys = {chr(i): i for i in range(32, 65)}
    #     # add letters and symbols
    #     keys.update({chr(i): i for i in range(91, 127)})
    #     keys.update({char: i for char, i in zip(['shift', 'ctrl', 'alt', 'enter'] + ['left','right','top','bottom'],
    #                                             list(range(65306, 65310)) + list(range(65295,65299)))})
    #
    #     self.keys = keys
    #     self.keystr = {value: key for key, value in keys.items()}
    #
    #     self.a = 97
    #     self.b = 98
    #     self.c = 99
    #     self.d = 100
    #     self.e = 101
    #     self.f = 102
    #     self.g = 103
    #     self.h = 104
    #     self.i = 105
    #     self.j = 106
    #     self.k = 107
    #     self.l = 108
    #     self.m = 109
    #     self.n = 110
    #     self.o = 111
    #     self.p = 112
    #     self.q = 113
    #     self.r = 114
    #     self.s = 115
    #     self.t = 116
    #     self.u = 117
    #     self.v = 118
    #     self.w = 119
    #     self.x = 120
    #     self.y = 121
    #     self.z = 122
    #     self.n0 = 48
    #     self.n1 = 49
    #     self.n2 = 50
    #     self.n3 = 51
    #     self.n4 = 52
    #     self.n5 = 53
    #     self.n6 = 54
    #     self.n7 = 55
    #     self.n8 = 56
    #     self.n9 =  57
    #     self.space = 32
    #     self.shift = 65306
    #     self.ctrl = 65307
    #     self.alt = 65308
    #     self.enter = 65309
    #     self.left_arrow = 65295
    #     self.right_arrow = 65296
    #     self.top_arrow = 65297
    #     self.bottom_arrow = 65298
    #
    #     self.nothing = 0
    #     self.down = 1
    #     self.triggered = 2
    #     self.pressed = 3
    #     self.released = 4


class Mouse(object):  # Bullet mouse

    # event type
    moving = 1
    button = 2

    # button index
    no_click = -1
    left_click = 0
    middle_click = 1  # scroll
    right_click = 2

    # button state
    # state
    nothing = 0
    down = 1        # (never observed)
    triggered = 2   # (never observed)
    pressed = 3
    released = 4
