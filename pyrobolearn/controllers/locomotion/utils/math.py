# -*- coding: utf-8 -*-
import numpy as np


# skew = lambda V: np.array([[0, -V[2], V[1]], [V[2], 0, -V[0]], [-V[1], V[0], 0]])
def skew(V):
    return np.array([[0, -V[2], V[1]], [V[2], 0, -V[0]], [-V[1], V[0], 0]])


# vex = lambda M: 0.5 * np.array([M[2, 1] - M[1, 2], M[0, 2] - M[2, 0], M[1, 0] - M[0, 1]])
def vex(M):
    return 0.5 * np.array([M[2, 1] - M[1, 2], M[0, 2] - M[2, 0], M[1, 0] - M[0, 1]])


def nullspace(M):
    return np.identity(M.shape[1]) - np.linalg.pinv(M).dot(M)


def sigmoid(x, shift_x=0.0, shift_y=0.0, scale_x=1.0, scale_y=1.0):
    y = 1 / (1 + np.exp(-(scale_x * x + shift_x))) * scale_y + shift_y
    return y


# homogeneous_vector = lambda P: np.append(P,1)
def homogeneous_vector(P):
    return np.hstack((P, 1))


# homogeneous_matrix = lambda rot=np.identity(3), pos=np.zeros(3): np.vstack((np.append(rot[0, :], pos[0]),
# np.append(rot[1, :], pos[1]), np.append(rot[2, :], pos[2]), np.array([0, 0, 0, 1])))
def homogeneous_matrix(rot=np.identity(3), pos=np.zeros(3)):
    transform_matrix = np.identity(4)
    transform_matrix[:3, :3] = rot[:3, :3]
    transform_matrix[:3, -1] = pos[:3]
    return transform_matrix


def sinwave(t, y_ini=0.0, y_max=1.0, y_min=-1.0, T=1):
    median = (y_max + y_min) / 2.0
    A = np.abs(y_max - y_min) / 2.0
    f = 1.0 / T
    phase = np.math.asin(np.clip((y_ini - median) / A, -1, 1))
    pos_t = A * np.sin(2 * np.pi * f * t + phase) + median
    vel_t = A * np.cos(2 * np.pi * f * t + phase) * 2 * np.pi * f
    acc_t = - A * np.sin(2 * np.pi * f * t + phase) * (2 * np.pi * f) ** 2
    return pos_t, vel_t, acc_t


class SineWave(object):
    def __init__(self, y_ini=0.0, y_max=1.0, y_min=-1.0, T=1):
        """
        sine wave is defined in the form:
            y(t) = A*sin(2*pi*f*t + phi) = A*sin(w*t + phi)
        A = the amplitude, the peak deviation of the function from zero.
        f = the ordinary frequency, the number of oscillations (cycles) that occur each second of time.
        w = 2*pi*f, the angular frequency, the rate of change of the function argument in units of radians per second
        phi = the phase, specifies (in radians) where in its cycle the oscillation is at t = 0.
        :param y_ini: the initial value of the sine wave
        :param y_max: the maximum value of the sine wave
        :param y_min: the minimum value of the sine wave
        :param T: cycle time
        """
        if y_ini < y_min or y_ini > y_max:
            print 'Error: the initial value should be between y_min and y_max values!'
        elif y_max < y_min:
            print 'Error: please change the order of inputs y_max and y_min!'

        self.median = (y_max + y_min) / 2.0
        self.A = np.abs(y_max - y_min) / 2.0
        self.f = 1.0 / T
        self.phase = np.math.asin(np.clip((y_ini - self.median) / self.A, -1, 1))

    def __call__(self, t):
        return self.A * np.sin(2 * np.pi * self.f * t + self.phase) + self.median

    def pos(self, t):
        return self.A * np.sin(2 * np.pi * self.f * t + self.phase) + self.median

    def vel(self, t):
        return self.A * np.cos(2 * np.pi * self.f * t + self.phase) * 2 * np.pi * self.f

    def acc(self, t):
        return - self.A * np.sin(2 * np.pi * self.f * t + self.phase) * (2 * np.pi * self.f) ** 2
