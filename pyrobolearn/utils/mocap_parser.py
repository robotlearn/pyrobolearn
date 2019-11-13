#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt


class MocapParser(object):

    def __init__(self, filename):
        """
        Parser for motion capture. By default, if the data is described in Cartesian space, the x-axis should
        be pointing in front of the human, the y-axis on his/her left, and z-axis upward.
        :param filename:
        """
        self.filename = filename
        self.num_samples = 0
        self.joint_names = []
        self.link_names = []
        self.marker_names = []

        self.data = self.loadFile(filename)


    def loadFile(self, filename):
        raise NotImplementedError("loadFile is not implemented.")

    def interpolate(self, data, method='cubic', axis=-1)
        """
        Interpolate the Mocap data such that it is between 0 and 1, along the given axis.

        :param data: mocap data
        :param method: 'linear', 'cubic', 'hermite' interpolation
        :param axis: The axis on which to interpolate. The length should be equal to the number of samples in the
                     mocap data
        :return: Interpolator - function that given the time [0,1] will give the corresponding data
        """
        self.num_samples = data.shape[axis]
        x = np.linspace(0., 1., self.num_samples)
        interpolator = CubicSpline(x, self.data, axis=axis)
        return interpolator

    def getMarkerName(self, marker_idx=None):
        if marker_idx is None:
            return self.getMarkerNames()
        else:
            return self.marker_names[marker_idx]

    def getMarkerNames(self):
        return self.marker_names

    def getJointName(self, joint_idx=None):
        if joint_idx is None:
            return self.getJointNames()
        else:
            return self.joint_names[joint_idx]

    def getJointNames(self):
        return self.joint_names

    def getLinkName(self, link_idx=None):
        if link_idx is None:
            return self.getLinkNames()
        else:
            return self.link_names[link_idx]

    def getLinkNames(self):
        return self.link_names

    def getMarkerPosition(self, marker_idx=None):
        if marker_idx is None:
            return self.getMarkerPositions()
        else:
            pass

    def getMarkerPositions(self):
        pass

    def getJointPosition(self, joint_idx=None):
        if joint_idx is None:
            return self.getJointPositions()
        else:
            pass

    def getJointPositions(self):
        pass

    def getJointVelocity(self, joint_idx=None):
        if joint_idx is None:
            return self.getJointVelocities()
        else:
            pass

    def getJointVelocities(self):
        pass

    def getLinkPosition(self, link_idx=None):
        if link_idx is None:
            return self.getLinkPositions()
        else:
            pass

    def getLinkPositions(self):
        pass

    def getLinkVelocity(self, link_idx=None):
        if link_idx is None:
            return self.getLinkVelocities()
        else:
            pass

    def getLinkVelocities(self):
        pass

    def getLinkOrientation(self, link_idx=None):
        if link_idx is None:
            return self.getLinkOrientations()
        else:
            pass

    def getLinkOrientations(self):
        pass

    def getLinkAngularVelocity(self, link_idx=None):
        if link_idx is None:
            return self.getLinkAngularVelocities()
        else:
            pass

    def getLinkAngularVelocities(self):
        pass


    ## Plotting ##
    def plot3d(self, ax=None):
        pass

    def plotJointProfile(self, ax=None, joint_idx=None, pos=True, vel=True, acc=True):
        pass

    def plotLinkProfile(self, ax=None, link_idx=None, pos=True, vel=True, acc=True, wrt='world'):
        pass

    def plotMarkerProfile(self, ax=None, link_idx=None, pos=True, vel=True, acc=True, wrt='world'):
        pass

    def animate3d(self, ax=None, title=None):
        pass




from amcparser.skeleton import Skeleton
from amcparser.motion import SkelMotion

class CMUMocapParser(MocapParser):

    def __init__(self, skeleton_filename, motion_filename, skeleton_scale=1.0):
        super(CMUMocapParser, self).__init__(motion_filename)
        self.joint_names = ['head', 'upperneck', 'lowerneck', 'upperback', 'thorax', 'lowerback', 'root', # Spine
                            'rclavicle', 'rhumerus', 'rradius', 'rwrist', 'rhand', 'rthumb', 'rfingers', # Right arm
                            'lclavicle', 'lhumerus', 'lradius', 'lwrist', 'lhand', 'lthumb', 'lfingers', # Left arm
                            'rhipjoint', 'rfemur', 'rtibia', 'rfoot', 'rtoes', # Right leg
                            'lhipjoint', 'lfemur', 'ltibia', 'lfoot', 'ltoes'] # Left leg
        self.link_names = self.joint_names
        self.marker_names = self.joint_names
        self.base_name = 'root'

        # Load skeleton
        self.skeleton = Skeleton(skeleton_filename, scale=skeleton_scale)

    def loadFile(self, filename, framerate=120.):
        self.skeleton_motion = SkelMotion(self.skeleton, filename, (1./framerate))
        # compute trajectories
        #self.data = self.skeleton_motion.traverse(bone, start, end)
        self.data = self.skeleton_motion.traverse(None, 0, -1)
        # make sure that given axis

    def animate3d(self, ax=None, title=None):
        if ax is None:
            fig = plt.figure()
            ax = fig.gca(projection='3d')

        # Rescaling such that the skeleton it is in the right proportion and at the middle
        xmin, xmax = X[..., 2].min(), X[..., 2].max()
        ymin, ymax = X[..., 0].min(), X[..., 0].max()
        zmin, zmax = X[..., 1].min(), X[..., 1].max()
        x_len, y_len, z_len = (xmax - xmin), (ymax - ymin), (zmax - zmin)
        max_len = max([x_len, y_len, z_len])
        xmin, xmax = xmin + (x_len - max_len) / 2., xmin + (x_len + max_len) / 2.
        ymin, ymax = ymin + (y_len - max_len) / 2., ymin + (y_len + max_len) / 2.
        zmin, zmax = zmin + (z_len - max_len) / 2., zmin + (z_len + max_len) / 2.

        # Plot trajectories
        x, y, z = skel.bones['rhand'].xyz_data.T
        T = len(x)

        def init():
            ax.set_title('movement')
            ax.set_xlabel('x')
            ax.set_xlim(xmin, xmax)
            ax.set_ylabel('y')
            ax.set_ylim(ymin, ymax)
            ax.set_zlabel('z')
            ax.set_zlim(zmin, zmax)
            # ax.scatter(x[0], y[0], z[0], marker='o')
            return fig,

        def animate(i):
            # ax.view_init(elev=10., azim=i)
            ax.scatter(x[i], y[i], z[i], marker='o')
            return fig,

        def animate_skeleton(i):
            ax.clear()
            init()
            # ax.scatter(X[:,2,i], X[:,0,i], X[:,1,i], marker='o', c='b')
            for d in [X_TO, X_RA, X_LA, X_RL, X_LL]:
                ax.plot(d[:, i, 2], d[:, i, 0], d[:, i, 1], marker='o', c='b')
            return [fig]  # fig,

        # Animate
        # anim = animation.FuncAnimation(fig, animate, init_func=init,
        #                                frames=T, interval=20, blit=True)
        anim = animation.FuncAnimation(fig, animate_skeleton, init_func=init,
                                       frames=T, interval=20, blit=False)

        plt.show()


# Test
if __name__ == "__main__":
    pass