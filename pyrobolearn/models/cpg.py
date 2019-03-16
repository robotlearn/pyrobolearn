#!/usr/bin/env python
"""Define the Central Pattern Generator Node and Network classes

Central Pattern Generators (CPGs) allows to model rhythmic movement primitives, and are often used in locomotion.
This file provides the CPG Node and Network, where the network is composed of CPG nodes connected / phased together.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, CheckButtons
from matplotlib.animation import FuncAnimation


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class CPGNode(object):
    r"""Central Pattern Generator Node

    Each CPG node :math:`i` is driven by the following differential equations:
    * phase: :math:`\dot{\phi}_i &= \omega_i + \sum_j a_j w_{ij} \sin(\phi_j - \phi_i - \varphi_{ij})`
    * amplitude: :math:`\ddot{a}_i &= K_a (A_i - a_i) - D_a \dot{a}_i`
    * offset: :math:`\ddot{o}_i &= K_o (O_i - o_i) - D_o \dot{o}_i`
    * target angle: :math:`\theta_i &= o_i + a_i \cos(\phi_i)`

    References:
        [1] "Central pattern generators for locomotion control in animals and robots: a review", Ijspeert, 2008
    """

    def __init__(self, id, phi=0, offset=0, amplitude=1., timesteps=100, freq=None):
        self.id = id
        self.timesteps = timesteps
        self.dt = 1./self.timesteps
        self.t = 0.

        # gains
        self.D_amp, self.D_offset = 20., 20.
        self.K_amp, self.K_offset = self.D_amp**2/4., self.D_offset**2/4.

        # state parameters
        self.phi, self.curr_phi = phi, phi
        self.amp, self.damp, self.curr_amp = amplitude, 0, amplitude
        self.offset, self.doffset, self.curr_offset = offset, 0, offset

        # control parameters
        self.des_offset = offset
        self.des_amp = amplitude
        self.des_freq = 1. if freq is None else freq  # 1./self.timesteps if freq is None else freq
        self.des_omega = 2 * np.pi * self.des_freq

        # init values
        self.init_phi = self.phi
        # self.init_offset = self.offset
        # self.init_amp = self.amp

        # angle
        self.theta = self.offset + self.amp * np.cos(self.phi)

        # dict containing the CPG nodes connected to this node with their coupling weight and bias
        self.nodes = {}
        self.sliderNodes = {}

        # plot
        self.frame = None
        self.fig = None
        self.do_plot_offset, self.do_plot_amp, self.do_plot_phi, self.do_plot_theta = False, False, False, True
        self.updated = True

        # keep in memory the previous `timesteps` values of offset, amplitude, phi, and theta
        self.offsets = [self.offset] * self.timesteps
        self.amps = [self.amp] * self.timesteps
        self.phis = [self.phi] * self.timesteps
        self.thetas = [self.theta] * self.timesteps

    ##############
    # Properties #
    ##############

    @property
    def des_offset(self):
        """Return the desired offset."""
        return self._des_offset

    @des_offset.setter
    def des_offset(self, offset):
        """Set the desired offset."""
        if offset is None:
            offset = 0.
        if not isinstance(offset, (int, float)):
            raise TypeError("Expecting the desired offset to be an integer or float, got instead: "
                            "{}".format(type(offset)))
        self._des_offset = offset

    @property
    def des_amp(self):
        """Return the desired amplitude."""
        return self._des_amp

    @des_amp.setter
    def des_amp(self, amplitude):
        """Set the desired amplitude."""
        if amplitude is None:
            amplitude = 0.
        if not isinstance(amplitude, (int, float)):
            raise TypeError("Expecting the desired offset to be an integer or float, got instead: "
                            "{}".format(type(amplitude)))
        self._des_amp = amplitude

    @property
    def des_freq(self):
        """Return the desired frequency."""
        return self._des_freq

    @des_freq.setter
    def des_freq(self, freq):
        """Set the desired frequency."""
        if freq is None:
            freq = 1.
        if not isinstance(freq, (int, float)):
            raise TypeError("Expecting the desired frequency to be an integer or float, got instead: "
                            "{}".format(type(freq)))
        self._des_freq = freq
        self.des_omega = 2 * np.pi * self._des_freq

    @property
    def num_coupling_nodes(self):
        """Return the number of coupling nodes."""
        return len(self.nodes)

    # @property
    # def phi(self):
    #     """Return the phase."""
    #     return self._phi
    #
    # @phi.setter
    # def phi(self, phi):
    #     """Set the phase."""
    #     self.curr_phi, self.phi = phi, phi
    #     self.theta = self.offset + self.amp * np.cos(self.phi)

    def set_phi(self, phi):
        self.curr_phi, self.phi = phi, phi
        self.theta = self.offset + self.amp * np.cos(self.phi)

    @property
    def num_parameters(self):
        return 3 + 2 * len(self.nodes)

    ###########
    # Methods #
    ###########

    def add_node(self, cpg_node, weight=0., bias=0.):
        """Add a coupling node."""
        self.nodes[cpg_node] = {'weight': weight, 'bias': bias}

    def remove_node(self, cpg_node):
        """Remove a coupling node"""
        if cpg_node in self.nodes:
            self.nodes.pop(cpg_node)

    def reset(self):
        """Reset the phase of the CPG node; this can be useful for phase resetting."""
        # Re-initialize the CPG
        self.curr_phi, self.phi = self.init_phi, self.init_phi
        self.theta = self.offset + self.amp * np.cos(self.phi)

    def parameters(self):
        """Returns an iterator over the model parameters."""
        # proper node parameters
        yield self.des_amp
        yield self.des_offset
        yield self.des_freq

        # coupling parameters
        for node in self.nodes:
            yield self.nodes[node]['weight']
            yield self.nodes[node]['bias']

    def named_parameters(self):
        """Returns an iterator over the model parameters, yielding both the name and the parameter itself"""
        # proper node parameters
        yield str(self) + ':amplitude', self.des_amp
        yield str(self) + ':offset', self.des_offset
        yield str(self) + ':frequency', self.des_freq

        # coupling parameters
        for node in self.nodes:
            yield str(self) + ':weight:' + str(node), self.nodes[node]['weight']
            yield str(self) + ':bias:' + str(node), self.nodes[node]['bias']

    def list_parameters(self):
        """Return a list of parameters"""
        return list(self.parameters())

    def get_vectorized_parameters(self, to_numpy=True):
        """Return a vectorized form of the parameters (weights, biases, offsets)."""
        if to_numpy:
            return np.array(self.list_parameters())
        else:
            return torch.from_numpy(np.array(self.list_parameters()))

    def set_vectorized_parameters(self, vector):
        """Set the vector parameters."""
        # set the parameters from the vectorized one
        if len(vector) != self.num_parameters:
            raise ValueError("Expecting the size of the vectorized parameters to match the number of parameters "
                             "of this node. Instead of having {}, I got {}.".format(self.num_parameters, len(vector)))
        self.des_amp = vector[0]
        self.des_offset = vector[1]
        self.des_freq = vector[2]

        # coupling parameters
        for idx, node in enumerate(self.nodes):
            self.nodes[node]['weight'] = vector[2*idx + 3]
            self.nodes[node]['bias'] = vector[2*idx + 4]

    def step(self):
        """
        Perform a step by integrating (i.e. Euler integration) the differential equations governing the CPG.

        The CPG equations for node :math:`i` are:

        .. math::

            \dot{\phi}_i &= \omega_i + \sum_j a_j w_{ij} \sin(\phi_j - \phi_i - \varphi_{ij}) \\
            \ddot{a}_i &= K_a (A_i - a_i) - D_a \dot{a}_i \\
            \ddot{o}_i &= K_o (O_i - o_i) - D_o \dot{o}_i \\
            \theta_i &= o_i + a_i \cos(\phi_i) \\

        where
        :math:`\phi` is the phase,
        :math:`\omega` is the desired angular velocity (desired frequency),
        :math:`A` and :math:`a` are the desired and current amplitude,
        :math:`O` and :math:`o` are the desired and current offset,
        :math:`K` and :math:`D` are the stiffness and damping gains (which are normally related such that the system
        is critically damped),
        :math:`w_{ij}` and :math:`\varphi_{ij}` are the coupling weights and phase biases, and finally,
        :math:`\theta` is the resulting (joint) angle (to be sent to the controller).

        .. note:: for each node, update() has to be called only after step() has been called for all the nodes!
        """

        # offset
        ddoffset = self.K_offset * (self.des_offset - self.offset) - self.D_offset * self.doffset
        self.doffset += ddoffset * self.dt
        self.curr_offset += self.doffset * self.dt

        # amplitude
        ddamp = self.K_amp * (self.des_amp - self.amp) - self.D_amp * self.damp
        self.damp += ddamp * self.dt
        self.curr_amp += self.damp * self.dt

        # phase
        dphi = self.des_omega
        for node, coupling in self.nodes.items():
            dphi += node.amp * coupling['weight'] * np.sin(node.phi - self.phi - coupling['bias'])
        self.curr_phi = (self.curr_phi + dphi * self.dt) % (2*np.pi)

        # self.t += self.dt

    def update(self):
        curr_t = self.t + self.dt
        curr_theta = self.offset + self.amp * np.cos(self.phi)

        # update plot
        if self.fig is not None:
            if self.do_plot_theta:
                self.ax_plot.plot([self.t, curr_t], [self.theta, curr_theta], 'b')  # , animated=True)
            if self.do_plot_amp:
                self.ax_plot.plot([self.t, curr_t], [self.amp, self.curr_amp], 'g')  # , animated=True)
            if self.do_plot_offset:
                self.ax_plot.plot([self.t, curr_t], [self.offset, self.curr_offset], 'r')  # , animated=True)
            if self.do_plot_phi:
                self.ax_plot.plot([self.t, curr_t], [self.phi, self.curr_phi], 'm')  # , animated=True)

            # move axis to get a feeling of online plotting. Warning: this slows down rendering
            # self.ax_plot.set_xlim(curr_t - 0.5, curr_t + 0.5)

        # update state params
        self.phi = self.curr_phi
        self.amp = self.curr_amp
        self.offset = self.curr_offset
        self.theta = curr_theta
        self.t = curr_t

        # update data
        # if len(self.phis) >= self.timesteps: self.phis = self.phis[1:]
        # if len(self.amps) >= self.timesteps: self.amps = self.amps[1:]
        # if len(self.offsets) >= self.timesteps: self.offsets = self.offsets[1:]
        # if len(self.thetas) >= self.timesteps: self.thetas = self.thetas[1:]
        # self.phis.append(self.phi)
        # self.amps.append(self.amp)
        # self.offsets.append(self.offset)
        # self.thetas.append(self.theta)
        # self.updated = True

        # update sliders
        if self.fig is not None:
            self.slider_offset.set_val(self.offset)
            self.slider_amp.set_val(self.amp)
            self.slider_phi.set_val(self.phi)

        # update canvas
        # self.fig.canvas.draw()
        # plt.show(block=False)

    def plot(self):
        if self.fig is None:
            self.fig = plt.figure('CPG '+str(self.id))

        # Plot signals #
        h = 0.65
        self.ax_plot = self.fig.add_axes([0.1, h, 0.8, 0.3])
        self.ax_plot.set_ylim(-np.pi, np.pi)
        self.ax_plot.set_xlim(0, 1)
        ax = self.fig.add_axes([0.91, h+0.06, 0.08, 0.18])
        # ax.axis('off')
        self.check_offset = CheckButtons(ax, ['o', 'a', 'phi', 'theta'],
                                         [self.do_plot_offset, self.do_plot_amp, self.do_plot_phi, self.do_plot_theta])

        def check_buttons(val):
            if val == 'o':
                self.do_plot_offset = not self.do_plot_offset
                # self.offset_line.set_visible(self.do_plot_offset)
            elif val == 'a':
                self.do_plot_amp = not self.do_plot_amp
                # self.amp_line.set_visible(self.do_plot_amp)
            elif val == 'phi':
                self.do_plot_phi = not self.do_plot_phi
                # self.phi_line.set_visible(self.do_plot_phi)
            elif val == 'theta':
                self.do_plot_theta = not self.do_plot_theta
                # self.theta_line.set_visible(self.do_plot_theta)
        self.check_offset.on_clicked(check_buttons)

        # x = np.linspace(0., 1., self.timesteps)
        # self.theta_line = self.ax_plot.plot(x, self.thetas)[0]
        # self.amp_line = self.ax_plot.plot(x, self.amps)[0]
        # self.offset_line = self.ax_plot.plot(x, self.offsets)[0]
        # self.phi_line = self.ax_plot.plot(x, self.phis)[0]
        #
        # self.theta_line.set_visible(self.do_plot_theta)
        # self.amp_line.set_visible(self.do_plot_amp)
        # self.offset_line.set_visible(self.do_plot_offset)
        # self.phi_line.set_visible(self.do_plot_phi)
        #
        # def update_plot(_):
        #     lines = []
        #     if self.updated:
        #         if self.do_plot_theta:
        #             self.theta_line.set_ydata(self.thetas)
        #             lines.append(self.theta_line)
        #         if self.do_plot_amp:
        #             self.amp_line.set_ydata(self.amps)
        #             lines.append(self.amp_line)
        #         if self.do_plot_offset:
        #             self.offset_line.set_ydata(self.offsets)
        #             lines.append(self.offset_line)
        #         if self.do_plot_phi:
        #             self.phi_line.set_ydata(self.phis)
        #             lines.append(self.phi_line)
        #         self.updated = False
        #     return lines
        #
        # self.anim = FuncAnimation(self.fig, update_plot, interval=100, blit=True)

        # State parameters #
        # offset
        h -= 0.1
        # ax = self.fig.add_axes([0.2, h, 0.6, 0.03])
        ax = self.fig.add_axes([0.1, h, 0.35, 0.03])
        self.slider_offset = Slider(ax, 'o', -np.pi, np.pi, valinit=self.offset, dragging=False)

        def set_offset(val):
            self.offset = self.slider_offset.val

        self.slider_offset.on_changed(set_offset)

        # amplitude
        # h -= 0.05
        # ax = self.fig.add_axes([0.2, h, 0.6, 0.03])
        ax = self.fig.add_axes([0.55, h, 0.35, 0.03])
        self.slider_amp = Slider(ax, 'a', 0, np.pi, valinit=self.amp, dragging=False)

        def set_amp(val):
            self.amp = self.slider_amp.val

        self.slider_amp.on_changed(set_amp)

        # phi
        h -= 0.05
        # ax = self.fig.add_axes([0.2, h, 0.6, 0.03])
        ax = self.fig.add_axes([0.1, h, 0.35, 0.03])
        self.slider_phi = Slider(ax, 'phi', 0, 2*np.pi, valinit=self.phi, dragging=False)

        def set_phi(val):
            self.phi = self.slider_phi.val

        self.slider_phi.on_changed(set_phi)

        # Control parameters #
        # desired offset
        h -= 0.05
        # ax = self.fig.add_axes([0.2, h, 0.6, 0.03])
        ax = self.fig.add_axes([0.1, h, 0.35, 0.03])
        self.slider_des_offset = Slider(ax, 'O', -np.pi, np.pi, valinit=self.des_offset)

        def set_des_offset(val):
            self.des_offset = self.slider_des_offset.val

        self.slider_des_offset.on_changed(set_des_offset)

        # desired amplitude
        # h -= 0.05
        # ax = self.fig.add_axes([0.2, h, 0.6, 0.03])
        ax = self.fig.add_axes([0.55, h, 0.35, 0.03])
        self.slider_des_amp = Slider(ax, 'A', 0, np.pi, valinit=self.des_amp)

        def set_des_amp(val):
            self.des_amp = self.slider_des_amp.val

        self.slider_des_amp.on_changed(set_des_amp)

        # desired frequency
        h -= 0.05
        # ax = self.fig.add_axes([0.2, h, 0.6, 0.03])
        ax = self.fig.add_axes([0.1, h, 0.35, 0.03])
        self.slider_des_freq = Slider(ax, 'f', 0, 50, valinit=self.des_freq, valfmt='%0.0f')

        def set_des_freq(val):
            self.des_freq = self.slider_des_freq.val
            self.des_omega = 2 * np.pi * self.des_freq

        self.slider_des_freq.on_changed(set_des_freq)

        # Coupling parameters #
        class Coupling(object):
            """
            Class to be used with sliders for the coupling parameters (weights and biases)
            """
            def __init__(self, node, nodes, sliderNodes):
                self.node = node
                self.nodes = nodes
                self.sliderNodes = sliderNodes

            def set_weight(self, val):
                self.nodes[self.node]['weight'] = self.sliderNodes[self.node]['weight'].val

            def set_bias(self, val):
                self.nodes[self.node]['bias'] = self.sliderNodes[self.node]['bias'].val

        for node, coupling in self.nodes.items():
            h -= 0.05
            # slider for coupling weight
            ax = self.fig.add_axes([0.1, h, 0.35, 0.03])
            slider = Slider(ax, 'w'+str(self.id)+str(node.id), -5, 5, valinit=coupling['weight'], valfmt='%0.1f')
            c = Coupling(node, self.nodes, self.sliderNodes)
            slider.on_changed(c.set_weight)
            self.sliderNodes[node] = {'weight': slider}
            # slider for phase bias
            ax = self.fig.add_axes([0.55, h, 0.35, 0.03])
            slider = Slider(ax, 'b' + str(self.id) + str(node.id), -np.pi, np.pi, valinit=coupling['bias'], valfmt='%0.2f')
            slider.on_changed(c.set_bias)
            self.sliderNodes[node]['bias'] = slider

        plt.show(block=False)
        # plt.draw()
        # self.fig.show()
        # self.fig.canvas.draw() # Problem: not reactive to keyboard/mouse when moving sliders...

    #############
    # Operators #
    #############

    def __str__(self):
        """Return a string describing the class."""
        return self.__class__.__name__ + "(" + str(self.id) + ")"


class CPGNetwork(object):
    r"""Central Pattern Generator Network

    """

    def __init__(self, nodes, timesteps=100):
        nodes = nodes if isinstance(nodes, dict) else self.fully_connected_network(nodes)

        # create each node
        self.nodes, init_params = {}, {'phi', 'offset', 'amplitude', 'freq'}
        for node_id in nodes.keys():
            d = {key: val for key, val in nodes[node_id].items() if key in init_params}
            self.nodes[node_id] = CPGNode(node_id, timesteps=timesteps, **d)

        # couple the nodes
        for node_id, node in self.nodes.items():
            if 'nodes' in nodes[node_id]:  # check if 'nodes' in dict
                for coupling_node in nodes[node_id]['nodes']:  # coupling_node = {'id':..., 'weight':..., 'bias': ...}
                    w = coupling_node['weight'] if 'weight' in coupling_node else 0.
                    b = coupling_node['bias'] if 'bias' in coupling_node else 0.
                    node.add_node(self.nodes[coupling_node['id']], weight=w, bias=b)

        self.fig = None
        self.nodes_id = self.nodes.keys()
        self.nodes_id.sort()

    @property
    def num_parameters(self):
        """Return the total number of parameters in this CPG network."""
        return sum([node.num_parameters for node in self.nodes.values()])

    def parameters(self):
        """Returns an iterator over the model parameters."""
        for node in self.nodes.values():
            yield np.array(list(node.parameters()))

    def named_parameters(self):
        """Returns an iterator over the model parameters, yielding both the name and the parameter itself"""
        for node in self.nodes.values():
            yield str(node), np.array(list(node.parameters()))

    def list_parameters(self):
        """Return a list of parameters"""
        return list(self.parameters())

    def get_vectorized_parameters(self, to_numpy=True):
        """Return a vectorized form of the parameters (amplitudes, offsets, frequencies, weights, biases)."""
        return np.concatenate([node.get_vectorized_parameters(to_numpy=to_numpy) for node in self.nodes.values()])

    def set_vectorized_parameters(self, vector):
        """Set the vector parameters."""
        # set the parameters from the vectorized one
        if len(vector) != self.num_parameters:
            raise ValueError("Expecting the size of the vectorized parameters to match the number of parameters "
                             "of this node. Instead of having {}, I got {}.".format(self.num_parameters, len(vector)))

        # convert from torch tensor to numpy array if necessary
        if isinstance(vector, torch.Tensor):
            if vector.requires_grad:
                vector = vector.detach().numpy()
            else:
                vector = vector.numpy()

        # set the parameters from the vectorized one
        idx = 0
        for node in self.nodes.values():
            size = node.num_parameters
            node.set_vectorized_parameters(vector[idx:idx+size])
            idx += size

    def add_node(self, node):
        """Add a node to the CPG network."""
        # TODO: update self.nodes_id
        self.nodes[node.id] = node

    def reset(self):
        """Reset the phase of each CPG node in the network; this can be useful for phase resetting."""
        for node in self.nodes.values():
            node.reset()

    def step(self):
        """Perform a step with the CPG network."""
        # perform one step
        for node in self.nodes.values():
            node.step()
        # perform update
        for node in self.nodes.values():
            node.update()
        return np.array([self.nodes[idx].theta for idx in self.nodes_id])

    def plot(self):
        """Plot each node."""
        for node in self.nodes.values():
            node.plot()

    @staticmethod
    def fully_connected_network(num_nodes, init_phi=0, offset=0, amplitude=1., weight=0, bias=0, freq=1.):
        """
        Create an initial dictionary describing a fully connected network of CPG nodes.

        Args:
            num_nodes (int): the total number of nodes in the network
            init_phi (float): initial phase of CPG node
            offset (float): desired and initial offset of CPG node
            amplitude (float): desired and initial amplitude of CPG node
            weight (float): weight between 2 nodes
            bias (float): phase bias between 2 nodes

        Returns:
            dict: dictionary describing how the nodes are connected
        """
        node_ids = range(1, num_nodes+1)
        d = {i: {'phi': init_phi,
                 'offset': offset,
                 'amplitude': amplitude,
                 'freq': freq,
                 'nodes': [{'id': n, 'weight': weight, 'bias': bias} for n in (node_ids[:i-1]+node_ids[i:])]}
             for i in node_ids}
        return d


# Tests
if __name__ == "__main__":
    # Define and parse command line arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--test', help='What to test', type=str,
                        choices=['1_node', '2_nodes', 'coman_gazebo', 'minitaur_pybullet'],
                        default='minitaur_pybullet')
    parser.add_argument('-T', '--nb_steps', help='Total time number of steps to run the simulation',
                        type=int, default=100)
    parser.add_argument('-dt', '--dt', help='Time to wait before next step', type=float, default=0.01)
    args = parser.parse_args()

    dt = args.dt
    T = args.nb_steps

    # Check a single node #
    if args.test == '1_node':
        # create CPG node
        node = CPGNode(1)
        node.plot()

        # Run for T steps
        for _ in range(T):
            node.step()
            node.update()
            # time.sleep(0.1) # Don't use time, instead use plt.pause!
            plt.pause(dt)
        plt.show()

    # Check 2 nodes #
    elif args.test == '2_nodes':
        # create CPG network
        nodes = {1: {'phi': -np.pi / 2, 'nodes': [{'id': 2, 'weight': 0, 'bias': 0}]},
                 2: {'phi': np.pi / 2, 'nodes': [{'id': 1, 'weight': 0, 'bias': 0}]}}
        network = CPGNetwork(nodes)
        network.plot()

        # Run for T steps
        for _ in range(T):
            network.step()
            plt.pause(dt)
        plt.show()

    # Check coman robot in gazebo #
    # Warning: Don't forget to launch gazebo with coman before running this code!
    # $ roslaunch coman_gazebo coman_world.launch
    elif args.test == 'coman_gazebo':
        from pyrobolearn.robots.ros.coman.comanpublisher import ComanPublisher
        import rospy

        # create CPG network of 2 nodes
        pub = ComanPublisher(joints=['RHipSag', 'LHipSag'])
        nodes = {1: {'phi': -np.pi/2, 'nodes': [{'id': 2, 'weight': 1, 'bias': 0}]},
                 2: {'phi': np.pi/2, 'nodes': [{'id': 1, 'weight': 0, 'bias': 0}]}}
        network = CPGNetwork(nodes)
        # network.plot()

        # Run for T steps
        # rate = rospy.Rate(30)
        # while not rospy.is_shutdown():
        for _ in range(T):
            network.step()
            pub.send({'RHipSag': network.nodes[1].theta,
                     'LHipSag': network.nodes[2].theta})
            plt.pause(dt)
            # rate.sleep()

    # Check CPG with Minitaur #
    elif args.test == 'minitaur_pybullet':
        from pybullet_envs.bullet.bullet_client import BulletClient
        from pybullet_envs.bullet.minitaur import Minitaur
        import pybullet
        import pybullet_data
        import time

        # create and configure pybullet simulator
        client = BulletClient(connection_mode=pybullet.GUI)
        client.setAdditionalSearchPath(pybullet_data.getDataPath())
        floor = client.loadURDF('plane.urdf')
        client.setGravity(0, 0, -9.81)
        minitaur = Minitaur(client, urdf_root=pybullet_data.getDataPath())

        # create CPG network
        phis = [0, 0, 0, 0]
        nodes = {1: {'phi': phis[0], 'offset': -np.pi/2, 'amplitude': np.pi/4, 'nodes': []},  # front left leg
                 2: {'phi': phis[1], 'offset': -np.pi/2, 'amplitude': np.pi/4, 'nodes': []},  # back left leg
                 3: {'phi': phis[2], 'offset': np.pi/2, 'amplitude': np.pi/4, 'nodes': []},  # front right leg
                 4: {'phi': phis[3], 'offset': np.pi/2, 'amplitude': np.pi/4, 'nodes': []},  # back right leg
                 }
        network = CPGNetwork(nodes)

        # Run simulation
        for _ in range(10*T):
            # Get angles from CPG network and set them to the minitaur
            act = network.step()
            for i in range(len(act)//2): # Left
                minitaur._SetDesiredMotorAngleById(minitaur._motor_id_list[2*i], act[i])
                minitaur._SetDesiredMotorAngleById(minitaur._motor_id_list[2*i+1], -np.pi-act[i])
            for i in range(len(act)//2, len(act)): # Right
                minitaur._SetDesiredMotorAngleById(minitaur._motor_id_list[2*i], act[i])
                minitaur._SetDesiredMotorAngleById(minitaur._motor_id_list[2*i+1], np.pi - act[i])

            # run one-step forward the simulation
            client.stepSimulation()
            time.sleep(dt)
            # print(client.getEulerFromQuaternion(minitaur.GetBaseOrientation()))
