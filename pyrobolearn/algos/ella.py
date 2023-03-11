#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Provide the 'Efficient Lifelong Learning Algorithm' (ELLA).

The ELLA algorithm is an online multi-task learning algorithm that maintains a shared knowledge database that can be
trained and used to incorporate new knowledge to improve the performance on multiple tasks [1,2,3].

The code presented here is based on [3,4].

References:
    [1] "Learning Task Grouping and Overlap in Multi-Task Learning" (GO-MTL), Kumar et al., 2012
    [2] "ELLA: An Efficient Lifelong Learning Algorithm", Ruvolo et al., 2013
    [3] "Online Multi-Task Learning for Policy Gradient Methods", Ammar et al., 2014
    [4] Implementation of ELLA on Github (by Paul Ruvolo): https://github.com/paulruvolo/ELLA
    [5] Implementation of PG-ELLA on Github (by ): https://github.com/cdcsai/Online_Multi_Task_Learning
"""

import torch
import numpy as np

from pyrobolearn.utils.torch_utils import kronecker


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse", "Paul Ruvolo", "Charles Dognin"]
__license__ = "Apache License 2.0"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class ELLA(object):
    r"""Efficient Lifelong Learning Algorithm (ELLA)

    Type:: online multi-task, parametric

    ELLA is an online multi-task learning algorithm that maintains and refines (given new tasks) a shared basis for all
    task models, allowing to transfer knowledge from previous tasks and improve the performance on all tasks (including
    previous ones).

    In the multi-task learning paradigm, we are given a series of learning tasks: math:`Z^{(1)}, \cdots, Z^{(T_{max})}`,
    where each task:
    - in the supervised learning (SL) case is expressed as :math:`Z^{(t)} = (\hat{f}^{(t)}, X^{(t)}, y^{(t)}`, with
    :math:`\hat{f}` being the hidden true function that maps the input set to the output set, and the goal is to find
    the parametric function :math:`y = f(x; \theta^{(t)})` where :math:`\theta^{(t)}` are the parameters.
    - in the reinforcement learning (RL) case is given by
    :math:`Z^{(t)} = \langle S_0^{(t)}, S^{(t)}, A^{(t)}, P^{(t)}, R^{(t)}, \gamma^{(t)} \rangle`, where the goal is
    to find the policy :math:`\pi_{\theta^{(t)}}(a_t | s_t)` that maps states to actions, given the parameters
    :math:`\theta^{(t)}`.

    ELLA [2,3] achieves this by following the approach proposed in [1]; by maintaining and sharing a library of latent
    model components :math:`L \in \mathbb{R}^{d \times k}` between the various tasks such that the parameters for any
    given task `t` are given by :math:`\theta^{(t)} = L s^{(t)}`, with :math:`s^{(t)} \in \mathbb{R}^{k}` being the
    sparse latent weight vector; that is, :math:`\theta^{(t)}` is given by the superposition of the latent basis vector
    in :math:`L` and where the coefficients for each basis vector is given by the elements in :math:`s^{(t)}`.

    The ELLA minimizes the following objective function:

    .. math:: e_T(L) = \frac{1}{T} \sum_{t=1}^T \min_{s^(t)} \left[ \mathcal{L}(\theta^{(t)}) + \mu ||s^{(t)}||_1
        \right] + \lambda ||L||^2_F

    where :math:`T` is the number of tasks encountered so far, and :math:`\mathcal{L}(\theta^{(t)})` is the loss
    function which is given by:
    - in the SL case: :math:`\mathcal{L}(\theta^{(t)}) = \frac{1}{n_t} \sum_{i=1}^{n_t} \mathcal{L}(f(x_i^{(t)};
    \theta^{(t)}), y_i^{(t)})`, where :math:`x_i^{(t)}` and :math:`y_i^{(t)}` are the input and output data for the
    task :math:`t`, and :math:`\theta^{(t)} = L s^{(t)}` are the parameters.
    - in the RL case: :math:`\mathcal{L}(\theta^{(t)}) = - \mathcal{J}(\theta^{(t)})`, where
    :math:`\mathcal{J}(\theta^{(t)}) = \int_{\mathbb{T}^{(t)}} p_{\theta^{(t)}}(\tau) R(\tau) d\tau` is the expected
    average return on task :math:`t`.

    Now, there are two problems with the formulation above:
    1. the explicit dependency to all the training data or trajectories including the ones from the previous tasks
    through the inner summation (i.e. :math:`\sum_{i=1}^{n_t}` in the SL case, and :math:`\int_{\mathbb{T}^{(t)}}` in
    the RL case). Ideally, we would like to only depend on the training data and trajectories for the current task.
    This is performed by approximating the objective function using the second-order Taylor expansion of
    :math:`\mathcal{L}(\theta^{(t)})` around the best parameters
    :math:`\theta^{(t)*} = \arg \min_{\theta} \mathcal{L}(\theta)`.
    2. the evaluation of the :math:`s^{(t)}`'s for each task by solving the minimization problem which can become
    increasingly expensive as the number of tasks increased. Ideally, we would like to only perform the optimization
    for the current task and exploit the fact that while we only update the current :math:`s^{(t)}`, the
    library :math:`L` is updated for each task, and thus the tasks can still benefit from this last one. In [2,3],
    it has been observed that the choice of only updating the current :math:`s^{(t)}` does not affect the quality of
    the model as the number of tasks grows large.

    ...

    Warnings:
        - The ELLA requires the computation of the parameters that minimize the loss function, and most importantly
         the Hessian matrix of the loss function with respect to the parameters evaluated at the best found above
         parameters. Depending on the number of parameters that matrix can be big and expensive to compute.
        - The ELLA assumes that the input and output dimension of the model as well as its number of parameters is
         constant between the various tasks. For different state and action spaces between the tasks, see inter-task
         mappings [6], or ELLA using task groups where tasks in the same group share a common state and action space,
         such that :math:`\theta^{(t)} = B^{(g)}s^{(t)}` with :math:`B^{(g)} = \Phi^{(g)} L` where :math:`g` denotes
         the task group, :math:`B^{(g)}` is the latent model components shared withing :math:`g`, :math:`L` is the
         global latent model components, and :math:`\Phi^{(g)}` is the mapping from :math:`L` to :math:`B^{(g)}` [7].
        - The complexity of each ELLA update is :math:`O(k^2 d^3, \xi(d, n_t))` where :math:`k` is the number of latent
         components, :math:`d` is the dimensionality of the parameters, :math:`n_t` is the number of data instances (in
         SL) or trajectories (in RL), and :math:`\xi` is the function that computes the complexity to compute the
         best set of parameters and the Hessian matrix for the current task :math:`t`.


    The current implementation is based on [4,5].


    Pseudo-algorithm
    ----------------

    1. Inputs: k=number of latent components, d=dimensionality of parameters, lambda=L2 norm coefficient,
        mu=L1 norm coefficient
    2. Init: T=0, A=zeros(k*d,k*d), b=zeros(k*d,1), L=zeros(d,k)
    3. ...

    References:
        [1] "Learning Task Grouping and Overlap in Multi-Task Learning" (GO-MTL), Kumar et al., 2012
        [2] "ELLA: An Efficient Lifelong Learning Algorithm", Ruvolo et al., 2013
        [3] "Online Multi-Task Learning for Policy Gradient Methods", Ammar et al., 2014
        [4] Implementation of ELLA on Github (by Paul Ruvolo): https://github.com/paulruvolo/ELLA
        [5] Implementation of PG-ELLA on Github (by ): https://github.com/cdcsai/Online_Multi_Task_Learning
        [6] "Transfer learning via inter-task mappings for temporal difference learning", Taylor et al., 2007
        [7] "Autonomous cross-domain knowledge transfer in lifelong policy gradient reinforcement learning", Ammar et
            al., 2015
    """

    def __init__(self, num_parameters, num_latent_component, l1_sparsity_coefficient=1., l2_library_coefficient=1.):
        r"""
        Initialize ELLA.

        Args:
            num_parameters (int): number of model parameters (in the papers [2,3], this is the `d` variable).
            num_latent_component (int): number of latent component (in the papers [2,3], this is the `k` variable).
            l1_sparsity_coefficient (float): coefficient for the L1 norm applied on the sparse weight vector
                :math:`s^{(t)}` (in the papers [2,3], this is the `\mu` variable).
            l2_library_coefficient (float): coefficient for the L2 norm applied on the shared library basis :math:`L`
                (in the papers [2,3], this is the `\lambda` variable).
        """
        d, k = num_parameters, num_latent_component
        self.d = num_parameters
        self.k = num_latent_component
        self.l1_coeff = l1_sparsity_coefficient
        self.l2_coeff = l2_library_coefficient
        self.A = torch.zeros((d*k, d*k))  # A matrix used to compute the shared library L=A^{-1}b
        self.b = torch.zeros((d*k, 1))  # b vector used to compute the shared library L=A^{-1}b
        self.s = torch.zeros(k)  # sparse weight vector
        self.L = torch.randn((self.d, self.k))  # shared library of basis vectors
        self.num_task = 0  # counter for the number of tasks encountered
        self.tasks = {}  # dict of encountered tasks

    def train(self, task, save_task_parameters=True):
        """
        Train using ELLA [2,3].

        Args:
            task (ILTask, RLTask): task. All the tasks must contain the same policy.

        Returns:

        """
        # if new task
        if task not in self.tasks:
            self.num_task += 1
            # update dataset or collect trajectory randomly
            # TODO
        else:
            # get previous theta, hessian, and s vector
            values = self.tasks[task]
            theta, hessian, s = values['theta'], values['hessian'], values['s']

            # update A and b with respect to these previous parameters
            self.A -= kronecker(s.matmul(s.t()), hessian)
            self.b -= kronecker(s.t(), theta.t().matmul(hessian)).view(-1, 1)

            # update dataset or collect trajectories using theta
            # TODO

        # compute the best parameters and hessian matrix evaluated at these best parameters
        # task.train()
        theta = task.best_parameters
        hessian = None  # TODO

        # reinitialize the columns of the library L
        self.reinitialize_zero_columns()

        # optimize the loss with respect to s to obtain the best sparse vector s
        s = self.s
        diff = (theta - self.L.matmul(s))
        loss = self.l1_coeff * torch.sum(torch.abs(s)) + diff.t().matmul(hessian.matmul(diff))
        # TODO

        # compute A
        self.A += kronecker(s.matmul(s.t()), hessian)
        # compute b
        self.b += kronecker(s.t(), theta.t().matmul(hessian)).view(-1, 1)
        # compute L=A^{-1}b
        self.L = torch.inverse(1./self.num_task * self.A + self.l2_coeff).matmul(1./self.num_task * self.b)

        # save the best parameters and hessian for the task
        # TODO: saving the hessian can be quite expensive, maybe we should just save the important components (using
        #  SVD)
        self.tasks.setdefault(task, {}).update({'theta': theta, 'hessian': hessian, 's': self.s})

    def reinitialize_zero_columns(self):
        """
        Reinitialize the library columns that are zeros.
        """
        for i, val in enumerate(np.sum(self.L, axis=0)):
            if abs(val) < 10 ** -8:
                self.L[:, i] = torch.randn(self.d,)

    def plot_latent(self):
        pass
