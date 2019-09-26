# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""Provide the Explorer class used in the first step of RL algorithms

It consists to explore and collect samples in the environment using the policy. The samples are stored in the
given memory/storage unit which will be used to evaluate the policy, and then update its parameters.
"""

import os
import copy
import inspect

import torch
import torch.multiprocessing as multiprocessing
import torch.distributed as distributed  # note that you can only send/receive tensors with P2P backends

from pyrobolearn.tasks import RLTask
from pyrobolearn.envs import Env
from pyrobolearn.policies import Policy
from pyrobolearn.exploration import Exploration
from pyrobolearn.storages import RolloutStorage, ExperienceReplay

from pyrobolearn import logger
from pyrobolearn.metrics import Metric


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class Explorer(object):
    r"""Explorer

    (Model-free) reinforcement learning algorithms requires 3 steps:
    1. Explore: Explore and collect samples in the environment using the policy. The samples are stored in the
                given memory/storage unit.
    2. Evaluate: Assess the quality of the actions/trajectories using the returns.
    3. Update: Update the policy (and/or value function) parameters based on the loss

    This class focuses on the first step of RL algorithms. It accepts the task (environment and policy), the
    exploration strategy which wraps the policy, and the rollout (for on-policy) or experience replay (for off-policy)
    storage unit.
    """

    def __init__(self, task, explorer, storage, num_workers=1, backend='multiprocessing', metrics=None):
        """
        Initialize the exploration phase.

        Args:
            task (Task, Env, tuple of Env and Policy): RL task or environment.
            explorer (Exploration): policies.
            storage (RolloutStorage, ExperienceReplay): Rollout (for on-policy) or experience replay (for off-policy)
                storage unit (=replay memory). It will save the trajectories / rollouts / transitions in the storage
                while exploring.
            num_workers (int): number of processes / workers to run in parallel. This number has to be equal or smaller
                than the number of CPUs on the computer. If bigger, it will automatically be clipped when using the
                'multiprocessing' backend (see below). If only :attr:`num_workers=1`, it doesn't
            backend: backend to be used when using multiple processes. The different possible backends are
                'multiprocessing' (by default), 'gloo' (good for CPUs), 'nccl' (good for GPUs), 'mpi' (only valid if
                PyTorch has been built from source with MPI support). For more information, we refer the reader to
                references [2,4]. If the backend is 'mpi', you have to run the code using the following command:
                `mpirun -n 4 python <code>.py`.
            metrics ((list of) Metric, None): metrics that are used to evaluate the algorithm.

        References:
            [1] Multiprocessing best practices: https://pytorch.org/docs/stable/notes/multiprocessing.html
            [2] torch multiprocessing: https://pytorch.org/docs/stable/multiprocessing.html
            [3] Writing Distributed Applications with PyTorch: https://pytorch.org/tutorials/intermediate/dist_tuto.html
            [4] torch distributed: https://pytorch.org/docs/stable/distributed.html
        """
        self.task = task
        self.explorer = explorer
        self.storage = storage
        self.metrics = metrics

        # check the number of workers
        if not isinstance(num_workers, (int, long)):
            raise TypeError("Expecting the number of workers to be an integer, instead got: {}".format(num_workers))
        self.num_workers = 1 if num_workers < 1 else int(num_workers)

        # make sure that the maximum number of worker/process is smaller or equal to the number of CPUs
        if self.num_workers > multiprocessing.cpu_count():
            self.num_workers = multiprocessing.cpu_count()

        # check backend
        if backend is None:
            backend = 'multiprocessing'
        if not isinstance(backend, str):
            raise TypeError("Expecting the given 'backend' to be a string, instead got: {}".format(type(backend)))
        backend = backend.lower()
        if backend not in {'multiprocessing', 'mpi', 'gloo', 'nccl'}:
            raise ValueError("Expecting the given 'backend' to be 'multiprocessing', 'mpi', 'gloo', or 'nccl', instead "
                             "got: {}".format(backend))
        self.backend = backend

        # create processes
        self.processes = []
        self.pipe = None
        self.process_id = 0  # only the master should have the id set to 0, the workers have a strictly positive id
        if self.num_workers > 1:
            if self.backend == 'multiprocessing':
                is_rendering = self.environment.is_rendering
                rendering_mode = self.environment.rendering_mode

                # hide the GUI of the environment
                if is_rendering:
                    self.environment.hide()

                # create processes
                self.queue = multiprocessing.Queue()
                self.pipes = [multiprocessing.Pipe() for _ in range(self.num_workers)]
                self.processes = [multiprocessing.Process(target=self.explore_in_parallel, args=(pipe[1], self.queue,
                                                                                                 task))
                                  for pipe in self.pipes]

                # start processes
                for process in self.processes:
                    process.start()

                # render the GUI if it was initially rendered
                if is_rendering:
                    self.environment.render(mode=rendering_mode)

            else:
                raise NotImplementedError("Currently, other backends are not provided.")

            # elif self.backend == 'gloo' or self.backend == 'nccl':
            #
            #     if self.backend == 'nccl' and not distributed.is_nccl_available():
            #         raise ValueError("The 'nccl' backend is not available on this computer.")
            #
            #     # create processes
            #     self.processes = [multiprocessing.Process(target=init_processes, args=(rank, size, function))]
            #
            #     # start processes
            #     for process in self.processes:
            #         process.start()
            #
            # elif self.backend == 'mpi':
            #     if not distributed.is_mpi_available():
            #         raise ValueError("The 'mpi' backend is not available on this computer.")
            #
            #     init_processes(0, 0, function, self.backend)

    ##############
    # Properties #
    ##############

    @property
    def task(self):
        """Return the RL task."""
        return self._task

    @task.setter
    def task(self, task):
        """Set the RL task."""
        if isinstance(task, (tuple, list)):
            env, policy = None, None
            for t in task:
                if isinstance(t, Env):
                    env = t
                if isinstance(t, Policy):  # TODO if multiple policies
                    policy = t
            if env is None or policy is None:
                raise ValueError("Expecting the task to be an instance of `RLTask` or a list/tuple of an environment "
                                 "and policy.")
            task = RLTask(env, policy)
        if not isinstance(task, RLTask):
            raise TypeError("Expecting the task to be an instance of `RLTask`, instead got: {}".format(type(task)))
        self._task = task

    @property
    def policy(self):
        """Return the policy."""
        return self.explorer.policy

    @property
    def env(self):
        """Return the environment."""
        return self.task.environment

    # alias
    @property
    def environment(self):
        """Return the environment"""
        return self.task.environment

    @property
    def explorer(self):
        """Return the exploration strategy."""
        return self._explorer

    @explorer.setter
    def explorer(self, explorer):
        """Set the exploration strategy."""
        if inspect.isclass(explorer):  # if it is a class
            explorer = explorer(self.policy)
        if not isinstance(explorer, Exploration):
            raise TypeError("Expecting explorer to be an instance of Exploration")
        self._explorer = explorer

    @property
    def storage(self):
        """Return the storage unit."""
        return self._storage

    @storage.setter
    def storage(self, storage):
        """Set the storage unit."""
        if not isinstance(storage, (RolloutStorage, ExperienceReplay)):  # DictStorage):
            raise TypeError("Expecting the storage to be an instance of `RolloutStorage` or `ExperienceReplay`, "
                            "instead got: {}".format(type(storage)))
        self._storage = storage

    @property
    def metrics(self):
        """Return the metric instances."""
        return self._metrics

    @metrics.setter
    def metrics(self, metrics):
        """Set the metrics."""
        # check metrics type
        if metrics is None:
            metrics = []
        elif isinstance(metrics, Metric):
            metrics = [metrics]
        elif not isinstance(metrics, list):
            raise TypeError("Expecting the given 'metrics' to be an instance of `Metric` or a list of `Metric`, but "
                            "got instead: {}".format(type(metrics)))

        # check each metric type
        for i, metric in enumerate(metrics):
            if not isinstance(metric, Metric):
                raise TypeError("The {}th metric is not an instance of `Metric`, but: {}".format(i, type(metric)))

        # set metrics
        self._metrics = metrics

    ###########
    # Methods #
    ###########

    def close(self):
        """End the processes."""
        for process in self.processes:
            process.terminate()

    def rollout(self, num_steps, deterministic=False, render=False, verbose=False):
        """Perform a rollout.

        Args:
            num_steps (int): number of steps.
            deterministic (bool): if the policy should act deterministically instead of exploring (based on the
                exploration strategy).
            render (bool): if we should render the environment.
            verbose (bool): if we should print information about the rollout.
        """
        # reset environment
        observation = self.env.reset()

        # reset explorer
        self.explorer.reset()

        if verbose:
            print("Start the rollout")

        # run RL task for T steps
        trajectory = []
        for step in range(num_steps):
            # if we need to render
            if render:
                self.env.render()

            # get action and corresponding distribution from policy
            action, distribution = self.explorer.act(observation, deterministic=deterministic)

            # perform one step in the environment
            next_observation, reward, done, info = self.env.step(action)

            # append the transition tuple
            trajectory.append({'states': observation, 'actions': action, 'next_states': next_observation,
                               'reward': reward, 'mask': (1 - done), 'distribution': distribution})

            # if verbose:
            #     print("\n\t1. Observation data: {}".format(observation))
            #     print("\t2. Action data: {}".format(action))
            #     print("\t3. Next observation data: {}".format(next_observation))
            #     print("\t4. Reward: {}".format(reward))
            #     print("\t5. \\pi(.|s): {}".format(distribution))
            #     print("\t6. log \\pi(a|s): {}".format([d.log_prob(action) for d in distribution]))

            # if done, get out of the loop
            if done:
                break

            # set current observation to current one
            observation = next_observation

        if verbose:
            print("End of the rollout")

        # return trajectory
        return trajectory

    def __send(self, msg, dst=0, rank=0):
        """
        Send the given message to the specified destination.

        - master: send to the specified worker process the information about the rollout (number of steps, parameters,
            etc).
        - worker: send its rank / process id and the trajectory to the master

        Args:
            msg (tuple of object): message to send.
            dst (int): destination rank or process id to send the message to. The master process has a rank of 0.
            rank (int): rank or process id. The master process has a rank/id of 0, while the workers have a strictly
                positive integer id.
        """
        if self.backend == 'multiprocessing':

            # master process: send to the specified worker process the information about the rollout (number of steps,
            # parameters, etc).
            if rank == 0:
                self.pipes[dst][0].send(msg)

            # worker process: add the message (process id and trajectory) to the queue
            else:
                self.queue.put(msg)
        else:
            # TODO: the messages have to be tensors!!
            # master process: send to the specified worker process the information about the rollout (number of steps,
            # parameters, etc).
            if rank == 0:
                num_steps, render, verbose, parameters = msg

                distributed.send(tensor=torch.tensor(num_steps).view(1), dst=dst)
                distributed.send(tensor=torch.tensor(render).view(1), dst=dst)
                distributed.send(tensor=torch.tensor(verbose).view(1), dst=dst)

                for parameter in parameters:
                    distributed.send(tensor=parameter, dst=dst)

            # worker process: send the message (process id and trajectory) to the queue
            else:
                process_id, trajectory = msg
                distributed.send(tensor=torch.tensor(process_id).view(1), dst=0)

                for transition in trajectory:
                    distributed.send(tensor=transition['states'], dst=0)
                    distributed.send(tensor=transition['actions'], dst=0)
                    distributed.send(tensor=transition['next_states'], dst=0)
                    distributed.send(tensor=transition['reward'], dst=0)
                    distributed.send(tensor=transition['mask'], dst=0)
                    # TODO: send distribution (its hyperparameters? because I can only send tensors... Or its entropy
                    #  and the log likelihood of the action evaluated with the distribution?)
                    # distributed.send(tensor=transition['distribution'], dst=0)

    def __recv(self, rank=0):
        """
        Receive the message from the master or worker process.

        - master: receive the message (i.e. the process id (or rank) and the trajectory) from the worker
        - worker: receive information about the rollout (num_steps, parameters, etc) from the master

        Args:
            rank (int): rank or process id. The master process has a rank/id of 0, while the workers have a strictly
                positive integer id.

        Returns:
            int: process id
            list of torch.Tensor: the trajectory.
        """
        if self.backend == 'multiprocessing':

            # master process: receive the process id and trajectory from the workers
            if rank == 0:
                process_id, trajectory = self.queue.get()
                return process_id, trajectory

            # worker process: receive information about the rollout (num_steps, parameters, etc) from the master
            msg = self.pipe.recv()
            return msg

        else:
            # TODO: can only receive tensors and they have to be allocated in advance with the correct dimensions!!
            # master process: receive the process id and trajectory from the workers
            if rank == 0:
                # preallocate the tensors
                scalar_tensor = torch.zeros(1)
                state_shapes = self.explorer.policy.states.merged_shape
                action_shapes = self.explorer.policy.actions.merged_shape

                # get process id / rank
                distributed.recv(tensor=scalar_tensor, src=None)
                src = int(scalar_tensor)

                # get trajectory length
                distributed.recv(tensor=scalar_tensor, src=src)
                trajectory_length = int(scalar_tensor)

                # copy trajectory
                trajectory = []
                state_tensors = [torch.zeros(shape) for shape in state_shapes]
                for i in range(trajectory_length):
                    # copy transition tuple
                    transition = {}

                    # allocate tensors in advance
                    action_tensors = [torch.zeros(shape) for shape in action_shapes]
                    next_state_tensors = [torch.zeros(shape) for shape in state_shapes]
                    reward_tensor = torch.zeros(1)
                    mask_tensor = torch.zeros(1)

                    # copy states/observations
                    for state_tensor in state_tensors:
                        distributed.recv(tensor=state_tensor, src=src)
                    transition['states'] = state_tensors

                    # copy actions
                    for action_tensor in action_tensors:
                        distributed.recv(tensor=action_tensor, src=src)
                    transition['actions'] = action_tensors

                    # copy next states/observations
                    for state_tensor in next_state_tensors:
                        distributed.recv(tensor=state_tensor, src=src)
                    transition['next_states'] = state_tensors

                    # copy reward
                    distributed.recv(tensor=reward_tensor, src=src)
                    transition['reward'] = reward_tensor

                    # copy mask/done
                    distributed.recv(tensor=mask_tensor, src=src)
                    transition['mask'] = mask_tensor

                    # TODO: copy distribution
                    # distributed.recv(tensor=distribution_tensor, src=src)
                    # transition['distribution'] = distribution_tensor

                    # add transition tuple in the trajectory
                    trajectory.append(transition)

                    # set the state tensors to the next one (to be memory and time efficient)
                    state_tensors = next_state_tensors

                return src, trajectory

            # worker process: receive information about the rollout (num_steps, parameters, etc) from the master
            # get number of steps
            scalar_tensor = torch.zeros(1)
            distributed.recv(tensor=scalar_tensor, src=0)
            num_steps = int(scalar_tensor)

            # get if we should render or not, and verbose
            distributed.recv(tensor=scalar_tensor, src=0)
            render = bool(scalar_tensor)
            distributed.recv(tensor=scalar_tensor, src=0)
            verbose = bool(scalar_tensor)

            # copy parameters
            parameters = []
            for parameter in self.explorer.policy.parameters():
                parameter = torch.zeros(parameter.shape)
                distributed.recv(tensor=parameter, src=0)
                parameters.append(parameter)

            return num_steps, render, verbose, parameters

    def _master_explore(self, num_steps, num_rollouts=1, render=False, verbose=False):
        """Master: send the add the trajectories or transition tuples in the storage unit."""
        # create set of integers
        pool = set(range(min(len(self.processes), num_rollouts)))
        process, rollout = 0, 0
        while True:
            if pool and process < num_rollouts:
                # send job to process
                process_id = pool.pop()
                # self.pipes[process_id][0].send((num_steps, self.explorer))
                self.__send((num_steps, render, verbose, self.explorer.policy.parameters()), dst=process_id, rank=0)
                process += 1

            elif process >= num_rollouts and rollout >= num_rollouts:
                # get out of the loop
                break

            else:
                # wait for result from queue
                process_id, trajectory = self.__recv(rank=0)  # self.queue.get()

                # put the process id in the pool to let know that it is free
                pool.add(process_id)

                # add the trajectory in the storage
                self.storage.add_trajectory(trajectory, rollout_idx=rollout)
                rollout += 1

        return self.storage

    def _worker_explore(self, pipe=None):
        """Worker explore in the environment."""
        # get process
        process = multiprocessing.current_process()
        self.process_id = int(process.name.split('-')[-1])

        if self.backend == 'multiprocessing':
            self.pipe = pipe
        else:
            os.environ['MASTER_ADDR'] = '127.0.0.1'
            os.environ['MASTER_PORT'] = '29500'
            os.environ['WORLD_SIZE'] = str(self.num_workers)
            os.environ['RANK'] = str(self.process_id)
            distributed.init_process_group(self.backend, rank=self.process_id, world_size=self.num_workers)

        # copy task
        self.task = copy.deepcopy(self.task)

        while True:
            # get the message (number of steps, parameters, etc).
            msg = self.__recv(rank=self.process_id)

            # end the process if specified, i.e. if the number of steps is negative
            if msg[0] == -1:
                break

            # decompose the received message
            num_steps, render, verbose, parameters = msg

            # set the parameters of the policy
            self.explorer.policy.copy_parameters(parameters)

            # perform a rollout
            trajectory = self.rollout(num_steps=num_steps, deterministic=False, render=render, verbose=verbose)

            # return the process id and the trajectory
            self.__send((self.process_id, trajectory), dst=0, rank=self.process_id)

    def _explore(self, num_steps, num_rollouts=1, deterministic=False, render=False, verbose=False):
        """
        Explore in the environment.

        Args:
            num_steps (int): number of steps in one episode in the on-policy case. This is also the number of updates
                in the off-policy case.
            num_rollouts (int): number of trajectories/rollouts (only valid in the on-policy case).
            deterministic (bool): if deterministic is True, then it does not explore in the environment.
            render (bool): if we should render the environment.
            verbose (int, bool): verbose level, select between {0=False, 1=True, 2}. If 1 or 2, it will print
                information about the exploration process. The level 2 will print more detailed information. Do not use
                it when the states / actions are big or high dimensional, as it could be very hard to make sense of
                the data.

        Returns:
            DictStorage: updated memory storage
        """
        if verbose:
            print("\n#### 1. Starting the Exploration phase ####")

        for rollout in range(num_rollouts):
            # reset environment
            observation = self.env.reset()

            if verbose:
                print("\nStart rollout: {}/{}".format(rollout + 1, num_rollouts))
                # print("Explorer - initial state: {}".format(observation))

            # reset storage
            self.storage.reset(init_states=observation, rollout_idx=rollout)

            # reset explorer
            self.explorer.reset()

            # run RL task for T steps
            step = 0
            for step in range(num_steps):
                # if we need to render
                if render:
                    self.env.render()

                # get action and corresponding distribution from policy
                action, distribution = self.explorer.act(observation, deterministic=deterministic)

                # perform one step in the environment
                next_observation, reward, done, info = self.env.step(action)

                # if verbose:
                #     print("\nExplorer:")
                #     print("1. Observation data: {}".format(observation))
                #     print("2. Action data: {}".format(action))
                #     print("3. Next observation data: {}".format(next_observation))
                #     print("4. Reward: {}".format(reward))
                #     print("5. \\pi(.|s): {}".format(distribution))
                #     print("6. log \\pi(a|s): {}".format([d.log_prob(action) for d in distribution]))

                # insert in storage
                self.storage.insert(observation, action, next_observation, reward, mask=(1 - done),
                                    distributions=distribution, rollout_idx=rollout)

                # set current observation to current one
                observation = next_observation

                # if done, get out of the loop
                if done:
                    break

            # fill remaining mask values
            self.storage.end(rollout)

            if verbose:
                print("End rollout: {}/{} with performed step: {}/{}".format(rollout + 1, num_rollouts,
                                                                             step + 1, num_steps))
                # print("states: {}".format(self.storage['states']))
                # print("actions: {}".format(self.storage['actions']))
                # print("rewards: {}".format(self.storage['rewards']))
                # print("masks: {}".format(self.storage['masks']))
                # print("distributions: {}".format(self.storage['distributions']))

            # # clear explorer
            # self.explorer.clear()

        if verbose:
            print("\n#### End of the Exploration phase ####")

        # return storage unit
        return self.storage

    def explore(self, num_steps, num_rollouts=1, deterministic=False, render=False, verbose=False):
        """
        Explore in the environment.

        Args:
            num_steps (int): number of steps in one episode in the on-policy case. This is also the number of updates
                in the off-policy case.
            num_rollouts (int): number of trajectories/rollouts (only valid in the on-policy case).
            deterministic (bool): if deterministic is True, then it does not explore in the environment.
            render (bool): if we should render the environment.
            verbose (bool): If true, print information on the standard output.

        Returns:
            RolloutStorage, ExperienceReplay: updated memory storage
        """
        if self.num_workers > 1:  # parallel exploration
            return self._master_explore(num_steps, num_rollouts=num_rollouts, render=render, verbose=verbose)
        else:  # simple exploration
            return self._explore(num_steps, num_rollouts=num_rollouts, deterministic=deterministic, render=render,
                                 verbose=verbose)

    #############
    # Operators #
    #############

    def __del__(self):
        """Delete the exploration phase; this will close the processes."""
        self.close()

    def __repr__(self):
        """Return a representation string about the class."""
        return self.__class__.__name__

    def __str__(self):
        """Return a string describing the class."""
        return self.__class__.__name__

    def __call__(self, num_steps, num_rollouts=1, deterministic=False, verbose=True):  # rollout_idx
        """Explore in the environment.

        Args:
            num_steps (int): number of steps in one episode in the on-policy case. This is also the number of updates
                in the off-policy case.
            num_rollouts (int): number of trajectories/rollouts (only valid in the on-policy case).
            deterministic (bool): if deterministic is True, then it does not explore in the environment.
            verbose (bool): If true, print information on the standard output.

        Returns:
            DictStorage: updated memory storage
        """
        self.explore(num_steps, num_rollouts=num_rollouts, deterministic=deterministic, verbose=verbose)
