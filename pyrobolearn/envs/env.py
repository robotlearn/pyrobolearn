#!/usr/bin/env python
"""Define the `Env` class which defines the world, states, and possible rewards. This is the main object
a policy interacts with.

Dependencies:
- `pyrobolearn.worlds`
- `pyrobolearn.states`
- `pyrobolearn.actions`
- (`pyrobolearn.rewards`)
- (`pyrobolearn.envs.terminal_condition`)
"""

# import gym

from pyrobolearn.worlds import World, BasicWorld
from pyrobolearn.states import State
from pyrobolearn.actions import Action
from pyrobolearn.rewards import Reward

from pyrobolearn.terminal_conditions import TerminalCondition
from pyrobolearn.physics import PhysicsRandomizer

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class Env(object):  # gym.Env):
    r"""Environment class.

    This class defines the environment as it described in a reinforcement learning setting [1]. That is, given an
    action :math:`a_t` performed by an agent (i.e. policy), the environment computes and returns the next state
    :math:`s_{t+1}` and reward :math:`r(s_t, a_t, s_{t+1})`. A policy can then interact with this environment,
    and be trained.

    The environment defines the world, the rewards, and the states that are returned by it.
    To allow our framework to be generic and modular, the world, rewards, and states are decoupled from the environment,
    and can be defined outside of this one and then provided as inputs to the `Env` class. That is, we favor
    'composition over inheritance' (see [2]). This is a different approach compared to what is usually done using
    the OpenAI gym framework (see [3]). That said, in order to be compatible with this framework, we inherit from
    the `gym.Env` class (see `core.py` in `https://github.com/openai/gym/blob/master/gym/core.py`).

    References:
        [1] "Reinforcement Learning: An Introduction", Sutton and Barto, 1998
        [2] "Wikipedia: Composition over Inheritance", https://en.wikipedia.org/wiki/Composition_over_inheritance
        [3] "OpenAI gym": https://gym.openai.com/   and    https://github.com/openai/gym

    """

    def __init__(self, world, states, rewards=None, terminal_conditions=None, initial_state_distribution=None,
                 physics_randomizer=None, extra_info=None):
        """
        Initialize the environment.

        Args:
            world (World): world of the environment. The world contains all the objects (including robots), and has
                           access to the simulator.
            states (State): states that are returned by the environment at each time step.
            rewards (None, Reward): The rewards can be None when for instance we are in an imitation learning setting,
                                    instead of a reinforcement learning one. If None, only the state is returned by
                                    the environment.
            terminal_conditions (None, callable): A callable function or object that check if the policy has failed
                                                 or succeeded the task.
            initial_state_distribution (None, callable): A callable function or object that is called at the beginning
                                                        when resetting the environment to generate the initial state
                                                        distribution.
            physics_randomizer (None, PhysicsRandomizer, list of PhysicsRandomizer): physics randomizers. This will be
                called each time you reset the environment.
            extra_info (None, callable): Extra info returned by the environment at each time step.
        """
        # Check and set parameters (see corresponding properties)
        self.world = world
        self.states = states
        self.rewards = rewards
        self.terminal_conditions = terminal_conditions
        self.physics_randomizers = physics_randomizer
        self.extra_info = extra_info if extra_info is not None else lambda: False

        self.rendering = False  # check with simulator

        # save the world state in memory
        self.initial_world_state = self.world.save()

    ##############
    # Properties #
    ##############

    @property
    def world(self):
        """Return an instance of the world."""
        return self._world

    @world.setter
    def world(self, world):
        """Set the world."""
        if not isinstance(world, World):
            raise TypeError("Expecting the 'world' argument to be an instance of World.")
        self._world = world
        self.sim = self._world.simulator

    @property
    def simulator(self):
        """Return an instance of the simulator."""
        return self._world.simulator

    @property
    def states(self):
        """Return the states."""
        return self._states

    @states.setter
    def states(self, states):
        """Set the states."""
        if not isinstance(states, State):
            raise TypeError("Expecting the 'states' argument to be an instance of State.")
        self._states = states

    @property
    def rewards(self):
        """Return the rewards."""
        return self._rewards

    @rewards.setter
    def rewards(self, rewards):
        """Set the rewards."""
        if rewards is not None:
            if not isinstance(rewards, Reward):
                raise TypeError("Expecting the 'rewards' argument to be an instance of Reward.")
        else:
            rewards = lambda: None
        self._rewards = rewards

    @property
    def terminal_conditions(self):
        """Return the terminal condition."""
        return self._terminal_conditions

    @terminal_conditions.setter
    def terminal_conditions(self, conditions):
        """Set the terminal condition."""
        if conditions is None:
            conditions = [TerminalCondition()]
        elif isinstance(conditions, TerminalCondition):
            conditions = [conditions]
        elif isinstance(conditions, (list, tuple)):
            for idx, condition in enumerate(conditions):
                if not callable(conditions):
                    raise TypeError("Expecting the {} item in the given terminal conditions to be an instance of "
                                    "`TerminalCondition`, instead got: {}".format(idx, type(condition)))
        else:
            raise TypeError("Expecting the terminal conditions to be an instance of `TerminalCondition`, or a list of "
                            "`TerminalCondition`, but instead got: {}".format(type(conditions)))
        self._terminal_conditions = conditions

    @property
    def physics_randomizers(self):
        """Return the list of physics randomizers used each time we reset the environment."""
        return self._physics_randomizers

    @physics_randomizers.setter
    def physics_randomizers(self, randomizers):
        """Set the physics randomizers."""
        if randomizers is None:
            randomizers = []
        elif isinstance(randomizers, PhysicsRandomizer):
            randomizers = [randomizers]
        elif isinstance(randomizers, (list, tuple)):
            for randomizer in randomizers:
                if not isinstance(randomizer, PhysicsRandomizer):
                    raise TypeError("Expecting the randomizer to be an instance of `PhysicsRandomizer`, instead got "
                                    "{}".format(randomizer))
        else:
            raise TypeError("Expecting the given randomizers to be None, a `PhysicsRandomizer`, or a list of them; "
                            "instead got: {}".format(type(randomizers)))
        self._physics_randomizers = randomizers

    ###########
    # Methods #
    ###########

    def reset(self):
        """
        Reset the environment; reset the world and states.

        Returns:
            list/np.array: list of state data
        """
        # reset world
        self.world.reset()

        # randomize the environment
        for randomizer in self.physics_randomizers:
            randomizer.randomize()

        # reset states and return first states/observations
        return self.states.reset()

    def step(self, actions=None):
        """
        Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.
        Accepts an action and returns a tuple (observation, reward, done, info).

        Args:
            action (Action, None): an action provided by the policy(ies) to the environment

        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (boolean): whether the episode has ended, in which case further step() calls will return undefined
                            results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        # if not isinstance(actions, (list, tuple)):
        #     actions = [actions]
        if actions is not None and not isinstance(actions, Action):
            raise TypeError("Expecting actions to be an instance of Action.")

        # apply each policy's action in the environment
        # for action in actions:
        #    action()
        # TODO: calling the actions should be done inside the policy(ies), and not in the environments. The policy
        #  decided when to execute an action. Think about when there are multiple policies, when using multiprocessing,
        #  or when the environment runs in real-time.
        if actions is not None and isinstance(actions, Action):
            actions()

        # perform a step forward in the simulation which computes all the dynamics
        self.world.step()

        # compute reward
        # rewards = [reward.compute() for reward in self.rewards]
        rewards = self.rewards()

        # compute terminating condition
        # done = [reward.is_done() for reward in self.rewards]
        done = any([condition() for condition in self.terminal_conditions])

        # get next state/obs for each policy
        # states = [state() for state in self.states]
        # TODO: this should be before computing the rewards as some rewards need the next state
        self.states()

        # get extra information
        info = self.extra_info()

        return self.states, rewards, done, info

    def render(self, mode='human'):
        # This is dependent on the simulator. Some simulators allow to show the GUI at any point in time,
        # while others like pybullet requires to specify it at the beginning (thus see SimuRealInterface).

        # Bullet: do nothing
        # if isinstance(self.sim, Bullet): pass
        # self.sim.configureDebugVisualizer(self.sim.COV_ENABLE_RENDERING, 1)
        pass

    def hide(self):
        # hide the GUI
        # self.sim.configureDebugVisualizer(self.sim.COV_ENABLE_RENDERING, 0)
        pass

    def close(self):
        pass

    def seed(self, seed=None):
        """
        Set the given seed for the simulator.

        Args:
            seed (int): seed for the random generator used in the simulator.
        """
        # set the simulator seed
        self.simulator.seed(seed)

        # set the seed for the physics randomizer
        for randomizer in self.physics_randomizers:
            randomizer.seed(seed)



class BasicEnv(Env):
    """Basic Environment class.

    It creates a basic environment with a basic world (a floor and with gravity), no rewards, and no states.
    """

    def __init__(self, states=None, rewards=None):
        world = BasicWorld()
        super(BasicEnv, self).__init__(world, states, rewards)


# Tests
if __name__ == '__main__':
    from pyrobolearn.simulators import BulletSim
    import time

    # create simulator
    sim = BulletSim()

    # create world
    world = BasicWorld(sim)
    robot = world.loadRobot('coman', useFixedBase=True)

    # create states
    states = State()

    # create rewards
    reward = Reward()

    # create Env
    env = Env(world, states, reward)

    # dummy action
    action = Action()

    # run the environment for n steps
    for _ in range(10000):
        env.step(action)
        time.sleep(1./240)
