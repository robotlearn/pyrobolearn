## Approximators

This folder provides the various approximators used in the PRL framework. Approximators are a layer on top of the learning models and can accept as inputs/outputs the `State` and `Action` classes defined in this framework in addition to normal tensors. In contrast, the learning models are completely independent of the various classes defined in this framework and can thus be used in other projects.

Approximators can be used to define:
- policies which map states to actions
- value function approximators which map states (and possibly actions) to a scalar.
- dynamic transition function approximators which map states and actions to the next state. This is notably useful in the model-based reinforcement learning paradigm.
- reward function approximator which map states/actions to a scalar. This is notably useful in the inverse reinforcement learning paradigm.

Approximators can be trained / optimized using a loss function and an optimizer.