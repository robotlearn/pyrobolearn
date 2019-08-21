Baselines
---------

This folder contains examples when using PRL environments and algorithms defined in the ``stable_baselines`` Python
library.

Few notes with respect to that:

1. ``stable_baselines`` uses the ``TensorFlow`` backend, and a ``DummyVecEnv`` has to be provided to the algorithms.
2. Normally, in PRL, the actions can be defined outside the environments and it is the policy that is responsible to
   apply the action in the world. However, in ``OpenAI gym``, it is the environment that has the ``action_space`` and
   apply the ``action``. To accommodate with that, the action can also be defined and provided to the PRL environment.
3. When using PRL with ``stable_baselines``, make sure that each states have the same dimensions; i.e. we can not
   return a 1D vector state with a 2D matrix state at the same time (at least, not currently).
