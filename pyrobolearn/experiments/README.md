## Experiments

In this folder, we define the `Experiment` class which is the highest-level class of our framework. More specifically, it allows to organize which tasks to run, which metrics to use, and allows to easily compare different models, algos, methods, and so on.

The idea would be to define everything (world, robot, states, actions, rewards, environment, policy, algorithm, etc) inside the class that inherits from the `Experiment` class, and we should be able to run it using few commands.
```python
experiment = MyExperiment(<args>)
print(experiment.description())
experiment.train()
experiment.test()
experiment.plot()
```

This would I hope ease the reproduction of experiments performed in the literature.

Experiments will be added in a later stage.
