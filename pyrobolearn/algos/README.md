## Algos

This folder contains the various learning algorithms. Learning algorithms describe how to optimize the (hyper-)parameters of a particular learning model using a given loss function and optimizer.
Learning algorithms should not be confused with the models.

They can be divided into two main categories:
* supervised / unsupervised learning algorithms: This type of algorithm describes how to update the (hyper-)parameters of a learning model given some input (and possibly output) data, a loss function to evaluate its performance, and an optimizer. 
* reinforcement learning algorithms:
	* In the model-free paradigm: the algorithm basically performs 3 main steps:
		1. Exploration: the algorithm describes how the policy should explore in the given environment and collect the various states, actions and rewards.
		2. Evaluation: it evaluates the actions taken by the policy using a certain estimator
		3. Update: this step is similar to supervised/unsupervised learning algorithms, where it updates the (hyper-)parameters of the various approximators (e.g. policies, value approximators, etc) using the given loss function and optimizer.
