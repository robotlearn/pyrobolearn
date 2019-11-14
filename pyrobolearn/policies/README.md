## Policies

Policies in this framework are controllers that can be trained and map states to actions. They use directly the 
learning model or the `Approximator` class (which uses the learning model).

In this framework, `State` and `Action` instances should be given to the `Policy` which would infer its input and 
output dimensions and build the model with the correct number of inputs/outputs. In contrast to the learning model, 
the policy should know how to feed the various input states to the inner learning model, such that if a picture and 
joint states are given to the policy it knows where to feed the corresponding input observations.
