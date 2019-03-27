## State Processors

This folder contains code to process states that are returned by the environment. This is a little bit different from the `processors` defined in the `pyrobolearn/processors` folder which can process the state data but let the state unchanged. That is, it doesn't change the state data, thus if you have a policy and a value function (which have and use the same state as input), each of these approximators can use their own processors on the state without one set of processors affecting the state data. This can lead to some overhead in processing time if the same processors have to be used for the various approximators / controllers. In contrast, the processors defined here wraps the states (i.e. they are also considered as `State` as they inherit from it), and the modified data will be apparent to all the approximators / controllers that take as inputs the state. Wrapping the states allows also to use different processors for different states, and inheriting from `State` allows to use the various operations defined there. For the latter case, you can for instance add two state processors.

Here is a pseudo-code to illustrate the above points:
```python
s1 = State1(args)
s2 = State2(args)

s = s1 + s2
proc_state_1 = StateProcessor(s, args)
proc_state_2 = StateProcessor1(s1, args1) + StateProcessor2(s2, args2)
proc_state_3 = StateProcessor2(StateProcessor1(s, args1), args2)

# to compute the states, just call them which would normally provide different data
state_data = s()
proc_state_1_data = proc_state_1()
proc_state_2_data = proc_state_2()
proc_state_3_data = proc_state_3()
```

Important state processors are processors that center, normalize, standardize, and/or clip the state data.

