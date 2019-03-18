## Backends

The general idea is to provide different backends such that different tensor frameworks can be used. These include for instance numpy (with autograd), pytorch, and tensorflow.
These frameworks use different data structures and different signatures for the methods.

It would be nice to have learning models that are more or less independent of the tensor framework as done in Keras.

This is mostly an idea that I had in a later stage, and thus is not operational for the moment. It would require to refactor a bit the code, as currently our code is coupled to the pytorch and numpy frameworks.
