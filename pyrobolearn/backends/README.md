## Backends

The general idea is to provide different backends such that different tensor frameworks can be used. These include for 
instance numpy (with autograd), pytorch, and tensorflow. These frameworks use different data structures and different 
method signatures. By defining a common API, it would ease the use of these various frameworks as the syntax 
would be the same. For instance, in numpy, the outer product between two arrays is performed using `np.outer` 
while in pytorch it is carried out by calling `torch.ger`. Defining a common API would solve these issues.
Also, using backends, we could convert inside each function the given data to the appropriate data structure. 
For instance, a pytorch function defined in the backend could easily accept a `np.array` as input and convert it 
automatically to a `torch.Tensor`.

It would be nice to have learning models that are more or less independent of the tensor framework as done in Keras.

This is mostly an idea that I had in a later stage, and thus is not operational for the moment. It would require to 
refactor a bit the code, as currently our code is coupled to the pytorch and numpy frameworks. Also, it would 
require to provide the same functionalities in the various frameworks, and thus implement their missing 
functionalities.
