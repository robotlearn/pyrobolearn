# The various backends allows to specify which framework we want to use

import os

# check if the 'PYROBOLEARN_BACKEND' environment variable has been defined
if 'PYROBOLEARN_BACKEND' not in os.environ:
    os.environ['PYROBOLEARN_BACKEND'] = 'torch'

# provide the various backends
if os.environ['PYROBOLEARN_BACKEND'] == 'torch':
    import torch_backend as backend
elif os.environ['PYROBOLEARN_BACKEND'] == 'numpy':
    import numpy_backend as backend
else:
    pass

# alias
b = backend


# Tests #

# create array
a = b.array([1, 2, 3])
print(type(a))
print(a)
print(a.reshape((3, 1)))
