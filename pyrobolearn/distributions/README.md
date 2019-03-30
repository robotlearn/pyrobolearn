## Probability distributions and layers

This folder mainly contains wrappers to the various `torch.distributions.*` and provides few additional features / 
functionalities. It also provides several `torch.nn.Module` layers/modules that accepts as input the base output 
or the output of a learning model (such as a linear model, or multilayer perceptron) and returns a distribution 
defined on the output of such layers/modules.

For instance, you can defined a Gaussian distribution module/layer as such:
```python
from pyrobolearn.distributions.modules import *

mean = MeanModule(num_inputs=10, num_outputs=5)
covariance = FullCovarianceModule(num_inputs=10, num_outputs=5)
gaussian = GaussianModule(mean=mean, covariance=covariance)
probs = gaussian(base_output)  # this will feed the `base_output` to the previously defined mean and covariance, 
                               # and will returned a Gaussian distribution based on their outputs.
```
