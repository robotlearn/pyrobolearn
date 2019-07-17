Models
======

Learning models versus algorithms.

As shown on the following figures:


Learning models can be divided into 2 categories:

- General function approximators (aka step-based learning models)
	- Linear models
	- Polynomial models
	- Deep Neural Networks (DNNs)
	- Gaussian processes (GPs)
- Trajectory based learning models
	- Dynamic Movement Primitives (DMPs)
	- Central Pattern Generators (CPGs)
	- Gaussian Mixture Models and Gaussian Mixture Regression (GMMs/GMRs)
	- Probabilistic Movement Primitives (ProMPs)
	- Kernel Movement Primitives (KMPs)

For few of these models, we provide a wrapper around popular libraries such as ``pytorch`` or ``gpytorch``. The other models have been reimplemented to be the most general possible.


Design
------

Models are independent of the other elements in PRL, but are used by other elements in PRL.

UML

The models is notably used by approximators and policies, and their (hyper-)parameters are optimized by algorithms.


How to use a learning model?
----------------------------

.. code-block:: python
    :linenos:

    import torch
    import pyrobolearn as prl


    x = torch.rand(4)
    model = prl.models.LinearModel(num_inputs=4, num_outputs=2)

    print(model.predict(y))


How to create your own model?
-----------------------------

.. code-block:: python
	:linenos:

	import pyrobolearn as prl


	class MyModel(prl.models.Model):
		"""Description"""

		def __init__(self, ...):
			pass

		# implement the various abstract methods
		def ...


Comparisons between the various models?
---------------------------------------

A question that you might have, especially if you are new to the field, what are the differences between the different models that have been proposed in the literature? In this section, I will try to provide the differences (strengths and weaknesses) of each model, and when you should favor one over another one.

The below table summarizes:

- General function approximator (aka step-based learning models) vs trajectory based learning models: trajectory based models accepts as inputs the time and outputs a trajectory (a sequence).
- Parametric vs Non-parametric: In a nutshell, parametric models have parameters that are tuned by the learning algorithm based on the given dataset. Depending on the number of parameters, they might require a lot of data or to have been pretrained on similar datasets. Once trained, parametric models do not need the dataset anymore. On the other hand, non-parametric models don't have parameters but few hyper-parameters. They remember each data point in the dataset, and when given a new input they compare that new input with previous ones, and outputs an estimate based on it. Non-parametric models are very good when you don't have a lot of data points and don't have a pretrained model.
- Linear vs Non-linear: Linear models are the most simple models that makes the least assumption about the data, but can be quite limited in their expressiveness.
- Deterministic vs Probabilistic: Deterministic models predicts a point estimate as output without any quantity that captures the uncertainty associated with that output. Meanwhile, probabilistic models provide a probability distribution for each output. 
- Discriminative vs Generative: discriminative models model learn the mapping ``p(y|x)`` where x is the input and y is the output, while generative models use to learn the data distribution ``p(x,y)``. Generative models are more powerful as given the prior ``p(x)`` or ``p(y)``, you can get back ``p(y|x)`` or ``p(x|y)``. Generative models might require more data.


Future works
------------

* add methods to combine different models together
* provide few other functionalities for the various models
