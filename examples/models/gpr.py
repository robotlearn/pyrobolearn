#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Provide some examples using GPR.
"""

import sys  # to check if Python 3 or 2
import numpy as np
import matplotlib.pyplot as plt

from pyrobolearn.models.gp import GPR
from pyrobolearn.utils.converter import torch_to_numpy


# create data: Generate random sample following a sine curve
# Ref: https://scikit-learn.org/stable/auto_examples/mixture/plot_gmm_sin.html#sphx-glr-auto-examples-mixture-\
# plot-gmm-sin-py
n_samples = 100
np.random.seed(0)
X = np.zeros((n_samples, 2))
step = 4. * np.pi / n_samples

for i in range(X.shape[0]):
    x = i * step - 6.
    X[i, 0] = x + np.random.normal(0, 0.1)
    X[i, 1] = 3. * (np.sin(x) + np.random.normal(0, .2))

x, y = X[:, 0], X[:, 1]
xlim, ylim = [-8, 8], [-8, 8]

# plot data
plt.title('Training data')
plt.scatter(x, y)
plt.xlim(xlim)
plt.ylim(ylim)
plt.show()

# create GPR
model = GPR()

# plot prior possible functions
f = model.sample(x, num_samples=10, to_numpy=True)
plt.title('Sampled functions from prior distribution')
plt.scatter(x, y, alpha=0.3)
plt.plot(x, f.T)
plt.xlim(xlim)
plt.ylim(ylim)
plt.show()

# compute log likelihoods
print("\nBefore training:")
print("Log likelihood: {}".format(model.log_likelihood(x, y, to_numpy=True)))
print("Log marginal likelihood: {}".format(model.log_marginal_likelihood(x, y, to_numpy=True)))

# fit the data
optimizer = 'adam'  # 'lbfgs'
model.fit(x, y, num_iters=100, optimizer=optimizer, verbose=True)

# compute log likelihoods
print("\nAfter training:")
print("Log likelihood: {}".format(model.log_likelihood(x, y, to_numpy=True)))
print("Log marginal likelihood: {}".format(model.log_marginal_likelihood(x, y, to_numpy=True)))

# sample function and plot it
f = model.sample(x, num_samples=1, to_numpy=True)
plt.title('one sampled function after training')
plt.scatter(x, y)
plt.plot(x, f.T, 'k', linewidth=2.)
plt.xlim(xlim)
plt.ylim(ylim)
plt.show()

# predict prob
x_test = np.linspace(-10, 10, 101)
xlim, ylim = [-10, 10], [-10, 10]
mean_y, var_y = model.predict_prob(x_test, to_numpy=True)
std_y = np.sqrt(var_y)
plt.title('Prediction')
plt.scatter(x, y)
plt.plot(x_test, mean_y, 'b')
plt.fill_between(x_test, mean_y-2*std_y, mean_y+2*std_y, facecolor='green', alpha=0.3)
plt.xlim(xlim)
plt.ylim(ylim)
plt.show()

# Another way to predict
pred = model.forward(x_test)
lower, upper = pred.confidence_region()

plt.title("Another way to predict (see code)")
plt.plot(x, y, 'k*')
if sys.version_info[0] < 3:  # Python 2
    plt.plot(x_test, torch_to_numpy(pred.mean()), 'b')
else:  # Python 3
    plt.plot(x_test, torch_to_numpy(pred.mean), 'b')
plt.fill_between(x_test, torch_to_numpy(lower), torch_to_numpy(upper), alpha=0.5)
plt.xlim(xlim)
plt.ylim(ylim)
plt.show()
