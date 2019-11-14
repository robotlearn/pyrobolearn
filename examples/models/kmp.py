#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Provide some examples using KMP.
"""

import numpy as np
import matplotlib.pyplot as plt

from pyrobolearn.models.kmp import KMP


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

xlim, ylim = [-8, 8], [-8, 8]

# plot data
plt.title('Training data')
plt.scatter(X[:, 0], X[:, 1])
plt.show()

# create KMP
print("Creating the KMP model")
kmp = KMP()

# fit a KMP on the data
print("Training the KMP...")
kmp.fit(X=X[:, 0].reshape(1, -1, 1), Y=X[:, 1].reshape(1, -1, 1), gmm_num_components=5, mean_reg=1.,
        covariance_reg=60., database_size_limit=n_samples)
print("Finished the training")

# predict using the KMP
means, std_devs = [], []
time_linspace = np.linspace(-6, 6, 100)
for t in time_linspace:
    g = kmp.predict_proba(t, return_gaussian=True)
    means.append(g.mean[0])
    std_devs.append(np.sqrt(g.covariance[0, 0]))

means, std_devs = np.asarray(means), np.asarray(std_devs)

plt.plot(time_linspace, means)
plt.fill_between(time_linspace, means - 2 * std_devs, means + 2 * std_devs, facecolor='green', alpha=0.3)
plt.fill_between(time_linspace, means - std_devs, means + std_devs, facecolor='green', alpha=0.5)
plt.title('KMP')
plt.scatter(X[:, 0], X[:, 1])
plt.show()
