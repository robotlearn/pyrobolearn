# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""Provide some examples using GMMs.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

from pyrobolearn.models.gmm import Gaussian, GMM, plot_gmm, plot_gmm_sklearn


# create manually a GMM
dim, num_components = 2, 5
gmm = GMM(gaussians=[Gaussian(mean=np.random.uniform(-1., 1., size=dim),
                              covariance=0.1*np.identity(dim)) for _ in range(num_components)])
gmm_sklearn = GaussianMixture(n_components=num_components)


# plot initial GMM
plot_gmm(gmm, title='Initial GMM')
plt.show()


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


# init GMM
init_method = 'k-means'  # 'random', 'k-means', 'uniform', 'sklearn', 'curvature'
gmm.init(X, method=init_method)
fig, ax = plt.subplots(1, 1)
plot_gmm(gmm, X=X, ax=ax, title='GMM after ' + init_method.capitalize(), xlim=xlim, ylim=ylim)
plt.show()


# fit a GMM using EM
result = gmm.fit(X, init=None)
gmm_sklearn.fit(X)

# plot EM optimization
plt.plot(result['losses'])
plt.title('EM per iteration')
plt.show()

# plot trained GMM
fig, ax = plt.subplots(1, 2)
plot_gmm(gmm, X=X, label=True, ax=ax[0], title='Our Trained GMM', option=1, xlim=xlim, ylim=ylim)
plot_gmm_sklearn(gmm_sklearn, X, label=True, ax=ax[1], title="Sklearn's Trained GMM", xlim=xlim, ylim=ylim)
plt.show()

# GMR: condition on the input variable and plot
means, std_devs = [], []
time_linspace = np.linspace(-6, 6, 100)
for t in time_linspace:
    g = gmm.condition(np.array([t]), idx_out=[1], idx_in=[0]).approximate_by_single_gaussian()
    means.append(g.mean[0])
    std_devs.append(np.sqrt(g.covariance[0, 0]))

means, std_devs = np.asarray(means), np.asarray(std_devs)

plt.plot(time_linspace, means)
plt.fill_between(time_linspace, means - 2 * std_devs, means + 2 * std_devs, facecolor='green', alpha=0.3)
plt.fill_between(time_linspace, means - std_devs, means + std_devs, facecolor='green', alpha=0.5)
plt.title('GMR')
plt.scatter(X[:, 0], X[:, 1])
plt.show()