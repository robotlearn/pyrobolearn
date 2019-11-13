#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Provide some examples using GMR.

See also the `gmm.py` example beforehand. In this example, we delve a bit deeper into GMR using 2D letters as training
data.
"""

import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

from pyrobolearn.models.gmm import Gaussian, GMM, plot_gmm, plot_gmr


# load the training data
G = loadmat('../../data/2Dletters/G.mat')  # dict
demos = G['demos']  # shape (1,N)
n_demos = demos.shape[1]
dim = demos[0, 0][0, 0][0].shape[0]
length = demos[0, 0][0, 0][0].shape[1]

# plot the training data (x,y)
X = []
xlim, ylim = [-10, 10], [-10, 10]
plt.xlim(xlim)
plt.ylim(ylim)
for i in range(0, n_demos, 2):
    demo = demos[0, i][0, 0][0]  # shape (2, 200)
    plt.plot(demo[0], demo[1])
    X.append(demo.T)
plt.show()

# reshape training data (add time in addition to (x,y), thus we now have (t,x,y))
time_linspace = np.linspace(0, 2., length)
times = np.asarray([time_linspace for _ in range(len(X))]).reshape(-1, 1)
X = np.vstack(X)  # shape (N*200, 2)
X = np.hstack((times, X))  # shape (N*200, 3)
print(X.shape)

# create GMM
dim, num_components = X.shape[1], 7
gmm = GMM(gaussians=[Gaussian(mean=np.concatenate((np.random.uniform(0, 2., size=1),
                                                   np.random.uniform(-8., 8., size=dim-1))),
                              covariance=0.1*np.identity(dim)) for _ in range(num_components)])

# init GMM
init_method = 'k-means'  # 'random', 'k-means', 'uniform', 'sklearn', 'curvature'
gmm.init(X, method=init_method)
fig, ax = plt.subplots(1, 1)
plot_gmm(gmm, dims=[1, 2], X=X, ax=ax, title='GMM after ' + init_method.capitalize(), xlim=xlim, ylim=ylim)
plt.show()

# fit a GMM on it
result = gmm.fit(X, init=None, num_iters=200)

# plot EM optimization
plt.plot(result['losses'])
plt.title('EM per iteration')
plt.show()

# plot trained GMM
fig, ax = plt.subplots(1, 1)
plot_gmm(gmm, dims=[1, 2], X=X, label=True, ax=ax, title='Our Trained GMM', option=1, xlim=xlim, ylim=ylim)
plt.show()

# GMR: condition on the input variable and plot
gaussians = []
for t in time_linspace:
    g = gmm.condition(np.array([t]), idx_out=[1, 2], idx_in=[0]).approximate_by_single_gaussian()
    gaussians.append(g)

# plot figures for GMR
plot_gmr(time_linspace, gaussians=gaussians, xlim=xlim, ylim=ylim)
plt.show()
