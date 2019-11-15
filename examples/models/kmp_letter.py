#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Provide some examples using KMP.

See also the `kmp.py` example beforehand. In this example, we delve a bit deeper into KMP using 2D letters as training
data.
"""

import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

from pyrobolearn.models.gmm import plot_gmr, plot_gmm
from pyrobolearn.models.kmp import KMP, RBF


# KMP parameters (play with them)
mean_reg = 1.  # 0.01, 0.1, 1.
covariance_reg = 100.  # 0.1
lengthscale = 1./6

# load the training data
G = loadmat('../../data/2Dletters/G.mat')  # dict
demos = G['demos']  # shape (1,N)
n_demos = demos.shape[1]
dim = demos[0, 0][0, 0][0].shape[0]
length = demos[0, 0][0, 0][0].shape[1]

# plot the training data (x,y)
X = []
xlim, ylim = [-10, 10], [-10, 10]
plt.title("Training Data")
plt.xlim(xlim)
plt.ylim(ylim)
for i in range(0, n_demos, 2):
    demo = demos[0, i][0, 0][0]  # shape (2, 200)
    plt.plot(demo[0], demo[1])
    X.append(demo.T)
plt.show()

# reshape training data (add time in addition to (x,y), thus we now have (t,x,y))
time_linspace = np.linspace(0, 2., length)  # shape (200,)
times = np.asarray([time_linspace for _ in range(len(X))])  # shape (N,200)
X = np.asarray(X)  # shape (N, 200, 2)
X = np.dstack((times, X))  # shape (N, 200, 3)
print(X.shape)

# create KMP
print("Creating the KMP model")
kernel = RBF(lengthscale=lengthscale)
kmp = KMP(kernel_fct=kernel)

# fit a KMP on the data
print("Training the KMP...")
kmp.fit(X=X[:, :, [0]], Y=X[:, :, 1:], gmm_num_components=7, mean_reg=mean_reg, covariance_reg=covariance_reg,
        gmm_num_iters=200, database_size_limit=200, verbose=True)
print("Finished the training")

# plot underlying GMM
gmm = kmp.reference_probability_distribution
plot_gmm(gmm, dims=[1, 2], X=X.reshape(-1, 3), label=True, title='Underlying trained GMM', option=1, xlim=xlim,
         ylim=ylim)
plt.show()

# predict with GMR
gaussians = []
for t in time_linspace:
    g = gmm.condition(t, idx_out=[1, 2], idx_in=0).approximate_by_single_gaussian()
    gaussians.append(g)

# plot figures for GMR
plot_gmr(time_linspace, gaussians=gaussians, xlim=xlim, ylim=ylim, suptitle='GMR')

# predict with the KMP
gaussians = []
for t in time_linspace:
    g = kmp.predict_proba(t, return_gaussian=True)
    gaussians.append(g)

# plot figures for KMP
plot_gmr(time_linspace, gaussians=gaussians, xlim=xlim, ylim=ylim, suptitle='KMP')
plt.show()
