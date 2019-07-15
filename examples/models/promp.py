#!/usr/bin/env python
"""Provide some examples using ProMPs.
"""

import numpy as np
import matplotlib.pyplot as plt

from pyrobolearn.models.promp.promp import DiscreteProMP, plot_state, plot_proba_state, plot_weighted_basis


# create data and plot it
N = 8
t = np.linspace(0., 1., 100)
# eps = 0.1
# y = np.array([np.sin(2*np.pi*t) + eps * np.random.rand(len(t)) for _ in range(N)])      # shape: NxT
# dy = np.array([2*np.pi*np.cos(2*np.pi*t) + eps * np.random.rand(len(t)) for _ in range(N)])     # shape: NxT
phi = np.random.uniform(low=-1., high=1., size=N)
y = np.array([np.sin(2 * np.pi * t + phi[i]) for i in range(int(N/2))])  # shape: NxT
y1 = np.array([np.cos(2 * np.pi * t + phi[i]) for i in range(int(N/2))])
y = np.vstack((y, y1))
dy = np.array([2 * np.pi * np.cos(2 * np.pi * t + phi[i]) for i in range(int(N/2))])  # shape: NxT
dy1 = np.array([2 * np.pi * np.sin(2 * np.pi * t + phi[i]) for i in range(int(N/2))])
dy = np.vstack((dy, dy1))
Y = np.dstack((y, dy))  # N,T,2D  --> why not N,2D,T
plot_state(Y, title='Training data')
plt.show()

# create discrete and rhythmic ProMP
promp = DiscreteProMP(num_dofs=1, num_basis=10, basis_width=1./20)

# plot the basis function activations
plt.plot(promp.Phi(t)[:, :, 0].T)
plt.title('basis functions')
plt.show()

# plot ProMPs
y_pred = promp.rollout()
fig, ax = plt.subplots(1, 2)
plot_state(y_pred[None], ax=ax, title='ProMP prediction before learning', linewidth=2.)    # shape: N,T,2D
plot_weighted_basis(t, promp, ax=ax)
plt.show()

# learn from demonstrations
promp.imitate(Y)
y_pred = promp.rollout()
fig, ax = plt.subplots(1, 2)
plot_state(y_pred[None], ax=ax, title='ProMP prediction after learning', linewidth=3.)   # N,T,2D
plot_weighted_basis(t, promp, ax=ax)
plt.show()

method = 'marginal'
means, covariances = promp.rollout_proba(method=method, return_gaussian=False)
fig, ax = plt.subplots(1, 2)
# plot_state(Y, ax=ax, title='Training data')
plot_proba_state(means, covariances, ax=ax, title='ProMP prediction after learning', linewidth=3.)
plt.show()
