#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Provide some examples using DMPs.
"""

from pyrobolearn.models.dmp import *


# tests canonical systems
discrete_cs = DiscreteCS()
rhythmic_cs = RhythmicCS()

# plot canonical systems
plt.subplot(1, 2, 1)
plt.title('Discrete CS')
for tau in [1., 0.5, 2.]:
    rollout = discrete_cs.rollout(tau=tau)
    plt.plot(np.linspace(0, 1., len(rollout)), rollout, label='tau='+str(tau))
plt.legend()

plt.subplot(1, 2, 2)
plt.title('Rhythmic CS')
for tau in [1., 0.5, 2.]:
    rollout = rhythmic_cs.rollout(tau=tau)
    plt.plot(np.linspace(0, 1., len(rollout)), rollout, label='tau='+str(tau))
plt.legend()
plt.show()

# tests basis functions
num_basis = 20
discrete_f = DiscreteForcingTerm(discrete_cs, num_basis)
rhythmic_f = RhythmicForcingTerm(rhythmic_cs, num_basis)

plt.subplot(1, 2, 1)
rollout = discrete_cs.rollout()
plt.title('discrete basis fcts')
plt.plot(rollout, discrete_f.psi(rollout))

plt.subplot(1, 2, 2)
rollout = rhythmic_cs.rollout()
plt.title('rhythmic basis fcts')
plt.plot(rollout, rhythmic_f.psi(rollout))
plt.show()

# tests forcing terms
f = np.sin(np.linspace(0, 2*np.pi, 100))
discrete_f.train(f, plot=True)

f = np.sin(np.linspace(0, 2*np.pi, int(2*np.pi*100)))
rhythmic_f.train(f, plot=True)

# Test discrete DMP
discrete_dmp = DiscreteDMP(num_dmps=1, num_basis=num_basis)
t = np.linspace(-6, 6, 100)
y_target = 1 / (1 + np.exp(-t))
discrete_dmp.imitate(y_target)
y, dy, ddy = discrete_dmp.rollout()

plt.plot(y_target, label='y_target')
plt.plot(y[0], label='y_pred')
# plt.plot(dy[0])
# plt.plot(ddy[0])
y, dy, ddy = discrete_dmp.rollout(new_goal=np.array([2.]))
plt.plot(y[0], label='y_scaled')
plt.title('Discrete DMP')
plt.legend()
plt.show()


# tests basis functions
num_basis = 100

# Test Biologically-inspired DMP
t = np.linspace(0., 1., 100)
y_d = np.sin(np.pi * t)
new_goal = np.array([[0.8, -0.25],
                     [0.8, 0.25],
                     [1.2, -0.25]])

discrete_dmp = DiscreteDMP(num_dmps=2, num_basis=num_basis)
discrete_dmp.imitate(np.array([t, y_d]))
y, dy, ddy = discrete_dmp.rollout()
init_points = np.array([discrete_dmp.y0, discrete_dmp.goal])
# print(discrete_dmp.generate_goal())
# print(discrete_dmp.generate_goal(f0=discrete_dmp.f_target[:,0]))

# check with standard discrete DMP when rescaling the goal
plt.subplot(1, 3, 1)
plt.title('Initial discrete DMP')
plt.scatter(init_points[:,0], init_points[:, 1], color='b')
plt.scatter(new_goal[:, 0], new_goal[:, 1], color='r')
plt.plot(y[0], y[1], 'b', label='original')

plt.subplot(1, 3, 2)
plt.title('Rescaled discrete DMP')
plt.scatter(init_points[:, 0], init_points[:, 1], color='b')
plt.scatter(new_goal[:, 0], new_goal[:, 1], color='r')
plt.plot(y[0], y[1], 'b', label='original')
for g in new_goal:
    y, dy, ddy = discrete_dmp.rollout(new_goal=g)
    plt.plot(y[0], y[1], 'g', label='scaled')
plt.legend(['original', 'scaled'])

# change goal with biologically-inspired DMP
new_goal = np.array([[0.8, -0.25],
                     [0.8, 0.25],
                     [0.4, 0.1],
                     [5., 0.15],
                     [1.2, -0.25],
                     [-0.8, 0.1],
                     [-0.8, -0.25],
                     [5., -0.25]])
bio_dmp = BioDiscreteDMP(num_dmps=2, num_basis=num_basis)
bio_dmp.imitate(np.array([t, y_d]))
y, dy, ddy = bio_dmp.rollout()
init_points = np.array([bio_dmp.y0, bio_dmp.goal])

plt.subplot(1, 3, 3)
plt.title('Biologically-inspired DMP')
plt.scatter(init_points[:, 0], init_points[:, 1], color='b')
plt.scatter(new_goal[:, 0], new_goal[:, 1], color='r')
plt.plot(y[0], y[1], 'b', label='original')
for g in new_goal:
    y, dy, ddy = bio_dmp.rollout(new_goal=g)
    plt.plot(y[0], y[1], 'g', label='scaled')
plt.legend(['original', 'scaled'])
plt.show()

# changing goal at the middle
y_list = []
for g in new_goal:
    bio_dmp.reset()
    y_traj = np.zeros((2, 100))
    for t in range(100):
        if t < 30:
            y, dy, ddy = bio_dmp.step()
        else:
            y, dy, ddy = bio_dmp.step(new_goal=g)
        y_traj[:, t] = y
    y_list.append(y_traj)
for y in y_list:
    plt.plot(y[0], y[1])
plt.scatter(bio_dmp.y0[0], bio_dmp.y0[1], color='b')
plt.scatter(new_goal[:, 0], new_goal[:, 1], color='r')
plt.title('change goal at the middle')
plt.show()

# changing goal at the middle but with a moving goal
g = np.hstack((np.arange(1.0, 2.0, 0.1).reshape(10, -1),
               np.arange(0.0, 1.0, 0.1).reshape(10, -1)))

bio_dmp.reset()
y_traj = np.zeros((2, 100))
y_list = []
for t in range(100):
    y, dy, ddy = bio_dmp.step(new_goal=g[int(t/10)])
    y_traj[:, t] = y
    if (t % 10) == 0:
        y_list.append(y)
y_list = np.array(y_list)

plt.plot(y_traj[0], y_traj[1])
plt.scatter(bio_dmp.y0[0], bio_dmp.y0[1], color='b')
plt.scatter(g[:, 0], g[:, 1], color='r')
plt.scatter(y_list[:, 0], y_list[:, 1], color='g')
plt.title('moving goal')
plt.show()
