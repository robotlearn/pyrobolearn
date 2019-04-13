## Locomotion controllers

**Note**: the code given here has not yet been integrated fully with the PRL framework, and several piece of codes are 
missing or duplicated. It is also very likely that it doesn't work yet. I will soon clean it, document it, make it 
more modular and flexible, and integrate it into the PRL framework. If you use this code, please cite [1].


This folder contains locomotion controllers, including:
- High-level controllers
    - Behavior controller which uses template simplified models (LIP, SLIP, etc).
- Middle-level controllers
    - Model-predictive controllers (MPCs)
- Low-level controllers
    - Marc Raibert's controller
    - Inverse kinematics controller (which uses QP to optimize several kinematic tasks and constraints)
    - Inverse dynamics controller (which uses QP to optimize several dynamic tasks and constraints)
- Hierarchical controllers (which can accept a high-level controller, middle-level controller, and a low-level controller)


Note that these controllers which are at different levels run at different frequencies (with the lower-level controllers running typically at higher frequencies than higher-level controllers).

Several of these controllers might be using priority tasks defined in `pyrobolearn/priorities`.

If you use this code, please cite [1].

References:
1. "Motion Planning and Control of Dynamic Humanoid Locomotion" (PhD thesis), Songyan Xin, 2018

