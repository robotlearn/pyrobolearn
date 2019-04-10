## Optimizers / Solvers

This folder provides the various optimizers. It mostly wraps already existing optimizers but provides a common API.

Optimizers allows to maximize/minimize a function with respect to its parameters/inputs under possibly various (equality and/or inequality) constraints with possibly different bounds on the parameters.

Optimizers:
- [`scipy.optimize`](https://docs.scipy.org/doc/scipy/reference/optimize.html)
- [`torch.optim`](https://pytorch.org/docs/stable/optim.html)
- [`nlopt`](https://nlopt.readthedocs.io/en/latest/)
- [`ipopt`](https://projects.coin-or.org/Ipopt)
- [`qpsolvers`](https://github.com/stephane-caron/qpsolvers)
- [`cma`](https://github.com/CMA-ES/pycma)
- [`GPyOpt`](https://sheffieldml.github.io/GPyOpt/)


TODO:
- [ ] clean and finish to implement the various optimizers + structure the code in a better way
