# This file describes the Linear Quadratic Gaussian


class LQG(object):
    r"""Linear Quadratic Gaussian

    Type: Model-based (optimal control)

    Notes: It assumes that the dynamics are described by a linear system of differential equations

    "LQG concerns uncertain linear systems disturbed by additive white Gaussian noise, having incomplete
    state information (i.e. not all the state variables are measured and available for feedback) and
    undergoing control subject to quadratic costs. Moreover, the solution is unique and constitutes a linear
    dynamic feedback control law that is easily computed and implemented."
    LQG = LQE + LQR, where LQE is a Linear Quadratic Estimator (i.e. Kalman Filter), and LQR is a Linear Quadratic
    Regressor.

    References:
        [1]

    See also:
        - `lqr.py`: LQR
        - `ilqr.py`: iterative LQR
        - `ilqg.py`: iterative LQG
    """

    def __init__(self):
        pass

    def compute(self):
        pass