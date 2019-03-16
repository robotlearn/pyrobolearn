# MPC: Model Predictive Control

class MPC(object):
    r"""Model Predictive Control

    Type: Optimal Control

    MPC optimizes for a finite time-horizon :math:`T` (i.e. compute the optimal :math:`\{u_t, u_{t+1},..., u_T\}`
    given the dynamical system :math:`x_{t+1} = f(x_t, u_t)` and cost :math:`c(x_t,u_t)`), executes the first best
    found control law :math:`u_t`, lets the system goes to the next state :math:`x_{t+1}`, and then re-optimize again
    for each next time step. The finite time-horizon allows to take into account close future events, while
    re-optimizing at each time step allows to deal with the discrepancy between the modeled and real dynamical systems.

    Notes:
        * MPC vs LQR: LQR assumes a linear dynamical system and optimizes for the whole time horizon providing us
            the single optimal solution, while MPC optimizes in a receding time window at each time step
            resulting in a suboptimal solution but more robust to various perturbations, uncertainties, and so on
            not accounted/modeled by our dynamical system.

    References:
        [1]
    """

    def __init__(self):
        pass

    def compute(self, x):
        pass