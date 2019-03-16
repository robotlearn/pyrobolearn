

class OptimalControlAlgo(object):
    """Optimal Control Algorithm

    Any optimal control schemes inherit from this class. Optimal control is also known as model-based reinforcement
    learning in the computer science and machine learning communities. They however use a different vocabulary and
    have different notations:
    * they minimize 'costs' :math:`c_t` instead of maximizing 'rewards' :math:`r_t`
    * 'controller' instead of 'policy' or 'agent'
    * 'controlled system'/'plant' instead of 'environment'
    * 'control signal' :math:`u_t` instead of 'actions' :math:`a_t`
    * 'states' are denoted by :math:`x_t` instead of `s_t`

    Optimal control assumes that the dynamic model :math:`p(x_{t+1}|x_{t},u_{t})` is given, and use this knowledge
    to optimize the controller.

    .. seealso::
        * `lqr.py`: Linear Quadratic Regulator
        * `ilqr.py`: iterative Linear Quadratic Regulator
        * `lqg.py`: Linear Quadratic Gaussian
        * `ilqg.py`: iterative Linear Quadratic Gaussian
        * `dp.py`: Dynamic Programming
        * `ddp.py`: Differential Dynamic Programming
        * `mpc.py`: Model Predictive Control

    References:
        [1] python-control: https://github.com/python-control/python-control
    """

    def __init__(self, physical_model, states, actions):
        pass