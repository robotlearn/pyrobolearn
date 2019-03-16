# This defines the PID controller


class PID(object):
    r"""Proportional-Integral-Derivative Controller

    The PID control scheme is given by:

    .. math: u(t) = K_p e(t) + K_i \int_0^t e(\tau) d\tau + K_d \frac{d e(t)}{dt}

    where :math:`u(t)` is the controller output, :math:`K_p, K_i, K_d` are respectively the proportional, integral,
    and derivative tuning gains (set by the user or an algorithm), :math:`e(t) = (x_{des} - x(t))` is the error
    between the desired point :math:`x_{des}` and the current point :math:`x(t)`.
    """

    def __init__(self, kp=0, kd=0, ki=0, dt=0.001):
        """
        Initialize the PID controller

        Args:
            kp (float): proportional gain
            kd (float): derivative gain
            ki (float): integral gain
            dt (float): time step
        """
        self.kp = kp
        self.kd = kd
        self.ki = ki
        self.dt = dt
        self.errorI = 0
        self.prev_error = 0

    def compute(self, xd, x, dt=None):
        """
        Compute the controller output using PID control scheme.

        Args:
            xd (float, array): desired point
            x (float, array): current point
            dt (float): time step

        Returns:
            float, array: control output
        """
        error = xd - x
        self.errorI += error
        errorD = (error - self.prev_error) / dt
        self.prev_error = error
        u = self.kp * error + self.ki * self.errorI + self.kd * errorD
        return u