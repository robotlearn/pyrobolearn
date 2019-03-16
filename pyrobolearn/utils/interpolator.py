
import numpy as np


class HermiteInterpolator(object):
    r"""5th order Hermite interpolator

    """

    def __init__(self, t, x):
        """Calculate the coefficients for the interpolation.

        Assuming a trajectory x(t) is described by a fifth order polynomial such that:
        .. math:: x(t) = a_5 t^5 + a_4 t^4 + a_3 t^3 + a_2 t^2 + a_1 t + a_0

        then taking the derivatives with respect to time give us:
        .. math::
            \dot{x}(t) = 5 a_5 t^4 + 4 a_4 t^3 + 3 a_3 t^2 + 2 a_2 t + a_1
            \ddot{x}(t) = 20 a_5 t^3 + 12 a_4 t^2 + 6 a_3 t + 2 a_2

        We further impose that the initial/final velocities/accelerations to be equal to 0, that is
        :math:`\dot{x}(t_0) = 0, \dot{x}(t_f) = 0, \ddot{x}(t_0) = 0, \ddot{x}(t_f) = 0`.

        Args:
            t (float[T]): time
            x (float[T]): signal/trajectory x(t) to interpolate
        """
        if not isinstance(t, (np.ndarray, list, tuple)):
            raise TypeError("Expecting an iterable for variable t")
        if not isinstance(x, (np.ndarray, list, tuple)):
            raise TypeError("Expecting an iterable for variable x")

        tf = t[-1]
        A = np.array([[1, 1, 1, 1, 1, 1],
                      [5, 4, 3, 2, 1, 0],
                      [20, 12, 6, 2, 0, 0],
                      [0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 0, 1]], dtype=np.float64)
        A *= np.array([tf**i for i in range(5,-1,-1)])
        L = len(t) - 2
        if L != 0:
            l = []
            for i in t[1:-1]:
                l.append([i**j for j in range(5,-1,-1)])
            A = np.vstack((A, np.array(l)))
            b = np.array([x[-1], 0, 0, 0, 0, x[0]] + list(x[1:-1]))
        else:
            b = np.array([x[-1], 0, 0, 0, 0, x[0]])
        #coeff = np.linalg.solve(A,b)[0]
        self.coeff = np.linalg.lstsq(A, b, rcond=None)[0]

    def __call__(self, t):
        """Interpolate the function.

        Args:
            t (float, float[T]): time

        Returns:
            float, float[T]: position
            float, float[T]: velocity
            float, float[T]: acceleration
        """
        x = np.sum(self.coeff * np.array([[ti**i for i in range(5,-1,-1)] for ti in t]), axis=1)
        xd = np.sum(self.coeff[:-1] * np.array([[5*ti**4, 4*ti**3, 3*ti**2, 2*ti, 1] for ti in t]), axis=1)
        xdd = np.sum(self.coeff[:-2] * np.array([[20*ti**3, 12*ti**2, 6*ti, 2] for ti in t]), axis=1)
        return x, xd, xdd


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    # define few points in the x-y plane parametrized by t
    t = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    x = np.array([0.5, 0.25, 0.5, 0.75, 0.5])
    y = np.array([1.0, 0.75, 0.5, 0.25, 0.0])

    # create 5th order Hermite interpolators
    x_interpolator = HermiteInterpolator(t, x)
    y_interpolator = HermiteInterpolator(t, y)

    # interpolate the data
    t = np.linspace(0., 1., 100)
    x,xd,xdd = x_interpolator(t)
    y,yd,ydd = y_interpolator(t)

    # plot figures
    gs = gridspec.GridSpec(4,4)
    plt.subplot(gs[0, 1:3])
    plt.title('Hermite Interpolator')
    plt.plot(x,y)
    plt.xlabel('x(t)')
    plt.ylabel('y(t)')

    y_labels = ['x(t)', 'y(t)', 'dx/dt', 'dy/dt', 'd^2x/dt^2', 'd^2y/dt^2']
    for i, (x_traj, y_traj) in enumerate(zip([x, xd, xdd], [y, yd, ydd])):
        plt.subplot(gs[i+1, :2])
        plt.plot(t, x_traj)
        plt.ylabel(y_labels[2*i])
        if i == 2:
            plt.xlabel('t')
        plt.subplot(gs[i+1, 2:])
        plt.plot(t, y_traj)
        plt.ylabel(y_labels[2*i+1])
        if i == 2:
            plt.xlabel('t')

    plt.tight_layout()
    plt.show()