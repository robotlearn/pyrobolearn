# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""Define several filters such as moving average and 1euro filters.
"""

import numpy as np

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2019, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class Filter(object):
    """Abstract Filter class"""
    pass


class MovingAverageFilter(Filter):
    r"""Moving Average Filter

    The moving average filter computes the moving mean given by:

    .. math:: \mu_{N+1} = \frac{N}{N+1} \mu_N + \frac{1}{N+1} x_{N+1}`

    where :math:`\mu_0 = 1`.

    If an :math:`\alpha` parameter is provided it will compute:

    .. math:: \mu_{N+1} = (1-\alpha) \mu_N + \alpha x_{N+1}
    """

    def __init__(self, alpha=None):
        """Initialize the moving average filter"""
        self.mu = None
        self.N = 0
        self.alpha = alpha

    def __call__(self, x):
        """
        Filter the given noisy sample input value.

        Args:
            x (float): noisy sample input value.

        Returns:
            float: moving average filtered value.
        """
        if self.mu is None:
            self.mu, self.N = x, 1
        elif self.alpha is None:
            self.N += 1
            self.mu = (self.N-1.)/self.N * self.mu + 1./self.N * x
        else:
            self.mu = (1. - self.alpha) * self.mu + self.alpha * x
        return self.mu


class MovingMedianFilter(Filter):
    """Moving Median Filter

    Compared to the moving average filter, it is a bit less sensitive to outliers especially for short spikes.
    """
    pass


class OneEuroFilter(Filter):
    """1 Euro Filter

    The code given here is taken from [4] which is distributed under the BSD-3 license:

    "OneEuroFilter.py -

    Author: Nicolas Roussel (nicolas.roussel@inria.fr)
    Copyright 2019 Inria

    BSD License https://opensource.org/licenses/BSD-3-Clause

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:

    1. Redistributions of source code must retain the above copyright notice, this list of conditions
    and the following disclaimer.

    2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions
    and the following disclaimer in the documentation and/or other materials provided with the distribution.

    3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or
    promote products derived from this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
    INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
    SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
    WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
    USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE."

    From the documentation presented in [1] and reproduced here for completeness purpose:

    "Tuning the filter

    To minimize jitter and lag when tracking human motion, the two parameters (fcmin and beta) can be set using a
    simple two-step procedure. First beta is set to 0 and fcmin (mincutoff) to a reasonable middle-ground value such
    as 1 Hz. Then the body part is held steady or moved at a very low speed while fcmin is adjusted to remove jitter
    and preserve an acceptable lag during these slow movements (decreasing fcmin reduces jitter but increases lag,
    fcmin must be > 0). Next, the body part is moved quickly in different directions while beta is increased with a
    focus on minimizing lag. First find the right order of magnitude to tune beta, which depends on the kind of data
    you manipulate and their units: do not hesitate to start with values like 0.001 or 0.0001. You can first multiply
    and divide beta by factor 10 until you notice an effect on latency when moving quickly. Note that parameters
    fcmin and beta have clear conceptual relationships: if high speed lag is a problem, increase beta; if slow speed
    jitter is a problem, decrease fcmin."

    References:
        - [1] Main webpage: http://cristal.univ-lille.fr/~casiez/1euro/
        - [2] Interactive demo: http://cristal.univ-lille.fr/~casiez/1euro/InteractiveDemo/
        - [3] Blog: https://jaantollander.com/2018-12-29-noise-filtering-using-one-euro-filter.html
        - [4] Python version: http://cristal.univ-lille.fr/~casiez/1euro/OneEuroFilter.py
    """

    class LowPassFilter(object):
        """Low Pass Filter

        This low pass filter is used in the One euro filter.
        """

        def __init__(self, alpha):
            """
            Initialize the Low-pass filter.

            Args:
                alpha (float): alpha value.
            """
            self.alpha = alpha
            self.__y = self.__s = None

        @property
        def alpha(self):
            return self.__alpha

        @alpha.setter
        def alpha(self, alpha):
            alpha = float(alpha)
            if alpha <= 0 or alpha > 1.0:
                raise ValueError("alpha (%s) should be in (0.0, 1.0]" % alpha)
            self.__alpha = alpha

        @property
        def last_value(self):
            return self.__y

        def __call__(self, value, timestamp=None, alpha=None):
            """
            Filter method of Low-pass filter.

            Args:
                value (float): noisy sample value.
                timestamp (float): time stamp value.
                alpha (float): alpha value.

            Returns:
                float: filtered value.
            """
            if alpha is not None:
                self.alpha = alpha
            if self.__y is None:
                s = value
            else:
                s = self.__alpha * value + (1.0 - self.__alpha) * self.__s
            self.__y = value
            self.__s = s
            return s

    def __init__(self, freq, mincutoff=1.0, beta=0.0, dcutoff=1.0):
        """
        Initialize the One euro filter.

        Args:
            freq (float): data update rate.
            mincutoff (float): minimum cutoff frequency.
            beta (float): cutoff slope.
            dcutoff (float): cutoff frequency for derivate.
        """
        if freq <= 0:
            raise ValueError("freq should be >0")
        if mincutoff <= 0:
            raise ValueError("mincutoff should be >0")
        if dcutoff <= 0:
            raise ValueError("dcutoff should be >0")
        self.__freq = float(freq)
        self.__mincutoff = float(mincutoff)
        self.__beta = float(beta)
        self.__dcutoff = float(dcutoff)
        self.__x = self.LowPassFilter(self.__alpha(self.__mincutoff))
        self.__dx = self.LowPassFilter(self.__alpha(self.__dcutoff))
        self.__last_time = None

    def __alpha(self, cutoff):
        """
        Alpha computation.

        Args:
            cutoff (float): cutoff frequency in Hz.

        Returns:
            float: alpha value for low-pass filter.
        """
        te = 1.0 / self.__freq
        tau = 1.0 / (2 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / te)

    def __call__(self, x, timestamp=None):
        """
        Filter the noisy sample input value(s).

        Args:
            x (float): noisy sample input value.
            timestamp (float): time stamp value.

        Returns:
            float: filtered sample value.
        """
        # update the sampling frequency based on timestamps
        if self.__last_time and timestamp:
            self.__freq = 1.0 / (timestamp - self.__last_time)
        self.__last_time = timestamp
        # estimate the current variation per second
        prev_x = self.__x.last_value
        dx = 0.0 if prev_x is None else (x - prev_x) * self.__freq  # FIXME: 0.0 or value?
        edx = self.__dx(dx, timestamp, alpha=self.__alpha(self.__dcutoff))
        # use it to update the cutoff frequency
        cutoff = self.__mincutoff + self.__beta * np.fabs(edx)
        # filter the given value
        return self.__x(x, timestamp, alpha=self.__alpha(cutoff))


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # Test the filters
    one_euro_filter = OneEuroFilter(freq=1, mincutoff=0.5, beta=0.1, dcutoff=1.0)
    moving_average = MovingAverageFilter(alpha=0.3)

    t = np.linspace(0, 1., 200)
    x = np.sin(4*np.pi*t) + 0.2 * (np.random.rand(200) - 0.5)

    plt.plot(t, x)
    plt.plot(t, [moving_average(i) for i in x])
    plt.plot(t, [one_euro_filter(i, timestamp=i) for i in t])
    plt.show()
