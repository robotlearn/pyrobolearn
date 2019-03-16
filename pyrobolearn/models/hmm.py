# This file describes the Hidden Markov Model

from gaussian import Gaussian
from model import Model
from hmmlearn.hmm import GaussianHMM


class HMM(object):
    r"""Hidden Markov Models

    Description: emission probabilities, transition probabilities,...

    References:
        [1] "Pattern Recognition and Machine Learning" (chap 13), Bishop, 2006

    The code was inspired by the following codes:
    * `hmmlearn`: https://github.com/hmmlearn/hmmlearn
    * `ghmm`: http://ghmm.sourceforge.net/
    * `pbdlib`: https://gitlab.idiap.ch/rli/pbdlib-python/tree/master/pbdlib
    """

    def __init__(self, emission_prob=None):
        if emission_prob is None or (isinstance(emission_prob, str) and emission_prob.lower() == 'gaussian'):
            emission_prob = Gaussian()

    ##############
    # Properties #
    ##############


    ##################
    # Static Methods #
    ##################

    @staticmethod
    def copy(other):
        if not isinstance(other, HMM):
            raise TypeError("Trying to copy an object which is not a HMM")

    @staticmethod
    def isParametric():
        """The HMM is a parametric model"""
        return True

    @staticmethod
    def isLinear():
        """The HMM is a non-linear model"""
        return True

    @staticmethod
    def isRecurrent():
        """The HMM is recurrent; current outputs depends on previous inputs and states"""
        return True

    @staticmethod
    def isProbabilistic():
        """The HMM a probabilistic model"""
        return False

    @staticmethod
    def isDiscriminative():
        """The HMM is a discriminative model"""
        return False

    @staticmethod
    def isGenerative():
        """The HMM is a generative model which models the joint distributions on states and outputs.
        This means we can sample from it."""
        return True

    ###########
    # Methods #
    ###########

    def likelihood(self):
        pass

    # alias
    pdf = likelihood

    def joint_pdf(self, X, Z):
        pass

    def sample(self, size=None, seed=None):
        """
        Sample from the HMM.

        Args:
            size:
            seed:

        Returns:

        """
        pass

    def expectation_step(self):
        """
        Expectation step in the expectation-maximization algorithm.

        Returns:

        """
        pass

    def maximization_step(self):
        pass

    def expectation_maximization(self, X):
        """Expectation-Maximization (EM) algorithm"""
        pass

    def forward_backward(self):
        """Forward backward algorithm"""
        pass

    def sum_product(self):
        """Sum-product algorithm"""
        pass

    def viterbi(self):
        """Viterbi algorithm"""
        pass


class HSMM(HMM):
    r"""Hidden semi-Markov Models

    """
    pass