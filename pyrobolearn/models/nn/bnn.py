#!/usr/bin/env python
r"""Provide Bayesian Neural Networks.

This file provides Bayesian neural networks which add uncertainty to the the prediction of NNs. In a bayesian neural
network, the weights and biases have a prior probability distribution attached to them. The posterior distribution on
these parameters is computed after training the model on some data.

Several approaches to learning Bayesian neural networks have been proposed in the literature:
- Laplace approximation [5,4]
- Monte Carlo
- MC Dropout [8,9]
- Variational Inference (Bayes by Backprop [7,10])

References:
    [1] "Deep Learning" (http://www.deeplearningbook.org/), Goodfellow et al., 2016
    [2] PyTorch: https://pytorch.org/
    [3] Pyro: https://pyro.ai/
    [4] "Pattern Recognition and Machine Learning" (section 5.7), Bishop, 2006
    [5] "Practical Bayesian Framework for Backpropagation Networks", MacKay, 1992
        (https://authors.library.caltech.edu/13793/1/MACnc92b.pdf)
    [6] "Bayesian Learning for Neural Networks" (PhD thesis), Neal, 1995
        (http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.446.9306&rep=rep1&type=pdf)
    [7] "Weight Uncertainty in Neural Networks", Blundell et al., 2015 (https://arxiv.org/abs/1505.05424)
    [8] "Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning", Gal et al., 2015
        (https://arxiv.org/abs/1506.02142  and   appendix: https://arxiv.org/abs/1506.02157)
    [9] "Uncertainty in Deep Learning" (PhD thesis), Gal, 2016
    [10] "A Comprehensive guide to Bayesian Convolutional Neural Network with Variational Inference", Shridhar et al.,
        2019 (https://arxiv.org/pdf/1901.02731.pdf)
    [11] Blog post: "Making your Neural Network Say 'I Don't Know' - Bayesian NNs using Pyro and PyTorch":
        https://towardsdatascience.com/making-your-neural-network-say-i-dont-know-bayesian-nns-using-pyro-and-pytorch\
        -b1c24e6ab8cd

Interesting implementations:
- https://github.com/kumar-shridhar/PyTorch-BayesianCNN
- https://github.com/anassinator/bnn
- https://github.com/paraschopra/bayesian-neural-network-mnist
"""

import torch

from pyrobolearn.models.nn.dnn import NN

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


# TODO: implement
class BNN(NN):
    r"""Bayesian Neural Networks"""
    pass
