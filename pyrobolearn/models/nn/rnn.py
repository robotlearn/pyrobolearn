#!/usr/bin/env python
"""Define the Recurrent Neural Network (RNN) learning model.

This file provides the RNN model; a parametric, generally non-linear, recurrent, discriminative,
and deterministic model. This model is convenient for sequential data. For instance, they can be used with language,
where each word in a sentence is conditioned on the previous words.

We decided to use the `pytorch` framework because of its popularity in the research community field, flexibility,
similarity with numpy (but with automatic differentiation: autograd), GPU capabilities, and more Pythonic approach.
While we thought about using other frameworks (such as Keras, Tensorflow, and others) as well, it would have
unnecessarily complexify the whole framework, as these frameworks would not only influence the learning models,
but also the losses, optimizers, and other modules. While we could have written some interfaces that makes the bridge
between these various frameworks and ours, we came to the conclusion that this would take a considerable amount of
efforts and time that we do not have for the moment.

References:
    [1] "Deep Learning" (http://www.deeplearningbook.org/), Goodfellow et al., 2016
    [2] PyTorch: https://pytorch.org/
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


class RNN(NN):
    r"""Recurrent Neural Network

    A recurrent neural network is a network that captures the sequential nature / aspect of the data. It possesses
    an internal memory (i.e. internal state) which is returned at the input.

    References:
        [1] "Deep Learning" (http://www.deeplearningbook.org/), Goodfellow et al., 2016
    """

    def __init__(self, model, input_shape, output_shape):
        super(RNN, self).__init__(model, input_shape, output_shape)

    ##################
    # Static methods #
    ##################

    @staticmethod
    def is_recurrent():  # unless RNN
        """RNNs are recurrent models."""
        return True

    ###########
    # Methods #
    ###########

    def forward(self, inputs):
        pass


class MLP_RNN(RNN):
    r"""Recurrent Multi-layer Perceptron using Elman RNNs.

    This uses `torch.nn.RNN` and `torch.nn.RNNCell`.

    References:
        [1]  "Finding structure in time.", Elman, 1990
        [2] "Deep Learning" (http://www.deeplearningbook.org/), Goodfellow et al., 2016
    """

    def __init__(self, units=(), activation=None, last_activation=None, dropout=None):
        """
        Initialize a recurrent MLP network.

        Args:
            units (list/tuple of int): number of units in each layer (this includes the input and output layer)
            activation (None, str, or list/tuple of str/None): activation function to be applied after each layer.
                                                                   If list/tuple, then it has to match the number of
                                                                   hidden layers. If None, it is a linear layer.
            last_activation (None or str): last activation function to be applied. If not specified, it will check
                                               if it is in the list/tuple of activation functions provided for the
                                               previous argument.
            dropout (None, float, or list/tuple of float/None): dropout probability.
        """
        pass
        # super(MLP_RNN, self).__init__(model, input_shape=tuple([units[0]]), output_shape=tuple([units[-1]]))


class MLP_LSTM(RNN):
    r"""Recurrent Multi-Layer Perceptron using Long-Short Term Memories (LSTMs).

    This uses `torch.nn.LSTM` and `torch.nn.LSTMCell`.

    References:
        [1] "Long Short Term Memory", Hochreiter et al., 1997
        [2] "Deep Learning" (http://www.deeplearningbook.org/), Goodfellow et al., 2016
    """

    def __init__(self, units=(), activation=None, last_activation=None, dropout=None):
        pass
        # super(MLP_LSTM, self).__init__(model, input_shape, output_shape)


class MLP_GRU(RNN):
    r"""Recurrent Multi-Layer Perceptron using Gated Recurrent Units (GRUs).

    This uses `torch.nn.GRU` and `torch.nn.GRUCell`.

    References:
        [1] "Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation", Cho et al.,
            2014
        [2] "Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling", Chung et al., 2014
        [3] "Deep Learning" (http://www.deeplearningbook.org/), Goodfellow et al., 2016
    """

    def __init__(self, model, input_shape, output_shape):
        pass
        # super(MLP_GRU, self).__init__(model, input_shape, output_shape)


# Tests
if __name__ == '__main__':
    pass
    # # create Recurrent MLP network
    # mlp_rnn = MLP_RNN(units=(2, 10, 3), activation='relu')
    # mlp_lstm = MLP_LSTM(num_units=(2, 10, 3), activation_fct='relu')
    # mlp_gru = MLP_GRU(num_units=(2,10,3), activation_fct='relu')
    #
    # print("RNN: {}".format(mlp_rnn))
    # print("LSTM: {}".format(mlp_lstm))
    # print("GRU: {}".format(mlp_gru))
    #
    # x = torch.rand(2)
    #
    # for t in range(3):
    #     y = mlp_rnn.forward(x)
    #     print("RNN: Input at t{}: {} - Output: {}".format(t, x, y))
    #     y = mlp_lstm.forward(x)
    #     print("LSTM: Input at t{}: {} - Output: {}".format(t, x, y))
    #     y = mlp_gru.forward(x)
    #     print("GRU: Input at t{}: {} - Output: {}".format(t, x, y))
