# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""Define the Generative Adversarial Network (GAN) learning model.

This file provides the GAN model; a parametric, generally non-linear, non-recurrent, generative, and stochastic model.
This is a generative model which works in a game theory setting by having a generator and discriminator compete
between each other. The goal of the generator is to generate data samples that are similar to the provided dataset
and fool the discriminator. The goal of the discriminator is to discriminate the given samples by identifying the fake
ones from the true ones.

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


class GAN(NN):
    r"""Generative Adversarial Network

    Type: generative model

    Generative adversarial networks are two networks, a generator and a discriminator competing against each other.
    The generator tries to produce fake data that are similar to the real data such that the discriminator can not
    distinguished the fake data from the real one [1,2]. Once trained the generator is trained to generate samples
    that are similar to the original data distribution.

    Several GAN models implemented in PyTorch are provided in [3].

    Note that with multiple layers, the networks become sensitive to the initial values of the weights and will fail
    to train. This problem can be solved by using batch normalization [2] for the layers (except the output layer for
    the generator and the input layer of the discriminator). The use of Leaky ReLU for the layers in the discriminator
    is also advised in order to propagate the gradients for negative values. The use of ReLU or Leaky ReLU is advised
    for the generator except the last layer which has been shown to perform the best with a tanh layer. The last layer
    for the discriminator should be a sigmoid function with 0 indicating that the data is fake and 1 the data comes
    from the real data distribution.

    The loss being minimized for the discriminator is the sum of the losses for real and fake images using the sigmoid
    cross-entropy loss for each one of them. That is, the predicted logit outputs of the discriminator on the real
    data should be as close as possible to the label 1, while on the fake data the labels should be 0.
    As for the generator, the loss being optimized is the sigmoid cross entropy loss taken on the discriminator output
    logits but such that the labels are 1 for the fake data.
    With these losses, it can be seen that the generator is trying to fool the discriminator while the discriminator
    is trying to distinguish between the real and fake data.

    .. seealso:: Variational Auto-Encoders

    References:
        [1] "Generative Adversarial Networks", Goodfellow et al., 2014
        [2] "Improved Techniques for Training GANs", Salimans et al., 2016
        [3] "PyTorch Implementations of GANs", Linder-Noren, 2018
    """

    def __init__(self, generator, discriminator, input_shape, output_shape):
        self.generator = generator
        self.discriminator = discriminator
        model = torch.nn.Sequential(generator, discriminator)
        super(GAN, self).__init__(model, input_shape, output_shape)

    ##################
    # Static methods #
    ##################

    @staticmethod
    def is_discriminative():
        """A neural network is a discriminative model which given inputs predicts some outputs"""
        return True

    @staticmethod
    def is_generative():  # unless VAE, GAN,...
        """Standard neural networks are not generative, and thus we can not sample from it. This is different,
        for instance, for generative adversarial networks (GANs) and variational auto-encoders (VAEs)."""
        return True

    ###########
    # Methods #
    ###########

    def forward(self, *inputs):
        """The inputs is discarded."""
        x = self.generate_latent(*inputs)
        x = self.generate(x)
        x = self.discriminator(x)
        return x

    def generate_latent(self, sample_shape=None):
        """
        Generate latent vector / matrix.

        Args:
            sample_shape (tuple of int, None): shape of the latent vectors to generate.

        Returns:
            torch.Tensor: latent vector / matrix.
        """
        if sample_shape is None:
            sample_shape = self.input_shape
        return torch.randn(*sample_shape)

    def generate(self, latent):
        """Generate data.

        Args:
            latent (torch.Tensor): latent vector / matrix.

        Returns:
            torch.Tensor: fake data generated by the generator
        """
        # generate the fake data
        x = self.generator(latent)
        return x

    def discriminate(self, data):
        """Discriminate the given data.

        Args:
            data (torch.Tensor): fake or real data to distinguish.

        Returns:
            torch.Tensor: value between 0 and 1.
        """
        return self.discriminator(data)


class MLP_GAN(GAN):
    r"""Generative Multi-Layer Perceptron.

    This implements Generative Adversarial Networks (GANs) using Multi-Layer Perceptrons (MLPs).
    Generative adversarial networks are two networks, a generator and a discriminator competing against each other.
    The generator tries to produce fake data that are similar to the real data such that the discriminator can not
    distinguished the fake data from the real one [1,2]. Once trained the generator is trained to generate samples
    that are similar to the original data distribution.

    Several GAN models implemented in PyTorch are provided in [3].

    Note that with multiple layers, the networks become sensitive to the initial values of the weights and will fail
    to train. This problem can be solved by using batch normalization [2] for the layers (except the output layer for
    the generator and the input layer of the discriminator). The use of Leaky ReLU for the layers in the discriminator
    is also advised in order to propagate the gradients for negative values. The use of ReLU or Leaky ReLU is advised
    for the generator except the last layer which has been shown to perform the best with a tanh layer. The last layer
    for the discriminator should be a sigmoid function with 0 indicating that the data is fake and 1 the data comes
    from the real data distribution.

    References:
        [1] "Generative Adversarial Networks", Goodfellow et al., 2014
        [2] "Improved Techniques for Training GANs", Salimans et al., 2016
        [3] "PyTorch Implementations of GANs", Linder-Noren, 2018
    """

    def __init__(self, generator_units=[], discriminator_units=[],
                 generator_activation='LeakyReLU', generator_last_activation='tanh',
                 discriminator_activation='LeakyReLU', discriminator_last_activation='sigmoid',
                 use_batch_norm=False):
        """
        Initialize the MLP-GAN.

        Args:
            generator_units (list/tuple of int): number of units per layer in the generator, including the data input
                dimension.
            discriminator_units (list/tuple of int): number of units per layer in the discriminator. The last unit
                must be one. If it is not the case, the one will automatically be added.
            generator_activation (str, torch.nn.Module, None): activation function to be used at each layer in the
                generator. They can be found in `torch.nn.modules.activation.*`. If None, it will use 'LeakyReLU'.
            generator_last_activation (str, torch.nn.Module, None): last activation function to be used on the output
                of the generator. By default, it will use 'tanh', so the output is between -1 and 1.
            discriminator_activation (str, torch.nn.Module): activation function to be used at each layer in the
                discriminator. They can be found in `torch.nn.modules.activation.*`. by default, it will use
                'LeakyReLU'.
            discriminator_last_activation (str, torch.nn.Module, None): last activation function to be used on the
                output of the discriminator. By default, it is the sigmoid where 1 means that the discriminator thinks
                that the data comes from the real data distribution while 0 means that it comes from the
            use_batch_norm (bool): If batch normalization should be used.
        """
        # check the generator length
        if len(generator_units) < 2:
            raise ValueError("Expecting more than the input layer for the generator.")

        # check discriminator units.
        if discriminator_units is None:
            discriminator_units = generator_units[::-1]  # reverse

        # the last output unit should be 1 for the sigmoid
        if discriminator_units[-1] != 1:
            discriminator_units = discriminator_units + [1]

        # if the number of units in the first layer of the discriminator is not the same as the number of units
        # in the last layer of the generator, just append it
        if discriminator_units[0] != generator_units[-1]:
            discriminator_units = generator_units[-1:] + discriminator_units

        if len(discriminator_units) == 0:
            raise ValueError("Expecting at least the output layer for the discriminator.")

        units = generator_units + discriminator_units[1:]

        # check for activation fcts
        activations = dir(torch.nn.modules.activation)
        activations = {act: act for act in activations}
        activations.update({act.lower(): act for act in activations})

        def check_activation(activation):
            if activation is None or activation.lower() == 'linear':
                activation = None
            elif activation in activations:
                activation = getattr(torch.nn, activations[activation])
            elif callable(activation):
                pass
            else:
                raise ValueError("The given activation function is not available")
            return activation

        generator_activation = check_activation(generator_activation)
        discriminator_activation = check_activation(discriminator_activation)
        generator_last_activation = check_activation(generator_last_activation)
        discriminator_last_activation = check_activation(discriminator_last_activation)

        # build network
        def build_layers(units, activation, last_activation):
            layers = []

            for i in range(len(units[:-2])):
                # add linear layer
                layer = torch.nn.Linear(units[i], units[i + 1])
                layers.append(layer)

                # if use batch normalization
                if use_batch_norm:
                    layers.append(torch.nn.BatchNorm1d(num_features=units[i + 1]))

                # add activation layer
                if activation is not None:
                    layers.append(activation())

            # last output layer
            layers.append(torch.nn.Linear(units[-2], units[-1]))
            if last_activation is not None:
                layers.append(last_activation())

            return layers

        # create generator and discriminator layers
        generator_layers = build_layers(generator_units, generator_activation, generator_last_activation)
        discriminator_layers = build_layers(discriminator_units, discriminator_activation,
                                            discriminator_last_activation)

        # create generator and discriminator networks
        generator = torch.nn.Sequential(*generator_layers)
        discriminator = torch.nn.Sequential(*discriminator_layers)

        super(MLP_GAN, self).__init__(generator, discriminator, input_shape=tuple([units[0]]),
                                      output_shape=tuple([units[-1]]))


class DCGAN(GAN):
    r"""Deep Convolutional Generative Adversarial Network.

    References:
        [1] "Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks", Radford et
            al., 2015
        [2] "DCGAN Tutorial": https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
        [3] "PyTorch Implementations of GANs", Linder-Noren, 2018
    """

    def __init__(self, generator_units=[], discriminator_units=[],
                 generator_activation='LeakyReLU', generator_last_activation='tanh',
                 discriminator_activation='LeakyReLU', discriminator_last_activation='sigmoid',
                 use_batch_norm=True):
        """
        Initialize the DCGAN.

        Args:
            generator_units (list/tuple of int): number of units per layer in the generator, including the data input
                dimension.
            discriminator_units (list/tuple of int): number of units per layer in the discriminator.
            generator_activation (str, torch.nn.Module, None): activation function to be used at each layer in the
                generator. They can be found in `torch.nn.modules.activation.*`. If None, it will use 'LeakyReLU'.
            generator_last_activation (str, torch.nn.Module, None): last activation function to be used on the output
                of the generator. If None, it will use 'tanh', so the output is between -1 and 1.
            discriminator_activation (str, torch.nn.Module): activation function to be used at each layer in the
                discriminator. They can be found in `torch.nn.modules.activation.*`. If None, it will use 'LeakyReLU'.
            discriminator_last_activation (str, torch.nn.Module, None): last activation function to be used on the
                output of the discriminator. By default, it is the sigmoid where 1 means that the discriminator thinks
                that the data comes from the real data distribution while 0 means that it comes from the
            use_batch_norm (bool): If batch normalization should be used.
        """
        # check the generator length
        if len(generator_units) < 2:
            raise ValueError("Expecting more than the input layer for the generator.")

        # check discriminator units.
        if discriminator_units is None:
            discriminator_units = generator_units[::-1]  # reverse

        # the last output unit should be 1 for the sigmoid
        if discriminator_units[-1] != 1:
            discriminator_units = discriminator_units + [1]

        # if the number of units in the first layer of the discriminator is not the same as the number of units
        # in the last layer of the generator, just append it
        if discriminator_units[0] != generator_units[-1]:
            discriminator_units = generator_units[-1:] + discriminator_units

        if len(discriminator_units) == 0:
            raise ValueError("Expecting at least the output layer for the discriminator.")

        units = generator_units + discriminator_units[1:]

        # check for activation fcts
        activations = dir(torch.nn.modules.activation)
        activations = {act: act for act in activations}
        activations.update({act.lower(): act for act in activations})

        def check_activation(activation):
            if activation is None or activation.lower() == 'linear':
                activation = None
            elif activation in activations:
                activation = getattr(torch.nn, activations[activation])
            elif callable(activation):
                pass
            else:
                raise ValueError("The given activation function is not available")
            return activation

        generator_activation = check_activation(generator_activation)
        discriminator_activation = check_activation(discriminator_activation)
        generator_last_activation = check_activation(generator_last_activation)
        discriminator_last_activation = check_activation(discriminator_last_activation)

        # keep track of (C, H, W) dimensions
        if len(units[0]) == 2:
            channel = 1
        else:
            channel = units[0][0]

        # build network
        def build_layers(units, activation, last_activation, generator=True):
            layers = []

            for i in range(1, len(units)):

                # currently we only support convolution layers
                if not isinstance(units[i], (tuple, list)):
                    raise ValueError("Expecting each unit to be a list or tuple of 2/3 ints for the `torch.nn.Conv*` "
                                     "and `torch.nn.ConvTranspose*`, instead got: {}".format(units[i]))

                # add (transposed) convolution layer based on if we have a generator or discriminator
                unit = units[i]
                if len(units[i]) == 2:
                    unit = (channel,) + units[i]

                if generator:  # generator
                    layer = torch.nn.Conv2d(*unit)
                else:  # discriminator
                    layer = torch.nn.ConvTranspose2d(*unit)
                layers.append(layer)

                # compute new dimensions
                channel = layer.out_channels
                # p, d, k, s = layer.padding, layer.dilation, layer.kernel_size, layer.stride
                # height = int(math.floor((height + 2. * p[0] - d[0] * (k[0] - 1) - 1.) / s[0] + 1))
                # width = int(math.floor((width + 2. * p[1] - d[1] * (k[1] - 1) - 1.) / s[1] + 1))

                # if use batch normalization
                if use_batch_norm:
                    layers.append(torch.nn.BatchNorm2d(num_features=unit[1]))

                # add activation layer
                if activation is not None:
                    layers.append(activation())

            # last output layer
            if last_activation is not None:
                layers.append(last_activation())

            return layers

        # create generator and discriminator layers
        generator_layers = build_layers(generator_units, generator_activation, generator_last_activation)
        discriminator_layers = build_layers(discriminator_units[:-1], discriminator_activation,
                                            discriminator_last_activation, generator=False)

        # create generator and discriminator networks
        generator = torch.nn.Sequential(*generator_layers)
        discriminator = torch.nn.Sequential(*discriminator_layers)

        super(DCGAN, self).__init__(generator, discriminator, input_shape=units[0][:3], output_shape=(1,))


# Tests
if __name__ == '__main__':
    # create GAN
    gan = MLP_GAN(generator_units=[100, 128, 28*28], discriminator_units=[128, 1])
    print(gan)

    # generate data
    z = gan.generate_latent()
    x = gan.generate(z)
    y = gan.discriminate(x)
    print("Latent vector shape: {}".format(z.shape))
    print("Generator output shape: {}".format(x.shape))
    print("Discriminator output: {}".format(y))

    dcgan = DCGAN(generator_units=[(100, 8*64, 4, 1, 0), (8*64, 8*4, 4, 2, 1), (4*64, 2*64, 4, 2, 1),
                                   (2*64, 64, 4, 2, 1), (64, 3, 4, 2, 1)],
                  discriminator_units=[(3, 64, 4, 2, 1), (64, 2*64, 4, 2, 1), (2*64, 4*64, 4, 2, 1),
                                       (4*64, 8*64, 4, 2, 1), (8*64, 1, 4, 1, 0)])
    print(dcgan)
