#!/usr/bin/env python
"""Provide the various memory-augmented neural networks (MANNs).

This file provides the various MANNs; a parametric, generally non-linear, recurrent with memory, discriminative,
and deterministic model.

These kinds of model includes "Neural Turing Machines".

The code presented here is mostly inspired by [4, 5, 6, 7] and sometimes loosely inspired by [8,9].

References:
    [1] "Deep Learning" (http://www.deeplearningbook.org/), Goodfellow et al., 2016
    [2] PyTorch: https://pytorch.org/
    [3] "Neural Turing Machines", Graves et al., 2014
    [4] "Meta-Learning with Memory-Augmented Neural Networks", Santoro et al., 2016
    [5] "One-shot Learning with Memory-Augmented Neural Networks", Santoro et al., 2016
    [6] http://rylanschaeffer.github.io/content/research/neural_turing_machine/main.html
    [7] https://rylanschaeffer.github.io/content/research/one_shot_learning_with_memory_augmented_nn/main.html
    [8] PyTorch implementation of NTM: https://github.com/loudinthecloud/pytorch-ntm
    [9] PyTorch implementation of NTM (based on [8]): https://github.com/vlgiitr/ntm-pytorch

Existing implementations that might interest other users:
- https://github.com/loudinthecloud/pytorch-ntm
- https://github.com/vlgiitr/ntm-pytorch
"""

import torch

from pyrobolearn.models.nn.dnn import NN


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


def circular_convolve(weight, shift):
    r"""
    Perform a circular convolution. Taken from [1].

    Args:
        weight (torch.Tensor): weight vector.
        shift (torch.Tensor): shift vector.

    Returns:
        torch.Tensor: convolved weight vector.

    References:
        [1] https://github.com/loudinthecloud/pytorch-ntm/blob/master/ntm/memory.py
    """
    aug_weight = torch.cat([weight[-1:], weight, weight[:1]])
    conv = torch.conv1d(aug_weight.view(1, 1, -1), shift.view(1, 1, -1)).view(-1)
    return conv


class Kernel(torch.nn.Module):
    r"""Kernel

    Measure the similarity between two tensors.
    """

    def __init__(self, kernel):
        super(Kernel, self).__init__()
        if not callable(kernel):
            raise TypeError("Expecting the given 'kernel' to be callable.")
        self.kernel = kernel

    def forward(self, memory, output):
        return self.kernel(memory, output)


class CosineSimilarityKernel(Kernel):
    r"""Cosine Similarity Kernel

    Measure the similarity between two tensors using the cosine similarity.
    """

    def __init__(self):
        # define closure
        def kernel(dim=-1):
            def cosine_similarity(memory, output):
                return torch.cosine_similarity(memory, output, dim=dim)
            return cosine_similarity

        # create kernel
        kernel = kernel()
        super(CosineSimilarityKernel, self).__init__(kernel)


class Memory(object):
    """Memory.

    Memory used in MANNs. It often consists of a 2D matrix.

    References:
        [1] "Neural Turing Machines", Graves et al., 2014
        [2] "Meta-Learning with Memory-Augmented Neural Networks", Santoro et al., 2016
        [3] "One-shot Learning with Memory-Augmented Neural Networks", Santoro et al., 2016
        [4] http://rylanschaeffer.github.io/content/research/neural_turing_machine/main.html
        [5] https://rylanschaeffer.github.io/content/research/one_shot_learning_with_memory_augmented_nn/main.html
    """

    def __init__(self, shape=(), memory=None):
        """
        Initialize the memory matrix module.

        Args:
            shape (tuple of int): shape of the 2D memory matrix. This has to be specified if no memory matrix/module
                is given.
            memory (torch.Tensor, Memory, None): 2D memory matrix. If None, it will just create one based on the given
                shape.
        """
        self._shape = shape
        self.memory = memory

    ##############
    # Properties #
    ##############

    @property
    def memory(self):
        """Return the memory matrix."""
        return self._memory

    @memory.setter
    def memory(self, memory):
        """Set the memory matrix."""
        if memory is None:
            memory = torch.zeros(self.shape)
        if not isinstance(memory, torch.FloatTensor):
            raise TypeError("Expecting the given 'memory' to be an instance of `torch.FloatTensor`, instead got: "
                            "{}".format(type(memory)))
        if len(memory.shape) != 2:
            raise ValueError("Expecting the given 'memory' to be a 2D matrix, instead got shape: "
                             "{}".format(memory.shape))
        self._memory = memory
        self._shape = memory.shape

    @property
    def shape(self):
        """Return the shape of the memory matrix module."""
        return self._shape

    ###########
    # Methods #
    ###########

    def read(self, address):
        r"""
        Read the content in the memory using an attention mechanism (i.e. a normalized weighted vector which describes
        the importance of each row in the memory). This is mathematically given by:

        .. math:: r_t \leftarrow \sum_i^R w_t[i] M_t[i]

        where :math:`M_t` is a 2D matrix of shape (R,C) representing the memory module, and :math:`M_t[i]` is the
        `i`th row of that matrix, :math:`r_t` is the returned read vector of shape (C,), and :math:`w_t` is the
        normalized weight vector of length `R` which is used to access the rows in the memory matrix using the
        attention mechanism.

        Args:
            address (torch.Tensor): normalized weight vector of length R which is used to access the rows in the memory
                module using the attention mechanism.

        Returns:
            torch.Tensor: read vector
        """
        # normalize the weight vector if not normalized
        if bool(torch.sum(address) != 1):
            address = address / torch.sum(address)
        return torch.sum(address.view(-1, 1) * self._memory, dim=0)

    def write(self, address, content, erase=None):
        r"""
        Write the given vector at the specified address.

        The writing operation consists of an erasing and adding stages.

        Args:
            address (torch.Tensor): normalized weight vector of length R which is used to access the rows in the memory
                module using the attention mechanism.
            content (torch.Tensor): write vector of length C to be incorporated in the memory matrix. This vector is
                often the one outputted by the NN controller.
            erase (torch.Tensor, None): erase vector of length C used in conjunction with the address/weight vector to
                specify which elements in a row should be erased. If None, it will not erase anything in the memory.
        """
        # if specified, clean / erase the content in the specified address
        if erase is not None:
            self.erase(address, erase)
        # add content at the specified address
        self.add(address, content)

    def erase(self, address, erase):
        r"""
        Erase parts of the memory using the specified erase vector at the specified soft-addresses.

        .. math:: M_t[i] \leftarrow M_{t-1}[i] (1 - w_t[i] e_t)

        where :math:`M_t` is a 2D matrix of shape (R,C) representing the memory module, and :math:`M_t[i]` is the
        `i`th row of that matrix, :math:`w_t` is the normalized weight vector of length `R` which is used to access
        the rows in the memory matrix using the attention mechanism, and :math:`e_t` is the erase vector of length
        `C`.

        Args:
            address (torch.Tensor): normalized weight vector of length R which is used to access the rows in the memory
                module using the attention mechanism.
            erase (torch.Tensor): erase vector of length C used in conjunction with the address/weight vector to
                specify which elements in a row should be erased.
        """
        self._memory = self._memory - torch.ger(address, erase) * self._memory

    def add(self, address, content):
        r"""
        Add the given vector at the specified soft-addresses.

        .. math:: M_t[i] \leftarrow M^{erased}_t[i] + w_t[i] a_t

        where :math:`M_t` is a 2D matrix of shape (R,C) representing the memory module, and :math:`M_t[i]` is the
        `i`th row of that matrix, :math:`w_t` is the normalized weight vector of length `R` which is used to access
        the rows in the memory matrix using the attention mechanism, and :math:`a_t` is the write vector of length
        `C`.

        Args:
            address (torch.Tensor): normalized weight vector of length R which is used to access the rows in the memory
                module using the attention mechanism.
            content (torch.Tensor): write vector of length C to be incorporated in the memory matrix. This vector is
                often the one outputted by the NN controller.
        """
        self._memory = self._memory + torch.ger(address, content)


class MemoryAccess(object):
    """Memory access.

    Describe how to access parts of the memory, that is it provides the address in the memory.

    Warnings: more sophisticated memory access capabilities are more powerful, but the controller requires more
    training.
    """

    def __init__(self, memory):
        """
        Initialize the memory access.

        Args:
            memory (Memory): memory instance.
        """
        self.memory = memory

    @property
    def memory(self):
        """Return the memory instance."""
        return self._memory

    @memory.setter
    def memory(self, memory):
        """Set the memory instance."""
        if not isinstance(memory, Memory):
            raise TypeError("Expecting the given 'memory' to be an instance of `Memory`, instead got: "
                            "{}".format(type(memory)))
        self._memory = memory

    def address(self, *args, **kwargs):
        """To be implemented in the child class."""
        pass


class ContentBasedAccess(MemoryAccess):
    r"""Content-based addressing.

    Access specific memory values using a similarity measure.

    Warnings: more sophisticated memory access capabilities are more powerful, but the controller requires more
    training.
    """

    def __init__(self, memory, kernel=None):
        """
        Initialize the content-based access.

        Args:
            memory (Memory): memory.
            kernel (Kernel): kernel that computes the similarity between two tensors.
        """
        super(ContentBasedAccess, self).__init__(memory)

        # set the kernel
        if kernel is None:
            kernel = CosineSimilarityKernel()
        if not isinstance(kernel, Kernel):
            raise TypeError("Expecting the given kernel to be an instance of `Kernel`, instead got: "
                            "{}".format(type(kernel)))
        self.kernel = kernel

    def content_addressing(self, key, strength=1.):
        r"""
        Check how similar each row in the memory is with the given output vector.
        This is the 1st phase in NTM.

        .. math:: w^c_t[i] = \frac{ \exp(\beta_t K(k_t, M_t[i])) }{ \sum_j \exp(\beta_t K(k_t, M_t[j])) }

        where the subscript :math:`t` means the current time step, :math:`w^c_t[i]` is the element of the current
        content weight vector, :math:`beta_t` is the key strength which determines how concentrated the weight vector
        should be, :math:`k_t` is the vector outputted by the NN controller, :math:`M_t[i]` is the `i` row of the
        memory matrix, and :math:`K(\cdot, \cdot)` is a similarity measure function (like a kernel).

        Args:
            key (torch.Tensor): output vector produced by the NN controller.
            strength (float): key strength (> 0) which determines how concentrated the weight vector should be.
                This parameter is also outputted by the NN controller. The condition > 0, can be achieved using the
                softplus function.

        Returns:
            torch.Tensor: content weight/address vector.
        """
        return torch.softmax(strength * self.kernel(self.memory.memory + 1e-16, key + 1e-16), dim=1)

    def address(self, output, key_strength=1):
        """Return the content address."""
        return self.content_addressing(output, key_strength)


class LocationBasedAccess(MemoryAccess):
    r"""Location-based addressing.

    Access specific memory locations.

    Warnings: more sophisticated memory access capabilities are more powerful, but the controller requires more
    training.
    """

    @staticmethod
    def interpolate(prev_weight, weight, gate=1):
        r"""
        Interpolate the previous weight with the given weight using the interpolation gate by polyak averaging.
        This is the 2nd phase in NTM.

        .. math:: w^g_t \leftarrow g_t w^c_t + (1 - g_t) w_{t-1}

        Args:
            prev_weight (torch.Tensor): previous weight vector used to access the memory.
            weight (torch.Tensor): current (content) weight vector.
            gate (float): interpolation gate = polyak scalar coefficient parameter (which is belongs to ]0, 1[). This
                parameter is also outputted by the NN controller. The condition that the gate must be between 0 and 1
                can be achieved using the sigmoid function.

        Returns:
            torch.Tensor: interpolated weight vector.
        """
        return gate * weight + (1 - gate) * prev_weight

    @staticmethod
    def shift(weight, shift):
        r"""
        Convolutional shift: allows for the controller to shift the focus to other rows.
        This is the 3rd phase in NTM.

        .. math:: w^s_t[i] \leftarrow \sum_{j=0}^{R-1} w^g_t[j] s_t[i - j]

        where :math:`s_t[i]` is the `i` element of the normalized shift weighting vector outputted by the NN
        controller.

        Args:
            weight (torch.Tensor): weight vector.
            shift (torch.Tensor): normalized shift weighting vector outputted by the NN controller. In order to be
                normalized, the softmax function can be used.

        Returns:
            torch.Tensor: shifted weight vector.
        """
        return circular_convolve(weight, shift)

    @staticmethod
    def sharpening(weight, gamma=1):
        r"""
        Sharpening phase: prevent the shifted weight from blurring.
        This the 4th and final stage in NTM.

        .. math:: w_t[i] \leftarrow \frac{ w^s_t[i]^{\gamma_t} }{ \sum_j w^s_t[j]^{\gamma_t} }

        Args:
            weight (torch.Tensor): weight vector.
            gamma (float): scalar parameter (>= 1) outputted by the NN controller. The condition >= 1, can be achieved
                using the softplus function and adding 1 to its output.

        Returns:
            torch.Tensor: memory access weight vector.
        """
        w = weight**gamma
        return w / torch.sum(w)

    def address(self, prev_weight, weight, shift, gate=1, gamma=1):
        """
        Return the soft-address.

        Args:
            prev_weight (torch.Tensor): previous weight vector used to access the memory.
            weight (torch.Tensor): current (content) weight vector.
            shift (torch.Tensor): normalized shift weighting vector outputted by the NN controller.
            gate (float): interpolation gate = polyak scalar coefficient parameter (which is belongs to ]0, 1[). This
                parameter is also outputted by the NN controller. The condition that the gate must be between 0 and 1
                can be achieved using the sigmoid function.
            gamma (float): scalar parameter (>= 1) outputted by the NN controller. The condition >= 1, can be achieved
                using the softplus function and adding 1 to its output.

        Returns:
            torch.Tensor: memory address weight vector.
        """
        weight = self.interpolate(prev_weight, weight, gate=gate)
        weight = self.shift(weight, shift)
        weight = self.sharpening(weight, gamma=gamma)
        return weight


class LRUA(MemoryAccess):
    r"""Least-Recent Used Access (LRUA).

    Read: ContentBasedAccess
    Write: LRUA

    References:
        [1] "Meta-Learning with Memory-Augmented Neural Networks", Santoro et al., 2016
        [2] "One-shot Learning with Memory-Augmented Neural Networks", Santoro et al., 2016
        [3] https://rylanschaeffer.github.io/content/research/one_shot_learning_with_memory_augmented_nn/main.html
    """

    def __init__(self, memory):
        """
        Initialize the LRUA.

        Args:
            memory (Memory): memory.
        """
        super(LRUA, self).__init__(memory)
        self.usage_weight_vector = torch.zeros(self.memory.shape[0])  # usage weight vector
        self.read = torch.zeros(self.memory.shape[0])  # previous read-weight vector

    def least_usage_vector(self, n=None):
        r"""Return the least usage vector.

        The least usage vector is a vector containing ones and zeros and is computed according to:

        .. math::

            w^{lu}_t[i] = \left\{ \begin{array}{cc}
                0 & \mbox{if } w^u_t[i] > m(w^u_t, n) \\
                1 & \mbox{if } w^u_t[i] <= m(w^u_t, n)
            \end{array} \right.

        where :math:`m(v, n)` denotes the `n`th smallest element of the vector v.

        Args:
            n (int): nth smallest element of the usage weight vector.
        """
        if n is None or n <= 0:
            value = torch.argmin(self.usage_weight_vector)
        else:
            if n > self.usage_weight_vector.shape[-1] - 1:
                n = self.usage_weight_vector.shape[-1] - 1
            indices = self.usage_weight_vector.sort()[1]
            value = self.usage_weight_vector[indices[n]]
        vector = torch.zeros_like(self.usage_weight_vector)
        vector[self.usage_weight_vector <= value] = 1.
        return vector

    def update(self, read, write, gamma=0.995):
        r"""
        Update the usage weight vector.

        .. math:: w^u_t \leftarrow \gamma w^u_{t-1} + w^r_t + w^w_t

        where :math:`w^u` is the usage weight vector, :math:`w^r_t` is the current read weight vector, and
        :math:`w^w_t` is the current write weight vector.

        Args:
            read (torch.Tensor): current read weight/soft-address vector.
            write (torch.Tensor): current write weight/soft-address vector.
            gamma (float): decay parameter.

        Returns:
            torch.Tensor: current usage weight vector.
        """
        self.usage_weight_vector = gamma * self.usage_weight_vector + read + write
        return self.usage_weight_vector

    def address(self, prev_read, gate, n=0):
        r"""
        Return the soft write address.

        .. math:: w^w_t \leftarrow g_t w^r_{t-1} + (1 - g_t) w^{lu}_{t-1}

        This :math:`w^w_t` is the write weight vector that is then send to the `memory.write()` along with the `key`
        that was also used for the content based reading.

        Args:
            prev_read (torch.Tensor): previous read weight vector used to access the memory.
            gate (float): interpolation gate = polyak scalar coefficient parameter (which is belongs to ]0, 1[). This
                parameter is also outputted by the NN controller. The condition that the gate must be between 0 and 1
                can be achieved using the sigmoid function.

        Returns:
            torch.Tensor: write weight vector.
        """
        return gate * prev_read + (1. - gate) * self.least_usage_vector()


class Head(object):
    pass


class Controller(object):
    pass


class NTM(NN):
    r"""Neural Turing Machines

    This class implements the Neural Turing Machines.
    """

    def __init__(self, controller, ouput_model, input_shape, output_shape, memory, model=None):
        self.memory = memory
        self.controller = controller
        self.output_model = ouput_model

        # combine controller, memory, and output model
        super(NTM, self).__init__(model, input_shape, output_shape)

    @property
    def memory(self):
        """Return the memory instance."""
        return self._memory

    @memory.setter
    def memory(self, memory):
        """Set the memory instance."""
        if isinstance(memory, tuple):
            memory = Memory(shape=memory)
        elif isinstance(memory, torch.Tensor):
            memory = Memory(memory=memory)
        elif not isinstance(memory, Memory):
            raise TypeError("Expecting the given 'memory' module to be a tuple of int describing the shape, a "
                            "`torch.FloatTensor`, or a `Memory`, instead got: {}".format(type(memory)))
        self._memory = memory

    def train(self):
        pass

    def eval(self):
        pass

    def forward(self, x):
        """Forward the inputs."""
        pass


# Tests
if __name__ == '__main__':
    pass
