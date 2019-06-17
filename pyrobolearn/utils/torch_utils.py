#!/usr/bin/env python
"""Provide few PyTorch utils methods.

For instance, it can evaluate the Hessian matrix of a scalar function.

References:
    [1] https://pytorch.org/
    [2] https://github.com/mariogeiger/hessian
    [3] https://github.com/Ageliss/For_shared_codes/blob/master/Second_order_gradients.py
"""

import torch
from torch.autograd import grad

# import hessian


__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


def kronecker(A, B):
    """
    Return the kronecker product between two matrices [1].

    Args:
        A (torch.Tensor): 1st matrix.
        B (torch.Tensor): 2nd matrix.

    Returns:
        torch.Tensor: kronecker product between two matrices

    References:
        [1] Kronecker product (Wikipedia): https://discuss.pytorch.org/t/kronecker-product/3919/7
        [2] https://discuss.pytorch.org/t/kronecker-product/3919/7
        [3] Another implementation:
            https://github.com/cornellius-gp/gpytorch/blob/master/gpytorch/lazy/kronecker_product_lazy_tensor.py
    """
    return torch.einsum("ab,cd->acbd", A, B).view(A.size(0)*B.size(0),  A.size(1)*B.size(1))


def hessian(f, model):
    r"""
    Get the Hessian matrix of the model.

    Args:
        f:
        model:

    Returns:
        torch.Tensor: Hessian matrix (which is a square symmetric and positive-definite real matrix)

    References:
        [1] https://github.com/mariogeiger/hessian
        [2] https://github.com/Ageliss/For_shared_codes/blob/master/Second_order_gradients.py
    """
    pass


def hessian_vector(f, model):
    r"""
    Get the Hessian-vector product.

    Args:
        f:
        model:

    Returns:
        torch.Tensor: vector

    References:
        [1] https://github.com/pytorch/pytorch/releases/tag/v0.2.0
    """
    pass
