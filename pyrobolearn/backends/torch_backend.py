
import torch
from torch import *


def array(data, dtype=None, copy=True, device=None, requires_grad=False, ndmin=0):
    return torch.tensor(data, dtype=dtype, device=device, requires_grad=requires_grad)


def inv(data, out=None):
    return torch.inverse(data, out=out)


def concatenate(data, axis=0, out=None):
    return torch.cat(data, axis, out)

# np.size vs torch.nelement() vs torch.size()

