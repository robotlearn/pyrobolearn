# This file provides the most common probability distributions

# References:
# [1] pyrobolearn/models/gaussian
# [2] Distributions in pytorch: https://pytorch.org/docs/stable/distributions.html
# [3] https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/distributions.py

import torch
import torch.nn as nn
import torch.nn.functional as F


FixedCategorical = torch.distributions.Categorical
FixedCategorical.sample = lambda self: FixedCategorical.sample(self).unsqueeze(-1)
FixedCategorical.log_probs = lambda self, actions: FixedCategorical.log_prob(self, actions.squeeze(-1)).unsqueeze(-1)
FixedCategorical.mode = lambda self: self.probs.argmax(dim=1, keepdim=True)

Normal = torch.distributions.Normal
Normal.log_probs = lambda self, actions: Normal.log_prob(self, actions).sum(-1, keepdim=True)
Normal.entropy = lambda self: Normal.entropy(self).sum(-1)
Normal.mode = lambda self: self.mean

MVN = torch.distributions.MultivariateNormal
MVN.log_probs = lambda self, actions: MVN.log_prob(self, actions)
MVN.mode = lambda self: self.mean


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


# https://github.com/openai/baselines/blob/master/baselines/common/tf_util.py#L87
def init_normc_(weight, gain=1):
    # initialize the weights
    weight.normal_(0, 1)
    weight *= gain / torch.sqrt(weight.pow(2).sum(1, keepdim=True))


class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)
        return x + bias


class Categorical(nn.Module):
    r"""Categorical distribution

    """
    def __init__(self, num_inputs, num_outputs):
        super(Categorical, self).__init__()

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=0.01)

        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x):
        x = self.linear(x)
        return FixedCategorical(logits=x)


class DiagonalGaussian(nn.Module):
    r"""Diagonal Gaussian distribution

    This multivariate gaussian distribution has a diagonal covariance matrix, that is, the variables are independent
    between each other.
    """
    def __init__(self, num_inputs, num_outputs):
        super(DiagonalGaussian, self).__init__()

        init_ = lambda m: init(m, init_normc_, lambda x: nn.init.constant_(x, 0))

        self.fc_mean = init_(nn.Linear(num_inputs, num_outputs))
        self.logstd = AddBias(torch.zeros(num_outputs))

    def forward(self, x):
        action_mean = self.fc_mean(x)

        #  An ugly hack for my KFAC implementation.
        zeros = torch.zeros(action_mean.size())
        if x.is_cuda:
            zeros = zeros.cuda()

        action_logstd = self.logstd(zeros)
        return Normal(action_mean, action_logstd.exp())


class FixedDiagonalMVN(nn.Module):
    r"""Fixed Diagonal Multivariate Normal

    """
    def __init__(self, num_outputs, variance=1.):
        super(FixedDiagonalMVN, self).__init__()
        self.cov = torch.diag(variance * torch.ones(num_outputs))

    def forward(self, x):
        return MVN(x, covariance_matrix=self.cov)
