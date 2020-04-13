'''
Experiment 3, 4

Implemented Adaptive Routing Algorithm (https://arxiv.org/abs/1911.08119)
'''


from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from torch.optim import lr_scheduler
from torch.autograd import Variable


def squash(x):
    lengths2 = x.pow(2).sum(dim=2)
    lengths = lengths2.sqrt()
    x = x * (lengths2 / (1 + lengths2) / lengths).view(x.size(0), x.size(1), 1)
    return x


class AgreementRouting(nn.Module):
    def __init__(self, input_caps, output_caps, n_iterations):
        super(AgreementRouting, self).__init__()
        self.n_iterations = n_iterations
        self.b = nn.Parameter(torch.zeros((input_caps, output_caps)))

    '''
    def forward(self, u_predict):
        batch_size, input_caps, output_caps, output_dim = u_predict.size()

        c = F.softmax(self.b)
        s = (c.unsqueeze(2) * u_predict).sum(dim=1)
        v = squash(s)

        if self.n_iterations > 0:
            b_batch = self.b.expand((batch_size, input_caps, output_caps))
            for r in range(self.n_iterations):
                v = v.unsqueeze(1)
                b_batch = b_batch + (u_predict * v).sum(-1)

                c = F.softmax(b_batch.view(-1, output_caps)).view(-1, input_caps, output_caps, 1)
                s = (c * u_predict).sum(dim=1)
                v = squash(s)

        return v
    '''

    def forward(self, u_predict):
        s = u_predict.sum(dim=1)
        v = squash(self.n_iterations * s)
        return v

class CapsLayer(nn.Module):
    def __init__(self, input_caps, input_dim, output_caps, output_dim, routing_module):
        super(CapsLayer, self).__init__()
        self.input_dim = input_dim
        self.input_caps = input_caps
        self.output_dim = output_dim
        self.output_caps = output_caps
        self.weights = nn.Parameter(torch.Tensor(input_caps, input_dim, output_caps * output_dim))
        self.routing_module = routing_module
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.input_caps)
        self.weights.data.uniform_(-stdv, stdv)

    def forward(self, caps_output):
        caps_output = caps_output.unsqueeze(2)
        u_predict = caps_output.matmul(self.weights)
        u_predict = u_predict.view(u_predict.size(0), self.input_caps, self.output_caps, self.output_dim)
        v = self.routing_module(u_predict)
        return v


class PrimaryCapsLayer(nn.Module):
    def __init__(self, input_channels, output_caps, output_dim, kernel_size, stride):
        super(PrimaryCapsLayer, self).__init__()
        self.conv = nn.Conv2d(input_channels, output_caps * output_dim, kernel_size=kernel_size, stride=stride)
        self.input_channels = input_channels
        self.output_caps = output_caps
        self.output_dim = output_dim

    def forward(self, input):
        out = self.conv(input)
        N, C, H, W = out.size()
        out = out.view(N, self.output_caps, self.output_dim, H, W)

        # will output N x OUT_CAPS x OUT_DIM
        out = out.permute(0, 1, 3, 4, 2).contiguous()
        out = out.view(out.size(0), -1, out.size(4))
        out = squash(out)
        return out


class DynamicRoutingCapsule(nn.Module):
    def __init__(self, routing_iterations=3):
        super(DynamicRoutingCapsule, self).__init__()
        self.primaryCaps = PrimaryCapsLayer(256, 32, 8, kernel_size=1, stride=1)  # outputs 6*6
        self.num_primaryCaps = 32 * 6 * 6
        routing_module = AgreementRouting(self.num_primaryCaps, 16, routing_iterations)
        self.digitCaps = CapsLayer(self.num_primaryCaps, 8, 16, 8, routing_module)

        self.unfold = torch.nn.Unfold(kernel_size=(6,6))
    
    def forward(self, z, x):
        bs = x.shape[0]

        x = self.unfold(x)                      # (bs, 9216, 289)
        x = x.permute(0, 2, 1).contiguous()     # (bs, 289, 9216)
        x = x.view(bs*289, 256, 6, 6)           # (bs*289, 256, 6, 6)

        x = self.primaryCaps(x)
        x = self.digitCaps(x)                   # (bs*289, 16, 8)
        x = x.view(bs,289,16,8)
        x = x.permute(1,0,2,3).contiguous()     # (289,bs,16,8)

        z = self.primaryCaps(z)
        z = self.digitCaps(z)                   # (bs, 16, 8)
       
        y = x * z                                # (289,bs,16,8)
        y = torch.sum(y.mean(-1), dim=-1)                  # (289,bs)
        y = y.permute(1,0).view(bs,-1,17,17)

        return y

