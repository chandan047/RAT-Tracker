'''
Experiment 1, 2
'''


from __future__ import absolute_import, division, print_function

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

import pdb

NUM_ROUTING_ITERATIONS = 3

def softmax(input, dim=1):
	transposed_input = input.transpose(dim, len(input.size()) - 1)
	softmaxed_output = F.softmax(transposed_input.contiguous().view(-1, transposed_input.size(-1)), dim=-1)
	return softmaxed_output.view(*transposed_input.size()).transpose(dim, len(input.size()) - 1)


class PrimaryCapsule(nn.Module):
	def __init__(self, num_capsules, in_channels, out_channels, kernel_size=None, stride=None):
		super(PrimaryCapsule, self).__init__()

		self.num_capsules = num_capsules

		self.capsules = nn.ModuleList(
			[nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=0) for _ in
				range(num_capsules)])

	def squash(self, tensor, dim=1):
		squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
		scale = squared_norm / (1 + squared_norm)
		return scale * tensor / torch.sqrt(squared_norm)

	def forward(self, x):
		outputs = [capsule(x).view(x.size(0), -1, 1) for capsule in self.capsules]
		outputs = torch.cat(outputs, dim=-1)
		outputs = self.squash(outputs)

		return outputs


class SiamCapsule(nn.Module):
	def __init__(self, num_capsules, num_route_nodes, in_channels, out_channels, kernel_size=None, stride=None, num_iterations=NUM_ROUTING_ITERATIONS):
		super(SiamCapsule, self).__init__()

		self.num_route_nodes = num_route_nodes
		self.num_iterations = num_iterations

		self.num_capsules = num_capsules

		self.route_weights = nn.Parameter(torch.randn(num_capsules, num_route_nodes, in_channels, out_channels))

	def squash(self, tensor, dim=-1):
		squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
		scale = squared_norm / (1 + squared_norm)
		return scale * tensor / torch.sqrt(squared_norm)

	def forward(self, x):
		priors = x[None, :, :, None, :] @ self.route_weights[:, None, :, :, :]
#		breakpoint()
		logits = Variable(torch.zeros(*priors.size())).cuda()
		for i in range(self.num_iterations):
			probs = softmax(logits, dim=2)
			outputs = self.squash((probs * priors).sum(dim=2, keepdim=True))

			if i != self.num_iterations - 1:
				delta_logits = (priors * outputs).sum(dim=-1, keepdim=True)
				logits = logits + delta_logits

		return outputs


class CapsuleLayer(nn.Module):
    def __init__(self, num_capsules, num_route_nodes, in_channels, out_channels, kernel_size=None, stride=None,
                 num_iterations=NUM_ROUTING_ITERATIONS):
        super(CapsuleLayer, self).__init__()

        self.num_route_nodes = num_route_nodes
        self.num_iterations = num_iterations

        self.num_capsules = num_capsules

        if num_route_nodes != -1:
            self.route_weights = nn.Parameter(torch.randn(num_capsules, num_route_nodes, in_channels, out_channels))
        else:
            self.capsules = nn.ModuleList(
                [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=0) for _ in
                 range(num_capsules)])

    def squash(self, tensor, dim=-1):
        squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * tensor / torch.sqrt(squared_norm)

    def forward(self, x):
        if self.num_route_nodes != -1:
            priors = x[None, :, :, None, :] @ self.route_weights[:, None, :, :, :]

            logits = Variable(torch.zeros(*priors.size())).cuda()
            for i in range(self.num_iterations):
                probs = softmax(logits, dim=2)
                outputs = self.squash((probs * priors).sum(dim=2, keepdim=True))

                if i != self.num_iterations - 1:
                    delta_logits = (priors * outputs).sum(dim=-1, keepdim=True)
                    logits = logits + delta_logits
        else:
            outputs = [capsule(x).view(x.size(0), -1, 1) for capsule in self.capsules]
            outputs = torch.cat(outputs, dim=-1)
            outputs = self.squash(outputs)

        return outputs


class DynamicRoutingCapsule(nn.Module):
    def __init__(self):
        super(DynamicRoutingCapsule, self).__init__()
        self.primary_capsules = CapsuleLayer(num_capsules=8, num_route_nodes=-1, 
                                                 in_channels=256, out_channels=32,
                                                 kernel_size=1, stride=1)
        self.siam_capsules = CapsuleLayer(num_capsules=16, num_route_nodes=32 * 6 * 6, 
                                               in_channels=8, out_channels=8)

        self.unfold = torch.nn.Unfold(kernel_size=(6,6))
        self.cos = torch.nn.CosineSimilarity(dim=-1,eps=1e-6)

        # TODO: self.decoder

    def squash(self, tensor, dim=1):
        squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * tensor / torch.sqrt(squared_norm)

    def forward(self, z, x):
        # z: (bs, 256, 6, 6)
        # x: (bs, 256, 20, 20)

        x_size = x.shape
        z_size = z.shape

        # print ("Shapes", x_size, z_size)

        x = self.unfold(x)              # (bs, 9216, 225)
        x = x.permute(0, 2, 1).contiguous()	# (bs, 255, 9216)
        x = x.view(x_size[0]*225, 256, 6, 6)	# (bs*225, 256, 6, 6)

        x = self.primary_capsules(x)
        x = self.siam_capsules(x).squeeze().transpose(0, 1)	# (bs*225, 16, 8)

        z = self.primary_capsules(z)
        z = self.siam_capsules(z).squeeze().transpose(0, 1)	# (bs, 16, 8)
        z = z.repeat(225,1,1)                                   # (bs*225, 16, 8)
        
        y = F.relu(self.cos(x,z))                        # (bs*225,16)
        y = torch.mean(y,dim=-1,keepdim=False) 
        y = y.view(x_size[0],225)          # (bs,225) 
        y = y.view(x_size[0], -1, 15, 15)

        return y
