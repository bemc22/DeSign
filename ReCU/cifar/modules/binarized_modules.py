import torch
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F

from torch.autograd import Function, Variable
from utils.options import args

import scipy.io as sio

class BinarizeConv2d(nn.Conv2d):

    def __init__(self, *kargs, **kwargs):
        super(BinarizeConv2d, self).__init__(*kargs, **kwargs)
        self.alpha = nn.Parameter(torch.rand(self.weight.size(0), 1, 1), requires_grad=True)
        self.register_buffer('tau', torch.tensor(1.))

    def forward(self, input):
        a = input
        w = self.weight

        w0 = w - w.mean([1,2,3], keepdim=True)
        w1 = w0 / (torch.sqrt(w0.var([1,2,3], keepdim=True) + 1e-5) / 2 / np.sqrt(2))
        EW = torch.mean(torch.abs(w1))
        Q_tau = (- EW * torch.log(2-2*self.tau)).detach().cpu().item()
        w2 = torch.clamp(w1, -Q_tau, Q_tau)

        if self.training:
            a0 = a / torch.sqrt(a.var([1,2,3], keepdim=True) + 1e-5)
        else: 
            a0 = a
        
        #* binarize
        bw = BinaryQuantize().apply(w2)
        ba = BinaryQuantize_a().apply(a0)
        #* 1bit conv
        output = F.conv2d(ba, bw, self.bias,
                          self.stride, self.padding,
                          self.dilation, self.groups)
        #* scaling factor
        output = output * self.alpha
        return output


class Threshold2D(nn.Module):

    def __init__(self):
        super(Threshold2D, self).__init__()

        filename = './thresholds/threshold_2x2_v1.mat'

        self.kernel = sio.loadmat(filename)['kernel']
        self.kernel = torch.tensor( self.kernel, dtype=torch.float32)[None, None, ...]
        self.kernel = self.kernel.cuda()
        self.max =   nn.MaxPool2d(2, stride=2)
    
    
    def forward(self, input):

        ones = torch.sum( input, axis=1, keepdims=True)
        ones = self.max( ones*0. + 1.)
        spatial_thresh = nn.functional.conv_transpose2d(ones, self.kernel, stride=2)

        return input - spatial_thresh


class Threshold3D(nn.Module):

    def __init__(self):
        super(Threshold3D, self).__init__()

        filename = './thresholds/threshold_2x2x4_shifting_n1_right_v1.mat'

        self.kernel = sio.loadmat(filename)['kernel']
        m, n , features = self.kernel.shape
        self.stride = (features, m, n)
        self.stride_conv = (m, n)
        self.kernel = np.transpose( self.kernel, axes=(2, 0, 1) )
        self.kernel = torch.tensor( self.kernel, dtype=torch.float32)[None, ...]
        self.kernel = self.kernel.cuda()
        self.max =  nn.MaxPool3d( self.stride, stride=self.stride)
    
    
    def forward(self, input):

        input, gamma = input
        temp = input[:, None, ...]
        #ones = torch.sum( input, axis=1, keepdims=True)
        temp = self.max( temp )
        replicates = temp.size(dim=2)
        ones = torch.sum( temp, axis=2, keepdims=False)
        ones = ones*0. + 1

        spatial_thresh = nn.functional.conv_transpose2d(ones, self.kernel, stride=self.stride_conv)
        spatial_thresh = torch.tile(spatial_thresh, (1, replicates, 1, 1))

        return input -  spatial_thresh*gamma
        # return (input - spatial_thresh )*gamma 

class BinaryQuantize(Function):
    @staticmethod
    def forward(ctx, input):
        out = torch.sign(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input


class BinaryQuantize_a(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        out = torch.sign(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors[0]
        grad_input = (2 - torch.abs(2*input))
        grad_input = grad_input.clamp(min=0) * grad_output.clone()
        return grad_input