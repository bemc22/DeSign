'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from modules import *


__all__ =['resnet18A_1w1a','resnet18B_1w1a','resnet18C_1w1a','resnet18_1w1a']


ENABLE_THRESHOLD = True
CENTER = False
SCALE = False

THRESHOLD_OPTS = {
    'threshold': ENABLE_THRESHOLD,
    'batch_center': CENTER,
    'batch_scale': SCALE,
}


def _weights_init(m):
    if isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum):
    # Use `is_grad_enabled` to determine whether the current mode is training
    # mode or prediction mode
    if not torch.is_grad_enabled():
        # If it is prediction mode, directly use the mean and variance
        # obtained by moving average
        X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
    else:
        assert len(X.shape) in (2, 4)
        if len(X.shape) == 2:
            # When using a fully-connected layer, calculate the mean and
            # variance on the feature dimension
            mean = X.mean(dim=0)
            var = ((X - mean) ** 2).mean(dim=0)
        else:
            # When using a two-dimensional convolutional layer, calculate the
            # mean and variance on the channel dimension (axis=1). Here we
            # need to maintain the shape of `X`, so that the broadcasting
            # operation can be carried out later
            mean = X.mean(dim=(0, 2, 3), keepdim=True)
            var = ((X - mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)
        # In training mode, the current mean and variance are used for the
        # standardization
        X_hat = (X - mean) / torch.sqrt(var + eps)
        # Update the mean and variance using moving average
        moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
        moving_var = momentum * moving_var + (1.0 - momentum) * var
    Y = gamma * X_hat + beta  # Scale and shift
    return Y, moving_mean.data, moving_var.data

class BatchNorm(nn.Module):
    # `num_features`: the number of outputs for a fully-connected layer
    # or the number of output channels for a convolutional layer. `num_dims`:
    # 2 for a fully-connected layer and 4 for a convolutional layer
    def __init__(self, num_features, num_dims, center=True, scale=True, return_gamma=False):
        super().__init__()
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
        # The scale parameter and the shift parameter (model parameters) are
        # initialized to 1 and 0, respectively
        self.return_gamma = return_gamma
        self.gamma = nn.Parameter(torch.ones(shape), requires_grad=scale)
        self.beta = nn.Parameter(torch.zeros(shape), requires_grad=center)
        # The variables that are not model parameters are initialized to 0 and 1
        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.ones(shape)

    def forward(self, X):
        # If `X` is not on the main memory, copy `moving_mean` and
        # `moving_var` to the device where `X` is located
        if self.moving_mean.device != X.device:
            self.moving_mean = self.moving_mean.to(X.device)
            self.moving_var = self.moving_var.to(X.device)
        # Save the updated `moving_mean` and `moving_var`
        Y, self.moving_mean, self.moving_var = batch_norm(
            X, self.gamma, self.beta, self.moving_mean,
            self.moving_var, eps=1e-5, momentum=0.9)

        if self.return_gamma:
            return Y, self.gamma
        return Y # N ( beta , gamma )

        
class BasicBlock_1w1a(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, threshold_opts=None):
        super(BasicBlock_1w1a, self).__init__()

        # THRESHOLD OPTIONS CONFIGURATION
        self.threshold = threshold_opts['threshold']
        center = threshold_opts['batch_center']
        scale = threshold_opts['batch_scale']
        return_gamma = self.threshold

        self.conv1 = BinarizeConv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = BatchNorm(planes, num_dims=4, center=center, scale=scale, return_gamma=return_gamma)
        self.conv2 = BinarizeConv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = BatchNorm(planes, num_dims=4, center=center, scale=scale, return_gamma=return_gamma)

        if self.threshold == True:
            self.threshold = Threshold3D()
        else:
            self.threshold = None

        self.shortcut = nn.Sequential()
        pad = 0 if planes == self.expansion*in_planes else planes // 4
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                        nn.AvgPool2d((2,2)), 
                        LambdaLayer(lambda x:
                        F.pad(x, (0, 0, 0, 0, pad, pad), "constant", 0)))

    def forward(self, x):
        out = self.bn1(self.conv1(x))

        if self.threshold:
            out = self.threshold(out)

        out = F.hardtanh(out, inplace=True)
        out = self.bn2(self.conv2(out))

        if self.threshold:
            out = self.threshold(out)

        out += self.shortcut(x)
        out = F.hardtanh(out, inplace=True)
        return out


class Bottleneck_1w1a(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1,  threshold_opts=None):
        super(Bottleneck_1w1a, self).__init__()

        # THRESHOLD OPTIONS CONFIGURATION
        self.threshold = threshold_opts['threshold']
        center = threshold_opts['batch_center']
        scale = threshold_opts['batch_scale']

        return_gamma = self.threshold

        self.conv1 = BinarizeConv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm(planes, num_dims=4, center=center, scale=scale, return_gamma=return_gamma)
        self.conv2 = BinarizeConv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = BatchNorm(planes, num_dims=4, center=center, scale=scale, return_gamma=return_gamma)
        self.conv3 = BinarizeConv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = BatchNorm(self.expansion*planes, num_dims=4, center=center, scale=scale, return_gamma=return_gamma)

        if self.threshold == True:
            self.threshold = Threshold3D()
        else:
            self.threshold = None

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                BinarizeConv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                BatchNorm(self.expansion*planes, num_dims=4, center=center, scale=scale, return_gamma=return_gamma)
            )

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        if self.threshold:
            out = self.threshold(out)
        out = F.hardtanh(out)
        
        out =self.bn2(self.conv2(out))
        if self.threshold:
            out = self.threshold(out)
        out = F.hardtanh(out)
        
        out = self.bn3(self.conv3(out))
        if self.threshold:
            out = self.threshold(out)

        out += self.shortcut(x)
        out = F.hardtanh(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_channel, num_classes=10, threshold_opts=THRESHOLD_OPTS):
        super(ResNet, self).__init__()
        self.in_planes = num_channel[0]

        center = threshold_opts['batch_center']
        scale = threshold_opts['batch_scale']
        return_gamma = threshold_opts['threshold']

        
        if threshold_opts['threshold'] == True:
            self.threshold = Threshold3D()
        else:
            self.threshold = None

        self.conv1 = nn.Conv2d(3, num_channel[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 =  BatchNorm(num_channel[0], num_dims=4, center=center, scale=scale, return_gamma=False)
        self.layer1 = self._make_layer(block, num_channel[0], num_blocks[0], stride=1, threshold_opts=threshold_opts)
        self.layer2 = self._make_layer(block, num_channel[1], num_blocks[1], stride=2, threshold_opts=threshold_opts)
        self.layer3 = self._make_layer(block, num_channel[2], num_blocks[2], stride=2, threshold_opts=threshold_opts)
        self.layer4 = self._make_layer(block, num_channel[3], num_blocks[3], stride=2, threshold_opts=threshold_opts)
        self.linear = nn.Linear(num_channel[3]*block.expansion, num_classes)
        self.bn2 = nn.BatchNorm1d(num_channel[3]*block.expansion)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride, threshold_opts):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, threshold_opts))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        # if self.threshold:
        #     out = self.threshold(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.bn2(out)
        out = self.linear(out)
        return out 


def resnet18A_1w1a(**kwargs):
    return ResNet(BasicBlock_1w1a, [2,2,2,2],[32,32,64,128],**kwargs)

def resnet18B_1w1a(**kwargs):
    return ResNet(BasicBlock_1w1a, [2,2,2,2],[32,64,128,256],**kwargs)

def resnet18C_1w1a(**kwargs):
    return ResNet(BasicBlock_1w1a, [2,2,2,2],[64,64,128,256],**kwargs)

def resnet18_1w1a(**kwargs):
    return ResNet(BasicBlock_1w1a, [2,2,2,2],[64,128,256,512],**kwargs)

def resnet34_1w1a(**kwargs):
    return ResNet(BasicBlock_1w1a, [3,4,6,3],[64,128,256,512],**kwargs)

def resnet50_1w1a(**kwargs):
    return ResNet(Bottleneck_1w1a, [3,4,6,3],[64,128,256,512],**kwargs)

def resnet101_1w1a(**kwargs):
    return ResNet(Bottleneck_1w1a, [3,4,23,3],[64,128,256,512],**kwargs)

def resnet152_1w1a(**kwargs):
    return ResNet(Bottleneck_1w1a, [3,8,36,3],[64,128,256,512],**kwargs)


def test(net):
    import numpy as np
    total_params = 0

    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += np.prod(x.data.numpy().shape)
    print("Total number of params", total_params)
    print("Total layers", len(list(filter(lambda name: 'conv' in name or 'linear' in name, [name[0] for name in list(net.named_modules())]))))


if __name__ == "__main__":
    for net_name in __all__:
        if net_name.startswith('resnet'):
            print(net_name)
            test(globals()[net_name]())
            print()
