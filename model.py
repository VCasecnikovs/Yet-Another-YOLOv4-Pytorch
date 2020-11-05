import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

# Need for Pi
import math

#Mypy
import typing as ty

# Model consists of
# - backbone
# - neck
# - head
# - yolo

# To implement:
# Mish (just download) DONE
# CSP (is in architecture) DONE
# MiWRC (attention_forward)
# SPP block (in architecture) DONE
# PAN (in architecture) DONE
# Implemented with
# https://lutzroeder.github.io/netron/?url=https%3A%2F%2Fraw.githubusercontent.com%2FAlexeyAB%2Fdarknet%2Fmaster%2Fcfg%2Fyolov4.cfg


class BadArguments(Exception):
    pass

import torch
import torch.nn.functional as F

# Mish as written in darknet speed check
class darknet_mish(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        ctx.save_for_backward(input)
        e = torch.exp(input)
        n = e * e + 2 * e
        mask = input <= -0.6
        input[mask] = (input * (n / (n + 2)))[mask]
        input[~mask] = ((input - 2 * (input / (n + 2))))[~mask]

        return input

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        input, = ctx.saved_tensors

        sp = F.softplus(input)
        grad_sp = -torch.expm1(sp)

        tsp = F.tanh(sp)
        grad_tsp = (1 - tsp * tsp) * grad_sp
        grad = input * grad_tsp + tsp
        return grad


class DarknetMish(nn.Module):
    def __init__(self):
        """
        Initialize the state

        Args:
            self: (todo): write your description
        """
        super().__init__()

    def forward(self, x):
        """
        Evaluate x todo.

        Args:
            self: (todo): write your description
            x: (todo): write your description
        """
        return darknet_mish.apply(x)

# @torch.jit.script
class HardMish(nn.Module):
    def __init__(self):
        """
        Initialize the state

        Args:
            self: (todo): write your description
        """
        super().__init__()
    def forward(self, x):
        """
        Calculate the forward.

        Args:
            self: (todo): write your description
            x: (todo): write your description
        """
        return (x/2) * torch.clamp(x+2, min=0, max=2)


# Taken from https://github.com/lessw2020/mish
class Mish(nn.Module):
    def __init__(self):
        """
        Initialize the state

        Args:
            self: (todo): write your description
        """
        super().__init__()

    def forward(self, x):
        """
        Forward computation.

        Args:
            self: (todo): write your description
            x: (todo): write your description
        """
        # inlining this saves 1 second per epoch (V100 GPU) vs having a temp x and then returning x(!)
        return x * torch.tanh(F.softplus(x))


# Taken from https://github.com/Randl/DropBlock-pytorch/blob/master/DropBlock.py
class DropBlock2D(nn.Module):
    r"""Randomly zeroes spatial blocks of the input tensor.
    As described in the paper
    `DropBlock: A regularization method for convolutional networks`_ ,
    dropping whole blocks of feature map allows to remove semantic
    information as compared to regular dropout.
    Args:
        keep_prob (float, optional): probability of an element to be kept.
        Authors recommend to linearly decrease this value from 1 to desired
        value.
        block_size (int, optional): size of the block. Block size in paper
        usually equals last feature map dimensions.
    Shape:
        - Input: :math:`(N, C, H, W)`
        - Output: :math:`(N, C, H, W)` (same shape as input)
    .. _DropBlock: A regularization method for convolutional networks:
       https://arxiv.org/abs/1810.12890
    """

    def __init__(self, keep_prob=0.9, block_size=7):
        """
        Initialize the state.

        Args:
            self: (todo): write your description
            keep_prob: (float): write your description
            block_size: (int): write your description
        """
        super(DropBlock2D, self).__init__()
        self.keep_prob = keep_prob
        self.block_size = block_size

    def forward(self, input):
        """
        Forward computation.

        Args:
            self: (todo): write your description
            input: (todo): write your description
        """
        # print("Before: ", torch.isnan(input).sum())
        if not self.training or self.keep_prob == 1:
            return input
        gamma = (1. - self.keep_prob) / self.block_size ** 2
        for sh in input.shape[2:]:
            gamma *= sh / (sh - self.block_size + 1)
        M = torch.bernoulli(torch.ones_like(input) * gamma).to(device=input.device)
        Msum = F.conv2d(M,
                        torch.ones((input.shape[1], 1, self.block_size, self.block_size)).to(device=input.device,
                                                                                             dtype=input.dtype),
                        padding=self.block_size // 2,
                        groups=input.shape[1])
        mask = (Msum < 1).to(device=input.device, dtype=input.dtype)
        # print("After: ", torch.isnan(input * mask * mask.numel() /mask.sum()).sum())
        return input * mask * mask.numel() / mask.sum()

class SAM(nn.Module):
    def __init__(self, in_channels):
        """
        Initialize the channel.

        Args:
            self: (todo): write your description
            in_channels: (int): write your description
        """
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels=1, kernel_size=1)

    def forward(self, x):
        """
        Forward computation.

        Args:
            self: (todo): write your description
            x: (todo): write your description
        """
        spatial_features = self.conv(x)
        attention = torch.sigmoid(spatial_features)
        return attention.expand_as(x) * x

#Got and modified from https://arxiv.org/pdf/2003.13630.pdf
class FastGlobalAvgPool2d():
    def __init__(self, flatten=False):
        """
        Initialize this object.

        Args:
            self: (todo): write your description
            flatten: (todo): write your description
        """
        self.flatten = flatten
    def __call__(self, x):
        """
        Return the size of x.

        Args:
            self: (todo): write your description
            x: (todo): write your description
        """
        if self.flatten:
            in_size = x.size()
            return x.view((in_size[0], in_size[1], -1)).mean(dim=2)
        else:
            return x.view(x.size(0), x.size(1), -1).mean(-1).view(x.size(0), x.size(1), 1, 1)

#As an example was taken https://github.com/BangguWu/ECANet/blob/master/models/eca_module.py
class ECA(nn.Module):
    def __init__(self, k_size=3):
        """
        Initialize k k k k_size k_size

        Args:
            self: (todo): write your description
            k_size: (int): write your description
        """
        super().__init__()
        self.avg_pool = FastGlobalAvgPool2d(flatten=False)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)

    def forward(self, x):
        """
        Forward computation.

        Args:
            self: (todo): write your description
            x: (todo): write your description
        """
        squized_channels = self.avg_pool(x)
        channel_features = self.conv(squized_channels.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        attention = torch.sigmoid(channel_features)
        return attention.expand_as(x) * x


#Taken from https://github.com/joe-siyuan-qiao/WeightStandardization modified with new std https://github.com/joe-siyuan-qiao/WeightStandardization/issues/1#issuecomment-528050344
class Conv2dWS(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        """
        Initialize the channel.

        Args:
            self: (todo): write your description
            in_channels: (int): write your description
            out_channels: (int): write your description
            kernel_size: (int): write your description
            stride: (int): write your description
            padding: (str): write your description
            dilation: (todo): write your description
            groups: (list): write your description
            bias: (float): write your description
        """
        super(Conv2dWS, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias)

    def forward(self, x):
        """
        Forward computation.

        Args:
            self: (todo): write your description
            x: (todo): write your description
        """
        # print("IN: ", (~torch.isfinite(x)).sum())
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2,
                                  keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - weight_mean
        std = torch.sqrt(torch.var(weight.view(weight.size(0), -1), dim=1) + 2e-5).view(-1, 1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)
        return F.conv2d(x, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

# From https://arxiv.org/pdf/2003.10152.pdf and https://github.com/Wizaron/coord-conv-pytorch/blob/master/coord_conv.py
class AddCoordChannels(nn.Module):
    def __init__(self, w=9, h=9, b=1):
        """
        Initialize the image.

        Args:
            self: (todo): write your description
            w: (int): write your description
            h: (int): write your description
            b: (int): write your description
        """
        super().__init__()
        self.w = w
        self.h = h
        self.b = b
        self.y_coords = 2.0 * torch.arange(h).unsqueeze(1).expand(h, w) / (h - 1.0) - 1.0
        self.x_coords = 2.0 * torch.arange(w).unsqueeze(0).expand(h, w) / (w - 1.0) - 1.0
        self.coords = torch.stack((self.x_coords, self.y_coords), dim=0)
        self.coords = torch.unsqueeze(self.coords, dim=0).repeat(b, 1, 1, 1)        

    def forward(self, x):
        """
        Forward computation of the image.

        Args:
            self: (todo): write your description
            x: (todo): write your description
        """
        b, c, h, w = x.shape
        if w != self.w or h != self.h or b != self.b:
            self.w = w
            self.h = h
            self.b = b
            self.y_coords = 2.0 * torch.arange(h).unsqueeze(1).expand(h, w) / (h - 1.0) - 1.0
            self.x_coords = 2.0 * torch.arange(w).unsqueeze(0).expand(h, w) / (w - 1.0) - 1.0
            self.coords = torch.stack((self.x_coords, self.y_coords), dim=0)
            self.coords = torch.unsqueeze(self.coords, dim=0).repeat(b, 1, 1, 1)

        return torch.cat((x, self.coords.to(x.device)), dim=1)

#Was taken from https://github.com/joe-siyuan-qiao/Batch-Channel-Normalization     
class BCNorm(nn.Module):
    def __init__(self, num_channels, num_groups=1, eps=1e-05, estimate=False):
        """
        Initialize the batch.

        Args:
            self: (todo): write your description
            num_channels: (int): write your description
            num_groups: (int): write your description
            eps: (float): write your description
            estimate: (todo): write your description
        """
        super(BCNorm, self).__init__()
        self.num_channels = num_channels
        self.num_groups = num_groups
        self.eps = eps
        self.weight = Parameter(torch.ones(1, num_groups, 1))
        self.bias = Parameter(torch.zeros(1, num_groups, 1))
        if estimate:
            self.bn = EstBN(num_channels)
        else:
            self.bn = nn.BatchNorm2d(num_channels)

    def forward(self, inp):
        """
        Forward computation.

        Args:
            self: (todo): write your description
            inp: (todo): write your description
        """
        out = self.bn(inp)
        out = out.view(1, inp.size(0) * self.num_groups, -1)
        out = torch.batch_norm(out, None, None, None, None, True, 0, self.eps, True)
        out = out.view(inp.size(0), self.num_groups, -1)
        out = self.weight * out + self.bias
        out = out.view_as(inp)
        return out

class EstBN(nn.Module):

    def __init__(self, num_features):
        """
        Initialize the gradient.

        Args:
            self: (todo): write your description
            num_features: (int): write your description
        """
        super(EstBN, self).__init__()
        self.num_features = num_features
        self.weight = Parameter(torch.ones(num_features))
        self.bias = Parameter(torch.zeros(num_features))
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        self.register_buffer('estbn_moving_speed', torch.zeros(1))

    def forward(self, inp):
        """
        Calculate forward computation.

        Args:
            self: (todo): write your description
            inp: (todo): write your description
        """
        ms = self.estbn_moving_speed.item()
        if self.training:
            with torch.no_grad():
                inp_t = inp.transpose(0, 1).contiguous().view(self.num_features, -1)
                running_mean = inp_t.mean(dim=1)
                inp_t = inp_t - self.running_mean.view(-1, 1)
                running_var = torch.mean(inp_t * inp_t, dim=1)
                self.running_mean.data.mul_(1 - ms).add_(ms * running_mean.data)
                self.running_var.data.mul_(1 - ms).add_(ms * running_var.data)
        out = inp - self.running_mean.view(1, -1, 1, 1)
        out = out / torch.sqrt(self.running_var + 1e-5).view(1, -1, 1, 1)
        weight = self.weight.view(1, -1, 1, 1)
        bias = self.bias.view(1, -1, 1, 1)
        out = weight * out + bias
        return out

# Taken and modified from https://github.com/Tianxiaomo/pytorch-YOLOv4/blob/master/models.py
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, activation, bn=True, bias=False, dropblock=False, sam=False, eca=False, ws=False, coord=False, hard_mish=False, bcn=False, mbn=False):
        """
        Initialize the module.

        Args:
            self: (todo): write your description
            in_channels: (int): write your description
            out_channels: (int): write your description
            kernel_size: (int): write your description
            stride: (int): write your description
            activation: (str): write your description
            bn: (int): write your description
            bias: (float): write your description
            dropblock: (todo): write your description
            sam: (todo): write your description
            eca: (todo): write your description
            ws: (int): write your description
            coord: (array): write your description
            hard_mish: (str): write your description
            bcn: (todo): write your description
            mbn: (str): write your description
        """
        super().__init__()

        # PADDING is (ks-1)/2
        padding = (kernel_size - 1) // 2

        modules: ty.List[ty.Union[nn.Module]] = []
        #Adding two more to input channels if coord
        if coord:
            in_channels += 2
            modules.append(AddCoordChannels())
        if ws:
            modules.append(Conv2dWS(in_channels, out_channels, kernel_size, stride, padding, bias=bias))            
        else:
            modules.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias))
        if bn:
            if bcn:
                modules.append(BCNorm(out_channels, estimate=True))
            elif mbn:
                modules.append(EstBN(out_channels))
            else:
                modules.append(nn.BatchNorm2d(out_channels, track_running_stats=not ws)) #IF WE ARE NOT USING track running stats and using WS, it just explodes.
        if activation == "mish":
            if hard_mish:
                modules.append(HardMish())
            else:
                modules.append(Mish())
        elif activation == "relu":
            modules.append(nn.ReLU(inplace=True))
        elif activation == "leaky":
            modules.append(nn.LeakyReLU(0.1, inplace=True))
        elif activation == "linear":
            pass
        else:
            raise BadArguments("Please use one of suggested activations: mish, relu, leaky, linear.")

        if sam:
            modules.append(SAM(out_channels))

        if eca:
            modules.append(ECA())

        if dropblock:
            modules.append(DropBlock2D())

        self.module = nn.Sequential(*modules)

    def forward(self, x):
        """
        Forward function.

        Args:
            self: (todo): write your description
            x: (todo): write your description
        """
        y = self.module(x)
        return y


# Taken and modified from https://github.com/Tianxiaomo/pytorch-YOLOv4/blob/master/models.py
class ResBlock(nn.Module):
    """
    Sequential residual blocks each of which consists of \
    two convolution layers.
    Args:
        ch (int): number of input and output channels.
        nblocks (int): number of residual blocks.
        shortcut (bool): if True, residual tensor addition is enabled.
    """
    # Creating few conv blocks. One with kernel 3, second with kernel 1. With residual skip connection
    def __init__(self, ch, nblocks=1, shortcut=True, dropblock=True, sam=False, eca=False, ws=False, coord=False, hard_mish=False, bcn=False, mbn=False):
        """
        Initialize the module.

        Args:
            self: (todo): write your description
            ch: (todo): write your description
            nblocks: (bool): write your description
            shortcut: (todo): write your description
            dropblock: (todo): write your description
            sam: (todo): write your description
            eca: (todo): write your description
            ws: (int): write your description
            coord: (array): write your description
            hard_mish: (str): write your description
            bcn: (todo): write your description
            mbn: (str): write your description
        """
        super().__init__()
        self.shortcut = shortcut
        self.module_list = nn.ModuleList()
        for i in range(nblocks):
            resblock_one = nn.ModuleList()
            resblock_one.append(ConvBlock(ch, ch, 1, 1, 'mish', dropblock=dropblock, sam=sam, eca=eca, ws=ws, coord=False, hard_mish=hard_mish, bcn=bcn, mbn=mbn))
            resblock_one.append(ConvBlock(ch, ch, 3, 1, 'mish', dropblock=dropblock, sam=sam, eca=eca, ws=ws, coord=coord, hard_mish=hard_mish, bcn=bcn, mbn=mbn))
            self.module_list.append(resblock_one)

        if dropblock:
            self.use_dropblock = True
            self.dropblock = DropBlock2D()
        else:
            self.use_dropblock = False

    def forward(self, x):
        """
        Perform module forward.

        Args:
            self: (todo): write your description
            x: (todo): write your description
        """
        for module in self.module_list:
            h = x
            for res in module:
                h = res(h)
            x = x + h if self.shortcut else h
            if self.use_dropblock:
                x = self.dropblock(x)

        return x


class DownSampleFirst(nn.Module):
    """
    This is first downsample of the backbone model.
    It differs from the other stages, so it is written as another Module
    Args:
        in_channels (int): Amount of channels to input, if you use RGB, it should be 3
    """
    def __init__(self, in_channels=3, dropblock=True, sam=False, eca=False, ws=False, coord=False, hard_mish=False, bcn=False, mbn=False):
        """
        Initialize output blocks.

        Args:
            self: (todo): write your description
            in_channels: (int): write your description
            dropblock: (todo): write your description
            sam: (todo): write your description
            eca: (todo): write your description
            ws: (int): write your description
            coord: (array): write your description
            hard_mish: (str): write your description
            bcn: (todo): write your description
            mbn: (str): write your description
        """
        super().__init__()

        self.c1 = ConvBlock(in_channels, 32, 3, 1, "mish", dropblock=dropblock, sam=sam, eca=eca, ws=ws, coord=coord, hard_mish=hard_mish, bcn=bcn, mbn=mbn)
        self.c2 = ConvBlock(32, 64, 3, 2, "mish", dropblock=dropblock, sam=sam, eca=eca, ws=ws, coord=coord, hard_mish=hard_mish, bcn=bcn, mbn=mbn)
        self.c3 = ConvBlock(64, 64, 1, 1, "mish", dropblock=dropblock, sam=sam, eca=eca, ws=ws, coord=False, hard_mish=hard_mish, bcn=bcn, mbn=mbn)
        self.c4 = ConvBlock(64, 32, 1, 1, "mish", dropblock=dropblock, sam=sam, eca=eca, ws=ws, coord=False, hard_mish=hard_mish, bcn=bcn, mbn=mbn)
        self.c5 = ConvBlock(32, 64, 3, 1, "mish", dropblock=dropblock, sam=sam, eca=eca, ws=ws, coord=coord, hard_mish=hard_mish, bcn=bcn, mbn=mbn)
        self.c6 = ConvBlock(64, 64, 1, 1, "mish", dropblock=dropblock, sam=sam, eca=eca, ws=ws, coord=False, hard_mish=hard_mish, bcn=bcn, mbn=mbn)

        # CSP Layer
        self.dense_c3_c6 = ConvBlock(64, 64, 1, 1, "mish", dropblock=dropblock, sam=sam, eca=eca, ws=ws, coord=False, hard_mish=hard_mish)

        self.c7 = ConvBlock(128, 64, 1, 1, "mish", dropblock=dropblock, sam=sam, eca=eca, ws=ws, coord=False, hard_mish=hard_mish)

    def forward(self, x):
        """
        Perform forward forward.

        Args:
            self: (todo): write your description
            x: (todo): write your description
        """
        x1 = self.c1(x)
        x2 = self.c2(x1)
        x3 = self.c3(x2)
        x4 = self.c4(x3)
        x5 = self.c5(x4)
        x5 = x5 + x3    # Residual block
        x6 = self.c6(x5)
        xd6 = self.dense_c3_c6(x2)  # CSP
        x6 = torch.cat([x6, xd6], dim=1)
        x7 = self.c7(x6)
        return x7


class DownSampleBlock(nn.Module):
    def __init__(self, in_c, out_c, nblocks=2, dropblock=True, sam=False, eca=False, ws=False, coord=False, hard_mish=False, bcn=False, mbn=False):
        """
        Initialize c2c4.

        Args:
            self: (todo): write your description
            in_c: (int): write your description
            out_c: (str): write your description
            nblocks: (bool): write your description
            dropblock: (todo): write your description
            sam: (todo): write your description
            eca: (todo): write your description
            ws: (int): write your description
            coord: (array): write your description
            hard_mish: (str): write your description
            bcn: (todo): write your description
            mbn: (str): write your description
        """
        super().__init__()

        self.c1 = ConvBlock(in_c, out_c, 3, 2, "mish", dropblock=dropblock, sam=sam, eca=eca, ws=ws, coord=coord, hard_mish=hard_mish, bcn=bcn, mbn=mbn)
        self.c2 = ConvBlock(out_c, in_c, 1, 1, "mish", dropblock=dropblock, sam=sam, eca=eca, ws=ws, coord=False, hard_mish=hard_mish, bcn=bcn, mbn=mbn)
        self.r3 = ResBlock(in_c, nblocks=nblocks, dropblock=dropblock, sam=sam, eca=eca, ws=ws, coord=coord, hard_mish=hard_mish, bcn=bcn, mbn=mbn)
        self.c4 = ConvBlock(in_c, in_c, 1, 1, "mish", dropblock=dropblock, sam=sam, eca=eca, ws=ws, coord=False, hard_mish=hard_mish, bcn=bcn, mbn=mbn)

        # CSP Layer
        self.dense_c2_c4 = ConvBlock(out_c, in_c, 1, 1, "mish", dropblock=dropblock, sam=sam, eca=eca, ws=ws, coord=False, hard_mish=hard_mish, bcn=bcn, mbn=mbn)

        self.c5 = ConvBlock(out_c, out_c, 1, 1, "mish", dropblock=dropblock, sam=sam, eca=eca, ws=ws, coord=False, hard_mish=hard_mish, bcn=bcn, mbn=mbn)

    def forward(self, x):
        """
        Perform forward forward.

        Args:
            self: (todo): write your description
            x: (todo): write your description
        """
        x1 = self.c1(x)
        x2 = self.c2(x1)
        x3 = self.r3(x2)
        x4 = self.c4(x3)
        xd4 = self.dense_c2_c4(x1)  # CSP
        x4 = torch.cat([x4, xd4], dim=1)
        x5 = self.c5(x4)

        return x5


class Backbone(nn.Module):
    def __init__(self, in_channels, dropblock=True, sam=False, eca=False, ws=False, coord=False, hard_mish=False, bcn=False, mbn=False):
        """
        Initialize output.

        Args:
            self: (todo): write your description
            in_channels: (int): write your description
            dropblock: (todo): write your description
            sam: (todo): write your description
            eca: (todo): write your description
            ws: (int): write your description
            coord: (array): write your description
            hard_mish: (str): write your description
            bcn: (todo): write your description
            mbn: (str): write your description
        """
        super().__init__()

        self.d1 = DownSampleFirst(in_channels=in_channels, dropblock=dropblock, sam=sam, eca=eca, ws=ws, coord=coord, hard_mish=hard_mish, bcn=bcn, mbn=mbn)
        self.d2 = DownSampleBlock(64, 128, nblocks=2, dropblock=dropblock, sam=sam, eca=eca, ws=ws, coord=coord, hard_mish=hard_mish, bcn=bcn, mbn=mbn)
        self.d3 = DownSampleBlock(128, 256, nblocks=8, dropblock=dropblock, sam=sam, eca=eca, ws=ws, coord=coord, hard_mish=hard_mish, bcn=bcn, mbn=mbn)
        self.d4 = DownSampleBlock(256, 512, nblocks=8, dropblock=dropblock, sam=sam, eca=eca, ws=ws, coord=coord, hard_mish=hard_mish, bcn=bcn, mbn=mbn)
        self.d5 = DownSampleBlock(512, 1024, nblocks=4, dropblock=dropblock, sam=sam, eca=eca, ws=ws, coord=coord, hard_mish=hard_mish, bcn=bcn, mbn=mbn)

    def forward(self, x):
        """
        Perform of the forward.

        Args:
            self: (todo): write your description
            x: (todo): write your description
        """
        x1 = self.d1(x)
        x2 = self.d2(x1)
        x3 = self.d3(x2)
        x4 = self.d4(x3)
        x5 = self.d5(x4)
        return (x5, x4, x3)


class PAN_Layer(nn.Module):
    def __init__(self, in_channels, dropblock=True, sam=False, eca=False, ws=False, coord=False, hard_mish=False, bcn=False, mbn=False):
        """
        Initialize c2.

        Args:
            self: (todo): write your description
            in_channels: (int): write your description
            dropblock: (todo): write your description
            sam: (todo): write your description
            eca: (todo): write your description
            ws: (int): write your description
            coord: (array): write your description
            hard_mish: (str): write your description
            bcn: (todo): write your description
            mbn: (str): write your description
        """
        super().__init__()

        in_c = in_channels
        out_c = in_c // 2

        self.c1 = ConvBlock(in_c, out_c, 1, 1, "leaky", dropblock=dropblock, sam=sam, eca=eca, ws=ws, coord=False, hard_mish=hard_mish, bcn=bcn, mbn=mbn)
        self.u2 = nn.Upsample(scale_factor=2, mode="nearest")
        # Gets input from d4
        self.c2_from_upsampled = ConvBlock(in_c, out_c, 1, 1, "leaky", dropblock=False, sam=sam, eca=eca, ws=ws, coord=False, hard_mish=hard_mish, bcn=bcn, mbn=mbn)
        # We use stack in PAN, so 512
        self.c3 = ConvBlock(in_c, out_c, 1, 1, "leaky", dropblock=False, sam=sam, eca=eca, ws=ws, coord=False, hard_mish=hard_mish, bcn=bcn, mbn=mbn)
        self.c4 = ConvBlock(out_c, in_c, 3, 1, "leaky", dropblock=dropblock, sam=sam, eca=eca, ws=ws, coord=coord, hard_mish=hard_mish, bcn=bcn, mbn=mbn)
        self.c5 = ConvBlock(in_c, out_c, 1, 1, "leaky", dropblock=False, sam=sam, eca=eca, ws=ws, coord=False, hard_mish=hard_mish, bcn=bcn, mbn=mbn)
        self.c6 = ConvBlock(out_c, in_c, 3, 1, "leaky", dropblock=False, sam=sam, eca=eca, ws=ws, coord=coord, hard_mish=hard_mish, bcn=bcn, mbn=mbn)
        self.c7 = ConvBlock(in_c, out_c, 1, 1, "leaky", dropblock=False, sam=sam, eca=eca, ws=ws, coord=False, hard_mish=hard_mish, bcn=bcn, mbn=mbn)

    def forward(self, x_to_upsample, x_upsampled):
        """
        Perform forward forward forward.

        Args:
            self: (todo): write your description
            x_to_upsample: (todo): write your description
            x_upsampled: (todo): write your description
        """
        x1 = self.c1(x_to_upsample)
        x2_1 = self.u2(x1)
        x2_2 = self.c2_from_upsampled(x_upsampled)
        # First is not upsampled!
        x2 = torch.cat([x2_2, x2_1], dim=1)
        x3 = self.c3(x2)
        x4 = self.c4(x3)
        x5 = self.c5(x4)
        x6 = self.c6(x5)
        x7 = self.c7(x6)
        return x7

#Taken and modified from https://github.com/ruinmessi/ASFF/blob/0ff0e3393675583f7da65a7b443ea467e1eaed65/models/network_blocks.py#L267-L330
class ASFF(nn.Module):
    def __init__(self, level, rfb=False, vis=False, bcn=False, mbn=False):
        """
        Initialize the level

        Args:
            self: (todo): write your description
            level: (int): write your description
            rfb: (int): write your description
            vis: (int): write your description
            bcn: (todo): write your description
            mbn: (str): write your description
        """
        super(ASFF, self).__init__()
        self.level = level
        self.dim = [512, 256, 128]
        self.inter_dim = self.dim[self.level]
        if level==0:
            self.stride_level_1 = ConvBlock(256, self.inter_dim, 3, 2, "leaky", bcn=bcn, mbn=mbn)
            self.stride_level_2 = ConvBlock(128, self.inter_dim, 3, 2, "leaky", bcn=bcn, mbn=mbn)
            self.expand = ConvBlock(self.inter_dim, 512, 3, 1, "leaky", bcn=bcn, mbn=mbn)
        elif level==1:
            self.compress_level_0 = ConvBlock(512, self.inter_dim, 1, 1, "leaky", bcn=bcn, mbn=mbn)
            self.stride_level_2 = ConvBlock(128, self.inter_dim, 3, 2, "leaky", bcn=bcn, mbn=mbn)
            self.expand = ConvBlock(self.inter_dim, 256, 3, 1, "leaky", bcn=bcn, mbn=mbn)
        elif level==2:
            self.compress_level_0 = ConvBlock(512, self.inter_dim, 1, 1, "leaky", bcn=bcn, mbn=mbn)
            self.compress_level_1 = ConvBlock(256, self.inter_dim, 1, 1, "leaky", bcn=bcn, mbn=mbn)
            self.expand = ConvBlock(self.inter_dim, 128, 3, 1, "leaky", bcn=bcn, mbn=mbn)

        compress_c = 8 if rfb else 16  #when adding rfb, we use half number of channels to save memory
        self.weight_level_0 = ConvBlock(self.inter_dim, compress_c, 1, 1, "leaky", bcn=bcn, mbn=mbn)
        self.weight_level_1 = ConvBlock(self.inter_dim, compress_c, 1, 1, "leaky", bcn=bcn, mbn=mbn)
        self.weight_level_2 = ConvBlock(self.inter_dim, compress_c, 1, 1, "leaky", bcn=bcn, mbn=mbn)
        self.weight_levels = nn.Conv2d(compress_c*3, 3, kernel_size=1, stride=1, padding=0)
    
        self.vis= vis


    def forward(self, x_level_0, x_level_1, x_level_2):
        """
        Forward computation.

        Args:
            self: (todo): write your description
            x_level_0: (todo): write your description
            x_level_1: (todo): write your description
            x_level_2: (todo): write your description
        """
        if self.level==0:
            level_0_resized = x_level_0 # 512 -> 512
            level_1_resized = self.stride_level_1(x_level_1) # 256 -> 512
            level_2_downsampled_inter =F.max_pool2d(x_level_2, 3, stride=2, padding=1)
            level_2_resized = self.stride_level_2(level_2_downsampled_inter) # 128 -> 512
        elif self.level==1:
            level_0_compressed = self.compress_level_0(x_level_0) # 512 -> 256
            level_0_resized =F.interpolate(level_0_compressed, scale_factor=2, mode='nearest')
            level_1_resized =x_level_1 # 256 -> 256
            level_2_resized =self.stride_level_2(x_level_2) # 128 -> 256
        elif self.level==2:
            level_0_compressed = self.compress_level_0(x_level_0) # 512 -> 128
            level_1_compressed = self.compress_level_1(x_level_1) # 256 -> 128
            level_0_resized =F.interpolate(level_0_compressed, scale_factor=4, mode='nearest')
            level_1_resized =F.interpolate(level_1_compressed, scale_factor=2, mode='nearest')
            level_2_resized =x_level_2 #128 -> 128

        level_0_weight_v = self.weight_level_0(level_0_resized)
        level_1_weight_v = self.weight_level_1(level_1_resized)
        level_2_weight_v = self.weight_level_2(level_2_resized)
        levels_weight_v = torch.cat((level_0_weight_v, level_1_weight_v, level_2_weight_v),1)
        levels_weight = self.weight_levels(levels_weight_v)
        levels_weight = F.softmax(levels_weight, dim=1)

        fused_out_reduced = level_0_resized * levels_weight[:,0:1,:,:]+\
                            level_1_resized * levels_weight[:,1:2,:,:]+\
                            level_2_resized * levels_weight[:,2:,:,:]

        out = self.expand(fused_out_reduced)

        if self.vis:
            return out, levels_weight, fused_out_reduced.sum(dim=1)
        else:
            return out

#Author: Vadims Casecnikovs creator of this repository
class ACFF(nn.Module):
    def __init__(self, level, rfb=False, vis=False, bcn=False, mbn=False):
        """
        Initialize the level

        Args:
            self: (todo): write your description
            level: (int): write your description
            rfb: (int): write your description
            vis: (int): write your description
            bcn: (todo): write your description
            mbn: (str): write your description
        """
        super(ACFF, self).__init__()
        self.level = level
        self.dim = [512, 256, 128]
        self.inter_dim = self.dim[self.level]
        if level==0:
            self.stride_level_1 = ConvBlock(256, self.inter_dim, 3, 2, "leaky", bcn=bcn, mbn=mbn)
            self.stride_level_2 = ConvBlock(128, self.inter_dim, 3, 2, "leaky", bcn=bcn, mbn=mbn)
            self.expand = ConvBlock(self.inter_dim, 512, 3, 1, "leaky", bcn=bcn, mbn=mbn)
        elif level==1:
            self.compress_level_0 = ConvBlock(512, self.inter_dim, 1, 1, "leaky", bcn=bcn, mbn=mbn)
            self.stride_level_2 = ConvBlock(128, self.inter_dim, 3, 2, "leaky", bcn=bcn, mbn=mbn)
            self.expand = ConvBlock(self.inter_dim, 256, 3, 1, "leaky", bcn=bcn, mbn=mbn)
        elif level==2:
            self.compress_level_0 = ConvBlock(512, self.inter_dim, 1, 1, "leaky", bcn=bcn, mbn=mbn)
            self.compress_level_1 = ConvBlock(256, self.inter_dim, 1, 1, "leaky", bcn=bcn, mbn=mbn)
            self.expand = ConvBlock(self.inter_dim, 128, 3, 1, "leaky", bcn=bcn, mbn=mbn)

        self.avg_pool = FastGlobalAvgPool2d()
        self.weights_spatial = torch.nn.Parameter(torch.ones((3, self.inter_dim, 3)))
        self.weights_spatial = torch.nn.init.kaiming_uniform_(self.weights_spatial, a=0, mode='fan_in', nonlinearity='relu')

        self.vis= vis


    def forward(self, x_level_0, x_level_1, x_level_2):
        """
        Forward computation.

        Args:
            self: (todo): write your description
            x_level_0: (todo): write your description
            x_level_1: (todo): write your description
            x_level_2: (todo): write your description
        """
        #In this part we are trying to construct the same channel features as in target level
        if self.level==0:
            level_0_resized = x_level_0 # 512 -> 512
            level_1_resized = self.stride_level_1(x_level_1) # 256 -> 512
            level_2_downsampled_inter =F.max_pool2d(x_level_2, 3, stride=2, padding=1)
            level_2_resized = self.stride_level_2(level_2_downsampled_inter) # 128 -> 512
        elif self.level==1: 
            level_0_compressed = self.compress_level_0(x_level_0) # 512 -> 256
            level_0_resized =F.interpolate(level_0_compressed, scale_factor=2, mode='nearest')
            level_1_resized =x_level_1 # 256 -> 256
            level_2_resized =self.stride_level_2(x_level_2) # 128 -> 256
        elif self.level==2:
            level_0_compressed = self.compress_level_0(x_level_0) # 512 -> 128
            level_1_compressed = self.compress_level_1(x_level_1) # 256 -> 128
            level_0_resized =F.interpolate(level_0_compressed, scale_factor=4, mode='nearest')
            level_1_resized =F.interpolate(level_1_compressed, scale_factor=2, mode='nearest')
            level_2_resized =x_level_2 #128 -> 128

        #In this part we are getting mean value of channel's(featuremaps) activations
        level_0_flattened = self.avg_pool(level_0_resized)
        level_1_flattened = self.avg_pool(level_1_resized)
        level_2_flattened = self.avg_pool(level_2_resized)

        #Concatenating all 3 levels, getting (B, C, L)        
        levels_weight = torch.cat([level_0_flattened, level_1_flattened, level_2_flattened], dim=2).squeeze(-1)

        #For each channel, getting 3 values, they would show how much attention should we give for each level's channel
        level_weight = torch.einsum("bci, kci-> bck", levels_weight, self.weights_spatial)

        levels_weight = torch.nn.functional.softmax(levels_weight, dim=2).unsqueeze(-1)

        fused_out_reduced = level_0_resized * levels_weight[:,:,0:1,:]+\
                            level_1_resized * levels_weight[:,:,1:2,:]+\
                            level_2_resized * levels_weight[:,:,2:,:]

        out = self.expand(fused_out_reduced)

        if self.vis:
            return out, levels_weight, fused_out_reduced.sum(dim=1)
        else:
            return out



class Neck(nn.Module):
    def __init__(self, spp_kernels=(5, 9, 13), PAN_layers=[512, 256], dropblock=True, sam=False, eca=False, ws=False, coord=False, hard_mish=False, asff=False, acff=False, bcn=False, mbn=False):
        """
        Initialize kernels dataset.

        Args:
            self: (todo): write your description
            spp_kernels: (str): write your description
            PAN_layers: (list): write your description
            dropblock: (todo): write your description
            sam: (todo): write your description
            eca: (todo): write your description
            ws: (int): write your description
            coord: (array): write your description
            hard_mish: (str): write your description
            asff: (str): write your description
            acff: (bool): write your description
            bcn: (todo): write your description
            mbn: (str): write your description
        """
        super().__init__()
        assert not(asff and acff)
        self.asff = asff
        self.acff = acff

        self.c1 = ConvBlock(1024, 512, 1, 1, "leaky", dropblock=False, sam=sam, eca=eca, ws=ws, coord=False, hard_mish=hard_mish, bcn=bcn, mbn=mbn)
        self.c2 = ConvBlock(512, 1024, 3, 1, "leaky", dropblock=False, sam=sam, eca=eca, ws=ws, coord=coord, hard_mish=hard_mish, bcn=bcn, mbn=mbn)
        self.c3 = ConvBlock(1024, 512, 1, 1, "leaky", dropblock=False, sam=sam, eca=eca, ws=ws, coord=False, hard_mish=hard_mish, bcn=bcn, mbn=mbn)

        # SPP block
        self.mp4_1 = nn.MaxPool2d(kernel_size=spp_kernels[0], stride=1, padding=spp_kernels[0] // 2)
        self.mp4_2 = nn.MaxPool2d(kernel_size=spp_kernels[1], stride=1, padding=spp_kernels[1] // 2)
        self.mp4_3 = nn.MaxPool2d(kernel_size=spp_kernels[2], stride=1, padding=spp_kernels[2] // 2)

        self.c5 = ConvBlock(2048, 512, 1, 1, "leaky", dropblock=False, sam=sam, eca=eca, ws=ws, coord=False, hard_mish=hard_mish, bcn=bcn, mbn=mbn)
        self.c6 = ConvBlock(512, 1024, 3, 1, "leaky", dropblock=dropblock, sam=sam, eca=eca, ws=ws, coord=coord, hard_mish=hard_mish, bcn=bcn, mbn=mbn)
        self.c7 = ConvBlock(1024, 512, 1, 1, "leaky", dropblock=False, sam=sam, eca=eca, ws=ws, coord=False, hard_mish=hard_mish, bcn=bcn, mbn=mbn)

        self.PAN8 = PAN_Layer(PAN_layers[0], dropblock=dropblock, sam=sam, eca=eca, ws=ws, coord=coord, hard_mish=hard_mish, bcn=bcn, mbn=mbn)
        self.PAN9 = PAN_Layer(PAN_layers[1], dropblock=dropblock, sam=sam, eca=eca, ws=ws, coord=coord, hard_mish=hard_mish, bcn=bcn, mbn=mbn)

        if asff: # branch inputs biggest objects: 512, medium objects: 256, smallest objects : 128
            self.ASFF_0 = ASFF(0)
            self.ASFF_1 = ASFF(1)
            self.ASFF_2 = ASFF(2)

        if acff:
            self.ACFF_0 = ACFF(0)
            self.ACFF_1 = ACFF(1)
            self.ACFF_2 = ACFF(2)

    def forward(self, input):
        """
        Perform forward operation.

        Args:
            self: (todo): write your description
            input: (todo): write your description
        """
        d5, d4, d3 = input

        x1 = self.c1(d5)
        x2 = self.c2(x1)
        x3 = self.c3(x2)

        x4_1 = self.mp4_1(x3)
        x4_2 = self.mp4_2(x3)
        x4_3 = self.mp4_3(x3)
        x4 = torch.cat([x4_1, x4_2, x4_3, x3], dim=1)

        x5 = self.c5(x4)
        x6 = self.c6(x5)
        x7 = self.c7(x6)

        x8 = self.PAN8(x7, d4)
        x9 = self.PAN9(x8, d3)

        if self.asff:
            x7ASFF = self.ASFF_0(x7, x8, x9)
            x8ASFF = self.ASFF_1(x7, x8, x9)
            x9ASFF = self.ASFF_2(x7, x8, x9)
            return x9ASFF, x8ASFF, x7ASFF

        if self.acff:
            x7ACFF = self.ACFF_0(x7, x8, x9)
            x8ACFF = self.ACFF_1(x7, x8, x9)
            x9ACFF = self.ACFF_2(x7, x8, x9)
            return x9ACFF, x8ACFF, x7ACFF


        return x9, x8, x7



class HeadPreprocessing(nn.Module):
    def __init__(self, in_channels, dropblock=True, sam=False, eca=False, ws=False, coord=False, hard_mish=False, bcn=False, mbn=False):
        """
        Initialize a channel.

        Args:
            self: (todo): write your description
            in_channels: (int): write your description
            dropblock: (todo): write your description
            sam: (todo): write your description
            eca: (todo): write your description
            ws: (int): write your description
            coord: (array): write your description
            hard_mish: (str): write your description
            bcn: (todo): write your description
            mbn: (str): write your description
        """
        super().__init__()
        ic = in_channels
        self.c1 = ConvBlock(ic, ic*2, 3, 2, 'leaky', dropblock=dropblock, sam=sam, eca=eca, ws=ws, coord=coord, hard_mish=hard_mish, bcn=bcn, mbn=mbn)
        self.c2 = ConvBlock(ic*4, ic*2, 1, 1, 'leaky', dropblock=dropblock, sam=sam, eca=eca, ws=ws, coord=False, hard_mish=hard_mish, bcn=bcn, mbn=mbn)
        self.c3 = ConvBlock(ic*2, ic*4, 3, 1, 'leaky', dropblock=dropblock, sam=sam, eca=eca, ws=ws, coord=coord, hard_mish=hard_mish, bcn=bcn, mbn=mbn)
        self.c4 = ConvBlock(ic*4, ic*2, 1, 1, 'leaky', dropblock=dropblock, sam=sam, eca=eca, ws=ws, coord=False, hard_mish=hard_mish, bcn=bcn, mbn=mbn)
        self.c5 = ConvBlock(ic*2, ic*4, 3, 1, 'leaky', dropblock=dropblock, sam=sam, eca=eca, ws=ws, coord=coord, hard_mish=hard_mish, bcn=bcn, mbn=mbn)
        self.c6 = ConvBlock(ic*4, ic*2, 1, 1, 'leaky', dropblock=dropblock, sam=sam, eca=eca, ws=ws, coord=False, hard_mish=hard_mish, bcn=bcn, mbn=mbn)

    def forward(self, input, input_prev):
        """
        Perform forward computation.

        Args:
            self: (todo): write your description
            input: (todo): write your description
            input_prev: (todo): write your description
        """
        x1 = self.c1(input_prev)
        x1 = torch.cat([x1, input], dim=1)
        x2 = self.c2(x1)
        x3 = self.c3(x2)
        x4 = self.c4(x3)
        x5 = self.c5(x4)
        x6 = self.c6(x5)

        return x6


class HeadOutput(nn.Module):
    def __init__(self, in_channels, out_channels, dropblock=True, sam=False, eca=False, ws=False, coord=False, hard_mish=False, bcn=False, mbn=False):
        """
        Initialize a channel.

        Args:
            self: (todo): write your description
            in_channels: (int): write your description
            out_channels: (int): write your description
            dropblock: (todo): write your description
            sam: (todo): write your description
            eca: (todo): write your description
            ws: (int): write your description
            coord: (array): write your description
            hard_mish: (str): write your description
            bcn: (todo): write your description
            mbn: (str): write your description
        """
        super().__init__()
        self.c1 = ConvBlock(in_channels, in_channels*2, 3, 1, "leaky", dropblock=False, sam=sam, eca=eca, ws=False, coord=coord, hard_mish=hard_mish, bcn=bcn, mbn=mbn)
        self.c2 = ConvBlock(in_channels*2, out_channels, 1, 1, "linear", bn=False, bias=True, dropblock=False, sam=False, eca=False, ws=False, coord=False, hard_mish=hard_mish, bcn=bcn, mbn=mbn)

    def forward(self, x):
        """
        Calculate of x1 and x1.

        Args:
            self: (todo): write your description
            x: (todo): write your description
        """
        x1 = self.c1(x)
        x2 = self.c2(x1)
        return x2


class Head(nn.Module):
    def __init__(self, output_ch, dropblock=True, sam=False, eca=False, ws=False, coord=False, hard_mish=False, bcn=False, mbn=False):
        """
        Initialize output.

        Args:
            self: (todo): write your description
            output_ch: (str): write your description
            dropblock: (todo): write your description
            sam: (todo): write your description
            eca: (todo): write your description
            ws: (int): write your description
            coord: (array): write your description
            hard_mish: (str): write your description
            bcn: (todo): write your description
            mbn: (str): write your description
        """
        super().__init__()

        self.ho1 = HeadOutput(128, output_ch, dropblock=dropblock, sam=sam, eca=eca, ws=ws, coord=coord, hard_mish=hard_mish, bcn=bcn, mbn=mbn)

        self.hp2 = HeadPreprocessing(128, dropblock=dropblock, sam=sam, eca=eca, ws=ws, coord=coord, hard_mish=hard_mish, bcn=bcn, mbn=mbn)
        self.ho2 = HeadOutput(256, output_ch, dropblock=dropblock, sam=sam, eca=eca, ws=ws, coord=coord, hard_mish=hard_mish, bcn=bcn, mbn=mbn)

        self.hp3 = HeadPreprocessing(256, dropblock=dropblock, sam=sam, eca=eca, ws=ws, coord=coord, hard_mish=hard_mish, bcn=bcn, mbn=mbn)
        self.ho3 = HeadOutput(512, output_ch, dropblock=dropblock, sam=sam, eca=eca, ws=ws, coord=coord, hard_mish=hard_mish, bcn=bcn, mbn=mbn)

    def forward(self, input):
        """
        Perform forward forward forward.

        Args:
            self: (todo): write your description
            input: (todo): write your description
        """
        input1, input2, input3 = input

        x1 = self.ho1(input1)
        x2 = self.hp2(input2, input1)
        x3 = self.ho2(x2)

        x4 = self.hp3(input3, x2)
        x5 = self.ho3(x4)

        return (x1, x3, x5)


class YOLOLayer(nn.Module):
    """Detection layer taken and modified from https://github.com/eriklindernoren/PyTorch-YOLOv3"""

    def __init__(self, anchors, num_classes, img_dim=608, grid_size=None, iou_aware=False, repulsion_loss=False):
        """
        Initialize the loss.

        Args:
            self: (todo): write your description
            anchors: (todo): write your description
            num_classes: (int): write your description
            img_dim: (int): write your description
            grid_size: (int): write your description
            iou_aware: (todo): write your description
            repulsion_loss: (str): write your description
        """
        super(YOLOLayer, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.ignore_thres = 0.5
        self.obj_scale = 1
        self.noobj_scale = 100
        self.metrics = {}
        self.img_dim = img_dim
        if grid_size:
            self.grid_size = grid_size
            self.compute_grid_offsets(self.grid_size)
        else:
            self.grid_size = 0  # grid size

        self.iou_aware = iou_aware
        self.repulsion_loss = repulsion_loss

    def compute_grid_offsets(self, grid_size, cuda=True):
        """
        Compute the grid grid offsets.

        Args:
            self: (todo): write your description
            grid_size: (int): write your description
            cuda: (todo): write your description
        """
        self.grid_size = grid_size
        g = self.grid_size
        FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        self.stride = self.img_dim / self.grid_size
        # Calculate offsets for each grid
        self.grid_x = torch.arange(g).repeat(g, 1).view([1, 1, g, g]).type(FloatTensor)
        self.grid_y = torch.arange(g).repeat(g, 1).t().view([1, 1, g, g]).type(FloatTensor)
        self.scaled_anchors = FloatTensor([(a_w / self.stride, a_h / self.stride) for a_w, a_h in self.anchors])
        self.anchor_w = self.scaled_anchors[:, 0:1].view((1, self.num_anchors, 1, 1))
        self.anchor_h = self.scaled_anchors[:, 1:2].view((1, self.num_anchors, 1, 1))

    def build_targets(self, pred_boxes, pred_cls, target, anchors, ignore_thres):
        """
        Builds the targets for the tensor.

        Args:
            self: (todo): write your description
            pred_boxes: (todo): write your description
            pred_cls: (str): write your description
            target: (todo): write your description
            anchors: (todo): write your description
            ignore_thres: (bool): write your description
        """

        ByteTensor = torch.cuda.BoolTensor if pred_boxes.is_cuda else torch.BoolTensor
        FloatTensor = torch.cuda.FloatTensor if pred_boxes.is_cuda else torch.FloatTensor

        nB = pred_boxes.size(0)
        nA = pred_boxes.size(1)
        nC = pred_cls.size(-1)
        nG = pred_boxes.size(2)

        # Output tensors
        obj_mask = ByteTensor(nB, nA, nG, nG).fill_(0)
        noobj_mask = ByteTensor(nB, nA, nG, nG).fill_(1)
        class_mask = FloatTensor(nB, nA, nG, nG).fill_(0)
        iou = FloatTensor(nB, nA, nG, nG).fill_(0)
        tx = FloatTensor(nB, nA, nG, nG).fill_(0)
        ty = FloatTensor(nB, nA, nG, nG).fill_(0)
        tw = FloatTensor(nB, nA, nG, nG).fill_(0)
        th = FloatTensor(nB, nA, nG, nG).fill_(0)
        tcls = FloatTensor(nB, nA, nG, nG, nC).fill_(0)

        target_boxes_grid = FloatTensor(nB, nA, nG, nG, 4).fill_(0)

        # 2 3 xy
        # 4 5 wh
        # Convert to position relative to box
        target_boxes = target[:, 2:6] * nG
        gxy = target_boxes[:, :2]
        gwh = target_boxes[:, 2:]

        # Get anchors with best iou
        ious = torch.stack([self.bbox_wh_iou(anchor, gwh) for anchor in anchors])
        best_ious, best_n = ious.max(0)

        # Separate target values
        b, target_labels = target[:, :2].long().t()
        gx, gy = gxy.t()
        gw, gh = gwh.t()
        gi, gj = gxy.long().t()

        # Setting target boxes to big grid, it would be used to count loss
        target_boxes_grid[b, best_n, gj, gi] = target_boxes

        # Set masks
        obj_mask[b, best_n, gj, gi] = 1
        noobj_mask[b, best_n, gj, gi] = 0

        # Set noobj mask to zero where iou exceeds ignore threshold
        for i, anchor_ious in enumerate(ious.t()):
            noobj_mask[b[i], anchor_ious > ignore_thres, gj[i], gi[i]] = 0

        # Coordinates
        tx[b, best_n, gj, gi] = gx - gx.floor()
        ty[b, best_n, gj, gi] = gy - gy.floor()

        # Width and height
        tw[b, best_n, gj, gi] = torch.log(gw / anchors[best_n][:, 0] + 1e-16)
        th[b, best_n, gj, gi] = torch.log(gh / anchors[best_n][:, 1] + 1e-16)

        # One-hot encoding of label (WE USE LABEL SMOOTHING)
        tcls[b, best_n, gj, gi, target_labels] = 0.9

        # Compute label correctness and iou at best anchor
        class_mask[b, best_n, gj, gi] = (pred_cls[b, best_n, gj, gi].argmax(-1) == target_labels).float()
        iou[b, best_n, gj, gi] = self.bbox_iou(pred_boxes[b, best_n, gj, gi], target_boxes, x1y1x2y2=False)

        tconf = obj_mask.float()

        return iou, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf, target_boxes_grid

    def bbox_wh_iou(self, wh1, wh2):
        """
        Returns the intersection of two boxes.

        Args:
            self: (todo): write your description
            wh1: (todo): write your description
            wh2: (todo): write your description
        """
        wh2 = wh2.t()
        w1, h1 = wh1[0], wh1[1]
        w2, h2 = wh2[0], wh2[1]
        inter_area = torch.min(w1, w2) * torch.min(h1, h2)
        union_area = (w1 * h1 + 1e-16) + w2 * h2 - inter_area
        return inter_area / union_area


    def bbox_iou(self, box1, box2, x1y1x2y2=True, get_areas = False):
        """
        Returns the IoU of two bounding boxes
        """
        if not x1y1x2y2:
            # Transform from center and width to exact coordinates
            b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
            b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
            b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
            b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
        else:
            # Get the coordinates of bounding boxes
            b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
            b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

        # get the coordinates of the intersection rectangle
        inter_rect_x1 = torch.max(b1_x1, b2_x1)
        inter_rect_y1 = torch.max(b1_y1, b2_y1)
        inter_rect_x2 = torch.min(b1_x2, b2_x2)
        inter_rect_y2 = torch.min(b1_y2, b2_y2)
        
        # Intersection area
        inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1, min=0) * torch.clamp(
            inter_rect_y2 - inter_rect_y1, min=0
        )
        # Union Area
        b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
        b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
        union_area = (b1_area + b2_area - inter_area + 1e-16)


        if get_areas:
            return inter_area, union_area

        iou = inter_area / union_area
        return iou


    def smallestenclosing(self, pred_boxes, target_boxes):
        """
        Calculate the smallest boxes.

        Args:
            self: (todo): write your description
            pred_boxes: (todo): write your description
            target_boxes: (todo): write your description
        """
        # Calculating smallest enclosing
        targetxc = target_boxes[..., 0]
        targetyc = target_boxes[..., 1]
        targetwidth = target_boxes[..., 2]
        targetheight = target_boxes[..., 3]

        predxc = pred_boxes[..., 0]
        predyc = pred_boxes[..., 1]
        predwidth = pred_boxes[..., 2]
        predheight = pred_boxes[..., 3]

        xc1 = torch.min(predxc - (predwidth/2), targetxc - (targetwidth/2))
        yc1 = torch.min(predyc - (predheight/2), targetyc - (targetheight/2))
        xc2 = torch.max(predxc + (predwidth/2), targetxc + (targetwidth/2))
        yc2 = torch.max(predyc + (predheight/2), targetyc + (targetheight/2))

        return xc1, yc1, xc2, yc2

    def xywh2xyxy(self, x):
        """
        Convert 2d 2d numpy. numpy. ndarray of 2d ndarrays to 2d 2darray.

        Args:
            self: (todo): write your description
            x: (int): write your description
        """
        # Convert bounding box format from [x, y, w, h] to [x1, y1, x2, y2]
        y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2
        y[:, 1] = x[:, 1] - x[:, 3] / 2
        y[:, 2] = x[:, 0] + x[:, 2] / 2
        y[:, 3] = x[:, 1] + x[:, 3] / 2
        return y

    def iou_all_to_all(self, a, b):
        """
        Convert a iouchch ) to a b.

        Args:
            self: (todo): write your description
            a: (array): write your description
            b: (array): write your description
        """
        #Calculates intersection over union area for each a bounding box with each b bounding box
        area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

        iw = torch.min(torch.unsqueeze(a[:, 2], dim=1), b[:, 2]) - torch.max(torch.unsqueeze(a[:, 0], 1), b[:, 0])
        ih = torch.min(torch.unsqueeze(a[:, 3], dim=1), b[:, 3]) - torch.max(torch.unsqueeze(a[:, 1], 1), b[:, 1])

        iw = torch.clamp(iw, min=0)
        ih = torch.clamp(ih, min=0)

        ua = torch.unsqueeze((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), dim=1) + area - iw * ih

        ua = torch.clamp(ua, min=1e-8)

        intersection = iw * ih

        IoU = intersection / ua

        return IoU

    def smooth_ln(self, x, smooth =0.5):
        """
        Smooth log - 1 ].

        Args:
            self: (todo): write your description
            x: (int): write your description
            smooth: (todo): write your description
        """
        return torch.where(
            torch.le(x, smooth),
            -torch.log(1 - x),
            ((x - smooth) / (1 - smooth)) - math.log(1 - smooth)
        )

    def iog(self, ground_truth, prediction):
        """
        Compute the prediction

        Args:
            self: (todo): write your description
            ground_truth: (array): write your description
            prediction: (array): write your description
        """

        inter_xmin = torch.max(ground_truth[:, 0], prediction[:, 0])
        inter_ymin = torch.max(ground_truth[:, 1], prediction[:, 1])
        inter_xmax = torch.min(ground_truth[:, 2], prediction[:, 2])
        inter_ymax = torch.min(ground_truth[:, 3], prediction[:, 3])
        Iw = torch.clamp(inter_xmax - inter_xmin, min=0)
        Ih = torch.clamp(inter_ymax - inter_ymin, min=0)
        I = Iw * Ih
        G = (ground_truth[:, 2] - ground_truth[:, 0]) * (ground_truth[:, 3] - ground_truth[:, 1])
        return I / G

    def calculate_repullsion(self, y, y_hat):
        """
        Calculate the empirical from target.

        Args:
            self: (todo): write your description
            y: (todo): write your description
            y_hat: (todo): write your description
        """
        batch_size = y_hat.shape[0]
        RepGTS = []
        RepBoxes = []
        for bn in range(batch_size):
            #Repulsion between prediction bbox and neighboring target bbox, which are not target for this bounding box. (pred bbox <- -> 2nd/3rd/... by iou target bbox)
            pred_bboxes = self.xywh2xyxy(y_hat[bn, :, :4])
            bn_mask = y[:, 0] == bn
            gt_bboxes = self.xywh2xyxy(y[bn_mask, 2:] * 608)
            iou_anchor_to_target = self.iou_all_to_all(pred_bboxes, gt_bboxes)
            val, ind = torch.topk(iou_anchor_to_target, 2)
            second_closest_target_index = ind[:, 1]
            second_closest_target = gt_bboxes[second_closest_target_index]
            RepGT = self.smooth_ln(self.iog(second_closest_target, pred_bboxes)).mean()
            RepGTS.append(RepGT)

            #Repulsion between pred bbox and pred bbox, which are not refering to the same target bbox.
            have_target_mask = val[:, 0] != 0
            anchors_with_target = pred_bboxes[have_target_mask]
            iou_anchor_to_anchor = self.iou_all_to_all(anchors_with_target, anchors_with_target)
            other_mask = (torch.eye(iou_anchor_to_anchor.shape[0]) == 0).to(iou_anchor_to_anchor.device)
            different_target_mask = (ind[have_target_mask, 0] != ind[have_target_mask, 0].unsqueeze(1))
            iou_atoa_filtered = iou_anchor_to_anchor[other_mask & different_target_mask]
            RepBox = self.smooth_ln(iou_atoa_filtered).sum()/iou_atoa_filtered.sum()
            RepBoxes.append(RepBox)
        return torch.stack(RepGTS).mean(), torch.stack(RepBoxes).mean()

    def forward(self, x, targets=None):
        """
        Forward computation.

        Args:
            self: (todo): write your description
            x: (todo): write your description
            targets: (todo): write your description
        """
        # Tensors for cuda support
        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor

        num_samples = x.size(0)
        grid_size = x.size(2)

        if self.iou_aware:
            not_class_channels = 6
        else:
            not_class_channels = 5
        prediction = (
            x.view(num_samples, self.num_anchors, self.num_classes + not_class_channels, grid_size, grid_size)
            .permute(0, 1, 3, 4, 2)
            .contiguous()
        )

        # Get outputs
        x = torch.sigmoid(prediction[..., 0])  # Center x
        y = torch.sigmoid(prediction[..., 1])  # Center y
        w = prediction[..., 2]  # Width
        h = prediction[..., 3]  # Height
        pred_conf = torch.sigmoid(prediction[..., 4])  # Conf
        if not self.iou_aware:
            pred_cls = torch.sigmoid(prediction[..., 5:])  # Cls pred
        else:
            pred_cls = torch.sigmoid(prediction[..., 5:-1])# Cls pred
            pred_iou = torch.sigmoid(prediction[..., -1]) #IoU pred

        # If grid size does not match current we compute new offsets
        if grid_size != self.grid_size or self.grid_x.is_cuda != x.is_cuda:
            self.compute_grid_offsets(grid_size, cuda=x.is_cuda)

        # Add offset and scale with anchors
        pred_boxes = FloatTensor(prediction[..., :4].shape)
        pred_boxes[..., 0] = x + self.grid_x
        pred_boxes[..., 1] = y + self.grid_y
        pred_boxes[..., 2] = torch.exp(w) * self.anchor_w
        pred_boxes[..., 3] = torch.exp(h) * self.anchor_h

        output = torch.cat(
            (
                pred_boxes.view(num_samples, -1, 4) * self.stride,
                pred_conf.view(num_samples, -1, 1),
                pred_cls.view(num_samples, -1, self.num_classes),
            ),
            -1,
        )

        # OUTPUT IS ALL BOXES WITH THEIR CONFIDENCE AND WITH CLASS
        if targets is None:
            return output, 0

        iou, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf, target_boxes = self.build_targets(
                pred_boxes=pred_boxes,
                pred_cls=pred_cls,
                target=targets,
                anchors=self.scaled_anchors,
                ignore_thres=self.ignore_thres
        )

        # Diagonal length of the smallest enclosing box (is already squared)
        xc1, yc1, xc2, yc2 = self.smallestenclosing(pred_boxes[obj_mask], target_boxes[obj_mask])
        c = ((xc2 - xc1) ** 2) + ((yc2 - yc1) ** 2) + 1e-7

        # Euclidean distance between central points
        d = (tx[obj_mask] - x[obj_mask]) ** 2 + (ty[obj_mask] - y[obj_mask]) ** 2

        rDIoU = d/c

        iou_masked = iou[obj_mask]
        v = (4 / (math.pi ** 2)) * torch.pow((torch.atan(tw[obj_mask]/th[obj_mask])-torch.atan(w[obj_mask]/h[obj_mask])), 2)

        with torch.no_grad():
            S = 1 - iou_masked
            alpha = v / (S + v + 1e-7)

        CIoUloss = (1 - iou_masked + rDIoU + alpha * v).sum(0)/num_samples
        # print(torch.isnan(pred_conf).sum())
        loss_conf_obj = F.binary_cross_entropy(pred_conf[obj_mask], tconf[obj_mask])
        loss_conf_noobj = F.binary_cross_entropy(pred_conf[noobj_mask], tconf[noobj_mask])
        loss_conf = self.obj_scale * loss_conf_obj + self.noobj_scale * loss_conf_noobj

        loss_cls = F.binary_cross_entropy(input=pred_cls[obj_mask], target=tcls[obj_mask])

        total_loss = CIoUloss + loss_cls + loss_conf

        if self.iou_aware:
            pred_iou_masked = pred_iou[obj_mask]

            # print("Pred iou", pred_iou.shape)
            # print("IOU masked", iou_masked.shape)
            # print("Pred iou", pred_iou)
            # print("IOU masked", iou_masked)
            # print("pred iou masked", pred_iou_masked.shape)
            # print("pred iou masked", pred_iou_masked)
            # print(F.binary_cross_entropy(pred_iou_masked, iou_masked.detach()))
            total_loss += F.binary_cross_entropy(pred_iou_masked, iou_masked.detach())   

        if self.repulsion_loss:
            repgt, repbox = self.calculate_repullsion(targets, output)
            total_loss += 0.5 * repgt + 0.5 * repbox

        # print(f"C: {c}; D: {d}")
        # print(f"Confidence is object: {loss_conf_obj}, Confidence no object: {loss_conf_noobj}")
        # print(f"IoU: {iou_masked}; DIoU: {rDIoU}; alpha: {alpha}; v: {v}")
        # print(f"CIoU : {CIoUloss.item()}; Confindence: {loss_conf.item()}; Class loss should be because of label smoothing: {loss_cls.item()}")
        return output, total_loss


class YOLOv4(nn.Module):
    def __init__(self, in_channels=3, n_classes=80, weights_path=None, pretrained=False, img_dim=608, anchors=None, dropblock=True, sam=False, eca=False, ws=False, iou_aware=False, coord=False, hard_mish=False, asff=False, repulsion_loss=False, acff=False, bcn=False, mbn=False):
        """
        Initialize the graph.

        Args:
            self: (todo): write your description
            in_channels: (int): write your description
            n_classes: (todo): write your description
            weights_path: (str): write your description
            pretrained: (bool): write your description
            img_dim: (int): write your description
            anchors: (todo): write your description
            dropblock: (todo): write your description
            sam: (todo): write your description
            eca: (todo): write your description
            ws: (int): write your description
            iou_aware: (todo): write your description
            coord: (array): write your description
            hard_mish: (str): write your description
            asff: (str): write your description
            repulsion_loss: (str): write your description
            acff: (bool): write your description
            bcn: (todo): write your description
            mbn: (str): write your description
        """
        super().__init__()
        if anchors is None:
            anchors = [[[10, 13], [16, 30], [33, 23]],
                       [[30, 61], [62, 45], [59, 119]],
                       [[116, 90], [156, 198], [373, 326]]]

        output_ch = (4 + 1 + n_classes) * 3
        if iou_aware:
            output_ch += 3 #1 for iou

        self.img_dim = img_dim

        self.backbone = Backbone(in_channels, dropblock=False, sam=sam, eca=eca, ws=ws, coord=coord, hard_mish=hard_mish, bcn=bcn, mbn=mbn)

        self.neck = Neck(dropblock=dropblock, sam=sam, eca=eca, ws=ws, coord=coord, hard_mish=hard_mish, asff=asff, acff=acff, bcn=bcn, mbn=mbn)

        self.head = Head(output_ch, dropblock=False, sam=sam, eca=eca, ws=ws, coord=coord, hard_mish=hard_mish, bcn=bcn)

        self.yolo1 = YOLOLayer(anchors[0], n_classes, img_dim, iou_aware=iou_aware, repulsion_loss=repulsion_loss)
        self.yolo2 = YOLOLayer(anchors[1], n_classes, img_dim, iou_aware=iou_aware, repulsion_loss=repulsion_loss)
        self.yolo3 = YOLOLayer(anchors[2], n_classes, img_dim, iou_aware=iou_aware, repulsion_loss=repulsion_loss)

        if weights_path:
            try:  # If we change input or output layers amount, we will have an option to use pretrained weights
                self.load_state_dict(torch.load(weights_path), strict=False)
            except RuntimeError as e:
                print(f'[Warning] Ignoring {e}')
        elif pretrained:
            try:  # If we change input or output layers amount, we will have an option to use pretrained weights
                self.load_state_dict(torch.hub.load_state_dict_from_url("https://github.com/VCasecnikovs/Yet-Another-YOLOv4-Pytorch/releases/download/V1.0/yolov4.pth"), strict=False)
            except RuntimeError as e:
                print(f'[Warning] Ignoring {e}')

    def forward(self, x, y=None):
        """
        Forward computation.

        Args:
            self: (todo): write your description
            x: (todo): write your description
            y: (todo): write your description
        """
        b = self.backbone(x)
        n = self.neck(b)
        h = self.head(n)

        h1, h2, h3 = h

        out1, loss1 = self.yolo1(h1, y)
        out2, loss2 = self.yolo2(h2, y)
        out3, loss3 = self.yolo3(h3, y)

        out1 = out1.detach()
        out2 = out2.detach()
        out3 = out3.detach()

        out = torch.cat((out1, out2, out3), dim=1)

        loss = (loss1 + loss2 + loss3)/3

        return out, loss


if __name__ == "__main__":
    import time
    import numpy as np

    model = YOLOv4().cuda().eval()
    x = torch.ones((1, 3, 608, 608)).cuda()
    y = torch.from_numpy(np.asarray([[0, 1, 0.5, 0.5, 0.3, 0.3]])).float().cuda()

    for i in range(1):
        t0 = time.time()
        y_hat, loss = model(x, y)
        t1 = time.time()
        print(t1 - t0)

    print(loss)
