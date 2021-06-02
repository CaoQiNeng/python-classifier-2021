# -*- coding: utf-8 -*-
"""
Created on Apr 29 11:40 2021

@author: LYM
"""

import torch.nn as nn
import torch
from functools import reduce
# from torchviz import make_dot
from torchsummary import summary
from torch.nn import functional as F
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
"""
SENet
"""
class mySENetBlock(nn.Module):
    def __init__(self, channel, r=16):

        super(mySENetBlock, self).__init__()
        # self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel//r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel//r, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # b, c, _, _ = x.size()
        b, c, _ = x.size()
        # Squeeze
        y = self.avg_pool(x).view(b, c)
        # print('y1_size:', y.shape)
        # Excitation
        # y = self.fc(y).view(b, c, 1, 1)
        y = self.fc(y).view(b, c, 1)
        # print('y2_size:', y.shape)
        # Fscale
        # print(x.size())
        # print(y.size())
        y = torch.mul(x, y)
        # print(y.size())
        return y

"""
SkNet
"""
class mySKNetBlock(nn.Module):
    def __init__(self,in_channels,out_channels,stride=1,M=2,r=16,L=32):
        super(mySKNetBlock, self).__init__()
        d = max(in_channels//r, L)   #
        self.M = M
        self.out_channels = out_channels
        self.conv = nn.ModuleList()  #
        for i in range(M):
            #
            self.conv.append(nn.Sequential(nn.Conv1d(in_channels, out_channels, 3, stride, padding=1+i, dilation=1+i, groups=32, bias=False),
                                           nn.BatchNorm1d(out_channels),
                                           nn.ReLU(inplace=True)))
        self.global_pool = nn.AdaptiveAvgPool1d(1)  #
        self.fc1 = nn.Sequential(nn.Conv1d(out_channels, d, 1, bias=False),
                               nn.BatchNorm1d(d),
                               nn.ReLU(inplace=True))   #
        self.fc2 = nn.Conv1d(d, out_channels*M, 1, 1, bias=False)  #
        self.softmax = nn.Softmax(dim=1)  #

    def forward(self, input):
        batch_size = input.size(0)
        print('batch_size:', batch_size)
        output = []
        # the part of split
        for i, conv in enumerate(self.conv):
            # print(i,conv(input).size())
            output.append(conv(input))
        # the part of fusion
        U = reduce(lambda x, y:x+y, output)
        s = self.global_pool(U)
        z = self.fc1(s)  # S->Z
        a_b = self.fc2(z)  # Z->a，b
        a_b = a_b.reshape(batch_size, self.M, self.out_channels)
        a_b = self.softmax(a_b) #
        # the part of selection
        a_b = list(a_b.chunk(self.M, dim=1))   #split to a and b
        a_b = list(map(lambda x: x.reshape(batch_size, self.out_channels, 1), a_b))
        V = list(map(lambda x, y: x*y, output, a_b))
        V = reduce(lambda x, y: x+y, V)
        return V


"""
CBAM
channel domain：ChannelAttention
Spatial domain：SpatialAttention
"""
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.fc1 = nn.Conv1d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv1d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        out = self.sigmoid(out)
        # print('shape of weigths', out.size())
        y = torch.mul(x, out)
        return y

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv1d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(out)
        out = self.sigmoid(out)
        # print('shape of weigths', out.size())
        y = torch.mul(x, out)
        return y



"""
GCNet
Global Context Attention Module
"""
class ContextBlock(nn.Module):
    def __init__(self, inplanes, ratio, pooling_type='att',
                    fusion_types = ('channel_add', )):
        super(ContextBlock, self).__init__()
        valid_fusion_types = ['channel_add', 'channel_mul']
        assert pooling_type in ['avg', 'att']
        assert isinstance(fusion_types, (list, tuple))
        assert all([f in valid_fusion_types for f in fusion_types])
        assert len(fusion_types) > 0, 'at least one fusion should be used'
        self.inplanes = inplanes
        self.ratio = ratio
        self.planes = int(inplanes * ratio)
        self.pooling_type = pooling_type
        self.fusion_types = fusion_types
        if pooling_type == 'att':
            # self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1)
            self.conv_mask = nn.Conv1d(inplanes, 1, kernel_size=1)
            self.softmax = nn.Softmax(dim=2)
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            # self.avg_pool = nn.AdaptiveAvgPool1d(1)
        if 'channel_add' in fusion_types:
            self.channel_add_conv = nn.Sequential(
                # nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.Conv1d(self.inplanes, self.planes, kernel_size=1),
                # nn.LayerNorm([self.planes, 1, 1]),
                nn.LayerNorm([self.planes, 1]),
                nn.ReLU(inplace=True),
                # yapf: disable
                # nn.Conv2d(self.planes, self.inplanes, kernel_size=1))
                nn.Conv1d(self.planes, self.inplanes, kernel_size=1))
        else:
            self.channel_add_conv = None
        if 'channel_mul' in fusion_types:
            self.channel_mul_conv = nn.Sequential(
                # nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.Conv1d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),       # yapf: disable
                # nn.Conv2d(self.planes, self.inplanes, kernel_size=1))
                nn.Conv1d(self.planes, self.inplanes, kernel_size=1))
        else:
            self.channel_mul_conv = None

    def spatial_pool(self, x):
        # batch, channel, height, width = x.size()
        # print(batch, channel, height, width)     #[2, 64, 128, 128]    128*128=16384
        batch, channel, d = x.size()
        print(batch, channel, d)                   #[2,64,72000]

        if self.pooling_type == 'att':
            input_x = x
            # [N, C, H * W]
            # input_x = input_x.view(batch, channel, height * width)
            input_x = input_x.view(batch, channel, -1)
            # print('input1=', input_x.shape)                 #[2, 64, 16384]        [2, 64, 72000]
            # [N, 1, C, H * W]
            input_x = input_x.unsqueeze(1)
            # print('input2=', input_x.shape)                 #[2, 1, 64, 16384]     [2, 1, 64, 72000]
            # [N, 1, H, W]
            context_mask = self.conv_mask(x)
            # print('context_mask1=', context_mask.shape)     #[2, 1, 128, 128]      [2, 1, 72000]
            # [N, 1, H * W]
            # context_mask = context_mask.view(batch, 1, height * width)
            context_mask = context_mask.view(batch, 1, -1)
            # print('context_mask2=', context_mask.shape)     #[2, 1, 16384]        [2, 1, 72000]
            # [N, 1, H * W]
            context_mask = self.softmax(context_mask)
            # print('context_mask3=', context_mask.shape)     #[2, 1, 16384]        [2, 1, 72000]
            # [N, 1, H * W, 1]
            context_mask = context_mask.unsqueeze(-1)
            # print('context_mask4=', context_mask.shape)     #[2, 1, 16384, 1]     [2, 1, 72000, 1]
            # [N, 1, C, 1]
            # print('input3=', input_x.shape)                 #[2, 1, 64, 16384]    [2, 1, 64, 72000]
            # print('context_mask5=', context_mask.shape)     #[2, 1, 16384, 1]     [2, 1, 72000, 1]
            context = torch.matmul(input_x, context_mask)
            # print('context1=', context.shape)               # [2, 1, 64, 1]       [2, 1, 64, 1]
            # [N, C, 1, 1]
            # context = context.view(batch, channel, 1, 1)
            context = context.view(batch, channel, -1)
            # print('context2=', context.shape)               #[2, 64, 1, 1]       [2, 64, 1]
        else:
            # [N, C, 1, 1]
            context = self.avg_pool(x)
            # print('context3=', context.shape)
        return context

    def forward(self, x):

        # [N, C, 1, 1]
        context = self.spatial_pool(x)
        # print('context4=', context.shape)           #[2, 64, 1, 1]              [2, 64, 1]
        out = x
        # print('out1=', out.shape)                   #[2, 64, 128, 128]          [2, 64, 72000]
        if self.channel_mul_conv is not None:
            # [N, C, 1, 1]
            channel_mul_term = torch.sigmoid(self.channel_mul_conv(context))
            # print('channel_mul_term=', channel_mul_term.shape)
            out = out * channel_mul_term
        if self.channel_add_conv is not None:
            # [N, C, 1, 1]
            channel_add_term = self.channel_add_conv(context)
            # print('channel_add_term=', channel_add_term.shape)   #[2, 64, 1, 1]        [2, 64, 1]
            out = out + channel_add_term
            # print('out2=', out.shape)                            #[2, 64, 128, 128]    [2, 64, 72000]
        return out

"""
CCNet
Criss-Cross Attention Module
"""
# def _check_contiguous(*args):
#     if not all([mod is None or mod.is_contiguous() for mod in args]):
#         raise ValueError("Non-contiguous input")
#
# class CA_Weight(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, t, f):
#         # Save context
#         n, c, h, w = t.size()
#         size = (n, h + w - 1, h, w)
#         weight = torch.zeros(size, dtype=t.dtype, layout=t.layout, device=t.device)
#         _ext.ca_forward_cuda(t, f, weight)
#         # Output
#         ctx.save_for_backward(t, f)
#         return weight
#
#     @staticmethod
#     @once_differentiable
#     def backward(ctx, dw):
#         t, f = ctx.saved_tensors
#         dt = torch.zeros_like(t)
#         df = torch.zeros_like(f)
#         _ext.ca_backward_cuda(dw.contiguous(), t, f, dt, df)
#         _check_contiguous(dt, df)
#         return dt, df

# class CA_Map(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, weight, g):
#         # Save context
#         out = torch.zeros_like(g)
#         _ext.ca_map_forward_cuda(weight, g, out)
#         # Output
#         ctx.save_for_backward(weight, g)
#         return out
#
#     @staticmethod
#     @once_differentiable
#     def backward(ctx, dout):
#         weight, g = ctx.saved_tensors
#         dw = torch.zeros_like(weight)
#         dg = torch.zeros_like(g)
#         _ext.ca_map_backward_cuda(dout.contiguous(), weight, g, dw, dg)
#         _check_contiguous(dw, dg)
#         return dw, dg
#
# ca_weight = CA_Weight.apply
# ca_map = CA_Map.apply
#
# class CrissCrossAttention(nn.Module):
#     """ Criss-Cross Attention Module"""
#     def __init__(self,in_dim):
#         super(CrissCrossAttention,self).__init__()
#         self.chanel_in = in_dim
#
#         self.query_conv = nn.Conv2d(in_channels = in_dim, out_channels = in_dim//8, kernel_size = 1)
#         self.key_conv = nn.Conv2d(in_channels = in_dim, out_channels = in_dim//8, kernel_size = 1)
#         self.value_conv = nn.Conv2d(in_channels = in_dim, out_channels = in_dim, kernel_size = 1)
#         self.gamma = nn.Parameter(torch.zeros(1))
#
#     def forward(self,x):
#         proj_query = self.query_conv(x)
#         proj_key = self.key_conv(x)
#         proj_value = self.value_conv(x)
#
#         energy = ca_weight(proj_query, proj_key)
#         attention = F.softmax(energy, 1)
#         out = ca_map(attention, proj_value)
#         out = self.gamma*out + x
#
#         return out



if __name__ == '__main__':
    # input = torch.randn(2, 32, 256, 1)
    input = torch.randn(2, 64, 72000)

    # net
    net = mySENetBlock(64,16)
    # net = mySKNetBlock(64, 64)

    # net = ChannelAttention(64)
    # net = SpatialAttention()

    # in_tensor = torch.ones((2, 64, 128, 128))
    # in_tensor = torch.ones((2, 64, 72000))
    # cb = ContextBlock(inplanes=64, ratio=1. / 16., pooling_type='att')
    # out_tensor = cb(in_tensor)
    # print(in_tensor.shape)
    # print(out_tensor.shape)


    # structure of net
    # summary(net, (64, 72000))
    # summary(cb, (64, 72000))

    # net = CrissCrossAttention(64)
    output = net(input)

    print('input-size', input.size())
    print('output-size', output.size())