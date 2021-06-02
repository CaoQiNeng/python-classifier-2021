# -*- coding: utf-8 -*-
"""
Created on Apr 29 16:00 2021
Attention and ResNet34
@author: LYM
"""

from common import *
import pretrainedmodels

import torch.nn as nn
import math

from torch.autograd import Variable
from graphviz import Digraph
from torchsummary import summary

from torchviz import make_dot
import torch

from torch.autograd import Variable
from torchkeras import summary
from model_AttenCNN import *


# from model_code.model_AttenCNN import *

class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, downsample=None,dropout_rate = 0.5):
        super(BasicBlock, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.stride = stride
        self.kernel_size = kernel_size
        self.bn1 = nn.BatchNorm1d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.conv1 = nn.Conv1d(in_planes, out_planes, kernel_size=self.kernel_size, stride=self.stride,padding=int(self.kernel_size / 2), bias=False)
        self.bn2 =  nn.BatchNorm1d(out_planes)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.conv2 = nn.Conv1d(out_planes, out_planes, kernel_size=self.kernel_size, stride=1,padding=int(self.kernel_size / 2))
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.dropout1(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout2(out)
        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class Net(nn.Module):

    def __init__(self, in_planes, num_classes=27, kernel_size=15, dropout_rate = 0.5):   #num_classes=1000
        super(Net, self).__init__()
        self.dilation = 1
        self.in_planes = in_planes
        self.out_planes = 64
        self.stride = 1
        self.kernel_size = kernel_size

        # pre conv
        self.conv1 = nn.Conv1d(self.in_planes, self.out_planes, kernel_size=self.kernel_size, stride=1, padding=int(self.kernel_size/2),bias=False)#(12,64,15,s=1,p=7)
        self.in_planes = self.out_planes
        self.bn1 = nn.BatchNorm1d(self.out_planes)
        self.relu = nn.ReLU(inplace=True)

        # # add attention----l
        # self.ca = ChannelAttention(self.in_planes)
        # self.sa = SpatialAttention()
        # end attention-------------------

        # first block
        self.conv2 = nn.Conv1d(self.out_planes, self.out_planes, kernel_size=self.kernel_size, stride=2, padding=int(self.kernel_size/2),bias=False)#(64,64,15,s=2,p=7)
        self.bn2 = nn.BatchNorm1d(self.out_planes)
        self.dropout = nn.Dropout(dropout_rate)
        self.conv3 = nn.Conv1d(self.out_planes, self.out_planes, kernel_size=self.kernel_size, stride=1, padding=int(self.kernel_size/2),bias=False)#(64,64,15,s=1,p=7)
        self.maxpool = nn.MaxPool1d(kernel_size=self.kernel_size, stride=2, padding=int(self.kernel_size/2))

        layers = []
        for i in range(1, 16):
            if i % 4 == 0 :
                self.out_planes = self.in_planes + 64

            if i % 4 == 0 :
                downsample = nn.Sequential(
                    nn.Conv1d(self.in_planes, self.out_planes, kernel_size=1, stride=2, bias=False),
                    nn.BatchNorm1d(self.out_planes))
                self.stride = 2
            elif i % 2 == 0 :
                downsample = self.maxpool
                self.stride = 2
            else :
                downsample = None
                self.stride = 1

            layers.append(BasicBlock(self.in_planes, self.out_planes, self.kernel_size, self.stride, downsample))

            self.in_planes = self.out_planes

        self.layers = nn.Sequential(*layers)

        # # add attention -----l
        # self.ca1 = ChannelAttention(self.in_planes)
        # self.sa1 = SpatialAttention()
        # end attention-------------------

        self.bn3 = nn.BatchNorm1d(self.out_planes)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(self.out_planes, num_classes)



    def forward(self, x):
        # pre conv
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        # add attention-------------------
        # x = self.ca(x) * x
        # x = self.sa(x) * x
        # x = self.maxpool(x)
        # end attention-------------------

        # first block
        identity = x
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv3(x)
        x = x + self.maxpool(identity)

        # res block x 15
        x = self.layers(x)

        x = self.bn3(x)
        x = self.relu(x)

        # add attention-------------------
        # x = self.ca1(x) * x
        # x = self.sa1(x) * x
        # end attention-------------------



        x = self.avgpool(x)

        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

# def run_check_net():
#     net = Net(12, num_classes=27, dropout_rate = 0.5)   #num_classes=9
#
#     input = torch.randn(1, 12, 72000)  #(batch-size, )
#     # print('input-size=',input.shape)
#     output = net(input)
#
#     # print('output-size=', output.shape)
#
#     # model graph
#     # g = make_dot(output)
#     # g.view()
#
#     summary(net, (12, 72000))
#     # summary(net, (1, 12, 72000))
#     print('======Network structure:======\n')
#     # print(net)
#     print('==============================\n')
#
#     # print(torch_summarize(net))
#
#     # model structure
#     # params = list(net.parameters())
#     # k = 0
#     # for i in params:
#     #     l = 1
#     #     print("Structure of Layer：" + str(list(i.size())))
#     #     for j in i.size():
#     #         l *= j
#     #     print("Sum parameters of Layer：" + str(l))
#     #     k = k + l
#     # print("Total parameters of model：" + str(k))



def metric(truth, predict):
    truth_for_cls = np.sum(truth, axis=0) + 1e-11
    predict_for_cls = np.sum(predict, axis=0) + 1e-11

    # TP
    count = truth + predict
    count[count != 2] = 0
    TP = np.sum(count, axis=0) / 2

    precision = TP / predict_for_cls
    recall = TP / truth_for_cls

    return precision, recall




# main #################################################################



if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))
    input = torch.randn(1, 12, 72000)  # (batch-size, )
    net = Net(12, num_classes=27, dropout_rate=0.5)
    # run_check_basenet()
    # run_check_net()
    # y = run_check_net()
    summary(net, (12, 72000))
    # summary(net, (1, 12, 72000))
    print('======Network structure:======\n')
    # print(net)
    print('==============================\n')

    output = net(input)
    print('\nsuccess!')


