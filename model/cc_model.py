import argparse
import os

import numpy as np

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from einops import rearrange , repeat
from einops.layers.torch import  Rearrange
from utils.SwinT import SwinT
# from utils.white import White_balance
# from utils.white import CLAHE
def make_model(args, parent=False):
    return CC_Module(args)

"""CBAM注意力机制"""

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=False, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes

        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = torch.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale

def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out) # broadcasting
        return x * scale

class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()
    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out






class MFFA(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(MFFA,self).__init__()
        self.R = nn.Conv2d(1,out_channels,kernel_size=1,stride=1,padding='same')
        self.G = nn.Conv2d(1,out_channels,kernel_size=3,stride=1,padding='same')
        self.B = nn.Conv2d(1,out_channels,kernel_size=5,stride=1,padding='same')
        self.C = nn.Conv2d(in_channels*3,out_channels,kernel_size=1,stride=1,padding='same')
        self.BN = nn.ReLU()
        self.CBAM = CBAM(in_channels)


    def forward(self,x):
        input_1 = torch.unsqueeze(x[:,0,:,:],dim=1)
        input_2 = torch.unsqueeze(x[:, 1, :, :], dim=1)
        input_3 = torch.unsqueeze(x[:, 2, :, :], dim=1)
        l1_1 = self.R(input_1)
        l1_2 = self.BN(l1_1)
        l2_1 = self.G(input_2)
        l2_2 = self.BN(l2_1)
        l3_1 = self.B(input_3)
        l3_2 = self.BN(l3_1)
        l_out = torch.cat((l1_2,l2_2,l3_2),1)
        l_out1 = self.C(l_out)
        finan_out = self.CBAM(l_out1)
        finan_out = torch.add(x,finan_out)
        return finan_out,l_out1





class TTT(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=3,kernel_size=3,stride=1,padding='same')
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=3,out_channels=3,kernel_size=3,stride=1,padding='same')
    def forward(self,x):
        l1 = self.conv1(x)
        l2 = self.relu(l1)
        l3 = self.conv2(l2)
        return l3






class Conv3(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv3, self).__init__()
        self.Conv3 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.Conv3(x)
        return x





class CC_Module(nn.Module):
    def __init__(self, args):
         super(CC_Module, self).__init__()
         scale = args.scale[0]
         self.Pretreatment = Conv3(3, 50)

         self.Conv0 = Conv3(50,50)
         self.Conv00 = Conv3(50, 50)
         self.Conv000 = Conv3(50, 50)
         self.Conv0000 = Conv3(50, 50)
         self.Conv00000 = Conv3(50, 50)
         self.Conv000000 = Conv3(50, 50)
         self.Conv0000000 = Conv3(50, 50)
         self.Conv00000000 = Conv3(50, 50)
         self.Conv000000000 = Conv3(50, 50)
         self.Conv0000000000 = Conv3(50, 50)
         self.Conv00000000000 = Conv3(50, 50)
         self.Conv000000000000 = Conv3(50, 50)

         self.head = Conv3(3, 50)

         self.RGB1 = MFFA(50, 50)
         self.RGB2 = MFFA(50, 50)
         self.RGB3 = MFFA(50, 50)
         self.RGB4 = MFFA(50, 50)
         self.RGB5 = MFFA(50, 50)
         self.RGB6 = MFFA(50, 50)
         self.RGB7 = MFFA(50, 50)
         self.RGB8 = MFFA(50, 50)
         self.RGB9 = MFFA(50, 50)
         self.RGB10 = MFFA(50, 50)
         self.RGB11 = MFFA(50, 50)
         self.RGB12 = MFFA(50, 50)
         self.RGB13 = MFFA(50, 50)
         self.RGB14 = MFFA(50, 50)
         self.RGB15 = MFFA(50, 50)
         self.RGB16 = MFFA(50, 50)
         self.RGB17 = MFFA(50, 50)
         self.RGB18 = MFFA(50, 50)
         self.RGB19 = MFFA(50, 50)
         self.RGB20 = MFFA(50, 50)
         self.RGB21 = MFFA(50, 50)
         self.RGB22 = MFFA(50, 50)
         self.RGB23 = MFFA(50, 50)
         self.RGB24 = MFFA(50, 50)

         self.Swin1 = SwinT()
         self.Swin2 = SwinT()
         self.Swin3 = SwinT()
         self.Swin4 = SwinT()
         self.Swin5 = SwinT()
         self.Swin6 = SwinT()
         self.Swin7 = SwinT()
         self.Swin8 = SwinT()
         self.Swin9 = SwinT()
         self.Swin10 = SwinT()
         self.Swin11 = SwinT()
         self.Swin12 = SwinT()

         self.Conv1 = nn.Conv2d(600, 50, kernel_size=1, stride=1, padding='same')
         self.Conv3 = nn.Conv2d(50, 50, kernel_size=3, stride=1, padding='same')
         self.ConvF = nn.Conv2d(50, 64, kernel_size=3, stride=1, padding='same')



         modules_tail = [nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True),
                         nn.Conv2d(64, 3 * (scale ** 2), kernel_size=3, padding=1, bias=True),
                         nn.PixelShuffle(scale)]

         self.tail = nn.Sequential(*modules_tail)





    def forward(self,input): #torch.Size([8, 3, 48, 48])
        # print(input.shape)
        # exit()
        F0 = self.Pretreatment(input)
        l1_1, d1 = self.RGB1(F0)
        # print(d1.shape)
        # exit()
        l1_2, d2 = self.RGB2(l1_1)
        C1 = self.Conv0(l1_2)
        add1 = C1 + F0 + d1 + d2

        Swin1 = self.Swin1(add1)

        l2_1, d3 = self.RGB3(Swin1)
        l2_2, d4 = self.RGB4(l2_1)
        C2 = self.Conv00(l2_2)
        add2 = Swin1 + C2 + d3 + d4

        Swin2 = self.Swin2(add2)

        l3_1, d5 = self.RGB5(Swin2)
        l3_2, d6 = self.RGB6(l3_1)
        C3 = self.Conv000(l3_2)
        add3 = C3 + Swin2 + d5 + d6

        Swin3 = self.Swin3(add3)

        l4_1, d7 = self.RGB7(Swin3)
        l4_2, d8 = self.RGB8(l4_1)
        C4 = self.Conv0000(l4_2)
        add4 = C4 + Swin3 + d7 + d8

        Swin4 = self.Swin4(add4)

        l5_1, d9 = self.RGB9(Swin4)
        l5_2, d10 = self.RGB10(l5_1)
        C5 = self.Conv00000(l5_2)
        add5 = C5 + Swin4 + d9 + d10

        Swin5 = self.Swin5(add5)

        l6_1, d11 = self.RGB11(Swin5)
        l6_2, d12 = self.RGB12(l6_1)
        C6 = self.Conv000000(l6_2)
        add6 = C6 + Swin5 + d11 + d12

        Swin6 = self.Swin6(add6)

        l7_1, d13 = self.RGB13(Swin6)
        l7_2, d14 = self.RGB14(l7_1)
        C7 = self.Conv0000000(l7_2)
        add7 = C7 + Swin6 + d13 + d14

        Swin7 = self.Swin7(add7)

        l8_1, d15 = self.RGB15(Swin7)
        l8_2, d16 = self.RGB16(l8_1)
        C8 = self.Conv00000000(l8_2)
        add8 = C8 + Swin7 + d15 + d16

        Swin8 = self.Swin8(add8)

        l9_1, d17 = self.RGB17(Swin8)
        l9_2, d18 = self.RGB18(l9_1)
        C9 = self.Conv000000000(l9_2)
        add9 = C9 + Swin8 + d17 + d18

        Swin9 = self.Swin9(add9)

        l10_1, d19 = self.RGB19(Swin9)
        l10_2, d20 = self.RGB20(l10_1)
        C10 = self.Conv0000000000(l10_2)
        add10 = C10 + Swin9 + d19 + d20

        Swin10 = self.Swin10(add10)

        l11_1, d21 = self.RGB21(Swin10)
        l11_2, d22 = self.RGB22(l11_1)
        C11 = self.Conv00000000000(l11_2)
        add11 = C11 + Swin10 + d21 + d22

        Swin11 = self.Swin11(add11)

        l12_1, d23 = self.RGB23(Swin11)
        l12_2, d24 = self.RGB24(l12_1)
        C12 = self.Conv000000000000(l12_2)
        add12 = C12 + Swin11 + d23 + d24

        Swin12 = self.Swin12(add12)

        F = self.head(input)

        final_output = torch.cat(
            (Swin1, Swin2, Swin3, Swin4, Swin5, Swin6, Swin7, Swin8, Swin9, Swin10, Swin11, Swin12), 1)
        final_output = self.Conv1(final_output)
        final_output = self.Conv3(final_output)
        final_output = final_output + F
        final_output = self.ConvF(final_output)
        # print(f"final-output:{final_output.shape}")

        final_output = self.tail(final_output)
        # print(final_output.shape)
        # exit()






        return final_output




