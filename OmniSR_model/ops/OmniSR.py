#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: OmniSR.py
# Created Date: Tuesday April 28th 2022
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Sunday, 23rd April 2023 3:06:36 pm
# Modified By: Chen Xuanhong
# Copyright (c) 2020 Shanghai Jiao Tong University
#############################################################

import  torch
import  torch.nn as nn
import numpy as np
from ops.OSAG import OSAG
from einops import rearrange, reduce, repeat
from ops.pixelshuffle import pixelshuffle_block
from torch.utils.checkpoint import checkpoint, checkpoint_sequential
from ops.Recon_Net import deep_rec_dq,deep_rec
from ops.INV import InvRescaleNet,INV
from ops.Quantization import Quantization_RS
# from ops.arb_rescaling.arbedrs import EDRS
from ops.downsacling.TAD import DownSample_TAD_x2
# from ops.edsr.edsr import EDSR,EDSR_CSUP
from ops.CNN_CR.CNN_CR import DownSample_CNNCR_x2,DownSample_CNNCR_x4

import torch.nn.functional as F


class DownSample_x2(nn.Module):
    def __init__(self):
        super(DownSample_x2, self).__init__()
        # self.layer=nn.Sequential(
        # nn.Conv2d(in_channels=3,out_channels=64,kernel_size=2,stride=2,padding=0),
        # nn.ReLU(),
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=3 // 2)
        self.down_x2_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=2, stride=2, padding=0)
        self.pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=3 // 2)
        # )

    def forward(self, x):
        x = self.conv1(x)

        redual = self.pool2d(x)
        out = self.down_x2_1(x)
        x = redual + out

        x = self.conv2(x)
        return x

class DownSample_unshuffle_x2(nn.Module):
    def __init__(self):
        super(DownSample_unshuffle_x2, self).__init__()
        # self.layer=nn.Sequential(
        # nn.Conv2d(in_channels=3,out_channels=64,kernel_size=2,stride=2,padding=0),
        # nn.ReLU(),
        self.layer=nn.Sequential(
            nn.PixelUnshuffle(2),
            nn.Conv2d(12,3,3,1,1)
        )

    def forward(self, x):
        x = self.layer(x)
        return x

class DownSample_x3(nn.Module):
    def __init__(self):
        super(DownSample_x3, self).__init__()
        # self.layer=nn.Sequential(
        # nn.Conv2d(in_channels=3,out_channels=64,kernel_size=2,stride=2,padding=0),
        # nn.ReLU(),
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=3 // 2)
        self.down_x2_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=3, padding=0)
        self.pool2d = nn.MaxPool2d(kernel_size=3, stride=3)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=3 // 2)
        # )

    def forward(self, x):
        x = self.conv1(x)

        redual = self.pool2d(x)
        out = self.down_x2_1(x)
        x = redual + out

        x = self.conv2(x)
        return x

class DownSample_x4(nn.Module):
    def __init__(self):
        super(DownSample_x4, self).__init__()
        # self.layer=nn.Sequential(
        # nn.Conv2d(in_channels=3,out_channels=64,kernel_size=2,stride=2,padding=0),
        # nn.ReLU(),
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=3 // 2)
        self.down_x2_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=2, stride=2, padding=0)
        self.pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.prelu = nn.PReLU()
        self.down_x2_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=3 // 2)
        # )

    def forward(self, x):
        x = self.conv1(x)

        redual = self.pool2d(x)
        out = self.down_x2_1(x)
        x = redual + out

        redual = self.pool2d(x)
        out = self.down_x2_2(x)
        x = redual + out
        # x = out

        x = self.conv2(x)
        return x

class DownSample_x8(nn.Module):
    def __init__(self):
        super(DownSample_x8, self).__init__()
        # self.layer=nn.Sequential(
        # nn.Conv2d(in_channels=3,out_channels=64,kernel_size=2,stride=2,padding=0),
        # nn.ReLU(),
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=3 // 2)
        self.down_x2_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=2, stride=2, padding=0)
        self.pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.prelu = nn.PReLU()
        self.down_x2_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=2, stride=2, padding=0)
        # self.prelu = nn.PReLU()
        self.down_x2_3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=3 // 2)
        # )

    def forward(self, x):
        x = self.conv1(x)

        redual = self.pool2d(x)
        out = self.down_x2_1(x)
        x = redual + out

        redual = self.pool2d(x)
        out = self.down_x2_2(x)
        x = redual + out

        redual = self.pool2d(x)
        out = self.down_x2_3(x)
        x = redual + out

        x = self.conv2(x)
        return x


# class Quantization_RS(nn.Module):
#     def __init__(self):
#         super(Quantization_RS, self).__init__()
#
#     def forward(self, input):
#         return Quant_RS.apply(input)


class CS_DownSample_x4(nn.Module):
    def __init__(self):
        super(CS_DownSample_x4, self).__init__()
        self.down1 = nn.Conv2d(in_channels=3, out_channels=8*8*3, kernel_size=32, stride=32, padding=0)
        self.prelu=nn.PReLU()

    def forward(self, x):
        x = self.down1(x)
        x = rearrange(x, 'b (c a1 a2) h w -> b c (h a1) (w a2)', a1=8, a2=8, c=3)
        return x

class CS_DownSample_x2(nn.Module):
    def __init__(self):
        super(CS_DownSample_x2, self).__init__()
        self.down1 = nn.Conv2d(in_channels=3, out_channels=16*16*3, kernel_size=32, stride=32, padding=0)
        self.prelu=nn.PReLU()

    def forward(self, x):
        x = self.down1(x)
        x = rearrange(x, 'b (c a1 a2) h w -> b c (h a1) (w a2)', a1=16, a2=16, c=3)
        return x



class OmniSR(nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, **kwargs):
        super(OmniSR, self).__init__()
        # print(kwargs)
        kwargs = kwargs['kwards']
        res_num = kwargs["res_num"]
        up_scale = kwargs["upsampling"]
        bias = kwargs["bias"]

        residual_layer = []
        self.res_num = res_num

        for _ in range(res_num):
            temp_res = OSAG(channel_num=num_feat, **kwargs)
            residual_layer.append(temp_res)
        self.residual_layer = nn.Sequential(*residual_layer)

        # self.osag1 = OSAG(channel_num=num_feat, **kwargs)
        # self.osag2 = OSAG(channel_num=num_feat, **kwargs)
        # self.osag3 = OSAG(channel_num=num_feat, **kwargs)
        # self.osag4 = OSAG(channel_num=num_feat, **kwargs)
        # self.osag5 = OSAG(channel_num=num_feat, **kwargs)

        self.input = nn.Conv2d(in_channels=num_in_ch, out_channels=num_feat, kernel_size=3, stride=1, padding=1,
                               bias=bias)
        self.output = nn.Conv2d(in_channels=num_feat, out_channels=num_feat, kernel_size=3, stride=1, padding=1,
                                bias=bias)
        self.up = pixelshuffle_block(num_feat, num_out_ch, up_scale, bias=bias)

        # self.tail   = pixelshuffle_block(num_feat,num_out_ch,up_scale,bias=bias)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, sqrt(2. / n))

        self.window_size = kwargs["window_size"]
        self.up_scale = up_scale

    def check_image_size(self, x):
        _, _, h, w = x.size()
        # import pdb; pdb.set_trace()
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
        # x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'constant', 0)
        return x

    def forward(self, x):
        # print(x.shape)
        H, W = x.shape[2:]
        # print('aa',x.shape)
        x = self.check_image_size(x)

        residual = self.input(x)
        # print(x.shape)
        out = self.residual_layer(residual)

        # origin
        out = torch.add(self.output(out), residual)
        out = self.up(out)
        # out = rearrange(out, 'b (c a1 a2) h w -> b c (h a1) (w a2)', a1=32, a2=32, c=3)

        out = out[:, :, :H * self.up_scale, :W * self.up_scale]
        return out


class OmniSR_CSUP(nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, **kwargs):
        super(OmniSR_CSUP, self).__init__()
        # print(kwargs)
        kwargs = kwargs['kwards']
        res_num = kwargs["res_num"]
        up_scale = kwargs["upsampling"]
        bias = kwargs["bias"]

        residual_layer = []
        self.res_num = res_num

        for _ in range(res_num):
            temp_res = OSAG(channel_num=num_feat, **kwargs)
            residual_layer.append(temp_res)
        self.residual_layer = nn.Sequential(*residual_layer)

        self.input = nn.Conv2d(in_channels=num_in_ch, out_channels=num_feat, kernel_size=3, stride=1, padding=1,
                               bias=bias)
        self.output = nn.Conv2d(in_channels=num_feat, out_channels=num_feat, kernel_size=3, stride=1, padding=1,
                                bias=bias)

        self.up1 = nn.Conv2d(in_channels=3, out_channels=16*16*3, kernel_size=8, stride=8, padding=0)
        self.up2 = pixelshuffle_block(num_feat, num_out_ch, 2, bias=bias)


        self.window_size = kwargs["window_size"]
        self.up_scale = up_scale

    def check_image_size(self, x):
        _, _, h, w = x.size()
        # import pdb; pdb.set_trace()
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
        # x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'constant', 0)
        return x

    def forward(self, x):
        # print(x.shape)
        H, W = x.shape[2:]
        # print('aa',x.shape)
        x = self.check_image_size(x)
        x = self.up1(x)
        x = rearrange(x, 'b (c a1 a2) h w -> b c (h a1) (w a2)', a1=16, a2=16, c=3)
        residual = self.input(x)
        out = self.residual_layer(residual)

        # origin
        out = torch.add(self.output(out), residual)
        out = self.up2(out)

        out = out[:, :, :H * self.up_scale, :W * self.up_scale]
        return out

class LR_SR_x2(nn.Module):
    def __init__(self, kwards,num_feat=64):
        super(LR_SR_x2, self).__init__()
        self.layer1 = DownSample_x2()
        self.layer2 = Quantization_RS()
        self.layer3 = OmniSR(num_feat=num_feat,kwards=kwards)

    def forward(self, x):
        LR = self.layer1(x)
        LR_processed = self.layer2(LR)

        # 把块置0
        # block_size = 32
        # scale = 2
        # n, _, w, h = x.shape
        # col = int(w / block_size)
        # row = int(h / block_size)
        # for j in range(n):
        #     randarr = list_arr[j]
        #     block_size = int(block_size / scale)
        #     for i in randarr[:int(len(randarr) / 2)]:
        #         LR_processed[j, :, int(i / col) * block_size:int(i / col) * block_size + block_size,
        #         (i % row) * block_size:(i % row) * block_size + block_size] = 0

        HR = self.layer3(LR_processed)
        return LR,HR

class LR_SR_x3(nn.Module):
    def __init__(self, kwards):
        super(LR_SR_x3, self).__init__()
        self.layer1 = DownSample_x3()
        self.layer2 = Quantization_RS()
        self.layer3 = OmniSR(kwards=kwards)

    def forward(self, x):
        LR = self.layer1(x)
        LR_processed = self.layer2(LR)
        HR = self.layer3(LR_processed)
        return LR, HR

class LR_SR_x4(nn.Module):
    def __init__(self, kwards):
        super(LR_SR_x4, self).__init__()
        self.layer1 = DownSample_x4()
        self.layer2 = Quantization_RS()
        self.layer3 = OmniSR(kwards=kwards)

    def forward(self, x):
        LR = self.layer1(x)
        LR_processed = self.layer2(LR)

        # 把块置0
        # block_size = 32
        # scale = 4
        # n, _, w, h = x.shape
        # col = int(w / block_size)
        # row = int(h / block_size)
        # for j in range(n):
        #     randarr = list_arr[j]
        #     block_size = int(block_size / scale)
        #     for i in randarr[:int(len(randarr) / 2)]:
        #         LR_processed[j, :, int(i / col) * block_size:int(i / col) * block_size + block_size,
        #         (i % row) * block_size:(i % row) * block_size + block_size] = 0

        HR = self.layer3(LR_processed)
        return LR, HR


class LR_SR_x4_adp_information(nn.Module):
    def __init__(self, kwards):
        super(LR_SR_x4_adp_information, self).__init__()
        self.layer1 = DownSample_x4()
        self.layer2 = Quantization_RS()
        self.layer3 = OmniSR(kwards=kwards)

    def forward(self, x):
        LR = self.layer1(x)
        LR_processed = self.layer2(LR)
        HR = self.layer3(LR_processed)
        return LR, HR

class CS_LR_SR_x4(nn.Module):
    def __init__(self, kwards):
        super(CS_LR_SR_x4, self).__init__()
        self.layer1 = CS_DownSample_x4()
        self.layer2 = Quantization_RS()
        self.layer3 = OmniSR(kwards=kwards)

    def forward(self, x):
        LR = self.layer1(x)
        # print(LR.shape)
        LR_processed = self.layer2(LR)
        HR = self.layer3(LR_processed)
        return LR, HR

class CS_LR_SR_x2(nn.Module):
    def __init__(self, kwards):
        super(CS_LR_SR_x2, self).__init__()
        self.layer1 = CS_DownSample_x2()
        self.layer2 = Quantization_RS()
        self.layer3 = OmniSR(kwards=kwards)

    def forward(self, x):
        LR = self.layer1(x)
        # print(LR.shape)
        LR_processed = self.layer2(LR)
        HR = self.layer3(LR_processed)
        return LR, HR

class unshuffle_LR_SR_x2(nn.Module):
    def __init__(self, kwards):
        super(unshuffle_LR_SR_x2, self).__init__()
        self.layer1 = DownSample_unshuffle_x2()
        self.layer2 = Quantization_RS()
        self.layer3 = OmniSR(kwards=kwards)

    def forward(self, x):
        LR = self.layer1(x)
        LR_processed = self.layer2(LR)
        # print(LR.shape)
        HR = self.layer3(LR_processed)
        return LR, HR

class arb_LR_SR_x2(nn.Module):
    def __init__(self, kwards):
        super(arb_LR_SR_x2, self).__init__()
        self.layer1 = EDRS(1/2)
        self.layer2 = Quantization_RS()
        self.layer3 = OmniSR(kwards=kwards)

    def forward(self, x):
        x=x/2+0.5
        LR = self.layer1(x)*2-1
        LR_processed = self.layer2(LR)
        HR = self.layer3(LR_processed)
        return LR, HR

class TAD_LR_SR_x2(nn.Module):
    def __init__(self, kwards):
        super(TAD_LR_SR_x2, self).__init__()
        self.layer1 = DownSample_TAD_x2()
        self.layer2 = Quantization_RS()
        self.layer3 = OmniSR(kwards=kwards)

    def forward(self, x):
        # x=x/2+0.5
        LR = self.layer1(x)
        LR_processed = self.layer2(LR)
        HR = self.layer3(LR_processed)
        return LR, HR

class CNNCR_EDSR_LR_SR_x2(nn.Module):
    def __init__(self):
        super(CNNCR_EDSR_LR_SR_x2, self).__init__()
        self.layer1 = DownSample_CNNCR_x2()
        self.layer2 = Quantization_RS()
        self.layer3 = EDSR(scale=2)

    def forward(self, x,LR_BICUBIC):
        # x=x/2+0.5
        LR = self.layer1(x,LR_BICUBIC)
        LR_processed = self.layer2(LR)
        HR = self.layer3(LR_processed)
        return LR, HR
class CNNCR_EDSR_CSUP_LR_SR_x2(nn.Module):
    def __init__(self):
        super(CNNCR_EDSR_CSUP_LR_SR_x2, self).__init__()
        self.layer1 = DownSample_CNNCR_x2()
        self.layer2 = Quantization_RS()
        self.layer3 = EDSR_CSUP(scale=2)

    def forward(self, x,LR_BICUBIC):
        # x=x/2+0.5
        LR = self.layer1(x,LR_BICUBIC)
        LR_processed = self.layer2(LR)
        HR = self.layer3(LR_processed)
        return LR, HR

class CNNCR_EDSR_LR_SR_x4(nn.Module):
    def __init__(self):
        super(CNNCR_EDSR_LR_SR_x4, self).__init__()
        self.layer1 = DownSample_CNNCR_x4()
        self.layer2 = Quantization_RS()
        self.layer3 = EDSR(scale=4)

    def forward(self, x,LR_BICUBIC):
        # x=x/2+0.5
        LR = self.layer1(x,LR_BICUBIC)
        LR_processed = self.layer2(LR)
        HR = self.layer3(LR_processed)
        return LR, HR

class LR_SR_x8(nn.Module):
    def __init__(self, kwards):
        super(LR_SR_x8, self).__init__()
        self.layer1 = DownSample_x8()
        self.layer2 = Quantization_RS()
        self.layer3 = OmniSR(kwards=kwards)

    def forward(self, x,list_arr):
        LR = self.layer1(x)
        LR_processed = self.layer2(LR)

        #把块置0
        block_size=32
        scale=8
        n, _, w, h = x.shape
        col = int(w / block_size)
        row = int(h / block_size)
        for j in range(n):
            randarr = list_arr[j]
            block_size = int(block_size / scale)
            for i in randarr[:int(len(randarr) / 2)]:
                LR_processed[j, :, int(i / col) * block_size:int(i / col) * block_size + block_size,
                (i % row) * block_size:(i % row) * block_size + block_size] = 0

        HR = self.layer3(LR_processed)
        return LR, HR

