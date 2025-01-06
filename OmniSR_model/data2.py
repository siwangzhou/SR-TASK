import torch
from torch.utils.data import Dataset
import json
import os
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from typing import Sequence
import numpy as np
from numpy import random
from einops import rearrange, reduce, repeat
import cv2
import math
import numpy as np

class MyRotateTransform:
    def __init__(self, angles: Sequence[int]):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle)


class MyDataset3(Dataset):
    def __init__(self,alist=["D:\code\Python\project\mechine Learning\Data\DIV2K\DIV2K_train_HR","D:\code\Python\project\mechine Learning\Data\Flickr2K\Flickr2K"]):
        super(MyDataset3, self).__init__()
        self.alist=alist
        self.hrimgs=[]
        self.lrimgs = []
        i=0
        str="\r{0}"
        for path in self.alist:
            train_list=os.listdir(path)
            for name in train_list:
                print(str.format(i),end='')
                i+=1
                img_path=path+"/"+name
                self.hrimgs.append(img_path)

    def __len__(self):
        return len(self.hrimgs)
    def __getitem__(self, index):
        lrsize=64
        scales=2
        self.transforms = transforms.Compose([
                transforms.RandomCrop(lrsize*scales),
                ])
        temp_img = Image.open(self.hrimgs[index]).convert('RGB')
        sourceImg = self.transforms(temp_img)
        cropimg = sourceImg.resize((lrsize,lrsize), Image.BICUBIC)

        sourceImg = sourceImg.resize((lrsize*scales,lrsize*scales), Image.BICUBIC)

        hr = transforms.ToTensor()(sourceImg)
        lr = transforms.ToTensor()(cropimg)
        flip_ran = random.randint(0, 2)

        if flip_ran == 0:
            # horizontal
            hr = torch.flip(hr, [1])
            lr = torch.flip(lr, [1])
        elif flip_ran == 1:
            # vertical
            hr = torch.flip(hr, [2])
            lr = torch.flip(lr, [2])

        rot_ran = random.randint(0, 3)

        if rot_ran != 0:
            # horizontal
            hr = torch.rot90(hr, rot_ran, [1, 2])
            lr = torch.rot90(lr, rot_ran, [1, 2])

        hr  = (hr - 0.5) * 2.0
        lr  = (lr - 0.5) * 2.0
        # temp = lr[0, :, :]
        # lr[0, :, :] = lr[2, :, :]
        # lr[2, :, :] = temp
        #
        # temp = hr[0, :, :]
        # hr[0, :, :] = hr[2, :, :]
        # hr[2, :, :] = temp

        return lr, hr

class Test(Dataset):
    def __init__(self,alist=["D:\code\Python\project\mechine Learning\Data\DIV2K_train_HR\DIV2K_train_HR","D:\code\Python\project\mechine Learning\Data\Flickr2K\Flickr2K"]):
        super(Test, self).__init__()
        # self.alist=alist
        # self.alist = ['D:\code\Python\project\mechine Learning\Data\Set5']
        # self.hr=['D:\code\Python\project\mechine Learning\Data\BSDS100\GTmod8']
        self.hr = ['D:\code\Python\project\mechine Learning\Data\Set5 (2)\Set5\image_SRF_4\HR']
        self.imgshr=[]
        self.imgslr = []
        i=0
        str="\r{0}"
        for path in self.hr:
            train_list=os.listdir(path)
            for name in train_list:
                print(str.format(i),end='')
                i+=1
                img_path=path+"/"+name
                self.imgshr.append(img_path)
        # for path in self.lr:
        #     train_list=os.listdir(path)
        #     for name in train_list:
        #         print(str.format(i),end='')
        #         i+=1
        #         img_path=path+"/"+name
        #         self.imgslr.append(img_path)
    def __len__(self):
        return len(self.imgshr)
    def __getitem__(self, index):
        scales=2
        temp_img = Image.open(self.imgshr[index]).convert('RGB')
        size = (np.array(temp_img.size) / scales).astype(int)
        # img1 = temp_img.resize(size * 4, Image.BICUBIC)
        w=size[0]
        h=size[1]
        sourceImg=transforms.RandomCrop((h*scales,w*scales))(temp_img)
        cropimg = sourceImg.resize((w*8, h*8), Image.BICUBIC)
        # sourceImg = sourceImg.resize((w*4, h*4), Image.BICUBIC)
        hr = transforms.ToTensor()(sourceImg)
        lr = transforms.ToTensor()(cropimg)
        # print(cropimg)
        hr = (hr - 0.5) * 2.0
        lr = (lr - 0.5) * 2.0
        # temp = lr[0, :, :]
        # lr[0, :, :]=lr[2,:,:]
        # lr[2, :, :]=temp
        #
        # temp = hr[0, :, :]
        # hr[0, :, :] = hr[2, :, :]
        # hr[2, :, :] = temp

        return lr, hr


class MyDataset(Dataset):
    def __init__(self):
        super(MyDataset, self).__init__()
        self.alist="D:\code\Python\project\mechine Learning\Data\VOCtrainval_11-May-2012\VOCdevkit\VOC2012/trainval.txt"
        # self.train_str='D:\code\Python\project\mechine Learning/block super resolution\pascal block SR image'
        # self.gt_str='D:\code\Python\project\mechine Learning\Data\VOCtrainval_11-May-2012\VOCdevkit\VOC2012\JPEGImages'
        self.train_str = 'D:\code\Python\project\mechine Learning/block super resolution\multi_scale_sr_df2k_method2'
        self.gt_str = 'D:\code\Python\project\mechine Learning/block super resolution\df2k_hr'

        self.gt_imgs=[]
        self.train_imgs = []

        # i=0
        # str="\r{0}"
        # with open(self.alist,'r') as f:
        #     for i in f.readlines():
        #         img=Image.open(self.gt_str+ '/'+i.split('\n')[0]+'.jpg').convert('RGB')
        #         if img.size[0]>256 and img.size[1]>256:
        #             self.gt_imgs.append(self.gt_str+ '/'+i.split('\n')[0]+'.jpg')
        #             self.train_imgs.append(self.train_str+ '/' + i.split('\n')[0] + '.png')

        i = 0
        str = "\r{0}"
        # for path in self.alist:
        train_list = os.listdir(self.train_str)
        for name in train_list:
            print(str.format(i), end='')
            i += 1
            # print(self.gt_str + '/' + name)
            self.gt_imgs.append(self.gt_str + '/' + name)
            self.train_imgs.append(self.train_str + '/' + name)

    def __len__(self):
        return len(self.gt_imgs)
    def __getitem__(self, index):
        size=512
        lrsize=256
        gt_img = Image.open(self.gt_imgs[index]).convert('RGB')
        train_img = Image.open(self.train_imgs[index]).convert('RGB')
        w,h=gt_img.size
        # print(w-size,h-size)
        rand_h = random.randint(0, h-size)
        rand_w = random.randint(0, w-size)
        # sourceImg.show()
        # cropimg = sourceImg.resize((lrsize,lrsize), Image.BICUBIC)
        gt_img = transforms.ToTensor()(gt_img)
        train_img = transforms.ToTensor()(train_img)
        gt_img=gt_img[:,rand_h:rand_h+size,rand_w:rand_w+size]
        train_img=train_img[:, int(rand_h/2):int(rand_h/2) + lrsize, int(rand_w/2):int(rand_w/2) + lrsize]

        flip_ran = random.randint(0, 2)
        if flip_ran == 0:
            # horizontal
            gt_img = torch.flip(gt_img, [1])
            train_img = torch.flip(train_img, [1])
        elif flip_ran == 1:
            # vertical
            gt_img = torch.flip(gt_img, [2])
            train_img = torch.flip(train_img, [2])

        rot_ran = random.randint(0, 3)

        if rot_ran != 0:
            # horizontal
            gt_img = torch.rot90(gt_img, rot_ran, [1, 2])
            train_img = torch.rot90(train_img, rot_ran, [1, 2])

        # hr  = (hr - 0.5) * 2.0
        # lr  = (lr - 0.5) * 2.0
        # print(train_img.shape,gt_img.shape)
        return train_img, gt_img


class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        # self.mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).cuda().view(1,3,1,1)
        # self.std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).cuda().view(1,3,1,1)
        # With Amp, it isn't necessary to manually convert data to half.
        # if args.fp16:
        #     self.mean = self.mean.half()
        #     self.std = self.std.half()
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda()
            self.next_target = self.next_target.cuda()
            # With Amp, it isn't necessary to manually convert data to half.
            # if args.fp16:
            #     self.next_input = self.next_input.half()
            # else:
            # self.next_input = self.next_input.float()
            # self.next_input = self.next_input.sub_(self.mean).div_(self.std)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        self.preload()
        return input, target



if __name__ == '__main__':
    a=MyDataset4()
    n=a.__len__()
    a.__getitem__(1)
    # for j in range(n):
    # #     for i in range(n):
    #         # print(i)
    #     a.__getitem__(j)
    #     print(j)