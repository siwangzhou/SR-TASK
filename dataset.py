# Copyright 2022 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Realize the function of dataset preparation."""
import os
import queue
import threading

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image


import imgproc
import json


# VOC
from collections import OrderedDict
import torchvision

__all__ = [
    "TrainValidImageDataset", "TestImageDataset","TrainValidContextDataset",
    "PrefetchGenerator", "PrefetchDataLoader", "CPUPrefetcher", "CUDAPrefetcher",
]


# class TrainValidImageDataset(Dataset):
#     """Define training/valid dataset loading methods.
#
#     Args:
#         image_dir (str): Train/Valid dataset address.
#         image_size (int): High resolution image size.
#         upscale_factor (int): Image up scale factor.
#         mode (str): Data set loading method, the training data set is for data enhancement, and the
#             verification dataset is not for data enhancement.
#     """
#
#     def __init__(self, image_dir: str, image_size: int, upscale_factor: int, mode: str) -> None:
#         super(TrainValidImageDataset, self).__init__()
#         # Get all image file names in folder
#         subset_path = "D:/jingou/SuperResolution/subset.txt"
#         with open(subset_path) as f:
#             filenames = [line.rstrip('\n') for line in f]
#         if mode == 'Train':
#             self.image_file_names = [os.path.join(image_dir, image_file_name) for image_file_name in filenames]
#         else:
#             self.image_file_names = [os.path.join(image_dir, image_file_name) for image_file_name in
#                                      os.listdir(image_dir)]
#         # Specify the high-resolution image size, with equal length and width
#         self.image_size = image_size
#         # How many times the high-resolution image is the low-resolution image
#         self.upscale_factor = upscale_factor
#         # Load training dataset or test dataset
#         self.mode = mode
#
#     def __getitem__(self, batch_index: int) -> [torch.Tensor, torch.Tensor]:
#         # Read a batch of image data [0-1]
#         image = cv2.imread(self.image_file_names[batch_index], cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.
#
#         # Image processing operations
#         if self.mode == "Train":
#             hr_image = imgproc.random_crop(image, self.image_size)
#             # hr_image = imgproc.random_rotate(hr_image, angles=[0, 90, 180, 270])
#             hr_image = imgproc.random_horizontally_flip(hr_image, p=0.5)
#             hr_image = imgproc.random_vertically_flip(hr_image, p=0.5)
#         elif self.mode == "Valid":
#             hr_image = imgproc.center_crop(image, self.image_size)
#         else:
#             raise ValueError("Unsupported data processing model, please use `Train` or `Valid`.")
#
#         lr_image = imgproc.image_resize(hr_image, 1 / self.upscale_factor)
#
#         # BGR convert to RGB
#         lr_image = cv2.cvtColor(lr_image, cv2.COLOR_BGR2RGB)
#         hr_image = cv2.cvtColor(hr_image, cv2.COLOR_BGR2RGB)
#
#         # Convert image data into Tensor stream format (PyTorch).
#         # Note: The range of input and output is between [0, 1]
#         lr_tensor = imgproc.image2tensor(lr_image, range_norm=False, half=False)
#         hr_tensor = imgproc.image2tensor(hr_image, range_norm=False, half=False)
#
#         return {"lr": lr_tensor, "hr": hr_tensor}
#
#     def __len__(self) -> int:
#         return len(self.image_file_names)


class TrainValidImageDataset(Dataset):
    """Define training/valid dataset loading methods.

    Args:
        image_dir (str): Train/Valid dataset address.
        image_size (int): High resolution image size.
        upscale_factor (int): Image up scale factor.
        mode (str): Data set loading method, the training data set is for data enhancement, and the
            verification dataset is not for data enhancement.
    """

    def __init__(self, image_dir: str, image_size: int, upscale_factor: int, mode: str) -> None:
        super(TrainValidImageDataset, self).__init__()
        # Get all image file names in folder
        subset_path = "D:/jingou/SuperResolution/subset.txt"
        with open(subset_path) as f:
            filenames = [line.rstrip('\n') for line in f]
        self.filenames = filenames
        if mode == 'Train':
            self.image_file_names = [os.path.join(image_dir, image_file_name) for image_file_name in filenames]
        else:
            self.image_file_names = [os.path.join(image_dir, image_file_name) for image_file_name in
                                     os.listdir(image_dir)]
        # Specify the high-resolution image size, with equal length and width
        self.image_size = image_size
        # How many times the high-resolution image is the low-resolution image
        self.upscale_factor = upscale_factor
        # Load training dataset or test dataset
        self.mode = mode
        # read class_indict
        json_file = './ImageNet_index_classes.json'  # 保存的类别对应的索引信息
        assert os.path.exists(json_file), "{} file not exist.".format(json_file)
        json_file = open(json_file, 'r')
        self.class_dict = json.load(json_file)
        json_file.close()
        # turn to dirname - label
        class_index_dict = {}
        for item in self.class_dict:
            class_index_dict[self.class_dict[item][0]] = item
        self.class_dict = class_index_dict

    def __getitem__(self, batch_index: int) -> [torch.Tensor, torch.Tensor]:
        # Read a batch of image data [0-1]
        image = cv2.imread(self.image_file_names[batch_index], cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.

        # Image processing operations
        if self.mode == "Train":
            hr_image = imgproc.random_crop(image, self.image_size)
            # hr_image = imgproc.random_rotate(hr_image, angles=[0, 90, 180, 270])
            hr_image = imgproc.random_horizontally_flip(hr_image, p=0.5)
            hr_image = imgproc.random_vertically_flip(hr_image, p=0.5)
        elif self.mode == "Valid":
            hr_image = imgproc.center_crop(image, self.image_size)
        else:
            raise ValueError("Unsupported data processing model, please use `Train` or `Valid`.")

        lr_image = imgproc.image_resize(hr_image, 1 / self.upscale_factor)

        # BGR convert to RGB
        lr_image = cv2.cvtColor(lr_image, cv2.COLOR_BGR2RGB)
        hr_image = cv2.cvtColor(hr_image, cv2.COLOR_BGR2RGB)

        # Convert image data into Tensor stream format (PyTorch).
        # Note: The range of input and output is between [0, 1]
        lr_tensor = imgproc.image2tensor(lr_image, range_norm=False, half=False)
        hr_tensor = imgproc.image2tensor(hr_image, range_norm=False, half=False)

        # # get label
        # labels = []
        # filename,_ = self.filenames[batch_index].split('/')
        # labels.append(self.class_dict[filename])
        # labels = torch.as_tensor(labels, dtype=torch.int64)


        # return {"lr": lr_tensor, "hr": hr_tensor, "label":labels}
        return {"lr": lr_tensor, "hr": hr_tensor}

    def __len__(self) -> int:
        return len(self.image_file_names)

class TrainValidImageDataset_PASCAL(Dataset):
    """Define training/valid dataset loading methods.

    Args:
        image_dir (str): Train/Valid dataset address.
        image_size (int): High resolution image size.
        upscale_factor (int): Image up scale factor.
        mode (str): Data set loading method, the training data set is for data enhancement, and the
            verification dataset is not for data enhancement.
    """

    def __init__(self, image_dir: str,
                 image_size: int,
                 upscale_factor: int,
                 txt_name: str,
                 mode: str) -> None:
        super(TrainValidImageDataset_PASCAL, self).__init__()
        # Get all image file names in folder

        self.image_dir = image_dir # 'D:/jingou/VOC'
        self.image_dir = os.path.join(self.image_dir, 'PASCAL_MT')
        # get image path
        image_dir = os.path.join(self.image_dir, 'JPEGImages')
        # get train segmentation and valid segmentation
        txt_path = os.path.join(self.image_dir, 'ImageSets/Context')
        assert os.path.exists(txt_path), "file '{}' does not exist.".format(txt_path)
        # 读取train.txt or valid.txt 文档内的 文件名称
        split_f = os.path.join(txt_path, txt_name)
        with open(os.path.join(split_f), "r") as f:
            file_names = [x.strip() for x in f.readlines() if len(x.strip()) > 0]

        self.image_file_names = [os.path.join(image_dir, x + ".jpg") for x in file_names]     # 获取Image

        # Specify the high-resolution image size, with equal length and width
        self.image_size = image_size
        # How many times the high-resolution image is the low-resolution image
        self.upscale_factor = upscale_factor
        # Load training dataset or test dataset
        self.mode = mode

        # # read class_indict
        # json_file = './ImageNet_index_classes.json'  # 保存的类别对应的索引信息
        # assert os.path.exists(json_file), "{} file not exist.".format(json_file)
        # json_file = open(json_file, 'r')
        # self.class_dict = json.load(json_file)
        # json_file.close()
        # # turn to dirname - label
        # class_index_dict = {}
        # for item in self.class_dict:
        #     class_index_dict[self.class_dict[item][0]] = item
        # self.class_dict = class_index_dict

    def __getitem__(self, batch_index: int) -> [torch.Tensor, torch.Tensor]:
        # Read a batch of image data [0-1]
        image = cv2.imread(self.image_file_names[batch_index], cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.

        # Image processing operations
        if self.mode == "Train":
            hr_image = imgproc.padding(image, self.image_size, range='float32') # 存在小于128的HorW
            hr_image = imgproc.random_crop(hr_image, self.image_size)
            hr_image = imgproc.random_horizontally_flip(hr_image, p=0.5)
            hr_image = imgproc.random_vertically_flip(hr_image, p=0.5)
        elif self.mode == "Valid":
            hr_image = imgproc.center_crop(image, self.image_size)
        else:
            raise ValueError("Unsupported data processing model, please use `Train` or `Valid`.")

        lr_image = imgproc.image_resize(hr_image, 1 / self.upscale_factor)

        # BGR convert to RGB
        lr_image = cv2.cvtColor(lr_image, cv2.COLOR_BGR2RGB)
        hr_image = cv2.cvtColor(hr_image, cv2.COLOR_BGR2RGB)

        # Convert image data into Tensor stream format (PyTorch).
        # Note: The range of input and output is between [0, 1]
        lr_tensor = imgproc.image2tensor(lr_image, range_norm=False, half=False)
        hr_tensor = imgproc.image2tensor(hr_image, range_norm=False, half=False)


        return {"lr": lr_tensor, "hr": hr_tensor}

    def __len__(self) -> int:

        return len(self.image_file_names)


class TestImageDataset(Dataset):
    """Define Test dataset loading methods.

    Args:
        test_lr_image_dir (str): Test dataset address for low resolution image dir.
        test_hr_image_dir (str): Test dataset address for high resolution image dir.
    """

    def __init__(self, test_lr_image_dir: str, test_hr_image_dir: str) -> None:
        super(TestImageDataset, self).__init__()
        # Get all image file names in folder
        self.lr_image_file_names = [os.path.join(test_lr_image_dir, x) for x in os.listdir(test_lr_image_dir)]
        self.hr_image_file_names = [os.path.join(test_hr_image_dir, x) for x in os.listdir(test_hr_image_dir)]

    def __getitem__(self, batch_index: int) -> [torch.Tensor, torch.Tensor]:
        # Read a batch of image data
        lr_image = cv2.imread(self.lr_image_file_names[batch_index], cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.
        hr_image = cv2.imread(self.hr_image_file_names[batch_index], cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.

        # BGR convert to RGB
        lr_image = cv2.cvtColor(lr_image, cv2.COLOR_BGR2RGB)
        hr_image = cv2.cvtColor(hr_image, cv2.COLOR_BGR2RGB)

        # Convert image data into Tensor stream format (PyTorch).
        # Note: The range of input and output is between [0, 1]
        lr_tensor = imgproc.image2tensor(lr_image, range_norm=False, half=False)
        hr_tensor = imgproc.image2tensor(hr_image, range_norm=False, half=False)

        return {"lr": lr_tensor, "hr": hr_tensor}

    def __len__(self) -> int:
        return len(self.lr_image_file_names)


class PrefetchGenerator(threading.Thread):
    """A fast data prefetch generator.

    Args:
        generator: Data generator.
        num_data_prefetch_queue (int): How many early data load queues.
    """

    def __init__(self, generator, num_data_prefetch_queue: int) -> None:
        threading.Thread.__init__(self)
        self.queue = queue.Queue(num_data_prefetch_queue)
        self.generator = generator
        self.daemon = True
        self.start()

    def run(self) -> None:
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)

    def __next__(self):
        next_item = self.queue.get()
        if next_item is None:
            raise StopIteration
        return next_item

    def __iter__(self):
        return self

# 优化pytorch DataLoader提升数据加载速度
class PrefetchDataLoader(DataLoader):
    """A fast data prefetch dataloader.

    Args:
        num_data_prefetch_queue (int): How many early data load queues.
        kwargs (dict): Other extended parameters.
    """

    def __init__(self, num_data_prefetch_queue: int, **kwargs) -> None:
        self.num_data_prefetch_queue = num_data_prefetch_queue
        super(PrefetchDataLoader, self).__init__(**kwargs)

    def __iter__(self):
        return PrefetchGenerator(super().__iter__(), self.num_data_prefetch_queue)


class CPUPrefetcher:
    """Use the CPU side to accelerate data reading.

    Args:
        dataloader (DataLoader): Data loader. Combines a dataset and a sampler, and provides an iterable over the given dataset.
    """

    def __init__(self, dataloader: DataLoader) -> None:
        self.original_dataloader = dataloader
        self.data = iter(dataloader)

    def next(self):
        try:
            return next(self.data)
        except StopIteration:
            return None

    def reset(self):
        self.data = iter(self.original_dataloader)

    def __len__(self) -> int:
        return len(self.original_dataloader)


class CUDAPrefetcher:
    """Use the CUDA side to accelerate data reading.

    Args:
        dataloader (DataLoader): Data loader. Combines a dataset and a sampler, and provides an iterable over the given dataset.
        device (torch.device): Specify running device.
    """

    def __init__(self, dataloader: DataLoader, device: torch.device):
        self.batch_data = None
        self.original_dataloader = dataloader
        self.device = device

        self.data = iter(dataloader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.batch_data = next(self.data)
        except StopIteration:
            self.batch_data = None
            return None

        with torch.cuda.stream(self.stream):
            for k, v in self.batch_data.items():
                if torch.is_tensor(v):
                    self.batch_data[k] = self.batch_data[k].to(self.device, non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch_data = self.batch_data
        self.preload()
        return batch_data

    def reset(self):
        self.data = iter(self.original_dataloader)
        self.preload()

    def __len__(self) -> int:
        return len(self.original_dataloader)




from torchvision.transforms.functional import InterpolationMode as IMode

class TrainValidContextDataset(Dataset):
    def __init__(self, voc_root='D:/jingou/VOC',
                 image_size=256,
                 upscale_factor=4,
                 image_set = 'val',
                 txt_name: str = "train.txt"):
        super(TrainValidContextDataset, self).__init__()
        # get train and val dataset
        self.voc_root = os.path.join(voc_root, 'PASCAL_MT')

        # get image path and segmentation path
        image_dir = os.path.join(self.voc_root, 'JPEGImages')
        semseg_dir = os.path.join(self.voc_root, 'semseg', 'pascal-context')

        SR_image_dir = "D:/jingou/RCM-master/SR/results/test/exp_SRResNet_seg/1_semseg/pascal_context"
        self.SR = False

        # get train segmentation and valid segmentation
        txt_path = os.path.join(self.voc_root, 'ImageSets/Context')
        assert os.path.exists(txt_path), "file '{}' does not exist.".format(txt_path)
        # 读取train.txt or valid.txt 文档内的 文件名称
        split_f = os.path.join(txt_path, txt_name)
        with open(os.path.join(split_f), "r") as f:
            file_names = [x.strip() for x in f.readlines() if len(x.strip()) > 0]

        # 分别保存img和mask的 图片路径
        self.im_ids = file_names
        self.images = [os.path.join(image_dir, x + ".jpg") for x in file_names]     # 获取Image
        self.images_SR = [os.path.join(SR_image_dir, x + ".png") for x in file_names]
        self.masks  = [os.path.join(semseg_dir, x + ".png") for x in file_names]   # 获取Image对应的seg mask GroundTruth
        assert (len(self.images) == len(self.masks))

        # Context类别数量
        self.num_classes = 20
        # 获取当前模式 train / val
        self.mode= image_set
        # 图像大小
        if self.mode=='train':
            self.image_size = image_size
        else:
            self.image_size = 512
        # 缩放倍数
        self.upscale_factor = upscale_factor
        # 语义分割和分类时 image的 预处理
        self.transform_bicubic = torchvision.transforms.Resize(self.image_size // self.upscale_factor, interpolation=IMode.BICUBIC)
        self.transform_totensor = torchvision.transforms.ToTensor()
        self.transform_norm = torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        # get image for semsegmentation and classification

        sample = OrderedDict()

        # get image
        image = np.array(Image.open(self.images[index]).convert('RGB')).astype(np.float32)
        if self.SR:
            image = np.array(Image.open(self.images_SR[index]).convert('RGB')).astype(np.float32) # 读取SR pascal context

        sample['image']    = image
        sample['image_lr'] = image

        sample['labels'] = OrderedDict()

        # get semseg mask
        semseg = np.array(Image.open(self.masks[index])).astype(np.float32)  # [H,W]


        if self.mode == 'train':
            # 随机裁剪为 image_size * image_size 大小的
            image_, semseg_ = imgproc.random_crop_2(image, semseg, crop_img_size=self.image_size)
            image_, semseg_ = imgproc.random_horizontally_flip2(image_, semseg_, p=0.5)
        elif self.mode == 'val':
            # 填充为 image_size * image_size
            image_, semseg_ = imgproc.padImg(image, semseg, pad_img_size=self.image_size)
        else:
            raise ValueError("Unsupported data processing model, please use `train` or `val`.")


        # 数据增强后的图像
        image_ = Image.fromarray(image_.astype('uint8'), 'RGB') # 转为PIL
        image_lr = self.transform_bicubic(image_) # PIL downsample

        # get label
        # 根据此时 经过 augmentation 的 img 获得对应的label ———— > 裁剪后的图像 对应 的 label
        classes = np.unique(semseg_)
        idx_255 = np.where(classes == 255.)
        if idx_255:  # semsge 经过裁剪 semseg_可能有的像素值填充了255 所以需要删掉255
            classes = np.delete(classes, idx_255)
        one_hot_class = np.zeros(self.num_classes + 1)
        one_hot_class[np.array(classes).astype(np.int)] = 1  # 转为onehot类型的标签 np.float64 类型的

        # toTensor
        image_tensor = self.transform_totensor(image_)     # 转 tensor 不标准化
        image_lr_tensor = self.transform_totensor(image_lr) # 转 tensor 不标准化 PIL - resize - toTensor 【0,1】
        semseg_tensor = torch.from_numpy(np.array(semseg_)).long() #
        classes_tensor = torch.from_numpy(one_hot_class).int() # 4 21

        # updata sample 的内容
        sample['image']  = image_tensor
        sample['image_lr']  = image_lr_tensor
        sample['labels']['semseg'] = semseg_tensor
        sample['labels']['label'] = classes_tensor

        sample['meta'] = {'image': str(self.im_ids),
                          'im_size': (np.shape(image)[0], np.shape(image)[1])} # 记录下图片的原始大小


        if self.SR: # 读取SR 为了 classify or semseg 任务进行测试
            sample['image'] = self.transform_norm(sample['image'])

        return sample


    def __len__(self):
        return len(self.images)

    def __len__(self):
        return len(self.images)



class TrainValidContextDataset_CV2(Dataset):
    def __init__(self, voc_root='./data/VOC',
                 image_size=256,
                 upscale_factor=4,
                 image_set = 'val',
                 txt_name: str = "train.txt"):
        super(TrainValidContextDataset_CV2, self).__init__()
        # get train and val dataset
        self.voc_root = os.path.join(voc_root, 'PASCAL_MT')

        # get image path and segmentation path
        image_dir = os.path.join(self.voc_root, 'JPEGImages')
        semseg_dir = os.path.join(self.voc_root, 'semseg', 'pascal-context')

        # get train segmentation and valid segmentation
        txt_path = os.path.join(self.voc_root, 'ImageSets/Context')
        assert os.path.exists(txt_path), "file '{}' does not exist.".format(txt_path)
        # 读取train.txt or valid.txt 文档内的 文件名称
        split_f = os.path.join(txt_path, txt_name)
        with open(os.path.join(split_f), "r") as f:
            file_names = [x.strip() for x in f.readlines() if len(x.strip()) > 0]

        # 分别保存img和mask的 图片路径
        self.im_ids = file_names
        self.images = [os.path.join(image_dir, x + ".jpg") for x in file_names]     # 获取Image
        self.masks  = [os.path.join(semseg_dir, x + ".png") for x in file_names]   # 获取Image对应的seg mask GroundTruth
        assert (len(self.images) == len(self.masks))

        # Context类别数量
        self.num_classes = 20
        # 获取当前模式 train / val
        self.mode= image_set
        # 图像大小
        if self.mode=='train':
            self.image_size = image_size
        else:
            self.image_size = 512
        # 缩放倍数
        self.upscale_factor = upscale_factor
        # 语义分割和分类时 image的 预处理
        self.transform_bicubic = torchvision.transforms.Resize(self.image_size // self.upscale_factor, interpolation=IMode.BICUBIC)
        self.transform_totensor = torchvision.transforms.ToTensor()
        self.transform_norm = torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        # get image for semsegmentation and classification

        sample = OrderedDict()

        # get image PIL
        # image = np.array(Image.open(self.images[index]).convert('RGB')).astype(np.float32)

        # get image CV
        image = cv2.imread(self.images[index], cv2.IMREAD_UNCHANGED).astype(np.float32) / 255. # 0 1 type = np.float32
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # to RGB


        sample['image']    = image
        sample['image_lr'] = image

        sample['labels'] = OrderedDict()

        # get semseg mask
        semseg = np.array(Image.open(self.masks[index])).astype(np.float32)  # [H,W]


        if self.mode == 'train':
            # 随机裁剪为 image_size * image_size 大小的
            image_, semseg_ = imgproc.random_crop_2(image, semseg, crop_img_size=self.image_size)
            image_, semseg_ = imgproc.random_horizontally_flip2(image_, semseg_, p=0.5)
        elif self.mode == 'val':
            # 填充为 image_size * image_size
            # image_, semseg_ = imgproc.padImg(image, semseg, pad_img_size=self.image_size,range='float32')
            # valid 时 统一进行裁剪
            image_  = imgproc.modcrop(image, self.upscale_factor)
            semseg_ = imgproc.modcrop(semseg, self.upscale_factor)
        else:
            raise ValueError("Unsupported data processing model, please use `train` or `val`.")


        # 数据增强后的图像
        # image_ = Image.fromarray(image_.astype('uint8'), 'RGB') # 转为PIL
        # image_lr = self.transform_bicubic(image_) # PIL downsample

        # get bicubic dowmsample LR image
        image_lr = imgproc.image_resize(image_, 1/self.upscale_factor)

        # get label
        # 根据此时 经过 augmentation 的 img 获得对应的label ———— > 裁剪后的图像 对应 的 label
        classes = np.unique(semseg_)
        idx_255 = np.where(classes == 255.)
        if idx_255:  # semsge 经过裁剪 semseg_可能有的像素值填充了255 所以需要删掉255
            classes = np.delete(classes, idx_255)

        # 不加入背景类计算
        idx_0 = np.where(classes == 0)
        if idx_0:
            classes = np.delete(classes, idx_0)

        # 从 1-20 到 0 - 19
        classes = classes - 1
        one_hot_class = np.zeros(self.num_classes)

        # one_hot_class = np.zeros(self.num_classes + 1)
        one_hot_class[np.array(classes).astype(np.int)] = 1  # 转为onehot类型的标签 np.float64 类型的



        # semseg_onehot
        semseg_onehot = imgproc.mask2one_hot(semseg_)
        semseg_onehot_tensor = torch.Tensor(semseg_onehot).float()


        # toTensor
        # image_tensor = self.transform_totensor(image_)     # 转 tensor 不标准化
        # image_lr_tensor = self.transform_totensor(image_lr) # 转 tensor 不标准化 PIL - resize - toTensor 【0,1】
        semseg_tensor = torch.from_numpy(np.array(semseg_)).long() #
        classes_tensor = torch.from_numpy(one_hot_class).int() # 4 21
        image_tensor = imgproc.image2tensor(image_,range_norm=False, half=False)
        image_lr_tensor = imgproc.image2tensor(image_lr,range_norm=False, half=False)

        # updata sample 的内容 并且要将tensor转为FloatTensor
        sample['image']  = image_tensor.float()
        sample['image_lr']  = image_lr_tensor.float()
        sample['labels']['semseg'] = semseg_tensor
        sample['labels']['label'] = classes_tensor
        sample['labels']['semseg_onehot'] = semseg_onehot_tensor
        sample['meta'] = {'image': str(self.im_ids),
                          'im_size': (np.shape(image)[0], np.shape(image)[1])} # 记录下图片的原始大小

        return sample


    def __len__(self):
        return len(self.images)

if __name__ == '__main__':
    subset_path = "D:/jingou/SuperResolution/subset.txt"
    with open(subset_path) as f:
        filenames = [line.rstrip('\n') for line in f]

    # read class_indict
    json_file = './ImageNet_index_classes.json'  # 保存的类别对应的索引信息
    assert os.path.exists(json_file), "{} file not exist.".format(json_file)
    json_file = open(json_file, 'r')
    class_dict = json.load(json_file)
    json_file.close()
    # turn to dirname - label
    class_index_dict = {}
    for item in class_dict:
        class_index_dict[class_dict[item][0]] = item
    class_dict = class_index_dict
    train_image_dir = "D:/jingou/ImageNet/ImageNet 2012 DataSets/ILSVRC2012_img_train"
    image_file_names = [os.path.join(train_image_dir, image_file_name) for image_file_name in filenames]
    filename,_ = filenames[9999].split('/')
    print(image_file_names[9999])
    print(filename)
    print(class_dict[filename])




