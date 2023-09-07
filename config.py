import random

import numpy as np
import torch
from torch.backends import cudnn

from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode as IMode
import os


# Random seed to maintain reproducible results
random.seed(0)
torch.manual_seed(0)
np.random.seed(0)
# Use GPU for training by default
device = torch.device("cuda", 0)
# Turning on when the image size does not change during training can speed up training
cudnn.benchmark = True
# When evaluating the performance of the SR model, whether to verify only the Y channel image data
only_test_y_channel = True
# Image magnification factor
upscale_factor = 12

# Current configuration parameter method
# task name
task_name = "semseg"
# train / valid mode
# mode = f"train_MD_ESRGAN"
mode = "valid"


# common parameters
# Dataset address PASCAL CONTEXT
train_image_dir = "./data/VOC"
valid_image_dir = "./data/VOC"

# in order to mod image, hr image_size corresponding to different scaling factors
if upscale_factor == 4:
    mod = 12
    image_size = 256
elif upscale_factor == 6:
    mod = 6
    image_size = 258
elif upscale_factor == 8:
    mod = 8
    image_size = 256
else:
    mod = 12  # X12
    image_size = 288

# Dataset address test Dataset
test_lr_image_dir = f"./data/test/Set5/LRbicx{upscale_factor}"
test_hr_image_dir = f"./data/test/Set5/GTmod{mod}"

batch_size = 4
num_workers = 4
# pascal context classes
num_classes = 20
start_epoch = 0

# Feature extraction layer parameter configuration Context_LOSS
feature_model_extractor_node = "features.34"
feature_model_normalize_mean = [0.485, 0.456, 0.406]
feature_model_normalize_std = [0.229, 0.224, 0.225]


if mode == "train_G":

    exp_name = f"G"

    #  semseg_network not training with sr generator, but resume weight to cal semseg loss for SR
    semseg_network_weight = ""
    # RRDBNet pre-trained weight as the initial weight for G
    resume = f".resluts/pre-trained/RRDBNet_PASCAL_X{upscale_factor}.pth.tar"
    resume_g = ""

    epochs = 100

    # Loss function weight
    pixel_weight = 0.01
    content_weight = 1.0
    semseg_weight = 0.05
    # Adam optimizer parameter
    model_lr = 1e-4
    model_betas = (0.9, 0.99)

    # Dynamically adjust the learning rate policy
    lr_scheduler_milestones = [int(epochs * 0.125), int(epochs * 0.250), int(epochs * 0.500), int(epochs * 0.750)]
    lr_scheduler_gamma = 0.5

    # How many iterations to print the training result
    print_frequency = 200

if mode == "train_T_ESRGAN":

    exp_name = f"T-ESRGAN"

    #  semseg_network not training with sr generator, but resume weight to cal semseg loss for SR
    semseg_network_weight = ""
    # RRDBNet pre-trained weight as the initial weight for G
    resume = f".resluts/pre-trained/ESRGAN_G_PASCAL_X{upscale_factor}.pth.tar"

    resume_d = ""
    resume_g = ""

    epochs = 100

    # Loss function weight
    pixel_weight = 0.01
    content_weight = 1.0
    semseg_weight = 0.05
    adversarial_weight = 0.001
    # Adam optimizer parameter
    model_lr = 1e-4
    model_betas = (0.9, 0.99)

    # Dynamically adjust the learning rate policy
    lr_scheduler_milestones = [int(epochs * 0.125), int(epochs * 0.250), int(epochs * 0.500), int(epochs * 0.750)]
    lr_scheduler_gamma = 0.5


if mode == "train_MD_ESRGAN":

    exp_name = f"MD-ESRGAN"

    # Incremental training and migration training
    epochs = 60

    # ESRGAN pre-trained generator weight as the initial weight for MD-ESRGAN generator
    resume = f".resluts/pre-trained/ESRGAN_G_PASCAL_X{upscale_factor}.pth.tar"

    # Continues training MD-ESRGAN
    resume_d = ""
    resume_g = ""


    # Loss function weight
    pixel_weight = 0.01
    content_weight = 1.0
    adversarial_weight = 0.001
    semseg_weight = 0.01

    # Adam optimizer parameter
    model_lr = 1e-4
    model_betas = (0.9, 0.99)

    # Dynamically adjust the learning rate policy
    lr_scheduler_milestones = [int(epochs * 0.125), int(epochs * 0.250), int(epochs * 0.500), int(epochs * 0.750)]
    lr_scheduler_gamma = 0.5

    # How many iterations to print the training result
    print_frequency = 200


if mode == "valid":

    exp_name = "MD-ESRGAN"

    # Test data address
    padding_image_size = 512
    trans_bicubic = transforms.Resize(padding_image_size//upscale_factor, interpolation=IMode.BICUBIC)
    trans_toTensor = transforms.ToTensor()

    # save generate images
    # if upscale_factor == 4 or upscale_factor == 12:
    #     mod = 12
    # elif upscale_factor == 6:
    #     mod = 6
    # else:
    #     mod = 8

    # Set
    sr_dir_set = f"./results/save_imgs/{exp_name}/X{upscale_factor}/Set"
    if not os.path.exists(sr_dir_set):
        os.makedirs(sr_dir_set)

    lr_dir_set5 = f"./data/test/Set5/LRbicx{upscale_factor}"
    hr_dir_set5 = f"./data/test/Set5/GTmod{upscale_factor}"

    lr_dir_set14 = f"./data/test/Set14/LRbicx{upscale_factor}"
    hr_dir_set14 = f"./data/test/Set14/GTmod{upscale_factor}"

    # BSDS100
    sr_dir_bsds = f"./results/save_imgs/{exp_name}/X{upscale_factor}/BSDS"
    if not os.path.exists(sr_dir_bsds):
        os.makedirs(sr_dir_bsds)
    lr_dir_bsds = f"./data/test/BSDS100/LRbicx{upscale_factor}"
    hr_dir_bsds = f"./data/test/BSDS100/GTmod{upscale_factor}"


    # PascalContext
    sr_path = f"./results/save_imgs/{exp_name}/X{upscale_factor}/PascalContext"
    if not os.path.exists(sr_path):
        os.makedirs(sr_path)
    pascal_context_path = f"./data/VOC/PASCAL_MT"



    # resume pretrained weights to reconsitution
    # exp_name choice which method : G / T-ESRGAN / MD-ESRGAN

    model_path = f"./results/{exp_name}/G_X{upscale_factor}.pth.tar"




