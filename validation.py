import os

import cv2
import torch
from natsort import natsorted
import numpy as np

import config as config
from imgproc import image2tensor,tensor2image,image_resize,modcrop,modcrop_up
from image_quality_assessment import PSNR, SSIM
from model import Generator,Generator_X8,Generator_X6,Generator_X12


def process_image_padding(hr_image,img_size=512):
    """
    Args:
        hr_image: Image.open 打开的文件 且转为 ndarray 了
    Returns:
        需要进行 边缘填充 以最大边为基准 且保证h 和 w 是 scale 的倍数
    """
    # W, H = hr_image.size[0], hr_image.size[1]  # size[0]W; size[1]H
    H = hr_image.shape[0]
    W = hr_image.shape[1]
    # paddingImag
    delta_height = max(img_size - H, 0)  # H 需要填充的像素大小
    delta_width = max(img_size - W, 0)  # W 需要填充的像素大小
    # Location to place image 图片在填充后的图像中的位置
    height_location = [delta_height // 2, (delta_height // 2) + H]
    width_location = [delta_width // 2, (delta_width // 2) + W]
    # padding 后 图像大小
    max_height = max(img_size, H)
    max_width = max(img_size, W)
    # 填充内容
    # fill_index = [255, 255, 255]  # 填充的值 img 0-255
    fill_index = [1., 1., 1.]  # 填充的值 img 0-1
    pad_value = fill_index
    padded = np.ones((max_height, max_width, 3)) * pad_value
    padded[height_location[0]:height_location[1], width_location[0]:width_location[1], :] = hr_image
    return padded
def process_image_depadding(sr_tensor,orig_h,orig_w):
    """
    :param sr_image:  重构的图片 tensor 512*512 BCHW
    :return: 减去增加的padding 得到 原始图像size 返回一个tensor类型的
    """
    # Cut image borders
    current_img_h = sr_tensor.shape[2]
    current_img_w = sr_tensor.shape[3]
    delta_height = current_img_h - orig_h
    delta_width  = current_img_w - orig_w
    height_location = [delta_height // 2, (delta_height // 2) + orig_h]
    width_location = [delta_width // 2, (delta_width // 2) + orig_w]
    return sr_tensor[:,:,height_location[0]:height_location[1],width_location[0]:width_location[1]]


def main() -> None:
    # Initialize the super-resolution model
    if config.upscale_factor==4:
        model = Generator().to(device=config.device, memory_format=torch.channels_last)
    elif config.upscale_factor==6:
        model = Generator_X6().to(device=config.device, memory_format=torch.channels_last)
    elif config.upscale_factor==8:
        model = Generator_X8().to(device=config.device, memory_format=torch.channels_last)
    else:
        model = Generator_X12().to(device=config.device, memory_format=torch.channels_last)
    print("Build ESRGAN model successfully.")

    # Load the super-resolution model weights
    checkpoint = torch.load(config.model_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint["state_dict"])
    # model.load_state_dict(checkpoint)
    print(f"Load ESRGAN model weights `{os.path.abspath(config.model_path)}` successfully.")

    # Create a folder of super-resolution experiment results
    results_dir = os.path.join("results", "save_imgs", config.exp_name,config.upscale_factor)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Start the verification mode of the model.
    model.eval()
    # Turn on half-precision inference.
    model.half()

    # Initialize the sharpness evaluation function
    psnr = PSNR(config.upscale_factor, config.only_test_y_channel)
    ssim = SSIM(config.upscale_factor, config.only_test_y_channel)

    # Set the sharpness evaluation function calculation device to the specified model
    psnr = psnr.to(device=config.device, memory_format=torch.channels_last, non_blocking=True)
    ssim = ssim.to(device=config.device, memory_format=torch.channels_last, non_blocking=True)

    # ======== Generate standard test images : Set / BSDS100 ==============
    # Get a list of test image file names.
    lr_file_dir = {"set5":config.lr_dir_set5,"set14":config.lr_dir_set14,"bsds":config.lr_dir_bsds}
    hr_file_dir = {"set5":config.hr_dir_set5,"set14":config.hr_dir_set14,"bsds":config.hr_dir_bsds}
    sr_dir      = {"set5":config.sr_dir_set,"set14":config.sr_dir_set,"bsds":config.sr_dir_bsds}

    for name in lr_file_dir.keys():
        file_names = natsorted(os.listdir(lr_file_dir[name]))
        total_files = len(file_names)

        # Initialize IQA metrics
        psnr_metrics = 0.0
        ssim_metrics = 0.0

        for index in range(total_files):
            lr_image_path = os.path.join(lr_file_dir[name], file_names[index])
            sr_image_path = os.path.join(sr_dir[name], file_names[index])
            hr_image_path = os.path.join(hr_file_dir[name], file_names[index])
            print(f"Processing `{os.path.abspath(lr_image_path)}`...")
            # Read LR image and HR image BGR
            lr_image = cv2.imread(lr_image_path, cv2.IMREAD_UNCHANGED)
            hr_image = cv2.imread(hr_image_path, cv2.IMREAD_UNCHANGED)
            # Convert BGR channel image format data to RGB channel image format data
            lr_image = cv2.cvtColor(lr_image, cv2.COLOR_BGR2RGB)
            hr_image = cv2.cvtColor(hr_image, cv2.COLOR_BGR2RGB)
            # Convert RGB channel image format data to Tensor channel image format data
            lr_tensor = image2tensor(lr_image, range_norm=False, half=True).unsqueeze_(0)
            hr_tensor = image2tensor(hr_image, range_norm=False, half=True).unsqueeze_(0)
            # Transfer Tensor channel image format data to CUDA device
            lr_tensor = lr_tensor.to(device=config.device, memory_format=torch.channels_last, non_blocking=True)
            hr_tensor = hr_tensor.to(device=config.device, memory_format=torch.channels_last, non_blocking=True)

            # Only reconstruct the Y channel image data.
            with torch.no_grad():
                sr_tensor = model(lr_tensor)
            # Save image to BGR
            sr_image = tensor2image(sr_tensor, range_norm=False, half=True)
            sr_image = cv2.cvtColor(sr_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(sr_image_path, sr_image)

            # Cal IQA metrics
            psnr_value = psnr(sr_tensor, hr_tensor).item()
            ssim_value = ssim(sr_tensor, hr_tensor).item()
            psnr_metrics += psnr_value
            ssim_metrics += ssim_value
            print("psnr: {}".format(psnr_value), "ssim: {}".format(ssim_value))
        # Calculate the average value of the sharpness evaluation index,
        # and all index range values are cut according to the following values
        # PSNR range value is 0~100
        # SSIM range value is 0~1
        avg_ssim = 1 if ssim_metrics / total_files > 1 else ssim_metrics / total_files
        avg_psnr = 100 if psnr_metrics / total_files > 100 else psnr_metrics / total_files
        print(f"TestDataset: {name}\n"
              f"PSNR: {avg_psnr:4.2f} dB\n"
              f"SSIM: {avg_ssim:4.4f} u")
    #
    #
    #
    #
    # file_names = natsorted(os.listdir(config.lr_dir))
    # # Get the number of test image files.
    # total_files = len(file_names)
    #
    # for index in range(total_files):
    #     lr_image_path = os.path.join(config.lr_dir, file_names[index])
    #     sr_image_path = os.path.join(config.sr_dir, file_names[index])
    #     hr_image_path = os.path.join(config.hr_dir, file_names[index])
    #
    #     print(f"Processing `{os.path.abspath(lr_image_path)}`...")
    #     # Read LR image and HR image BGR
    #     lr_image = cv2.imread(lr_image_path, cv2.IMREAD_UNCHANGED)
    #     hr_image = cv2.imread(hr_image_path, cv2.IMREAD_UNCHANGED)
    #
    #     # Convert BGR channel image format data to RGB channel image format data
    #     lr_image = cv2.cvtColor(lr_image, cv2.COLOR_BGR2RGB)
    #     hr_image = cv2.cvtColor(hr_image, cv2.COLOR_BGR2RGB)
    #
    #     # Convert RGB channel image format data to Tensor channel image format data
    #     lr_tensor = image2tensor(lr_image, range_norm=False, half=True).unsqueeze_(0)
    #     hr_tensor = image2tensor(hr_image, range_norm=False, half=True).unsqueeze_(0)
    #
    #     # Transfer Tensor channel image format data to CUDA device
    #     lr_tensor = lr_tensor.to(device=config.device, memory_format=torch.channels_last, non_blocking=True)
    #     hr_tensor = hr_tensor.to(device=config.device, memory_format=torch.channels_last, non_blocking=True)
    #
    #     # Only reconstruct the Y channel image data.
    #     with torch.no_grad():
    #         sr_tensor = model(lr_tensor)
    #
    #     # Save image to BGR
    #     sr_image = tensor2image(sr_tensor, range_norm=False, half=True)
    #     sr_image = cv2.cvtColor(sr_image, cv2.COLOR_RGB2BGR)
    #     cv2.imwrite(sr_image_path, sr_image)
    #
    #     # Cal IQA metrics
    #     psnr_value = psnr(sr_tensor, hr_tensor).item()
    #     ssim_value = ssim(sr_tensor, hr_tensor).item()
    #     psnr_metrics += psnr_value
    #     ssim_metrics += ssim_value
    #     print("psnr: {}".format(psnr_value), "ssim: {}".format(ssim_value))
    #
    # # Calculate the average value of the sharpness evaluation index,
    # # and all index range values are cut according to the following values
    # # PSNR range value is 0~100
    # # SSIM range value is 0~1
    # avg_ssim = 1 if ssim_metrics / total_files > 1 else ssim_metrics / total_files
    # avg_psnr = 100 if psnr_metrics / total_files > 100 else psnr_metrics / total_files
    #
    # print(f"PSNR: {avg_psnr:4.2f} dB\n"
    #       f"SSIM: {avg_ssim:4.4f} u")


    # ======== Generate standard test images : Set / BSDS100 ==============
    # get VOC HR images
    image_set = 'val'
    image_dir = os.path.join(config.pascal_context_path, 'JPEGImages')
    splits_dir = os.path.join(config.pascal_context_path, 'ImageSets/Context')
    split_f = os.path.join(splits_dir, image_set.rstrip('\n') + '.txt')
    if not os.path.exists(split_f):
        raise ValueError('Wrong image_set entered! Please select an appropriate one.')
    with open(os.path.join(split_f), "r") as f:
        file_names = [x.strip() for x in f.readlines()]
    images = [os.path.join(image_dir, x + ".jpg") for x in file_names]

    for index in range(len(images)):
        # CV2 imread
        # BRG 0 - 1
        hr_img = cv2.imread(images[index], cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.
        # RGB 0 - 1
        hr_img = cv2.cvtColor(hr_img, cv2.COLOR_BGR2RGB)
        # mod 到 upscale factor的倍数
        hr_img_mod = modcrop(hr_img, config.upscale_factor)
        # matlab bicubic resize 64 * 64
        lr_img = image_resize(hr_img_mod, 1 / config.upscale_factor)
        # lr to tensor
        lr_tensor = image2tensor(lr_img, range_norm=False, half=True).to(config.device).unsqueeze(0)
        # generate sr
        with torch.no_grad():
            sr_tensor = model(lr_tensor) # 送入小图 生成 超分辨图像 BCHW 512*512
        # Save image to BGR
        sr_image = tensor2image(sr_tensor, range_norm=False, half=True)
        sr_image = cv2.cvtColor(sr_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(config.sr_path,file_names[index]+'.png'), sr_image)

        print("save reconsitution image:",file_names[index])
#
# def save_pascal_context_SR_image():
#     # Load model weights.
#     # model = Generator().to(config.device) # X4
#     # model = Generator_X8().to(device=config.device) # X8
#     # model = Generator_X6().to(device=config.device) # X6
#     model = Generator_X12().to(device=config.device)
#     state_dict = torch.load(config.model_path, map_location=config.device)
#     model.load_state_dict(state_dict['state_dict'])
#     # model.load_state_dict(state_dict)
#
#     # Start the verification mode of the model.
#     model.eval()
#     # Turn on half-precision inference.
#     model.half()
#     # transfors
#     # get VOC HR images
#     image_set = 'val'
#     image_dir = os.path.join(config.pascal_context_path, 'JPEGImages')
#     splits_dir = os.path.join(config.pascal_context_path, 'ImageSets/Context')
#     split_f = os.path.join(splits_dir, image_set.rstrip('\n') + '.txt')
#     if not os.path.exists(split_f):
#         raise ValueError('Wrong image_set entered! Please select an appropriate one.')
#     with open(os.path.join(split_f), "r") as f:
#         file_names = [x.strip() for x in f.readlines()]
#     images = [os.path.join(image_dir, x + ".jpg") for x in file_names]
#
#     for index in range(len(images)):
#
#         # # PIL 读取
#         # hr_img = np.array(Image.open(images[index]).convert('RGB')).astype(np.float32)
#         # hr_img_pad = process_image_padding(hr_img)  # 填充 512*512
#         # hr_img_pad = Image.fromarray(np.uint8(hr_img_pad)) # ndarry 转为 PIL格式 方便后续 transforms
#         #
#         # lr_img = config.trans_bicubic(hr_img_pad) # PIL 进行resize 到下采样upsaler_factor倍
#         # lr_tensor = config.trans_toTensor(lr_img).to(config.device).unsqueeze(0) # 转tensor 并且 to GPU 且 增加一维 1 C H W
#         #
#         # # 类型转换
#         # lr_tensor = lr_tensor.half()
#         # with torch.no_grad():
#         #     sr_tensor = model(lr_tensor) # 送入小图 生成 超分辨图像 BCHW 512*512
#         #     sr_tensor_depad = process_image_depadding(sr_tensor,hr_img.shape[0],hr_img.shape[1]) # BCHW
#         # # save by tensor
#         # torchvision.utils.save_image(sr_tensor_depad, os.path.join(config.sr_path,file_names[index]+'.png')) # 保存生成的超分图像
#
#         # CV2 imread
#         # BRG 0 - 1
#         hr_img = cv2.imread(images[index], cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.
#         # RGB 0 - 1
#         hr_img = cv2.cvtColor(hr_img, cv2.COLOR_BGR2RGB)
#         # 填充到 512 * 512 大小的image
#         # hr_img_pad = process_image_padding(hr_img, img_size=config.padding_image_size)
#         # 修改到 upscale factor的倍数
#         hr_img_mod = modcrop(hr_img, config.upscale_factor)
#         # matlab bicubic resize 64 * 64
#         lr_img = image_resize(hr_img_mod, 1 / config.upscale_factor)
#         # lr to tensor
#         lr_tensor = image2tensor(lr_img, range_norm=False, half=True).to(config.device).unsqueeze(0)
#         # generate sr
#         with torch.no_grad():
#             sr_tensor = model(lr_tensor) # 送入小图 生成 超分辨图像 BCHW 512*512
#             # sr_tensor = process_image_depadding(sr_tensor,hr_img.shape[0],hr_img.shape[1]) # BCHW
#         # Save image to BGR
#         sr_image = tensor2image(sr_tensor, range_norm=False, half=True)
#         sr_image = cv2.cvtColor(sr_image, cv2.COLOR_RGB2BGR)
#         cv2.imwrite(os.path.join(config.sr_path,file_names[index]+'.png'), sr_image)
#
#         print("save reconsitution image:",file_names[index])

# def get_voc_sr_imgs(model, images, file_names):
#
#     for index in range(len(images)):
#
#         hr_img = cv2.imread(images[index], cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.
#         # RGB 0 - 1
#         hr_img = cv2.cvtColor(hr_img, cv2.COLOR_BGR2RGB)
#         # 修改到 upscale factor的倍数 - padding 0
#         hr_img_mod = modcrop_up(hr_img, config.upscale_factor)
#         hr_img = hr_img_mod['img']
#         hr_org_H, hr_org_W = hr_img_mod['H'],hr_img_mod['W']
#         # matlab bicubic resize
#         lr_img = image_resize(hr_img, 1 / config.upscale_factor)
#         # lr to tensor
#         lr_tensor = image2tensor(lr_img, range_norm=False, half=True).to(config.device).unsqueeze(0)
#         # generate sr
#         with torch.no_grad():
#             sr_tensor = model(lr_tensor)  # 送入小图 生成 超分辨图像 BCHW 512*512
#         # Save image to BGR
#         sr_image = tensor2image(sr_tensor, range_norm=False, half=True)
#         # resume original img size
#         sr_image_save = sr_image[:hr_org_H,:hr_org_W,:]
#         sr_image_save = cv2.cvtColor(sr_image_save, cv2.COLOR_RGB2BGR)
#
#         cv2.imwrite(os.path.join(config.sr_path, file_names[index] + '.png'), sr_image_save)
#         print("save reconsitution image:", file_names[index])
#
# def save_DIV2K_SR_image():
#     # Load model weights.
#     # model = Generator().to(config.device) # X4
#     # model = Generator_X6().to(device=config.device) # X6
#     # model = Generator_X8().to(device=config.device) # X8
#     model = Generator_X12().to(device=config.device)
#     # White
#     model_path =  f"E:/jingou/ESRGAN/save_weights/ESRGAN_baseline/add_seg_classify/add_classify/X4/g_best_1202.pth.tar"
#     # model_path = "D:/jingou/SuperResolution/ESRGAN/results/ESRGAN_baseline/add_seg_classify/add_classify/X4/g_best_1202.pth.tar"
#     # Black
#     # model_path = "D:/jingou/SuperResolution/ESRGAN/results/ESRGAN_baseline/Signle_D/X4/classify/11_no_bg_class/g_best.pth.tar"
#     state_dict = torch.load(model_path, map_location=config.device)
#     model.load_state_dict(state_dict['state_dict'])
#     # model.load_state_dict(state_dict)
#
#     # Start the verification mode of the model.
#     model.eval()
#     # Turn on half-precision inference.
#     model.half()
#     # transfors
#     # get DIV2K HR images
#     image_dir = "D:/jingou/SuperResolution/testdataset/Datasets/SuperResolution/Common/DIV2K/DIV2K_valid_HR"
#     # White
#     sr_path   = "E:/jingou/ESRGAN/save_imgs/ESRGAN_baseline/Add_single_loss/classify/X4/1_concat&ss/DIV2K"
#     # Black
#     # sr_path   = "E:/jingou/ESRGAN/save_imgs/ESRGAN_baseline/Signle_D/classify/X4/DIV2K"
#
#     if not os.path.exists(sr_path):
#         os.makedirs(sr_path)
#     file_names = os.listdir(image_dir)
#     # print(file_names) # 0805.png
#     images = [os.path.join(image_dir, x) for x in file_names]
#     # print(images) # 'D:/jingou/SuperResolution/testdataset/Datasets/SuperResolution/Common/DIV2K/DIV2K_valid_HR\\0801.png
#     for index in range(len(images)):
#
#         # CV2 imread
#         # BRG 0 - 1
#         hr_img = cv2.imread(images[index], cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.
#         # RGB 0 - 1
#         hr_img = cv2.cvtColor(hr_img, cv2.COLOR_BGR2RGB)
#         # 修改到 upscale factor的倍数
#         hr_img_mod = modcrop(hr_img, config.upscale_factor)
#         # matlab bicubic resize 64 * 64
#         lr_img = image_resize(hr_img_mod, 1 / config.upscale_factor)
#         # lr to tensor
#         lr_tensor = image2tensor(lr_img, range_norm=False, half=True).to(config.device).unsqueeze(0)
#         # generate sr
#         with torch.no_grad():
#             sr_tensor = model(lr_tensor) # 送入小图 生成 超分辨图像 BCHW 512*512
#         # Save image to BGR
#         sr_image = tensor2image(sr_tensor, range_norm=False, half=True)
#         sr_image = cv2.cvtColor(sr_image, cv2.COLOR_RGB2BGR)
#         cv2.imwrite(os.path.join(sr_path,file_names[index]), sr_image)
#
#         print("save reconsitution image:",file_names[index])


if __name__ == "__main__":
    # test
    main()
