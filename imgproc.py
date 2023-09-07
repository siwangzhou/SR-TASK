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
import math
import random
from typing import Any

import cv2
import numpy as np
import torch
from torchvision.transforms import functional as F

__all__ = [
    "image2tensor", "tensor2image",
    "image_resize",
    "expand_y", "rgb2ycbcr", "bgr2ycbcr", "ycbcr2bgr", "ycbcr2rgb",
    "center_crop", "random_crop", "random_vertically_flip", "random_horizontally_flip",
    # "random_rotate",
    "random_scaling","padImg","random_crop_2",
    "mask2one_hot"
]


def image2tensor(image: np.ndarray, range_norm: bool, half: bool) -> torch.Tensor:
    """Convert the image data type to the Tensor (NCWH) data type supported by PyTorch

    Args:
        image (np.ndarray): The image data read by ``OpenCV.imread``, the data range is [0,255] or [0, 1]
        range_norm (bool): Scale [0, 1] data to between [-1, 1]
        half (bool): Whether to convert torch.float32 similarly to torch.half type

    Returns:
        tensor (torch.Tensor): Data types supported by PyTorch

    Examples:
        >>> example_image = cv2.imread("lr_image.bmp")
        >>> example_tensor = image2tensor(example_image, range_norm=True, half=False)

    """
    # Convert image data type to Tensor data type
    tensor = F.to_tensor(image)

    # Scale the image data from [0, 1] to [-1, 1]
    if range_norm:
        tensor = tensor.mul(2.0).sub(1.0)

    # Convert torch.float32 image data type to torch.half image data type
    if half:
        tensor = tensor.half()

    return tensor


def tensor2image(tensor: torch.Tensor, range_norm: bool, half: bool) -> Any:
    """Convert the Tensor(NCWH) data type supported by PyTorch to the np.ndarray(WHC) image data type

    Args:
        tensor (torch.Tensor): Data types supported by PyTorch (NCHW), the data range is [0, 1]
        range_norm (bool): Scale [-1, 1] data to between [0, 1]
        half (bool): Whether to convert torch.float32 similarly to torch.half type.

    Returns:
        image (np.ndarray): Data types supported by PIL or OpenCV

    Examples:
        >>> example_image = cv2.imread("lr_image.bmp")
        >>> example_tensor = image2tensor(example_image, range_norm=False, half=False)

    """
    if range_norm:
        tensor = tensor.add(1.0).div(2.0)
    if half:
        tensor = tensor.half()

    image = tensor.squeeze(0).permute(1, 2, 0).mul(255).clamp(0, 255).cpu().numpy().astype("uint8")

    return image


# Code reference `https://github.com/xinntao/BasicSR/blob/master/basicsr/utils/matlab_functions.py`
def _cubic(x: Any) -> Any:
    """Implementation of `cubic` function in Matlab under Python language.

    Args:
        x: Element vector.

    Returns:
        Bicubic interpolation

    """
    absx = torch.abs(x)
    absx2 = absx ** 2
    absx3 = absx ** 3
    return (1.5 * absx3 - 2.5 * absx2 + 1) * ((absx <= 1).type_as(absx)) + (
            -0.5 * absx3 + 2.5 * absx2 - 4 * absx + 2) * (
               ((absx > 1) * (absx <= 2)).type_as(absx))


# Code reference `https://github.com/xinntao/BasicSR/blob/master/basicsr/utils/matlab_functions.py`
def _calculate_weights_indices(in_length: int,
                               out_length: int,
                               scale: float,
                               kernel_width: int,
                               antialiasing: bool) -> [np.ndarray, np.ndarray, int, int]:
    """Implementation of `calculate_weights_indices` function in Matlab under Python language.

    Args:
        in_length (int): Input length.
        out_length (int): Output length.
        scale (float): Scale factor.
        kernel_width (int): Kernel width.
        antialiasing (bool): Whether to apply antialiasing when down-sampling operations.
            Caution: Bicubic down-sampling in PIL uses antialiasing by default.

    Returns:
       weights, indices, sym_len_s, sym_len_e

    """
    if (scale < 1) and antialiasing:
        # Use a modified kernel (larger kernel width) to simultaneously
        # interpolate and antialiasing
        kernel_width = kernel_width / scale

    # Output-space coordinates
    x = torch.linspace(1, out_length, out_length)

    # Input-space coordinates. Calculate the inverse mapping such that 0.5
    # in output space maps to 0.5 in input space, and 0.5 + scale in output
    # space maps to 1.5 in input space.
    u = x / scale + 0.5 * (1 - 1 / scale)

    # What is the left-most pixel that can be involved in the computation?
    left = torch.floor(u - kernel_width / 2)

    # What is the maximum number of pixels that can be involved in the
    # computation?  Note: it's OK to use an extra pixel here; if the
    # corresponding weights are all zero, it will be eliminated at the end
    # of this function.
    p = math.ceil(kernel_width) + 2

    # The indices of the input pixels involved in computing the k-th output
    # pixel are in row k of the indices matrix.
    indices = left.view(out_length, 1).expand(out_length, p) + torch.linspace(0, p - 1, p).view(1, p).expand(
        out_length, p)

    # The weights used to compute the k-th output pixel are in row k of the
    # weights matrix.
    distance_to_center = u.view(out_length, 1).expand(out_length, p) - indices

    # apply cubic kernel
    if (scale < 1) and antialiasing:
        weights = scale * _cubic(distance_to_center * scale)
    else:
        weights = _cubic(distance_to_center)

    # Normalize the weights matrix so that each row sums to 1.
    weights_sum = torch.sum(weights, 1).view(out_length, 1)
    weights = weights / weights_sum.expand(out_length, p)

    # If a column in weights is all zero, get rid of it. only consider the
    # first and last column.
    weights_zero_tmp = torch.sum((weights == 0), 0)
    if not math.isclose(weights_zero_tmp[0], 0, rel_tol=1e-6):
        indices = indices.narrow(1, 1, p - 2)
        weights = weights.narrow(1, 1, p - 2)
    if not math.isclose(weights_zero_tmp[-1], 0, rel_tol=1e-6):
        indices = indices.narrow(1, 0, p - 2)
        weights = weights.narrow(1, 0, p - 2)
    weights = weights.contiguous()
    indices = indices.contiguous()
    sym_len_s = -indices.min() + 1
    sym_len_e = indices.max() - in_length
    indices = indices + sym_len_s - 1
    return weights, indices, int(sym_len_s), int(sym_len_e)


# Code reference `https://github.com/xinntao/BasicSR/blob/master/basicsr/utils/matlab_functions.py`
def image_resize(image: Any, scale_factor: float, antialiasing: bool = True) -> Any:
    """Implementation of `imresize` function in Matlab under Python language.

    Args:
        image: The input image.
        scale_factor (float): Scale factor. The same scale applies for both height and width.
        antialiasing (bool): Whether to apply antialiasing when down-sampling operations.
            Caution: Bicubic down-sampling in `PIL` uses antialiasing by default. Default: ``True``.

    Returns:
        out_2 (np.ndarray): Output image with shape (c, h, w), [0, 1] range, w/o round

    """
    squeeze_flag = False
    if type(image).__module__ == np.__name__:  # numpy type
        numpy_type = True
        if image.ndim == 2:
            image = image[:, :, None]
            squeeze_flag = True
        image = torch.from_numpy(image.transpose(2, 0, 1)).float()
    else:
        numpy_type = False
        if image.ndim == 2:
            image = image.unsqueeze(0)
            squeeze_flag = True

    in_c, in_h, in_w = image.size()
    out_h, out_w = math.ceil(in_h * scale_factor), math.ceil(in_w * scale_factor)
    kernel_width = 4

    # get weights and indices
    weights_h, indices_h, sym_len_hs, sym_len_he = _calculate_weights_indices(in_h, out_h, scale_factor, kernel_width,
                                                                              antialiasing)
    weights_w, indices_w, sym_len_ws, sym_len_we = _calculate_weights_indices(in_w, out_w, scale_factor, kernel_width,
                                                                              antialiasing)
    # process H dimension
    # symmetric copying
    img_aug = torch.FloatTensor(in_c, in_h + sym_len_hs + sym_len_he, in_w)
    img_aug.narrow(1, sym_len_hs, in_h).copy_(image)

    sym_patch = image[:, :sym_len_hs, :]
    inv_idx = torch.arange(sym_patch.size(1) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(1, inv_idx)
    img_aug.narrow(1, 0, sym_len_hs).copy_(sym_patch_inv)

    sym_patch = image[:, -sym_len_he:, :]
    inv_idx = torch.arange(sym_patch.size(1) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(1, inv_idx)
    img_aug.narrow(1, sym_len_hs + in_h, sym_len_he).copy_(sym_patch_inv)

    out_1 = torch.FloatTensor(in_c, out_h, in_w)
    kernel_width = weights_h.size(1)
    for i in range(out_h):
        idx = int(indices_h[i][0])
        for j in range(in_c):
            out_1[j, i, :] = img_aug[j, idx:idx + kernel_width, :].transpose(0, 1).mv(weights_h[i])

    # process W dimension
    # symmetric copying
    out_1_aug = torch.FloatTensor(in_c, out_h, in_w + sym_len_ws + sym_len_we)
    out_1_aug.narrow(2, sym_len_ws, in_w).copy_(out_1)

    sym_patch = out_1[:, :, :sym_len_ws]
    inv_idx = torch.arange(sym_patch.size(2) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(2, inv_idx)
    out_1_aug.narrow(2, 0, sym_len_ws).copy_(sym_patch_inv)

    sym_patch = out_1[:, :, -sym_len_we:]
    inv_idx = torch.arange(sym_patch.size(2) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(2, inv_idx)
    out_1_aug.narrow(2, sym_len_ws + in_w, sym_len_we).copy_(sym_patch_inv)

    out_2 = torch.FloatTensor(in_c, out_h, out_w)
    kernel_width = weights_w.size(1)
    for i in range(out_w):
        idx = int(indices_w[i][0])
        for j in range(in_c):
            out_2[j, :, i] = out_1_aug[j, :, idx:idx + kernel_width].mv(weights_w[i])

    if squeeze_flag:
        out_2 = out_2.squeeze(0)
    if numpy_type:
        out_2 = out_2.numpy()
        if not squeeze_flag:
            out_2 = out_2.transpose(1, 2, 0)

    return out_2


def expand_y(image: np.ndarray) -> np.ndarray:
    """Convert BGR channel to YCbCr format,
    and expand Y channel data in YCbCr, from HW to HWC

    Args:
        image (np.ndarray): Y channel image data

    Returns:
        y_image (np.ndarray): Y-channel image data in HWC form

    """
    # Normalize image data to [0, 1]
    image = image.astype(np.float32) / 255.

    # Convert BGR to YCbCr, and extract only Y channel
    y_image = bgr2ycbcr(image, only_use_y_channel=True)

    # Expand Y channel
    y_image = y_image[..., None]

    # Normalize the image data to [0, 255]
    y_image = y_image.astype(np.float64) * 255.0

    return y_image


# Code reference `https://github.com/xinntao/BasicSR/blob/master/basicsr/utils/matlab_functions.py`
def rgb2ycbcr(image: np.ndarray, only_use_y_channel: bool) -> np.ndarray:
    """Implementation of rgb2ycbcr function in Matlab under Python language

    Args:
        image (np.ndarray): Image input in RGB format.
        only_use_y_channel (bool): Extract Y channel separately

    Returns:
        image (np.ndarray): YCbCr image array data

    """
    if only_use_y_channel:
        image = np.dot(image, [65.481, 128.553, 24.966]) + 16.0
    else:
        image = np.matmul(image, [[65.481, -37.797, 112.0], [128.553, -74.203, -93.786], [24.966, 112.0, -18.214]]) + [
            16, 128, 128]

    image /= 255.
    image = image.astype(np.float32)

    return image


# Code reference `https://github.com/xinntao/BasicSR/blob/master/basicsr/utils/matlab_functions.py`
def bgr2ycbcr(image: np.ndarray, only_use_y_channel: bool) -> np.ndarray:
    """Implementation of bgr2ycbcr function in Matlab under Python language.

    Args:
        image (np.ndarray): Image input in BGR format
        only_use_y_channel (bool): Extract Y channel separately

    Returns:
        image (np.ndarray): YCbCr image array data

    """
    if only_use_y_channel:
        image = np.dot(image, [24.966, 128.553, 65.481]) + 16.0
    else:
        image = np.matmul(image, [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786], [65.481, -37.797, 112.0]]) + [
            16, 128, 128]

    image /= 255.
    image = image.astype(np.float32)

    return image


# Code reference `https://github.com/xinntao/BasicSR/blob/master/basicsr/utils/matlab_functions.py`
def ycbcr2rgb(image: np.ndarray) -> np.ndarray:
    """Implementation of ycbcr2rgb function in Matlab under Python language.

    Args:
        image (np.ndarray): Image input in YCbCr format.

    Returns:
        image (np.ndarray): RGB image array data

    """
    image_dtype = image.dtype
    image *= 255.

    image = np.matmul(image, [[0.00456621, 0.00456621, 0.00456621],
                              [0, -0.00153632, 0.00791071],
                              [0.00625893, -0.00318811, 0]]) * 255.0 + [-222.921, 135.576, -276.836]

    image /= 255.
    image = image.astype(image_dtype)

    return image


# Code reference `https://github.com/xinntao/BasicSR/blob/master/basicsr/utils/matlab_functions.py`
def ycbcr2bgr(image: np.ndarray) -> np.ndarray:
    """Implementation of ycbcr2bgr function in Matlab under Python language.

    Args:
        image (np.ndarray): Image input in YCbCr format.

    Returns:
        image (np.ndarray): BGR image array data

    """
    image_dtype = image.dtype
    image *= 255.

    image = np.matmul(image, [[0.00456621, 0.00456621, 0.00456621],
                              [0.00791071, -0.00153632, 0],
                              [0, -0.00318811, 0.00625893]]) * 255.0 + [-276.836, 135.576, -222.921]

    image /= 255.
    image = image.astype(image_dtype)

    return image


def rgb2ycbcr_torch(tensor: torch.Tensor, only_use_y_channel: bool) -> torch.Tensor:
    """Implementation of rgb2ycbcr function in Matlab under PyTorch

    References from：`https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.601_conversion`

    Args:
        tensor (torch.Tensor): Image data in PyTorch format
        only_use_y_channel (bool): Extract only Y channel

    Returns:
        tensor (torch.Tensor): YCbCr image data in PyTorch format

    """
    if only_use_y_channel:
        weight = torch.Tensor([[65.481], [128.553], [24.966]]).to(tensor)
        tensor = torch.matmul(tensor.permute(0, 2, 3, 1), weight).permute(0, 3, 1, 2) + 16.0
    else:
        weight = torch.Tensor([[65.481, -37.797, 112.0],
                               [128.553, -74.203, -93.786],
                               [24.966, 112.0, -18.214]]).to(tensor)
        bias = torch.Tensor([16, 128, 128]).view(1, 3, 1, 1).to(tensor)
        tensor = torch.matmul(tensor.permute(0, 2, 3, 1), weight).permute(0, 3, 1, 2) + bias

    tensor /= 255.

    return tensor


def bgr2ycbcr_torch(tensor: torch.Tensor, only_use_y_channel: bool) -> torch.Tensor:
    """Implementation of bgr2ycbcr function in Matlab under PyTorch

    References from：`https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.601_conversion`

    Args:
        tensor (torch.Tensor): Image data in PyTorch format
        only_use_y_channel (bool): Extract only Y channel

    Returns:
        tensor (torch.Tensor): YCbCr image data in PyTorch format

    """
    if only_use_y_channel:
        weight = torch.Tensor([[24.966], [128.553], [65.481]]).to(tensor)
        tensor = torch.matmul(tensor.permute(0, 2, 3, 1), weight).permute(0, 3, 1, 2) + 16.0
    else:
        weight = torch.Tensor([[24.966, 112.0, -18.214],
                               [128.553, -74.203, -93.786],
                               [65.481, -37.797, 112.0]]).to(tensor)
        bias = torch.Tensor([16, 128, 128]).view(1, 3, 1, 1).to(tensor)
        tensor = torch.matmul(tensor.permute(0, 2, 3, 1), weight).permute(0, 3, 1, 2) + bias

    tensor /= 255.

    return tensor


def center_crop(image: np.ndarray, image_size: int) -> np.ndarray:
    """Crop small image patches from one image center area.

    Args:
        image (np.ndarray): The input image for `OpenCV.imread`.
        image_size (int): The size of the captured image area.

    Returns:
        patch_image (np.ndarray): Small patch image

    """
    image_height, image_width = image.shape[:2]

    # Just need to find the top and left coordinates of the image
    top = (image_height - image_size) // 2
    left = (image_width - image_size) // 2

    # Crop image patch
    patch_image = image[top:top + image_size, left:left + image_size, ...]

    return patch_image


def random_crop(image: np.ndarray, image_size: int) -> np.ndarray:
    """Crop small image patches from one image.

    Args:
        image (np.ndarray): The input image for `OpenCV.imread`.
        image_size (int): The size of the captured image area.

    Returns:
        patch_image (np.ndarray): Small patch image

    """
    image_height, image_width = image.shape[:2]

    # Just need to find the top and left coordinates of the image
    top = random.randint(0, image_height - image_size)
    left = random.randint(0, image_width - image_size)

    # Crop image patch
    patch_image = image[top:top + image_size, left:left + image_size, ...]

    return patch_image


def padding(image:np.ndarray, pad_img_size:int = None, range = 'uint8'):

    size = tuple([int(pad_img_size), int(pad_img_size)])
    # 填充的值
    if range == 'uint8':
        fill_index = [103,116,123] # BGR
    else:
        fill_index = [0.406, 0.456, 0.485] # BGR

    unpadded_shape = np.shape(image)
    # 需要padding的像素
    delta_height = max(size[0] - unpadded_shape[0], 0)
    delta_width  = max(size[1] - unpadded_shape[1], 0)

    # Location to place image
    height_location = [delta_height // 2, (delta_height // 2) + unpadded_shape[0]]
    width_location  = [delta_width // 2, (delta_width // 2) + unpadded_shape[1]]

    max_height = max(size[0], unpadded_shape[0])
    max_width  = max(size[1], unpadded_shape[1])

    # # padding image
    # pad_value = fill_index
    # padded_img = np.ones((max_height, max_width, 3)) * pad_value
    # padded_img[height_location[0]:height_location[1], width_location[0]:width_location[1], :] = image


    # cv2 padding image [top, bottom, left, right]: 对应的上下左右四边界扩充像素数
    padded_img = cv2.copyMakeBorder(image,height_location[0],max_height-height_location[1],width_location[0],max_width-width_location[1],cv2.BORDER_CONSTANT, value=fill_index)

    return padded_img

# def random_rotate(image: np.ndarray,
#                   angles: list,
#                   center: tuple[int, int] = None,
#                   scale_factor: float = 1.0) -> np.ndarray:
#     """Rotate an image by a random angle
#
#     Args:
#         image (np.ndarray): Image read with OpenCV
#         angles (list): Rotation angle range
#         center (optional, tuple[int, int]): High resolution image selection center point. Default: ``None``
#         scale_factor (optional, float): scaling factor. Default: 1.0
#
#     Returns:
#         rotated_image (np.ndarray): image after rotation
#
#     """
#     image_height, image_width = image.shape[:2]
#
#     if center is None:
#         center = (image_width // 2, image_height // 2)
#
#     # Random select specific angle
#     angle = random.choice(angles)
#     matrix = cv2.getRotationMatrix2D(center, angle, scale_factor)
#     rotated_image = cv2.warpAffine(image, matrix, (image_width, image_height))
#
#     return rotated_image


def random_horizontally_flip(image: np.ndarray, p: float = 0.5) -> np.ndarray:
    """Flip the image upside down randomly

    Args:
        image (np.ndarray): Image read with OpenCV
        p (optional, float): Horizontally flip probability. Default: 0.5

    Returns:
        horizontally_flip_image (np.ndarray): image after horizontally flip

    """
    if random.random() < p:
        horizontally_flip_image = cv2.flip(image, 1)
    else:
        horizontally_flip_image = image

    return horizontally_flip_image


def random_vertically_flip(image: np.ndarray, p: float = 0.5) -> np.ndarray:
    """Flip an image horizontally randomly

    Args:
        image (np.ndarray): Image read with OpenCV
        p (optional, float): Vertically flip probability. Default: 0.5

    Returns:
        vertically_flip_image (np.ndarray): image after vertically flip

    """
    if random.random() < p:
        vertically_flip_image = cv2.flip(image, 0)
    else:
        vertically_flip_image = image

    return vertically_flip_image


def random_scaling(image:np.ndarray, mask:np.ndarray, min_scale_factor:float=1.0, max_scale_factor:float=1.0, step_size:float=0):

    # 保证image 和 mask 一样大
    assert np.shape(image)[0:2] == np.shape(mask)[0:2]

    # get random_scale_factor
    if min_scale_factor<0 or min_scale_factor > max_scale_factor:
       raise ValueError('Unexpected value of min_scale_factor')

    if min_scale_factor == max_scale_factor:
       random_scale = float(min_scale_factor)

    elif step_size == 0:
        random_scale = np.random.uniform(low=min_scale_factor, high=max_scale_factor)

    else:
        num_steps = int((max_scale_factor - min_scale_factor) / step_size + 1)
        rand_step = np.random.randint(num_steps)
        random_scale = min_scale_factor + rand_step * step_size

    # scale
    if random_scale == 1.0:
        return image,mask

    else:
        image_shape = np.shape(image)[0:2]
        new_dim = tuple([int(x * random_scale) for x in image_shape])
        img_scaled  = cv2.resize(image, new_dim[::-1], interpolation=cv2.INTER_LINEAR)
        mask_scaled = cv2.resize(mask , new_dim[::-1], interpolation=cv2.INTER_NEAREST)

    return img_scaled, mask_scaled

def padImg(image:np.ndarray, mask:np.ndarray,pad_img_size:int = 512,range='uint8'):

    assert np.shape(image)[0:2] == np.shape(mask)[0:2]

    size = tuple([int(pad_img_size), int(pad_img_size)])
    # 填充的值
    if range == 'uint8':
        fill_index = {
            'image':[123,116,103],
            'semseg':255
        }
    else:
        fill_index = {
            'image': [0.485, 0.456, 0.406],
            'semseg': 255
        }

    unpadded_shape = np.shape(image)
    # 需要padding的像素
    delta_height = max(size[0] - unpadded_shape[0], 0)
    delta_width  = max(size[1] - unpadded_shape[1], 0)

    # Location to place image
    height_location = [delta_height // 2, (delta_height // 2) + unpadded_shape[0]]
    width_location  = [delta_width // 2, (delta_width // 2) + unpadded_shape[1]]

    max_height = max(size[0], unpadded_shape[0])
    max_width  = max(size[1], unpadded_shape[1])

    # padding image
    pad_value = fill_index['image']
    padded_img = np.ones((max_height, max_width, 3)) * pad_value
    padded_img[height_location[0]:height_location[1], width_location[0]:width_location[1], :] = image
    # padding semseg mask
    pad_value = fill_index['semseg']
    padded_mask = np.ones((max_height, max_width)) * pad_value
    padded_mask[height_location[0]:height_location[1], width_location[0]:width_location[1]] = mask

    return padded_img, padded_mask


def random_crop_2(image:np.ndarray, mask:np.ndarray,crop_img_size:int = 512):

    assert np.shape(image)[0:2] == np.shape(mask)[0:2]

    size = tuple([int(crop_img_size), int(crop_img_size)])

    # 对不满足crop_img_size大小的 h or w 先进行一个填充 满足 crop_img_size
    image, mask = padImg(image, mask, pad_img_size=crop_img_size)

    uncropped_shape = np.shape(image)
    img_height = uncropped_shape[0]
    img_width  = uncropped_shape[1]

    desired_height = size[0]
    desired_width  = size[1]

    # get crop_loc
    if img_height == desired_height and img_width == desired_width:
        crop_loc =  None
    else:
        # Get random offset uniformly from [0, max_offset)
        max_offset_height = img_height - desired_height
        max_offset_width  = img_width - desired_width

        offset_height = random.randint(0, max_offset_height)
        offset_width = random.randint(0, max_offset_width)
        crop_loc = {'height': [offset_height, offset_height + desired_height],
                    'width': [offset_width, offset_width + desired_width],
                    }

    if not crop_loc:
        return image, mask

    else:
        cropped_img = image[crop_loc['height'][0]:crop_loc['height'][1],crop_loc['width'][0]:crop_loc['width'][1], :]
        cropped_mask = mask[crop_loc['height'][0]:crop_loc['height'][1],crop_loc['width'][0]:crop_loc['width'][1]]

    return cropped_img, cropped_mask

def random_horizontally_flip2(image:np.ndarray, mask:np.ndarray,p:float=0.5):

    assert np.shape(image)[0:2] == np.shape(mask)[0:2]

    # 随机翻转
    if random.random() < p:

        new_img = np.zeros_like(image)
        new_mask = np.zeros_like(mask)

        for i in range(np.shape(image)[1]):
            new_img[:, i, :] = image[:, -(i + 1), :]
            new_mask[:, i]   = mask[:, -(i + 1)]

    # 不翻转
    else:
        new_img  = image
        new_mask = mask

    return new_img, new_mask


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
    # height_location = [delta_height // 2, (delta_height // 2) + orig_h]
    # width_location = [delta_width // 2, (delta_width // 2) + orig_w]
    height_location = [torch.div(delta_height,2,rounding_mode='floor'), torch.div(delta_height,2,rounding_mode='floor') + orig_h]
    width_location  = [torch.div(delta_width,2,rounding_mode='floor'), torch.div(delta_width,2,rounding_mode='floor') + orig_w]
    # sr_ndarry = sr_tensor.cpu().data.numpy()
    # sr_ndarry = np.squeeze(sr_ndarry) #HWC
    return sr_tensor[:,:,height_location[0]:height_location[1],width_location[0]:width_location[1]]



# 取模
def modcrop(img_in, scale):
    # img_in: Numpy, HWC or HW
    img = np.copy(img_in)
    if img.ndim == 2:
        H, W = img.shape
        H_r, W_r = H % scale, W % scale
        img = img[:H - H_r, :W - W_r]
    elif img.ndim == 3:
        H, W, C = img.shape
        H_r, W_r = H % scale, W % scale
        img = img[:H - H_r, :W - W_r, :]
    else:
        raise ValueError('Wrong img ndim: [{:d}].'.format(img.ndim))
    return img

def modcrop_up(img_in, scale):
    # img_in: Numpy, HWC or HW
    img = np.copy(img_in)
    if img.ndim == 2:
        # 原始图像的size
        H, W = img.shape
        # scale的倍数
        H_p, W_p = math.ceil(H/scale), math.ceil(W/scale)
        H_r, W_r = H_p * scale, W_p * scale
        # 零填充扩大后的图像
        img_zeros = np.zeros((H_r,W_r))
        # 图片放进去, 从左上角开始填充图像
        img_zeros[:H,:W] = img

    elif img.ndim == 3:
        H, W, C = img.shape
        H_p, W_p = math.ceil(H/scale), math.ceil(W/scale)
        H_r, W_r = H_p * scale, W_p * scale
        img_zeros = np.zeros((H_r,W_r,C))
        img_zeros[:H,:W,:] = img

    else:
        raise ValueError('Wrong img ndim: [{:d}].'.format(img.ndim))

    return {'img':img_zeros,'H':H,'W':W}

def mask2one_hot(label, num_classes=21):
   """
   label: 标签图像 # （h, w) 输入的label是np.noarry类型的
   out: 分类类别数
   """
   current_label = label
   h,w = current_label.shape[0],current_label.shape[1]
   # print(h,w)
   one_hots = []
   for i in range(num_classes):
       tmplate = torch.ones(h, w) # （h, w)
       # 如果第i层处的像素点不是i类别 该像素点上的值就设置为0
       tmplate[current_label != i] = 0
       tmplate = tmplate.view(1,h,w) # （h, w) --> （1, h, w)
       one_hots.append(tmplate)

   onehot = torch.cat(one_hots, dim=0)

   return onehot


if __name__ == '__main__':

    import os
    import cv2
    from natsort import natsorted

    upscale_factor = 12

    # 制作测试集
    # originla_set = f'D:/jingou/SuperResolution/testdataset/Datasets/SuperResolution/Common/Urban100/original'
    # set_files = [os.path.join(originla_set,file_name) for file_name in os.listdir(originla_set)]
    # file_names = natsorted(os.listdir(originla_set))
    # file_names = os.listdir(originla_set)
    save_hr_path = f'D:/jingou/SuperResolution/testdataset/Datasets/SuperResolution/Common/Urban100/GTmod{upscale_factor}'
    save_lr_path = f'D:/jingou/SuperResolution/testdataset/Datasets/SuperResolution/Common/Urban100/LRbicx{upscale_factor}'
    if not os.path.exists(save_hr_path):
        os.makedirs(save_hr_path)
    if not os.path.exists(save_lr_path):
        os.makedirs(save_lr_path)

    # # mod
    # for idx,file in enumerate(set_files):
    #     # imread img BGR
    #     img_org = cv2.imread(file, cv2.IMREAD_UNCHANGED )
    #     # Convert BGR channel image format data to RGB channel image format data
    #     img_org = cv2.cvtColor(img_org, cv2.COLOR_BGR2RGB)
    #     # modcrop
    #     img_mod = modcrop(img_org,upscale_factor)
    #     # save
    #     mod_img = cv2.cvtColor(img_mod, cv2.COLOR_RGB2BGR)
    #     cv2.imwrite(os.path.join(save_hr_path, file_names[idx]), mod_img)

    #
    #
    # get LR
    # HR imgs
    set_files = [os.path.join(save_hr_path, file_name) for file_name in os.listdir(save_hr_path)]
    # file_names = natsorted(os.listdir(save_hr_path)) # 如果文件名是string的就需要排序
    file_names = os.listdir(save_hr_path)

    for idx, file in enumerate(set_files):
        # imread img BGR
        img_org = cv2.imread(file, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.
        # Convert BGR channel image format data to RGB channel image format data
        img_org = cv2.cvtColor(img_org, cv2.COLOR_BGR2RGB)
        # bicubic
        img_lr = image_resize(img_org, 1/upscale_factor)
        # save
        img_lr = cv2.cvtColor(img_lr*255., cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(save_lr_path, file_names[idx]), img_lr)
    #
    # test_img = np.ones((375,500,3))
    # test_new = modcrop_up(test_img,6)
    # print(test_new['img'].shape)
    # print(test_new['H'],test_new['W'])