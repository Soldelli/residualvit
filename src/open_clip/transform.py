import warnings
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torchvision.transforms.functional as F

from torchvision.transforms import Normalize, Compose, RandomResizedCrop, InterpolationMode, ToTensor, Resize, \
    CenterCrop

import math
import numpy as np

import random

from .constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD


@dataclass
class AugmentationCfg:
    scale: Tuple[float, float] = (0.9, 1.0)
    ratio: Optional[Tuple[float, float]] = None
    color_jitter: Optional[Union[float, Tuple[float, float, float]]] = None
    interpolation: Optional[str] = None
    re_prob: Optional[float] = None
    re_count: Optional[int] = None
    use_timm: bool = False


class ResizeMaxSize(nn.Module):

    def __init__(self, max_size, interpolation=InterpolationMode.BICUBIC, fn='max', fill=0):
        super().__init__()
        if not isinstance(max_size, int):
            raise TypeError(f"Size should be int. Got {type(max_size)}")
        self.max_size = max_size
        self.interpolation = interpolation
        self.fn = min if fn == 'min' else min
        self.fill = fill

    def forward(self, img):
        if isinstance(img, torch.Tensor):
            height, width = img.shape[:2]
        else:
            width, height = img.size
        scale = self.max_size / float(max(height, width))
        if scale != 1.0:
            new_size = tuple(round(dim * scale) for dim in (height, width))
            img = F.resize(img, new_size, self.interpolation)
            pad_h = self.max_size - new_size[0]
            pad_w = self.max_size - new_size[1]
            img = F.pad(img, padding=[pad_w//2, pad_h//2, pad_w - pad_w//2, pad_h - pad_h//2], fill=self.fill)
        return img


def _convert_to_rgb(image):
    return image.convert('RGB')


def image_transform(
        image_size: int,
        is_train: bool,
        mean: Optional[Tuple[float, ...]] = None,
        std: Optional[Tuple[float, ...]] = None,
        resize_longest_max: bool = False,
        fill_color: int = 0,
        aug_cfg: Optional[Union[Dict[str, Any], AugmentationCfg]] = None,
):
    mean = mean or OPENAI_DATASET_MEAN
    if not isinstance(mean, (list, tuple)):
        mean = (mean,) * 3

    std = std or OPENAI_DATASET_STD
    if not isinstance(std, (list, tuple)):
        std = (std,) * 3

    if isinstance(image_size, (list, tuple)) and image_size[0] == image_size[1]:
        # for square size, pass size as int so that Resize() uses aspect preserving shortest edge
        image_size = image_size[0]

    if isinstance(aug_cfg, dict):
        aug_cfg = AugmentationCfg(**aug_cfg)
    else:
        aug_cfg = aug_cfg or AugmentationCfg()
    normalize = Normalize(mean=mean, std=std)
    if is_train:
        aug_cfg_dict = {k: v for k, v in asdict(aug_cfg).items() if v is not None}
        use_timm = aug_cfg_dict.pop('use_timm', False)
        if use_timm:
            from timm.data import create_transform  # timm can still be optional
            if isinstance(image_size, (tuple, list)):
                assert len(image_size) >= 2
                input_size = (3,) + image_size[-2:]
            else:
                input_size = (3, image_size, image_size)
            # by default, timm aug randomly alternates bicubic & bilinear for better robustness at inference time
            aug_cfg_dict.setdefault('interpolation', 'random')
            aug_cfg_dict.setdefault('color_jitter', None)  # disable by default
            train_transform = create_transform(
                input_size=input_size,
                is_training=True,
                hflip=0.,
                mean=mean,
                std=std,
                re_mode='pixel',
                **aug_cfg_dict,
            )
        else:
            train_transform = Compose([
                RandomResizedCrop(
                    image_size,
                    scale=aug_cfg_dict.pop('scale'),
                    interpolation=InterpolationMode.BICUBIC,
                ),
                _convert_to_rgb,
                ToTensor(),
                normalize,
            ])
            if aug_cfg_dict:
                warnings.warn(f'Unused augmentation cfg items, specify `use_timm` to use ({list(aug_cfg_dict.keys())}).')
        return train_transform
    else:
        if resize_longest_max:
            transforms = [
                ResizeMaxSize(image_size, fill=fill_color)
            ]
        else:
            transforms = [
                Resize(image_size, interpolation=InterpolationMode.BICUBIC),
                CenterCrop(image_size),
            ]
        transforms.extend([
            _convert_to_rgb,
            ToTensor(),
            normalize,
        ])
        return Compose(transforms)


class VideoToTensor(torch.nn.Module):
    def forward(self, frames):
        frames = [np.array(x) for x in frames] 
        frames = np.array(frames, dtype=np.float32)
        frames = torch.tensor(frames)
        return frames.permute(0, 3, 1, 2) / 255.0


class VideoResize(torch.nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size

    def forward(self, frames):
        B, C, H, W = frames.shape
        new_frames = []
        for i in range(B):
            frame = frames[i]
            if H < W:
                new_h, new_w = self.size, int(self.size * W / H)
            else:
                new_h, new_w = int(self.size * H / W), self.size
            resized_frame = torch.nn.functional.interpolate(frame.unsqueeze(0), size=(new_h, new_w), mode='bilinear', align_corners=False)
            new_frames.append(resized_frame.squeeze(0))
        return torch.stack(new_frames)


class VideoRandomHorizontalFlip(torch.nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Args:
            frames (Tensor): Tensor frames to be flipped

        Returns:
            Tensor: Flipped Tensor image.
        """
        if torch.rand(1).item() < self.p:
            flipped_frames = torch.flip(frames, dims=[3])  # Flip along width dimension (W)
            return flipped_frames
        return frames

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p})"
    

class VideoRandomResizeCrop(torch.nn.Module):
    def __init__(self, size=(224, 224), scale=(0.8, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0)):
        """
        Args:
            size: Desired height and width after cropping.
            scale: Scale range of Inception-style area based random resizing.
            ratio: Aspect ratio range of Inception-style area based random resizing.
        """
        super().__init__()
        self.size = size
        self.scale = scale
        self.ratio = ratio

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Args:
            frames (Tensor): Tensor frames to be resized and cropped.

        Returns:
            Tensor: Resized and cropped Tensor image.
        """
        B, C, H, W = frames.shape
        i, j, h, w = self._get_param_spatial_crop(H, W)
        cropped_frames = frames[:, :, i:i + h, j:j + w]

        resized_frames = torch.nn.functional.interpolate(
            cropped_frames,
            size=self.size,
            mode="bilinear",
            align_corners=False,
        )

        return resized_frames

    def _get_param_spatial_crop(self, height, width, num_repeat=10, log_scale=True, switch_hw=False):
        """
        Given scale, ratio, height, and width, return sampled coordinates of the videos.
        """
        for _ in range(num_repeat):
            area = height * width
            target_area = random.uniform(*self.scale) * area

            if log_scale:
                log_ratio = (math.log(self.ratio[0]), math.log(self.ratio[1]))
                aspect_ratio = math.exp(random.uniform(*log_ratio))
            else:
                aspect_ratio = random.uniform(*self.ratio)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if np.random.uniform() < 0.5 and switch_hw:
                w, h = h, w

            if 0 < w <= width and 0 < h <= height:
                i = random.randint(0, height - h)
                j = random.randint(0, width - w)
                return i, j, h, w

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if in_ratio < min(self.ratio):
            w = width
            h = int(round(w / min(self.ratio)))
        elif in_ratio > max(self.ratio):
            h = height
            w = int(round(h * max(self.ratio)))
        else:  # whole image
            w = width
            h = height

        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size}, scale={self.scale}, ratio={self.ratio})"
    
    
class VideoCrop(torch.nn.Module):
    def __init__(self, size):
        super(VideoCrop, self).__init__()
        self.size = size

    def forward(self, frames):
        B, C, H, W = frames.shape
        target_h, target_w = self.size
        top = (H - target_h) // 2
        left = (W - target_w) // 2
        return frames[:, :, top:top + target_h, left:left + target_w]

    def __repr__(self):
        return f"{self.__class__.__name__}(size={self.size})"

# IMG_TRANSFORM = Compose([
#                     RandomResizedCrop(224, scale=(0.9, 1.0), interpolation=InterpolationMode.BICUBIC),
#                     ToTensor(),
#                     Normalize(mean=(0.48145466, 0.4578275, 0.40821073), 
#                               std=(0.26862954, 0.26130258, 0.27577711))
#                 ])

# IMG_TRANSFORM2 = Compose([
#                     Resize(256, interpolation=InterpolationMode.BICUBIC), 
#                     CenterCrop((224, 224)),  
#                     ToTensor(),
#                     Normalize(mean=(0.48145466, 0.4578275, 0.40821073), 
#                               std=(0.26862954, 0.26130258, 0.27577711))
#                 ])

Video_Transform_train = Compose([
                    VideoToTensor(),
                    VideoResize(size=256),
                    VideoRandomResizeCrop(size=(224,224)),
                    VideoRandomHorizontalFlip(p=0.5),
                    Normalize(mean=(0.48145466, 0.4578275, 0.40821073), 
                              std=(0.26862954, 0.26130258, 0.27577711))
                    ])

Video_Transform_val = Compose([
                    VideoToTensor(),
                    VideoResize(size=224),
                    VideoCrop(size=(224,224)),
                    Normalize(mean=(0.48145466, 0.4578275, 0.40821073), 
                              std=(0.26862954, 0.26130258, 0.27577711))
                    ])

def _modify_transforms_image_resolution(image_resolution):
    global Video_Transform_train, Video_Transform_val
    assert image_resolution <= 256, "image_resolution should be less than or equal to 256"

    # Update the train transform
    Video_Transform_train.transforms[2] = VideoRandomResizeCrop(size=(image_resolution, image_resolution))
    Video_Transform_train.transforms[1] = VideoResize(size=256)  # Assuming you always want to start with resizing to 256

    # Update the validation transform
    Video_Transform_val.transforms[1] = VideoResize(size=image_resolution)
    Video_Transform_val.transforms[2] = VideoCrop(size=(image_resolution, image_resolution))


def preprocess_video_train(video, transform=Video_Transform_train):
    return transform(video)

def preprocess_video_val(video, transform=Video_Transform_val):
    return transform(video)