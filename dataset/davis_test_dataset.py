"""
Modified from https://github.com/seoungwugoh/STM/blob/master/dataset.py
"""

import os
from os import path
import numpy as np
from PIL import Image

import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torch.utils.data.dataset import Dataset
from dataset.range_transform import im_normalization
from dataset.util import all_to_onehot
from torchvision.transforms import functional as F
import matplotlib.pyplot as plt


class DAVISTestDataset(Dataset):
    def __init__(
        self,
        root,
        imset="2017/val.txt",
        resolution=480,
        single_object=False,
        target_name=None,
        tta=False,
    ):
        self.root = root
        self.tta = tta
        if resolution == 480:
            res_tag = "480p"
        else:
            res_tag = "Full-Resolution"
        self.mask_dir = path.join(root, "Annotations", res_tag)
        self.mask480_dir = path.join(root, "Annotations", "480p")
        self.image_dir = path.join(root, "JPEGImages", res_tag)
        self.resolution = resolution
        _imset_dir = path.join(root, "ImageSets")
        _imset_f = path.join(_imset_dir, imset)

        self.videos = []
        self.num_frames = {}
        self.num_objects = {}
        self.shape = {}
        self.size_480p = {}
        with open(path.join(_imset_f), "r") as lines:
            for line in lines:
                _video = line.rstrip("\n")
                if target_name is not None and target_name != _video:
                    continue
                self.videos.append(_video)
                self.num_frames[_video] = len(
                    os.listdir(path.join(self.image_dir, _video))
                )
                _mask = np.array(
                    Image.open(path.join(self.mask_dir, _video, "00000.png")).convert(
                        "P"
                    )
                )
                self.num_objects[_video] = np.max(_mask)
                self.shape[_video] = np.shape(_mask)
                _mask480 = np.array(
                    Image.open(
                        path.join(self.mask480_dir, _video, "00000.png")
                    ).convert("P")
                )
                self.size_480p[_video] = np.shape(_mask480)

        self.single_object = single_object

        if resolution == 480:
            self.im_transform = transforms.Compose(
                [transforms.ToTensor(), im_normalization,]
            )
        else:
            self.im_transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    im_normalization,
                    transforms.Resize(
                        resolution, interpolation=InterpolationMode.BICUBIC
                    ),
                ]
            )
            self.mask_transform = transforms.Compose(
                [
                    transforms.Resize(
                        resolution, interpolation=InterpolationMode.NEAREST
                    ),
                ]
            )

    def __len__(self):
        return len(self.videos)

    def apply_color_transform(self, img, b, c, s, h):
        img = F.adjust_brightness(img, b)
        img = F.adjust_contrast(img, c)
        img = F.adjust_saturation(img, s)
        img = F.adjust_hue(img, h)
        return img

    def apply_tta(self, img, ismask, transform=None):
        output = []
        augmented_masks = F.five_crop(img, (img.size[1] / 2, img.size[0] / 2))
        # augmented_masks += F.five_crop(img, (img.size[1] / 1.2, img.size[0] / 1.2))
        for augmented_mask in augmented_masks:
            if ismask:
                augmented_mask = F.resize(
                    augmented_mask,
                    (img.size[1], img.size[0]),
                    interpolation=InterpolationMode.NEAREST,
                )
                # plt.imshow(augmented_mask), plt.axis("off"), plt.show()
                output.append(np.array(augmented_mask, dtype=np.uint8))
            else:
                augmented_mask = F.resize(
                    augmented_mask,
                    (img.size[1], img.size[0]),
                    interpolation=InterpolationMode.BICUBIC,
                    # interpolation=InterpolationMode.NEAREST,
                )
                # plt.imshow(augmented_mask), plt.axis("off"), plt.show()
                output.append(transform(augmented_mask))
        return output

    def apply_tta2(self, img, ismask, transform=None):
        aug_imgs = []
        # angles = [0]
        # translations = [[0, 0]]
        # scales = [1]
        # shears = [0]
        angles = [0, -1, 1]
        translations = [[0, 0]]
        scales = [1, 1.1]
        shears = [0, -1, 1]
        # brightnesses = [0.55, 0.4, 0.65]
        # contrasts = [1, 0.95, 1.15]
        # saturations = [1, 0.95, 1.15]
        # hues = [0, -0.07, 0.07]
        brightnesses = [1]
        contrasts = [1]
        saturations = [1]
        hues = [0]
        for angle in angles:
            for translation in translations:
                for scale in scales:
                    for shear in shears:
                        for b in brightnesses:
                            for c in contrasts:
                                for s in saturations:
                                    for h in hues:
                                        # if (np.count_nonzero([angle, translation[0]+translation[1],scale-1, shear]) > 1):
                                        #     continue
                                        if transform is None:
                                            if ismask:
                                                aug_imgs.append(
                                                    np.array(
                                                        F.affine(
                                                            img,
                                                            angle=angle,
                                                            translate=translation,
                                                            scale=scale,
                                                            shear=shear,
                                                            interpolation=InterpolationMode.NEAREST,
                                                        ),
                                                        dtype=np.uint8,
                                                    )
                                                )
                                            else:
                                                aug_imgs.append(
                                                    self.apply_color_transform(
                                                        F.affine(
                                                            img,
                                                            angle=angle,
                                                            translate=translation,
                                                            scale=scale,
                                                            shear=shear,
                                                            interpolation=InterpolationMode.BICUBIC,
                                                        ),
                                                        b,
                                                        c,
                                                        s,
                                                        h,
                                                    )
                                                )

                                        else:
                                            if ismask:
                                                augmented_mask = F.affine(
                                                    img,
                                                    angle=angle,
                                                    translate=translation,
                                                    scale=scale,
                                                    shear=shear,
                                                    interpolation=InterpolationMode.NEAREST,
                                                )
                                                aug_imgs.append(
                                                    np.array(
                                                        transform(augmented_mask),
                                                        dtype=np.uint8,
                                                    )
                                                )
                                            else:
                                                augmented = self.apply_color_transform(
                                                    F.affine(
                                                        img,
                                                        angle=angle,
                                                        translate=translation,
                                                        scale=scale,
                                                        shear=shear,
                                                        interpolation=InterpolationMode.NEAREST,
                                                    ),
                                                    b,
                                                    c,
                                                    s,
                                                    h,
                                                )
                                                plt.imshow(augmented), plt.axis(
                                                    "off"
                                                ), plt.show()
                                                aug_imgs.append(transform(augmented))

        return aug_imgs

    def __getitem__(self, index):
        video = self.videos[index]
        info = {}
        info["name"] = video
        info["frames"] = []
        info["num_frames"] = self.num_frames[video]
        info["size_480p"] = self.size_480p[video]
        images = []
        augimages = []
        masks = []
        augmasks = []

        for f in range(self.num_frames[video]):
            img_file = path.join(self.image_dir, video, "{:05d}.jpg".format(f))
            if f == 0 and self.tta:
                augimages = self.apply_tta(
                    Image.open(img_file).convert("RGB"),
                    ismask=False,
                    transform=self.im_transform,
                )
            images.append(self.im_transform(Image.open(img_file).convert("RGB")))
            info["frames"].append("{:05d}.jpg".format(f))

            mask_file = path.join(self.mask_dir, video, "{:05d}.png".format(f))
            if path.exists(mask_file):
                if f == 0 and self.tta:
                    augmasks = self.apply_tta(
                        Image.open(mask_file).convert("P"), ismask=True
                    )
                masks.append(
                    np.array(Image.open(mask_file).convert("P"), dtype=np.uint8)
                )

            else:
                masks.append(np.zeros_like(masks[0]))

        images = torch.stack(images, 0)
        masks = np.stack(masks, 0)
        if self.tta:
            augmasks = np.stack(augmasks, 0)
            augimages = torch.stack(augimages, 0)

        if self.single_object:
            labels = [1]
            masks = (masks > 0.5).astype(np.uint8)
            masks = torch.from_numpy(all_to_onehot(masks, labels)).float()
        else:
            labels = np.unique(masks[0])
            labels = labels[labels != 0]
            masks = torch.from_numpy(all_to_onehot(masks, labels)).float()
            if self.tta:
                augmasks = torch.from_numpy(all_to_onehot(augmasks, labels)).float()

        if self.resolution != 480:
            masks = self.mask_transform(masks)
            if self.tta:
                augmasks = self.mask_transform(augmasks)
        masks = masks.unsqueeze(2)
        if self.tta:
            augmasks = augmasks.unsqueeze(2)

        info["labels"] = labels
        if self.tta:
            data = {
                "rgb": images,
                "rgb_aug": augimages,
                "gt": masks,
                "gt_aug": augmasks,
                "info": info,
            }
        else:
            data = {
                "rgb": images,
                "gt": masks,
                "info": info,
            }

        return data
