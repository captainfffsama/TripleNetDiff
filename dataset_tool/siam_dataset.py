# -*- coding: utf-8 -*-

# @Description:
# @Author: CaptainHu
# @Date: 2021-05-12 16:13:09
# @LastEditors: CaptainHu

import os
import random
from concurrent import futures

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import cv2
import albumentations as A
import torchvision.transforms as transforms
from tqdm import tqdm

from .coco_dataset import COCODataset


def get_all_file_path(file_dir: str, filter_=('.jpg')) -> list:
    #遍历文件夹下所有的file
    return [os.path.join(maindir,filename) for maindir,_,file_name_list in os.walk(file_dir) \
        for filename in file_name_list \
        if os.path.splitext(filename)[1] in filter_ ]


class SiamTripData(Dataset):
    def __init__(self,
                 img_dir,
                 anchor_trans,
                 posi_trans,
                 p_dataset,
                 patch_size=(480, 480),
                 patch_shift=20,
                 skip_check=True):
        super(SiamTripData, self).__init__()
        self.all_img_path = get_all_file_path(img_dir)
        self.patch_shift = 20
        self.img_min_size = ((patch_size[0] + patch_shift) * 2 + 2,
                             patch_size[1] + 2 * patch_shift + 2)
        if not skip_check:
            self._filter_img(self.img_min_size)
        self.patch_size = patch_size
        self.shift_trans = A.ShiftScaleRotate(shift_limit=0,
                                              scale_limit=0,
                                              rotate_limit=(-10, 10),
                                              border_mode=cv2.BORDER_REFLECT)
        self.anchor_trans = anchor_trans
        self.posi_trans = posi_trans
        self.p_dataset: COCODataset = p_dataset
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def _is_legal(self, img_path, min_img_shape) -> bool:
        img = cv2.imread(img_path)
        if img.shape[0] < min_img_shape[0] or img.shape[1] < min_img_shape[1]:
            return False, img_path
        else:
            return True, img_path

    def _filter_img(self, min_img_shape, max_work=48):
        legal_img_list = []
        with futures.ThreadPoolExecutor(max_work) as exec:
            tasks = [
                exec.submit(self._is_legal, img_path, min_img_shape)
                for img_path in self.all_img_path
            ]
            for task in tqdm(futures.as_completed(tasks),
                             total=len(self.all_img_path)):
                flag, img_path = task.result()
                if flag:
                    legal_img_list.append(img_path)
        self.all_img_path = legal_img_list

    def __len__(self):
        return len(self.all_img_path)

    def get_random_roi(self, img, roi_size, border_limit=(0, 0, 0, 0)):
        """返回roi左上点所在img中的位置

            Args:
                img: np.ndarray
                    待截取的图片
                roi_size: tuple
                    w,h 截取的roi的大小
                border_limit: tuple
                    分别是距离左,上,右,下的边界的距离,单位像素

            Returns:
                roi_pos:tuple
                    x,y roi的左上点在图片中的座标
        """
        candicate_range = [
            border_limit[0],
            border_limit[1],
            img.shape[1] - 1 - border_limit[2] - roi_size[0],
            img.shape[0] - 1 - border_limit[3] - roi_size[1],
        ]
        return (random.randint(candicate_range[0], candicate_range[2]),
                random.randint(candicate_range[1], candicate_range[3]))

    def get_random_idx(self):
        return random.choice(self.all_img_path)

    def get_p_sample(self, img):
        img_s = self.anchor_trans(image=img)["image"]
        anchor_patch_pos = self.get_random_roi(
            img, self.patch_size, (self.patch_shift, self.patch_shift,
                                   self.patch_shift, self.patch_shift))
        posi_patch_pos = [
            x + random.randint(-self.patch_shift, self.patch_shift)
            for x in anchor_patch_pos
        ]
        neg_patch_pos = [
            x + random.randint(-self.patch_shift, self.patch_shift)
            for x in anchor_patch_pos
        ]

        anchor_patch = img_s[anchor_patch_pos[1]:anchor_patch_pos[1] +
                             self.patch_size[1],
                             anchor_patch_pos[0]:anchor_patch_pos[0] +
                             self.patch_size[0], :]
        posi_patch = img[posi_patch_pos[1]:posi_patch_pos[1] +
                         self.patch_size[1],
                         posi_patch_pos[0]:posi_patch_pos[0] +
                         self.patch_size[0], :]
        neg_patch = img_s[neg_patch_pos[1]:neg_patch_pos[1] +
                          self.patch_size[1],
                          neg_patch_pos[0]:neg_patch_pos[0] +
                          self.patch_size[0], :]

        posi_patch = self.shift_trans(image=posi_patch)["image"]
        posi_patch = self.posi_trans(image=posi_patch)["image"]

        _, p_obj, p_obj_mask = next(self.p_dataset)
        p_obj_size_rate = random.randint(5, 8) / 10.0
        p_obj_size = [int(p_obj_size_rate * x) for x in self.patch_size]

        p_obj = cv2.resize(p_obj, tuple(p_obj_size))
        p_obj_mask = cv2.resize(p_obj_mask,
                                tuple(p_obj_size),
                                interpolation=cv2.INTER_NEAREST)

        p_obj_pos = self.get_random_roi(neg_patch, p_obj_size, (0, 0, 0, 0))
        p_obj_pos_center = (p_obj_pos[0] + p_obj_size[0] // 2,
                            p_obj_pos[1] + p_obj_size[1] // 2)

        neg_patch = cv2.seamlessClone(p_obj, neg_patch, p_obj_mask,
                                      p_obj_pos_center, cv2.NORMAL_CLONE)

        return anchor_patch, posi_patch, neg_patch

    def get_no_p_sample(self, img):
        img_s = self.anchor_trans(image=img)["image"]
        anchor_patch_pos = self.get_random_roi(
            img,
            self.patch_size,
            border_limit=(self.patch_shift, self.patch_shift,
                          img.shape[1] // 2 - self.patch_size[0] // 2,
                          self.patch_shift))
        posi_patch_pos = [
            x + random.randint(-self.patch_shift, self.patch_shift)
            for x in anchor_patch_pos
        ]
        neg_patch_pos = self.get_random_roi(
            img,
            self.patch_size,
            border_limit=(img.shape[1] // 2, self.patch_shift,
                          self.patch_shift, self.patch_shift))
        anchor_patch = img_s[anchor_patch_pos[1]:anchor_patch_pos[1] +
                             self.patch_size[1],
                             anchor_patch_pos[0]:anchor_patch_pos[0] +
                             self.patch_size[0], :]
        posi_patch = img[posi_patch_pos[1]:posi_patch_pos[1] +
                         self.patch_size[1],
                         posi_patch_pos[0]:posi_patch_pos[0] +
                         self.patch_size[0], :]
        neg_patch = img_s[neg_patch_pos[1]:neg_patch_pos[1] +
                          self.patch_size[1],
                          neg_patch_pos[0]:neg_patch_pos[0] +
                          self.patch_size[0], :]

        posi_patch = self.shift_trans(image=posi_patch)["image"]
        posi_patch = self.posi_trans(image=posi_patch)["image"]
        return anchor_patch, posi_patch, neg_patch

    def __getitem__(self, idx):
        img_path = self.all_img_path[idx]
        img = cv2.imread(img_path)
        if img.shape[1] < self.img_min_size[0] or img.shape[
                0] < self.img_min_size[1]:
            img = cv2.resize(img, tuple(self.img_min_size))
        img = self.shift_trans(image=img)["image"]
        if random.randint(0, 1):
            anchor, posi, neg = self.get_no_p_sample(img)
        else:
            anchor, posi, neg = self.get_p_sample(img)

        anchor = self.transform(anchor)
        posi = self.transform(posi)
        neg = self.transform(neg)
        return anchor, posi, neg


class Collector(object):
    def __init__(self, scale_size: tuple):
        self.scale_size = scale_size

    def __call__(self, data):
        current_size = random.choice(self.scale_size)
        anchor, posi, neg = zip(*data)
        anchor_batch = torch.stack(anchor, dim=0)
        neg_batch = torch.stack(neg, dim=0)
        posi_batch = torch.stack(posi, dim=0)

        anchor_batch = F.interpolate(anchor_batch,
                                     size=current_size,
                                     mode="bilinear",
                                     align_corners=True)
        neg_batch = F.interpolate(neg_batch,
                                  size=current_size,
                                  mode="bilinear",
                                  align_corners=True)
        posi_batch = F.interpolate(posi_batch,
                                   size=current_size,
                                   mode="bilinear",
                                   align_corners=True)

        return anchor_batch, posi_batch, neg_batch
