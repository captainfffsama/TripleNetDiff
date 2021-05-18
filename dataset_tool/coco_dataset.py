# -*- coding: utf-8 -*-

# @Description:
# @Author: CaptainHu
# @Date: 2021-05-17 14:30:45
# @LastEditors: CaptainHu

import os
import random
from typing import Iterable

from pycocotools.coco import COCO
import cv2
import numpy as np


class BasicDataset(object):
    def __init__(self, sampler: str = 'random'):
        self.sampler = self._get_sampler(sampler)
        self._idx = 0
        raise AttributeError(
            "You must over write __init__,and set sampler and _idx")

    def __len__(self):
        raise AttributeError("You must overwrite __len__")

    def __next__(self):
        return self.sampler(self)

    def _get_sampler(self, sampler_name: str):
        def random_(self):
            idx = random.randint(0, len(self) - 1)
            return self[idx]

        def normal_(self):
            idx = self._idx
            self._idx = self._idx + 1 if self._idx + 1 < len(self) else 0
            return self[idx]

        if sampler_name == "random":
            return random_
        elif sampler_name == "normal":
            return normal_
        else:
            print(
                "Oh,here is no sampler named {},bro,please add it in sampler function"
                .format(sampler_name))
            print("I will use random sampler")
            return random_

    def __getitem__(self, idx):
        bg = self._get_bg(idx)
        fg, mask = self._get_fg(idx)
        return bg, fg, mask

    def _get_bg(self, idx):
        raise NotImplementedError(
            "You must overwrite _get_bg(self,idx),and return a pic")

    def _get_fg(self, idx):
        raise NotImplementedError(
            "You must overwrite _get_fg(self),and return a pic and it mask")


class COCODataset(BasicDataset):
    def __init__(self,
                 json_path: str,
                 pic_dir: str = None,
                 cats: list = None,
                 sampler='random'):
        self._coco = COCO(json_path)
        self._cats = cats
        self.sampler = self._get_sampler(sampler)
        self._imgIDs = self._coco.getImgIds()
        if pic_dir is not None:
            self._pic_dir = pic_dir
        else:
            data_name = os.path.basename(json_path).split('.')[0].split(
                '_')[-1]
            self._pic_dir = os.path.realpath(
                os.path.join(os.path.dirname(json_path), '..', data_name))
        print('COCO init done!!!')

    def _imread(self, imgID):
        imgs_info = self._coco.loadImgs(imgID)
        return cv2.imread(
            os.path.join(self._pic_dir,
                         imgs_info[0]['file_name'])), imgs_info[0]

    def set_cats(self, cats: list = None):
        self._cats = cats

    def __len__(self):
        return len(self._imgIDs)

    def _get_bg(self, idx):
        result, _ = self._imread(self._imgIDs[idx])
        return result

    def _pic_valid(self, pic):
        if pic.shape[0] < 30 or pic.shape[1] < 30:
            return False
        return True

    def _get_fg(self, idx):
        # idx_=idx
        if self._cats is not None and isinstance(self._cats, Iterable):
            catsIDs = self._coco.getCatIds(catNms=self._cats)
            imgIds = self._coco.getImgIds(catIds=catsIDs)
        else:
            imgIds = self._imgIDs
        while True:
            while True:
                # XXX:OPENCV有bug，有图片会炸，这里先去掉随机，查查图片是啥样子
                # 这里简单处理下 不符合的直接再读
                fg, img_info = self._imread(random.choice(imgIds))
                # fg,img_info=self._imread(imgIds[idx_])
                # print('idx:{},  idx_:{}'.format(idx,idx_))
                # idx_=idx_+1
                annIds = self._coco.getAnnIds(imgIds=img_info['id'],
                                              iscrowd=None)
                if annIds:
                    break
            anns = self._coco.loadAnns(annIds)
            mask = self._coco.annToMask(anns[0]) * 255
            box = self._deal_limit(mask.shape, anns[0]["bbox"])
            mask = mask[box[1]:box[3], box[0]:box[2]]
            fg = fg[box[1]:box[3], box[0]:box[2]]
            if self._pic_valid(fg) and self._pic_valid(mask):
                break
        return fg, mask

    #NOTE:注意COCO原始box是(x,y,w,h)，这里变换完之后变成tr和bl
    def _deal_limit(self, img_shape, box):
        box_ = [1] * 4
        box_[0] = max(0, round(box[0]))
        box_[1] = max(0, round(box[1]))
        box_[2] = min(img_shape[1] - 1, round(box[0] + box[2]))
        box_[3] = min(img_shape[0] - 1, round(box[1] + box[3]))
        return box_
