# -*- coding: utf-8 -*-

# @Description: 
# @Author: CaptainHu
# @Date: 2021-05-12 15:26:04
# @LastEditors: CaptainHu

import torch
from torch.utils.tensorboard import SummaryWriter

import cv2
import numpy as np
import albumentations as A
import debug_tool as D

from dataset_tool.coco_dataset import COCODataset
from dataset_tool.siam_dataset import SiamTripData
from transform import v1, v2

json_path="/home/chiebotgpuhq/MyCode/dataset/coco/annotations/instances_val2017.json"
dataset=COCODataset(json_path)

dataset_1=SiamTripData("/home/chiebotgpuhq/MyCode/dataset/patrol",v1,v2,dataset)
for data in dataset_1:
    D.show_img(data)