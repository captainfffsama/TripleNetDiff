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

t=cv2.imread("/home/chiebotgpuhq/MyCode/dataset/cruisepic/select1/2/t.jpg")[:,:,::-1]
m=cv2.imread("/home/chiebotgpuhq/MyCode/dataset/cruisepic/select1/2/2_2#2021-05-09-16-33-30.jpg")[:,:,::-1]

aug = A.Compose([A.FDA([t], beta_limit=0.01,p=1, read_fn=lambda x: x)])
result=aug(image=m)["image"]
D.show_img(result)
