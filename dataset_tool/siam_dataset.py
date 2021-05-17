# -*- coding: utf-8 -*-

# @Description: 
# @Author: CaptainHu
# @Date: 2021-05-12 16:13:09
# @LastEditors: CaptainHu

import os
import random

from torch.utils.data import Dataset
import cv2
import albumentations as A
import torchvision.transforms as transforms

from .coco_dataset import COCODataset

def get_all_file_path(file_dir:str,filter_=('.jpg')) -> list:
    #遍历文件夹下所有的file
    return [os.path.join(maindir,filename) for maindir,_,file_name_list in os.walk(file_dir) \
        for filename in file_name_list \
        if os.path.splitext(filename)[1] in filter_ ]

class SiamTripData(Dataset):
    def __init__(self,img_dir,anchor_trans,posi_trans,max_roi:int=256,):
        super(SiamTripData, self).__init__()
        self.all_img_path=get_all_file_path(img_dir)
        self.max_roi=max_roi
        self.shift_trans=A.ShiftScaleRotate(shift_limit=0,scale_limit=0,rotate_limit=(-10,10),border_mode=cv2.BORDER_REFLECT)
        self.anchor_trans=anchor_trans
        self.posi_trans=posi_trans
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.all_img_path)

    def get_random_idx(self):
        return random.choice(self.all_img_path)

    def __getitem__(self, idx):
        img_path=self.all_img_path[idx]
        img=cv2.imread(img_path)
        while img.shape[0]<552 or img.shape[1]<552:
            img_path=self.get_random_idx()
            img=cv2.imread(img_path)
        img=self.shift_trans(image=img)["image"]
        img_s=self.shift_trans(image=img)["image"]

        img=self.anchor_trans(image=img)["image"]

        roi_size=256
        try:
            w1_pos=random.randint(min(20,roi_size//3),img.shape[1]//2-roi_size-min(20,roi_size//3)-1)
            h1_pos=random.randint(min(20,roi_size//3),img.shape[0]//2-roi_size-min(20,roi_size//3)-1)
            w2_pos=random.randint(img.shape[1]//2+min(20,roi_size//3),img.shape[1]-roi_size-min(20,roi_size//3)-1)
            h2_pos=random.randint(img.shape[0]//2+min(20,roi_size//3),img.shape[0]-roi_size-min(20,roi_size//3)-1)
        except Exception as e:
            print(img_path)
            raise e

        w_shift=random.randint(-min(20,roi_size//3),min(20,roi_size//3))
        h_shift=random.randint(-min(20,roi_size//3),min(20,roi_size//3))

        if random.randint(0,1):
            anchor=img[h1_pos:h1_pos+roi_size,w1_pos:w1_pos+roi_size,:]
            neg=img[h2_pos:h2_pos+roi_size,w2_pos:w2_pos+roi_size,:]
            posi=img_s[h1_pos+h_shift:h1_pos+h_shift+roi_size,w1_pos+w_shift:w1_pos+w_shift+roi_size,:]
        else:
            neg=img[h1_pos:h1_pos+roi_size,w1_pos:w1_pos+roi_size,:]
            anchor=img[h2_pos:h2_pos+roi_size,w2_pos:w2_pos+roi_size,:]
            posi=img_s[h2_pos+h_shift:h2_pos+h_shift+roi_size,w2_pos+w_shift:w2_pos+w_shift+roi_size,:]
        posi=self.posi_trans(image=posi)["image"]
        
        anchor=self.transform(anchor)
        posi=self.transform(posi)
        neg=self.transform(neg)

        return anchor,posi,neg

