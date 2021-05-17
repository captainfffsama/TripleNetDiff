# -*- coding: utf-8 -*-

# @Description: 
# @Author: CaptainHu
# @Date: 2021-05-17 10:39:41
# @LastEditors: CaptainHu
import math
import matplotlib.pyplot as plt
import numpy as np
import torch

def show_img(img):
    if isinstance(img,list) or isinstance(img,tuple):
        img_num=len(img)
        row_n=math.ceil(math.sqrt(img_num))
        col_n=max(math.ceil(img_num/row_n),1)
        fig, axs = plt.subplots(row_n, col_n, figsize=(15*row_n, 15*col_n))
        for idx,img_ in enumerate(img):
            if isinstance(img_,torch.Tensor):
                img_:np.ndarray=img_.detach().cpu().numpy()
            img_=img_.squeeze()
            if 2==len(axs.shape):
                axs[idx%row_n][idx//row_n].imshow(img_)
                axs[idx%row_n][idx//row_n].set_title(str(idx))
            else:
                axs[idx%row_n].imshow(img_)
                axs[idx%row_n].set_title(str(idx))
    else:
        if isinstance(img,torch.Tensor):
            img:np.ndarray=img.detach().cpu().numpy().squeeze()
        if img.shape[0] == 1 or img.shape[0] == 3:
            img=np.transpose(img,(1,2,0))
        plt.imshow(img)
    plt.show()
