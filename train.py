# -*- coding: utf-8 -*-

# @Description:
# @Author: CaptainHu
# @Date: 2021-05-12 16:12:17
# @LastEditors: CaptainHu

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch import nn


from siam_dataset import SiamTripData
from transform import v1, v2
from model import SiamTripleNet

import debug_tool as D


def main(imgs_dir):
    batch_size=4
    epochs=10
    dataset = SiamTripData(imgs_dir, v1, v2)
    trainloader = DataLoader(dataset, batch_size=batch_size,
                                          shuffle=True, num_workers=1)
    model=SiamTripleNet().cuda()
    model.train=True
    optimizer=optim.Adam(model.parameters(),lr=3e-4)
    criterion=nn.TripletMarginLoss()
    for epoch in range(epochs):
        runing_loss=0.0
        for data in trainloader:
            data=[x.cuda() for x in data]
            optimizer.zero_grad()
            out=model(*data)

            loss=criterion(*out)
            loss.backward()

            optimizer.step()

            print(loss)



if __name__ == "__main__":
    img_dir = "/home/chiebotgpuhq/MyCode/dataset/nanjingbisai/bj_game_20200430/train"
    main(img_dir)
