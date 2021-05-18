# -*- coding: utf-8 -*-

# @Description:
# @Author: CaptainHu
# @Date: 2021-05-12 16:12:17
# @LastEditors: CaptainHu

import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from torchvision.utils import make_grid

from dataset_tool.siam_dataset import SiamTripData,Collector
from dataset_tool.coco_dataset import COCODataset
from transform import v1, v2
from model import SiamTripleNet

import debug_tool as D


def main(imgs_dir):
    batch_size = 2
    epochs = 10
    torch.backends.cudnn.benchmark = True
    json_path="/home/chiebotgpuhq/MyCode/dataset/coco/annotations/instances_val2017.json"
    dataset_t=COCODataset(json_path)
    dataset = SiamTripData(imgs_dir, v1, v2,dataset_t,skip_check=True)
    trainloader = DataLoader(dataset,
                             batch_size=batch_size,
                             shuffle=True,
                             num_workers=1,collate_fn=Collector([(256,256),(480,480)]))
    model = SiamTripleNet().cuda()
    model.train = True
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    criterion = nn.TripletMarginLoss()
    with SummaryWriter() as summary:
        for epoch in range(epochs):
            runing_loss = 0.0
            for i,data in enumerate(trainloader):

                data = [x.cuda() for x in data]
                optimizer.zero_grad()
                out = model(*data)

                loss = criterion(*out)
                loss.backward()

                optimizer.step()
                runing_loss += loss.item()
                if i%500:
                    runing_loss =runing_loss/500
                    summary.add_scalar("train_Loss:",runing_loss,epoch*len(trainloader)+i)
                    summary.add_image("anchor_pic",make_grid(data[0].detach().cpu()),epoch*len(trainloader)+i)
                    summary.add_image("posi_pic",make_grid(data[1].detach().cpu()),epoch*len(trainloader)+i)
                    summary.add_image("neg_pic",make_grid(data[2].detach().cpu()),epoch*len(trainloader)+i)
                    print("{} loss is:{}".format(str(epoch*len(trainloader)+i),runing_loss))
                    runing_loss=0.0

if __name__ == "__main__":
    img_dir = "/home/chiebotgpuhq/MyCode/dataset/nanjingbisai/bj_game_20200430/train"
    main(img_dir)
