# -*- coding: utf-8 -*-

# @Description:
# @Author: CaptainHu
# @Date: 2021-05-12 16:12:17
# @LastEditors: CaptainHu
import argparse
import os
from pprint import pprint

import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from torchvision.utils import make_grid

from dataset_tool.siam_dataset import SiamTripData, Collector
from dataset_tool.coco_dataset import COCODataset
from transform import v1, v2
from model import SiamTripleNet
import base_cfg as cfg

import debug_tool as D

def parse_args():
    """Parse arguments for training script"""
    parser = argparse.ArgumentParser(description='training script')
    parser.add_argument('--local_rank',default=0,type=int, help='node rank for distributed training')
    parser.add_argument('--launcher',choices=['nccl','none'],default='none',help='job launcher')
    parser.add_argument('-c','--cfg_path',type=str,default='')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] =  str(args.local_rank)

    return args

def init_train(args):
    cfg.merge_param(args.cfg_path)
    args_dict=cfg.param
    print("=====train args=====")
    pprint(args_dict)
    if 'nccl'==args.launcher: 
        torch.cuda.set_device(args.local_rank) 
        dist.init_process_group(backend=args.launcher)
    torch.backends.cudnn.benchmark = True
    return args_dict    

def main(args):
    args_dict=init_train(args)
    batch_size = args_dict.get('batch_size',1)
    epochs = args_dict.get('epochs',10)
    dataset_t = COCODataset(args_dict["coco_json"])
    dataset = SiamTripData(args_dict["train_data"], v1, v2, dataset_t, skip_check=True)
    trainloader = DataLoader(dataset,
                             batch_size=batch_size,
                             shuffle=True,
                             num_workers=1,
                             collate_fn=Collector([(256, 256), (480, 480)]))
    model = SiamTripleNet()
    if 'nccl'==args.launcher:
        model=model.to(args.local_rank)
        model=torch.nn.parallel.DistributedDataParallel(model,device_ids=[args.local_rank,],output_device=args.local_rank)
    else:
        model=model.cuda()
    model.train = True
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    criterion = nn.TripletMarginLoss()
    with SummaryWriter() as summary:
        for epoch in range(epochs):
            runing_loss = 0.0
            for i, data in enumerate(trainloader):

                data = [x.cuda() for x in data]
                optimizer.zero_grad()
                out = model(*data)

                loss = criterion(*out)
                loss.backward()

                optimizer.step()
                runing_loss += loss.item()
                if i % 500:
                    if 0==args.local_rank:
                        runing_loss = runing_loss / 500
                        summary.add_scalar("train_Loss:", runing_loss,
                                        epoch * len(trainloader) + i)
                        summary.add_image("anchor_pic",
                                        make_grid(data[0].detach().cpu()),
                                        epoch * len(trainloader) + i)
                        summary.add_image("posi_pic",
                                        make_grid(data[1].detach().cpu()),
                                        epoch * len(trainloader) + i)
                        summary.add_image("neg_pic",
                                        make_grid(data[2].detach().cpu()),
                                        epoch * len(trainloader) + i)
                        print("{} loss is:{}".format(
                            str(epoch * len(trainloader) + i), runing_loss))
                        runing_loss = 0.0
            if 0==args.local_rank:
                torch.save(model.state_dict(),args.ckpt_save)

if __name__ == "__main__":
    args=parse_args()
    main(args)
