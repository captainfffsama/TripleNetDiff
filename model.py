# -*- coding: utf-8 -*-

# @Description:  模型主体
# @Author: CaptainHu
# @Date: 2021-05-12 14:37:48
# @LastEditors: CaptainHu

import torch
from torch import nn

from torchvision.models import resnext50_32x4d

class SiamTripleNet(nn.Module):
    def __init__(self):
        super(SiamTripleNet, self).__init__()
        feat_net=resnext50_32x4d(pretrained=True)
        self.feat_net=nn.Sequential(*list(feat_net.children())[:-1])
    
    def forward(self,anchor,pos,neg):
        anchor_feat=self.feat_net(anchor)
        anchor_feat=anchor_feat.squeeze(dim=-1).squeeze(dim=-1)
        pos_feat=self.feat_net(pos)
        pos_feat=pos_feat.squeeze(dim=-1).squeeze(dim=-1)
        neg_feat=self.feat_net(neg)
        neg_feat=neg_feat.squeeze(dim=-1).squeeze(dim=-1)
        return anchor_feat,pos_feat,neg_feat
        