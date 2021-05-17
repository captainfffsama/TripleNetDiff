# -*- coding: utf-8 -*-

# @Description: 放置一些图像变换
# @Author: CaptainHu
# @Date: 2021-05-12 17:21:53
# @LastEditors: CaptainHu


import albumentations as A

v1=A.Compose(
    [
        A.RandomBrightnessContrast(),
        A.RandomGamma(),
        A.RandomSunFlare(),
    ]
)

v2=A.Compose([
    A.RandomBrightnessContrast(p=1),
    A.RandomGamma(),
    A.RandomFog(),
    A.RandomShadow(),
])
