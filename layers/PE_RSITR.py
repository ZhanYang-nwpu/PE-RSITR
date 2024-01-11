#!/usr/bin/env python
# -*-coding: utf -8-*-
"""
@ Author: ZhanYang
@ Email: zhanyangnwpu@gmail.com
@ Github: https://github.com/ZhanYang-nwpu/PE-RSITR
@ Paper: https://ieeexplore.ieee.org/document/10231134
"""

from torch import nn
from .MRS_Adapter import PE_RSITR

class Model(nn.Module):
    def __init__(self, opt):
        super(Model, self).__init__()
        self.Eiters = 0

        self.PE_RSITR = PE_RSITR()

    def forward(self, image, text):
        image_features = self.PE_RSITR.forward_image(image)
        text_features = self.PE_RSITR.forward_text(text)
        image_features_augme = self.PE_RSITR.forward_image(image, True)
        text_features_augme = self.PE_RSITR.forward_text(text, True)

        g_image = image_features.float()
        g_text = text_features.float()
        g_image_augme = image_features_augme.float()
        g_text_augme = text_features_augme.float()

        return g_image, g_text, g_image_augme, g_text_augme


def factory(opt, cuda=True, data_parallel=True):
    model = Model(opt)

    if data_parallel:
        model = nn.DataParallel(model).cuda()
        if not cuda:
            raise ValueError
    if cuda:
        model.cuda()
    return model

