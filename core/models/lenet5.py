"""
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2021-11-17 20:51:27
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2021-11-17 21:40:41
"""
from collections import OrderedDict

import torch.nn.functional as F
from core.models.layers.super_conv2d import SuperBlockConv2d
from core.models.layers.super_linear import SuperBlockLinear
from torch import nn

from .cnn import ConvBlock, LinearBlock, SuperOCNN

__all__ = ["CNNLeNet5"]


class CNNLeNet5(SuperOCNN):
    _conv_linear = (SuperBlockConv2d, SuperBlockLinear)
    _conv = (SuperBlockConv2d,)
    _linear = (SuperBlockLinear,)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def build_layers(self) -> None:
        self.features = OrderedDict()
        max_pool_kernel_size = 2
        for idx, out_channels in enumerate(self.kernel_list, 0):
            layer_name = "conv" + str(idx + 1)
            in_channels = self.in_channels if (idx == 0) else self.kernel_list[idx - 1]
            self.features[layer_name] = ConvBlock(
                in_channels,
                out_channels,
                self.kernel_size_list[idx],
                stride=self.stride_list[idx],
                padding=self.padding_list[idx],
                mini_block=self.block_list[idx],
                bias=self.bias,
                in_bit=self.in_bit,
                w_bit=self.w_bit,
                v_max=self.v_max,
                photodetect=self.photodetect,
                device=self.device,
                activation=True,
                act_thres=self.act_thres,
                bn_affine=self.bn_affine,
                verbose=self.verbose,
            )
            layer_name = "pool" + str(idx + 1)
            self.features[layer_name] = nn.MaxPool2d(max_pool_kernel_size)
        self.features = nn.Sequential(self.features)

        if self.pool_out_size > 0:
            self.pool2d = nn.AdaptiveAvgPool2d(self.pool_out_size)
            feature_size = self.kernel_list[-1] * self.pool_out_size * self.pool_out_size
        else:
            self.pool2d = None
            img_height, img_width = self.img_height, self.img_width
            for layer in self.modules():
                if isinstance(layer, self._conv):
                    img_height, img_width = layer.get_output_dim(img_height, img_width)
                    img_height //= max_pool_kernel_size
                    img_width //= max_pool_kernel_size  # maxpool
            feature_size = img_height * img_width * self.kernel_list[-1]

        self.classifier = OrderedDict()
        for idx, hidden_dim in enumerate(self.hidden_list, 0):
            layer_name = "fc" + str(idx + 1)
            in_features = feature_size if idx == 0 else self.hidden_list[idx - 1]
            out_features = hidden_dim
            self.classifier[layer_name] = LinearBlock(
                in_features,
                out_features,
                mini_block=self.block_list[idx],
                bias=self.bias,
                w_bit=self.w_bit,
                in_bit=self.in_bit,
                v_max=self.v_max,
                photodetect=self.photodetect,
                device=self.device,
                activation=True,
                act_thres=self.act_thres,
                verbose=self.verbose,
            )

        layer_name = "fc" + str(len(self.hidden_list) + 1)
        self.classifier[layer_name] = LinearBlock(
            self.hidden_list[-1] if len(self.hidden_list) > 0 else feature_size,
            self.num_classes,
            mini_block=self.block_list[-1],
            bias=self.bias,
            w_bit=self.w_bit,
            in_bit=self.in_bit,
            v_max=self.v_max,
            photodetect=self.photodetect,
            device=self.device,
            activation=False,
            act_thres=self.act_thres,
            verbose=self.verbose,
        )
        self.classifier = nn.Sequential(self.classifier)
