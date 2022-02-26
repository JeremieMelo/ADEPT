"""
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2021-11-18 22:48:11
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2021-11-18 22:53:53
"""

import torch
from core.models.layers.super_conv2d import SuperBlockConv2d
from core.models.layers.super_linear import SuperBlockLinear
from torch import Tensor, nn

from .cnn import ConvBlock, LinearBlock, SuperOCNN

__all__ = [
    "VGG8",
    "VGG11",
    "VGG13",
    "VGG16",
    "VGG19",
]

cfg_32 = {
    "vgg8": [64, "M", 128, "M", 256, "M", 512, "M", 512, "M"],
    "vgg11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "vgg13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "vgg16": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "vgg19": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        512,
        "M",
    ],
}

cfg_64 = {
    "vgg8": [64, "M", 128, "M", 256, "M", 512, "M", 512, "GAP"],
    "vgg11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "GAP"],
    "vgg13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "GAP"],
    "vgg16": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "GAP"],
    "vgg19": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        512,
        "GAP",
    ],
}


class VGG(SuperOCNN):
    """MZI VGG (Shen+, Nature Photonics 2017). Support sparse backpropagation. Blocking matrix multiplication."""

    _conv_linear = (SuperBlockConv2d, SuperBlockLinear)
    _conv = (SuperBlockConv2d,)
    _linear = (SuperBlockLinear,)

    def __init__(
        self,
        vgg_name: str,
        *args,
        **kwargs,
    ) -> None:
        self.vgg_name = vgg_name
        super().__init__(*args, **kwargs)

    def build_layers(self):
        cfg = cfg_32 if self.img_height == 32 else cfg_64
        self.features, convNum = self._make_layers(cfg[self.vgg_name])
        # build FC layers
        ## lienar layer use the last miniblock
        if self.img_height == 64 and self.vgg_name == "vgg8":  ## model is too small, do not use dropout
            classifier = []
        else:
            classifier = [nn.Dropout(0.5)]

        classifier += [
            LinearBlock(
                512,
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
        ]
        self.classifier = nn.Sequential(*classifier)

    def _make_layers(self, cfg):
        layers = []
        in_channel = self.in_channels
        convNum = 0

        for x in cfg:
            # MaxPool2d
            if x == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif x == "GAP":
                layers += [nn.AdaptiveAvgPool2d((1, 1))]
            else:
                # conv + BN + RELU
                layers += [
                    ConvBlock(
                        in_channel,
                        x,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        mini_block=self.block_list[0],
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
                ]
                in_channel = x
                convNum += 1
        return nn.Sequential(*layers), convNum

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x


def VGG8(*args, **kwargs):
    return VGG("vgg8", *args, **kwargs)


def VGG11(*args, **kwargs):
    return VGG("vgg11", *args, **kwargs)


def VGG13(*args, **kwargs):
    return VGG("vgg13", *args, **kwargs)


def VGG16(*args, **kwargs):
    return VGG("vgg16", *args, **kwargs)


def VGG19(*args, **kwargs):
    return VGG("vgg19", *args, **kwargs)


if __name__ == "__main__":
    pass
