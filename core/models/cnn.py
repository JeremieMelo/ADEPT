"""
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2021-09-28 04:26:34
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2021-09-28 04:33:35
"""

from collections import OrderedDict
from typing import Dict, List, Union

import torch
from core.models.layers.activation import ReLUN
from core.models.layers.super_conv2d import SuperBlockConv2d
from core.models.layers.super_linear import SuperBlockLinear
from torch import Tensor, nn
from torch.types import Device, _size

from .super_model_base import SuperModel_CLASS_BASE

__all__ = ["SuperOCNN", "LinearBlock", "ConvBlock"]


class LinearBlock(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        mini_block: int = 4,
        bias: bool = False,
        w_bit: int = 16,
        in_bit: int = 16,
        v_max=2.0,
        photodetect: bool = True,
        device: Device = torch.device("cuda"),
        activation: bool = True,
        act_thres: float = 6.0,
        verbose: bool = False,
    ) -> None:
        super().__init__()
        self.linear = SuperBlockLinear(
            in_features,
            out_features,
            mini_block=mini_block,
            bias=bias,
            w_bit=w_bit,
            in_bit=in_bit,
            v_max=v_max,
            photodetect=photodetect,
            device=device,
            # verbose=verbose,
        )

        self.activation = ReLUN(act_thres, inplace=True) if activation else None

    def forward(self, x: Tensor) -> Tensor:
        x = self.linear(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        mini_block: int = 8,
        bias: bool = False,
        stride: Union[int, _size] = 1,
        padding: Union[int, _size] = 0,
        w_bit: int = 16,
        in_bit: int = 16,
        v_max=2.0,
        photodetect: bool = True,
        device: Device = torch.device("cuda"),
        activation: bool = True,
        act_thres: float = 6.0,
        bn_affine: bool = False,
        verbose: bool = False,
    ) -> None:
        super().__init__()
        self.conv = SuperBlockConv2d(
            in_channels,
            out_channels,
            kernel_size,
            mini_block=mini_block,
            bias=bias,
            stride=stride,
            padding=padding,
            w_bit=w_bit,
            in_bit=in_bit,
            v_max=v_max,
            photodetect=photodetect,
            device=device,
            # verbose=verbose,
        )

        self.bn = nn.BatchNorm2d(out_channels, affine=bn_affine, track_running_stats=bn_affine)

        self.activation = ReLUN(act_thres, inplace=True) if activation else None

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class SuperOCNN(SuperModel_CLASS_BASE):
    _conv_linear = (SuperBlockConv2d, SuperBlockLinear)

    def __init__(
        self,
        img_height: int,
        img_width: int,
        in_channels: int,
        num_classes: int,
        kernel_list: List[int] = [16],
        kernel_size_list: List[int] = [3],
        stride_list: List[int] = [1],
        padding_list: List[int] = [1],
        hidden_list: List[int] = [],
        block_list: List[int] = [4, 4],
        pool_out_size=5,
        in_bit: int = 32,
        w_bit: int = 32,
        norm: str = "bn",
        act_thres: float = 6,
        bias: bool = False,
        v_max=2.0,
        photodetect: bool = True,
        super_layer_name: str = "ps_dc_cr",
        super_layer_config: Dict = {},
        bn_affine: bool = False,
        device: Device = torch.device("cuda"),
        verbose: bool = False,
    ):
        super().__init__(
            super_layer_name=super_layer_name, super_layer_config=super_layer_config, device=device
        )
        self.img_height = img_height
        self.img_width = img_width
        self.in_channels = in_channels
        self.kernel_list = kernel_list
        self.kernel_size_list = kernel_size_list
        self.block_list = block_list
        self.hidden_list = hidden_list
        self.stride_list = stride_list
        self.padding_list = padding_list
        self.pool_out_size = pool_out_size
        self.num_classes = num_classes
        self.norm = None if norm.lower() == "none" else norm
        self.act_thres = act_thres
        self.w_bit = w_bit
        self.in_bit = in_bit
        self.v_max = v_max
        self.photodetect = photodetect
        self.bn_affine = bn_affine

        self.bias = bias

        self.device = device
        self.verbose = verbose

        self.build_layers()
        self.reset_parameters()
        self.build_super_layer(super_layer_name, arch=super_layer_config, device=device)

    def build_layers(self) -> None:
        self.features = OrderedDict()
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

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)

        if self.pool2d is not None:
            x = self.pool2d(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    pass
