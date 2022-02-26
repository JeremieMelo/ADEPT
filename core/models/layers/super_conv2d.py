"""
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2021-04-03 01:54:42
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2021-09-26 18:19:18
"""
import logging
from typing import Dict, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from pyutils.compute import get_complex_energy, im2col_2d
from pyutils.quantize import input_quantize_fn
from torch import Tensor
from torch.nn import Parameter, init, Module
from torch.types import Device, _size

__all__ = ["SuperBlockConv2d"]


class SuperBlockConv2d(torch.nn.Module):
    """
    description: SVD-based Linear layer. Blocking matrix multiplication.
    """

    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        kernel_size: int = 3,
        mini_block: int = 8,
        bias: bool = False,
        stride: Union[int, _size] = 1,
        padding: Union[int, _size] = 0,
        dilation: Union[int, _size] = 1,
        groups: int = 1,
        mode: str = "weight",
        v_max: float = 4.36,  # 0-pi for clements, # 6.166 is v_2pi, 0-2pi for reck
        v_pi: float = 4.36,
        w_bit: int = 16,
        in_bit: int = 16,
        photodetect: bool = False,
        super_layer: Module = None,
        device: Device = torch.device("cuda"),
    ) -> None:
        super(SuperBlockConv2d, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.mini_block = mini_block
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.super_layer = super_layer
        self.super_ps_layers = None

        self.mode = mode
        self.v_max = v_max
        self.v_pi = v_pi
        self.gamma = np.pi / self.v_pi ** 2
        self.w_bit = w_bit
        self.in_bit = in_bit
        self.photodetect = photodetect
        self.device = device

        # build parameters
        self.build_parameters()

        # quantization tool
        self.input_quantizer = input_quantize_fn(self.in_bit, device=self.device)

        # default set to slow forward
        self.disable_fast_forward()
        # default set no phase noise
        self.set_phase_variation(0)
        # default set no gamma noise
        self.set_gamma_noise(0)
        # default set no crosstalk
        self.set_crosstalk_factor(0)
        # zero pad for input
        self.x_zero_pad = None

        if bias:
            self.bias = Parameter(torch.Tensor(out_channel).to(self.device))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def build_parameters(self) -> None:
        ### TODO, construct balanced weight
        mini_block = self.mini_block
        n = self.in_channel * self.kernel_size ** 2
        if self.in_channel % 2 == 0:  # even channel
            self.grid_dim_x = 2 * int(np.ceil(n / 2 / mini_block))
        else:  # odd channel, mostly in the first conv layer
            self.grid_dim_x = int(np.ceil((n // 2) / mini_block)) + int(np.ceil((n - n // 2) / mini_block))

        self.grid_dim_y = int(np.ceil(self.out_channel / mini_block))
        self.in_channel_pad = self.grid_dim_x * mini_block
        self.out_channel_pad = self.grid_dim_y * mini_block

        self.weight = Parameter(
            torch.empty(
                self.grid_dim_y, self.grid_dim_x, self.mini_block, dtype=torch.cfloat, device=self.device
            )
        )
        self.eye = torch.eye(self.mini_block, self.mini_block, dtype=torch.cfloat, device=self.device)
        self.U = self.V = self.eye

    def reset_parameters(self) -> None:
        temp = torch.empty(self.grid_dim_y*self.mini_block, self.grid_dim_x*self.mini_block, device=self.device)
        init.kaiming_normal_(temp)
        temp = temp.view(self.grid_dim_y, self.mini_block, self.grid_dim_x, self.mini_block).permute(0,2,1,3)
        _, s, _ = torch.svd(temp, compute_uv=False)
        self.weight.data.copy_(s)
        if self.bias is not None:
            init.uniform_(self.bias, 0, 0)

    def set_super_layer_transfer_matrices(self, U: Tensor, V: Tensor) -> None:
        self.U = U
        self.V = V

    def build_weight(self) -> Tensor:
        # [k,k] -> [k,k]
        # [p, q, k, 1] * [1, 1, k, k] complex = [p, q, k, k] complex
        weight = self.super_layer.get_weight_matrix(self.super_ps_layers, self.weight)
        weight = weight.permute(0, 2, 1, 3).reshape(self.out_channel_pad, self.in_channel_pad)[
            : self.out_channel, : self.in_channel * self.kernel_size ** 2
        ]

        return weight

    def enable_fast_forward(self) -> None:
        self.fast_forward_flag = True

    def disable_fast_forward(self) -> None:
        self.fast_forward_flag = False

    def set_phase_variation(self, noise_std: float, random_state: Optional[int] = None) -> None:
        self.phase_noise_std = noise_std

    def set_crosstalk_factor(self, crosstalk_factor: float) -> None:
        self.crosstalk_factor = crosstalk_factor

    def set_gamma_noise(self, noise_std: float = 0, random_state: Optional[int] = None) -> None:
        self.gamma_noise_std = noise_std

    def set_weight_bitwidth(self, w_bit: int) -> None:
        self.w_bit = w_bit
        # self.phase_U_quantizer.set_bitwidth(w_bit)
        # self.phase_S_quantizer.set_bitwidth(w_bit)
        # self.phase_V_quantizer.set_bitwidth(w_bit)

    def load_parameters(self, param_dict: Dict) -> None:
        """
        description: update parameters based on this parameter dictionary\\
        param param_dict {dict of dict} {layer_name: {param_name: param_tensor, ...}, ...}
        """
        for name, param in param_dict.items():
            getattr(self, name).data.copy_(param)
        if self.mode == "phase":
            self.build_weight(update_list=param_dict)

    def switch_mode_to(self, mode: str) -> None:
        self.mode = mode

    def get_power(self, mixtraining_mask: Optional[Tensor] = None) -> float:
        masks = (
            mixtraining_mask
            if mixtraining_mask is not None
            else (self.mixedtraining_mask if self.mixedtraining_mask is not None else None)
        )
        if masks is not None:
            power = ((self.phase_U.data * masks["phase_U"]) % (2 * np.pi)).sum()
            power += ((self.phase_S.data * masks["phase_S"]) % (2 * np.pi)).sum()
            power += ((self.phase_V.data * masks["phase_V"]) % (2 * np.pi)).sum()
        else:
            power = ((self.phase_U.data) % (2 * np.pi)).sum()
            power += ((self.phase_S.data) % (2 * np.pi)).sum()
            power += ((self.phase_V.data) % (2 * np.pi)).sum()
        return power.item()

    def get_output_dim(self, img_height, img_width):
        h_out = (img_height - self.kernel_size + 2 * self.padding) / self.stride + 1
        w_out = (img_width - self.kernel_size + 2 * self.padding) / self.stride + 1
        return int(h_out), int(w_out)

    def forward(self, x: Tensor) -> Tensor:
        if self.in_bit < 16:
            x = self.input_quantizer(x)

        if not self.fast_forward_flag or self.weight is None:
            weight = self.build_weight()  # [p, q, k, k] or u, s, v
        else:
            weight = self.weight
        _, x, h_out, w_out = im2col_2d(
            W=None,
            X=x,
            stride=self.stride,
            padding=self.padding,
            w_size=(self.out_channel, self.in_channel, self.kernel_size, self.kernel_size),
        )
        # inc_pos = int(np.ceil(self.grid_dim_x / 2) * self.mini_block)

        inc_pos = int(np.ceil(weight.size(1) / 2))
        x = x.to(torch.complex64)
        x_pos = weight[:, :inc_pos].matmul(x[:inc_pos])  # [outc, h*w*bs]
        x_neg = weight[:, inc_pos:].matmul(x[inc_pos:])  # [outc, h*w*bs]
        if self.photodetect:
            x = get_complex_energy(torch.view_as_real(x_pos)) - get_complex_energy(torch.view_as_real(x_neg))
        else:
            x = x_pos - x_neg
        out = x.view(self.out_channel, h_out, w_out, -1).permute(3, 0, 1, 2)

        # out_real = F.conv2d(
        #     x,
        #     weight.real,
        #     bias=None,
        #     stride=self.stride,
        #     padding=self.padding,
        #     dilation=self.dilation,
        #     groups=self.groups,
        # )
        # out_imag = F.conv2d(
        #     x,
        #     weight.imag,
        #     bias=None,
        #     stride=self.stride,
        #     padding=self.padding,
        #     dilation=self.dilation,
        #     groups=self.groups,
        # )
        # out = torch.complex(out_real, out_imag)

        # if self.photodetect:
        #     out = get_complex_energy(out)

        if self.bias is not None:
            out = out + self.bias.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

        return out
