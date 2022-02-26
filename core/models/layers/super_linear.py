"""
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2021-02-09 16:57:59
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2021-09-27 23:16:34
"""
import logging
from typing import Dict, Optional

import numpy as np
import torch
from torch.nn import Module
import torch.nn.functional as F
from pyutils.compute import get_complex_energy
from pyutils.quantize import input_quantize_fn
from torch import Tensor
from torch.nn import Parameter, init
from torch.types import Device

__all__ = ["SuperBlockLinear"]


class SuperBlockLinear(torch.nn.Module):
    """
    description: SVD-based Linear layer. Blocking matrix multiplication.
    """

    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        mini_block: int = 8,
        bias: bool = False,
        mode: str = "weight",
        v_max: float = 4.36,  # 0-pi for clements, # 6.166 is v_2pi, 0-2pi for reck
        v_pi: float = 4.36,
        w_bit: int = 16,
        in_bit: int = 16,
        photodetect: bool = False,
        super_layer: Module = None,
        device: Device = torch.device("cuda"),
    ) -> None:
        super(SuperBlockLinear, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.mini_block = mini_block
        self.grid_dim_x = int(np.ceil(self.in_channel / mini_block))
        self.grid_dim_y = int(np.ceil(self.out_channel / mini_block))
        self.in_channel_pad = self.grid_dim_x * mini_block
        self.out_channel_pad = self.grid_dim_y * mini_block
        self.mode = mode
        self.super_layer = super_layer
        self.super_ps_layers = None
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

    def build_parameters(self, mode: str = "weight") -> None:
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
        weight = self.super_layer.get_weight_matrix(self.super_ps_layers, self.weight)
        weight = weight.permute(0, 2, 1, 3).reshape(self.out_channel_pad, self.in_channel_pad)[
            : self.out_channel, : self.in_channel
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

    def forward(self, x: Tensor) -> Tensor:
        if self.in_bit < 16:
            x = self.input_quantizer(x)

        if not self.fast_forward_flag or self.weight is None:
            weight = self.build_weight()  # [p, q, k, k] or u, s, v
        else:
            weight = self.weight

        inc_pos = int(np.ceil(weight.size(1)/2))
        weight = weight.t()
        x = x.to(torch.complex64)
        x_pos = x[..., :inc_pos].matmul(weight[:inc_pos, :])  # [bs, outc]
        x_neg = x[..., inc_pos:].matmul(weight[inc_pos:, :])  # [outc, bs]
        if self.photodetect:
            out = get_complex_energy(torch.view_as_real(x_pos)) - get_complex_energy(
                torch.view_as_real(x_neg)
            )
        else:
            out = x_pos - x_neg

        if self.bias is not None:
            out = out + self.bias.unsqueeze(0)

        return out
