"""
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2021-04-01 19:43:53
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2021-09-27 21:42:08
"""
import os
import sys
from typing import Optional

import numpy as np
import torch
from pyutils.general import logger
from pyutils.quantize import uniform_quantize
from pyutils.torch_train import set_torch_deterministic
from torch import Tensor
from torch.types import _size

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))


__all__ = [
    "sinkhorn",
    "assert_unitary",
    "GradientMask",
    "weight_quantize_fn",
    "clip_grad_value_",
    "diff_round",
    "hard_diff_round",
    "PermutationFunction",
]


def sinkhorn(w, n_step=20, t_min=0.1, noise_std=0.01, svd=False):
    with torch.no_grad():
        if svd:
            u, _, v = w.svd()
            w = u.matmul(v.permute(-1, -2))
        if noise_std > 0:
            w = (
                w
                + torch.eye(w.size(-1), device=w.device).mul_(noise_std)
                + torch.randn_like(w).mul_(noise_std)
            )
        w = w.abs()
        for step in range(n_step):
            t = t_min ** (step / n_step)
            # t = t_min
            w /= w.sum(dim=-2, keepdim=True)
            w /= w.sum(dim=-1, keepdim=True)
            w = w.div(t).softmax(dim=-1)
        return w


class GradientMask(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, idx: Optional[int], grad_ratio: float):
        ctx.idx = idx
        ctx.grad_ratio = grad_ratio
        return x

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        idx = ctx.idx
        if idx is None:  # penalize all
            grad_input = grad_output.clone()
        elif idx == -1:  # no penalty
            grad_input = torch.zeros_like(grad_output)
        else:  # penalize selected
            grad_input = torch.zeros_like(grad_output)
            grad_input[idx] = grad_output[idx] * ctx.grad_ratio
        return grad_input, None, None


def clip_grad_value_(parameters, clip_value: float):
    for p in parameters:
        if p.grad is not None:
            if p.grad.is_complex():
                p.grad.data.real.clamp_(min=-clip_value, max=clip_value)
                p.grad.data.imag.clamp_(min=-clip_value, max=clip_value)
            else:
                p.grad.data.clamp_(min=-clip_value, max=clip_value)


class RoundFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor) -> Tensor:
        mask = (x.max(dim=1, keepdim=True)[0] > 0.95).repeat(1, x.size(-1))
        return torch.where(mask, x.round(), x)

    def backward(ctx, grad_output: Tensor) -> Tensor:
        return grad_output.clone()


class HardRoundFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor) -> Tensor:
        # x_max, indices = x.max(dim=1, keepdim=True)
        # illegal_indices = [k for k, v in Counter(indices.view(-1).cpu().numpy().tolist()).items() if v > 1]
        # mask = x_max > 0.95
        # for i in illegal_indices:

        mask = (x.max(dim=1, keepdim=True)[0] > 0.9).repeat(1, x.size(-1))
        ctx.mask = mask
        return torch.where(mask, x.round(), x)

    def backward(ctx, grad_output: Tensor) -> Tensor:
        return grad_output.clone().masked_fill_(ctx.mask, 0)


def diff_round(x: Tensor) -> Tensor:
    """Project to closest permutation matrix"""
    assert x.size(-1) == x.size(-2), f"input x has to be a square matrix, but got {x.size()}"
    return RoundFunction.apply(x)


def hard_diff_round(x: Tensor) -> Tensor:
    """Project to closest permutation matrix"""
    assert x.size(-1) == x.size(-2), f"input x has to be a square matrix, but got {x.size()}"
    return HardRoundFunction.apply(x)


class weight_quantize_fn(torch.nn.Module):
    def __init__(self, w_bit, mode="oconv", alg="dorefa", quant_ratio=1.0):
        """Differentiable weight quantizer. Support different algorithms. Support Quant-Noise with partial quantization.

        Args:
            w_bit (int): quantization bitwidth
            mode (str, optional): Different mode indicates different NN architectures. Defaults to "oconv".
            alg (str, optional): Quantization algorithms. [dorefa, dorefa_sym, qnn, dorefa_pos] Defaults to "dorefa".
            quant_ratio (float, optional): Quantization ratio to support full-precision gradient flow. Defaults to 1.0.
        """
        super(weight_quantize_fn, self).__init__()
        assert 1 <= w_bit <= 32, logger.error(f"Only support 1 - 32 bit quantization, but got {w_bit}")
        self.w_bit = w_bit
        self.alg = alg
        self.mode = mode
        assert alg in {"dorefa", "dorefa_sym", "qnn", "dorefa_pos"}, logger.error(
            f"Only support (dorefa, dorefa_sym, qnn, dorefa_pos) algorithms, but got {alg}"
        )
        self.quant_ratio = quant_ratio
        assert 0 <= quant_ratio <= 1, logger.error(f"Wrong quant ratio. Must in [0,1], but got {quant_ratio}")
        self.uniform_q = uniform_quantize(k=w_bit, gradient_clip=True)

    def set_quant_ratio(self, quant_ratio=None):
        if quant_ratio is None:
            ### get recommended value
            quant_ratio = [
                None,
                0.2,
                0.3,
                0.4,
                0.5,
                0.55,
                0.6,
                0.7,
                0.8,
                0.83,
                0.86,
                0.89,
                0.92,
                0.95,
                0.98,
                0.99,
                1,
            ][min(self.w_bit, 16)]
        assert 0 <= quant_ratio <= 1, logger.error(f"Wrong quant ratio. Must in [0,1], but got {quant_ratio}")
        self.quant_ratio = quant_ratio

    def forward(self, x):
        if self.quant_ratio < 1 and self.training:
            ### implementation from fairseq
            ### must fully quantize during inference
            quant_noise_mask = torch.empty_like(x, dtype=torch.bool).bernoulli_(1 - self.quant_ratio)
        else:
            quant_noise_mask = None

        if self.w_bit == 32:
            weight_q = torch.tanh(x)
            weight_q = weight_q / torch.max(torch.abs(weight_q))
        elif self.w_bit == 1:
            if self.mode == "ringonn":
                weight_q = (self.uniform_q(x) / 4) + 0.5
            else:
                if self.alg == "dorefa":
                    weight_q = (
                        self.uniform_q(x).add(1).mul((1 - 2 ** 0.5 / 2) / 2).add(2 ** 0.5 / 2)
                    )  # [0.717, 1]
                    if quant_noise_mask is not None:
                        x = x.add((2 + 2 ** 0.5) / 4)  # mean is (0.717+1)/2
                        noise = weight_q.data.sub_(x.data).masked_fill_(quant_noise_mask, 0)
                        ### unquantized weights have to follow reparameterization, i.e., tanh and scale
                        weight_q = x + noise
                elif self.alg == "dorefa_sym":
                    E = x.data.abs().mean()
                    weight_q = self.uniform_q(x / E) * E  # [-E, E]
                    if quant_noise_mask is not None:
                        noise = weight_q.data.sub_(x.data).masked_fill_(quant_noise_mask, 0)
                        ### unquantized weights have to follow reparameterization, i.e., tanh and scale
                        weight_q = x + noise
                else:
                    assert NotImplementedError
        else:
            if self.alg == "dorefa":
                weight = torch.tanh(x)  # [-1, 1]
                weight = weight / 2 / torch.max(torch.abs(weight.data)) + 0.5
                # weight = weight / 2 + 0.5
                weight_q = self.uniform_q(weight)
                if quant_noise_mask is not None:
                    noise = weight_q.data.sub_(weight.data).masked_fill_(quant_noise_mask, 0)
                    ### unquantized weights have to follow reparameterization, i.e., tanh and scale
                    weight_q = weight + noise

            elif self.alg == "dorefa_sym":
                weight = torch.tanh(x)  # [-1, 1]
                r = torch.max(torch.abs(weight.data))
                # weight = weight / 2 + 0.5
                weight_q = self.uniform_q(weight / (2 * r) + 0.5) * (2 * r) - r
                if quant_noise_mask is not None:
                    noise = weight_q.data.sub_(weight.data).masked_fill_(quant_noise_mask, 0)
                    ### unquantized weights have to follow reparameterization, i.e., tanh
                    weight_q = weight + noise
            elif self.alg == "dorefa_pos":
                weight = torch.tanh(x)  # [-1, 1]
                r = torch.max(torch.abs(weight.data))
                weight = weight + r
                # weight = weight / 2 + 0.5
                weight_q = self.uniform_q(weight / (2 * r)) * 2 * r
                if quant_noise_mask is not None:
                    noise = weight_q.data.sub_(weight.data).masked_fill_(quant_noise_mask, 0)
                    ### unquantized weights have to follow reparameterization, i.e., tanh
                    weight_q = weight + noise

            elif self.alg == "qnn":
                x_min = torch.min(x.data)
                x_max = torch.max(x.data)
                x_range = x_max - x_min
                weight_q = self.uniform_q((x - x_min) / x_range) * x_range + x_min
                if quant_noise_mask is not None:
                    noise = weight_q.data.sub_(x.data).masked_fill_(quant_noise_mask, 0)
                    ### unquantized weights have to follow reparameterization, i.e., tanh
                    weight_q = x + noise
            else:
                assert NotImplementedError

        return weight_q


class PermutationFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, forward_indices):
        ctx.forward_indices = forward_indices
        output = input[..., forward_indices]
        return output

    @staticmethod
    def backward(ctx, grad_output):
        forward_indices = ctx.forward_indices
        grad_input = grad_output.clone()
        grad_input[..., forward_indices] = grad_output
        return grad_input, None


def assert_unitary(x):

    if x.is_complex():
        x_t = x.conj().transpose(-1, -2)
    else:
        x_t = x.transpose(-1, -2)

    x = x_t.matmul(x)
    I = torch.eye(x.size(-1), device=x.device, dtype=x.dtype) + x.mul(0)
    assert torch.allclose(x, I, rtol=1e-3, atol=1e-5), f"{x}"


if __name__ == "__main__":
    pass
