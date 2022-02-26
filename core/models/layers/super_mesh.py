from functools import lru_cache
from core.models.layers.utils import hard_diff_round
import torch
from torch import nn
import itertools
import numpy as np

from typing import List, Iterable, Optional, Tuple
from torch.functional import Tensor

from torch.types import Device
from .utils import weight_quantize_fn
from pyutils.torch_train import set_torch_deterministic
from pyutils.compute import gen_gaussian_noise

__all__ = [
    "SuperOpticalModule",
    "SuperBatchedPSLayer",
    "SuperDCFrontShareLayer",
    "SuperCRLayer",
    "super_layer_name_dict",
]


def get_combinations(inset: List, n=None) -> List[List]:
    all_combs = []
    if n is None:
        # all possible combinations, with different #elements in a set
        for k in range(1, len(inset) + 1):
            all_combs.extend(list(map(list, itertools.combinations(inset, k))))
    elif isinstance(n, int):
        all_combs.extend(list(map(list, itertools.combinations(inset, n))))
    elif isinstance(n, Iterable):
        for k in n:
            all_combs.extend(list(map(list, itertools.combinations(inset, k))))

    return all_combs


class SuperOpticalModule(nn.Module):
    def __init__(self, n_waveguides):
        super().__init__()
        self.n_waveguides = n_waveguides
        self.sample_arch = None

    def set_sample_arch(self, sample_arch):
        # a structure that can repersent the architecture
        # e.g., for front share layer, a scalar can represent the arch
        # e.g., for permutation layer, a permuted index array can represent the arch
        self.sample_arch = sample_arch

    @property
    def arch_space(self):
        return None

    def count_sample_params(self):
        raise NotImplementedError


class SuperBatchedPSLayer(SuperOpticalModule):
    _share_uv_list = {"global", "row", "col", "none"}

    def __init__(
        self,
        grid_dim_x: int,
        grid_dim_y: int,
        n_waveguides: int,
        n_front_share_waveguides: int,
        share_uv: str = "global",
        trainable: bool = True,
        device: Device = torch.device("cuda:0"),
    ):
        super().__init__(n_waveguides=n_waveguides)
        self.grid_dim_x = grid_dim_x
        self.grid_dim_y = grid_dim_y
        self.n_front_share_waveguides = n_front_share_waveguides
        self.share_uv = share_uv.lower()
        assert (
            self.share_uv in self._share_uv_list
        ), f"share_uv only supports {self._share_uv_list}, but got {share_uv}"
        self.trainable = trainable
        self.device = device
        self.build_parameters()
        self.reset_parameters()
        self.set_phase_noise(0)

    def build_parameters(self):
        """weight is the phase shift of the phase shifter"""
        if self.share_uv == "global":
            self.weight = nn.Parameter(
                torch.empty(self.n_waveguides, device=self.device), requires_grad=self.trainable
            )
        elif self.share_uv == "row":
            ## use the same PS within a row => share U
            self.weight = nn.Parameter(
                torch.empty(self.grid_dim_y, self.n_waveguides, device=self.device),
                requires_grad=self.trainable,
            )
        elif self.share_uv == "col":
            ## use the same PS within a column => share V
            self.weight = nn.Parameter(
                torch.empty(self.grid_dim_x, self.n_waveguides, device=self.device),
                requires_grad=self.trainable,
            )
        elif self.share_uv == "none":
            ## independent PS for each block
            self.weight = nn.Parameter(
                torch.empty(self.grid_dim_y, self.grid_dim_x, self.n_waveguides, device=self.device),
                requires_grad=self.trainable,
            )
        else:
            raise ValueError(f"Not supported share_uv: {self.share_uv}")

    def reset_parameters(self, alg="uniform"):
        assert alg in {"normal", "uniform", "identity"}
        if alg == "normal":
            nn.init.normal_(self.weight)
        elif alg == "uniform":
            nn.init.uniform_(self.weight, -np.pi / 2, np.pi)
        elif alg == "identity":
            self.weight.data.zero_()

    def build_weight(self):
        if self.phase_noise_std > 0:
            weight = self.weight + gen_gaussian_noise(
                torch.zeros_like(self.weight), noise_mean=0, noise_std=self.phase_noise_std
            )
        else:
            weight = self.weight
        return weight

    def forward(self, x: Tensor) -> Tensor:
        # x[..., q, n_waveguides] complex
        if not x.is_complex():
            x = x.to(torch.cfloat)

        weight = self.build_weight()
        weight = torch.exp(1j * weight)
        if self.share_uv == "global":
            # [..., n_waveguides] * [n_waveguides] = [..., n_waveguides]
            x = x.mul(weight)
        elif self.share_uv == "row":
            # [..., p, n_waveguides] * [p, n_waveguides] = [..., p, n_waveguides]
            x = x.mul(weight)
        elif self.share_uv == "col":
            # [..., q, n_waveguides] * [q, n_waveguides] = [..., q, n_waveguides]
            x = x.mul(weight)
        elif self.share_uv == "none":
            # [..., p, q, n_waveguides] * [p, q, n_waveguides] = [..., p, q, n_waveguides]
            x = x.mul(weight)
        else:
            raise ValueError(f"Not supported share_uv: {self.share_uv}")

        return x

    @property
    def arch_space(self):
        ## do not sample PS, we always put a whole column of PS
        choices = [self.n_waveguides]
        return get_combinations(choices)

    def count_sample_params(self):
        return len(self.sample_arch) if self.weight.requires_grad else 0

    def extra_repr(self) -> str:
        s = f"grid_dim_x={self.grid_dim_x}, grid_dim_y={self.grid_dim_y}, share_uv={self.share_uv}, n_waveguides={self.n_waveguides}, n_front_share_waveguides={self.n_front_share_waveguides}, sample_arch={self.sample_arch}, trainable={self.trainable}"
        return s

    def set_phase_noise(self, noise_std: float = 0.0):
        self.phase_noise_std = noise_std


class SuperDCFrontShareLayer(SuperOpticalModule):
    def __init__(
        self,
        n_waveguides: int,
        n_front_share_waveguides: int,
        offset: int = 0,
        trainable: bool = False,
        binary: bool = False,
        device: Device = torch.device("cuda:0"),
    ):
        super().__init__(n_waveguides=n_waveguides)
        self.offset = offset
        self.n_front_share_waveguides = n_front_share_waveguides
        self.max_arch = (self.n_waveguides - self.offset) // 2
        self.trainable = trainable
        self.binary = binary
        self.device = device
        self.build_parameters()
        self.reset_parameters()
        if self.binary:
            self.weight_quantizer = weight_quantize_fn(w_bit=1, alg="dorefa")
        else:
            self.weight_quantizer = None

        self.set_dc_noise(0)
        self.fast_mode = False
        self.fast_weight = None

    def build_parameters(self):
        """weight is the transmission factor t in the DC transfer matrix"""
        self.weight = nn.Parameter(
            torch.empty(self.max_arch, device=self.device), requires_grad=self.trainable
        )

    def reset_parameters(self):
        if self.binary:
            nn.init.uniform_(self.weight, -0.01, 0.01)
        else:
            nn.init.constant_(self.weight, 2 ** 0.5 / 2)

    def build_weight(self):
        if self.sample_arch < self.max_arch:
            weight = self.weight[: self.sample_arch]
        else:
            weight = self.weight
        if self.binary:
            weight = self.weight_quantizer(weight)  # binarize to sqrt(2)/2 and 1

        if self.dc_noise_std > 0:
            mask = weight.data > 0.9  # only inject noise when t=sqrt(2)/2
            noise = gen_gaussian_noise(torch.zeros_like(weight), noise_mean=0, noise_std=self.dc_noise_std)
            noise.masked_fill_(mask, 0)
            weight = weight + noise

        t = weight
        k = (1 - weight.square() + 1e-6).sqrt()  # when t=1, k=0, the grad is nan !!
        w11 = w22 = t.to(torch.complex64)
        w12 = w21 = k.mul(1j)
        weight = torch.stack([w11, w12, w21, w22], dim=-1).view(-1, 2, 2)

        return weight

    def forward(self, x: Tensor) -> Tensor:
        # x[..., n_waveguides] complex

        if not x.is_complex():
            x = x.to(torch.cfloat)

        if self.fast_mode and self.fast_weight is not None:
            weight = self.fast_weight
        else:
            weight = self.build_weight()
        sample_arch = min(self.sample_arch, self.max_arch)
        n_sample_waveguides = int(sample_arch * 2)

        if n_sample_waveguides < x.size(-1):
            out = x[..., self.offset : self.offset + n_sample_waveguides]
            # [1, n//2, 2, 2] x [bs, n//2, 2, 1] = [bs, n//2, 2, 1] -> [bs, n]
            out = (
                weight.unsqueeze(0)
                .matmul(out.view(-1, sample_arch, 2, 1))
                .view(list(x.shape[:-1]) + [n_sample_waveguides])
            )
            out = torch.cat([x[..., : self.offset], out, x[..., self.offset + n_sample_waveguides :]], dim=-1)
        else:
            out = (
                weight.unsqueeze(0)
                .matmul(x.reshape(-1, sample_arch, 2, 1))
                .view(list(x.shape[:-1]) + [n_sample_waveguides])
            )

        return out

    @property
    def arch_space(self):
        if self.trainable:
            return [self.max_arch]
        else:
            return list(range(1, self.max_arch + 1))

    def count_sample_params(self):
        return len(self.sample_arch) if self.weight.requires_grad else 0

    def extra_repr(self) -> str:
        s = f"n_waveguides={self.n_waveguides}, n_front_share_waveguides={self.n_front_share_waveguides}, sample_arch={self.sample_arch}, offset={self.offset}, trainable={self.trainable}"
        return s

    def set_dc_noise(self, noise_std: float = 0.0):
        self.dc_noise_std = noise_std

    def fix_arch_solution(self):
        with torch.no_grad():
            if self.binary:
                weight = self.weight_quantizer(self.weight.data)
            else:
                weight = self.weight.data
            t = weight
            k = (1 - weight.square()).sqrt()  # when t=1, k=0, the grad is nan !!
            w11 = w22 = t.to(torch.complex64)
            w12 = w21 = k.mul(1j)
            weight = torch.stack([w11, w12, w21, w22], dim=-1).view(-1, 2, 2)

            self.fast_weight = weight
            self.fast_mode = True
            self.weight.requires_grad_(False)


class SuperCRLayer(SuperOpticalModule):
    def __init__(
        self,
        n_waveguides: int,
        trainable: bool = True,
        symmetry: bool = False,
        device: Device = torch.device("cuda:0"),
    ):
        super().__init__(n_waveguides=n_waveguides)
        self.trainable = trainable
        self.symmetry = symmetry
        self.device = device
        self.build_parameters()
        self.reset_parameters()
        self.identity_forward = False
        self.set_cr_noise(0)
        self.fast_mode = False
        self.indices = None

    def build_parameters(self):
        self.weight = nn.Parameter(
            torch.empty(self.n_waveguides, self.n_waveguides, device=self.device),
            requires_grad=self.trainable,
        )
        self.eye = torch.eye(self.n_waveguides, device=self.device)
        # ALM multiplier
        self.alm_multiplier = nn.Parameter(
            torch.empty(2, self.n_waveguides, device=self.device), requires_grad=False
        )

    def reset_parameters(self, alg="noisy_identity"):
        assert alg in {
            "ortho",
            "uniform",
            "normal",
            "identity",
            "near_identity",
            "noisy_identity",
            "perm",
            "near_perm",
        }
        if alg == "ortho":
            nn.init.orthogonal_(self.weight)
        elif alg == "uniform":
            nn.init.constant_(self.weight, 1 / self.n_waveguides)
            set_torch_deterministic(0)
            self.weight.data += torch.randn_like(self.weight).mul(0.01)
        elif alg == "normal":
            set_torch_deterministic(0)
            torch.nn.init.xavier_normal_(self.weight)
            self.weight.data += self.weight.data.std() * 3
        elif alg == "identity":
            self.weight.data.copy_(torch.eye(self.weight.size(0), device=self.device))
        elif alg == "near_identity":
            self.weight.data.copy_(torch.eye(self.weight.size(0), device=self.device))
            margin = 0.5
            self.weight.data.mul_(margin - (1 - margin) / (self.n_waveguides - 1)).add_(
                (1 - margin) / (self.n_waveguides - 1)
            )
        elif alg == "noisy_identity":
            self.weight.data.copy_(torch.eye(self.weight.size(0), device=self.device))
            margin = 0.3
            self.weight.data.mul_(margin - (1 - margin) / (self.n_waveguides - 1)).add_(
                (1 - margin) / (self.n_waveguides - 1)
            )
            self.weight.data.add_(torch.randn_like(self.weight.data) * 0.05)

        elif alg == "perm":
            self.weight.data.copy_(
                torch.eye(self.weight.size(0), device=self.device)[torch.randperm(self.weight.size(0))]
            )
        elif alg == "near_perm":
            set_torch_deterministic(0)
            self.weight.data.copy_(
                torch.eye(self.weight.size(0), device=self.device)[torch.randperm(self.weight.size(0))]
            )
            margin = 0.9
            self.weight.data.mul_(margin - (1 - margin) / (self.n_waveguides - 1)).add_(
                (1 - margin) / (self.n_waveguides - 1)
            )
        nn.init.constant_(self.alm_multiplier, 0)

    def enable_identity_forward(self):
        self.identity_forward = True

    def set_butterfly_forward(self, forward=True, level=0):
        initial_indices = torch.arange(0, self.n_waveguides, dtype=torch.long, device=self.device)
        block_size = 2 ** (level + 2)
        if forward:
            indices = (
                initial_indices.view(-1, self.n_waveguides // block_size, 2, block_size // 2)
                .transpose(dim0=-2, dim1=-1)
                .contiguous()
                .view(-1)
            )
        else:
            indices = initial_indices.view(-1, self.n_waveguides // block_size, block_size)
            indices = torch.cat([indices[..., ::2], indices[..., 1::2]], dim=-1).contiguous().view(-1)
        eye = torch.eye(self.n_waveguides, device=self.device)[indices, :]
        self.weight.data.copy_(eye)

    def build_weight(self):
        if self.identity_forward:
            return self.weight
        """Enforce DN hard constraints with reparametrization"""

        if self.symmetry:
            weight1 = self.weight[: self.weight.size(0) // 2]
            weight2 = torch.flipud(torch.fliplr(weight1))
            weight = torch.cat([weight1, weight2], dim=0)
        else:
            weight = self.weight
        weight = weight.abs()  # W >= 0
        weight = weight / weight.data.sum(dim=0, keepdim=True)  # Wx1=1
        weight = weight / weight.data.sum(dim=1, keepdim=True)  # W^Tx1=1

        with torch.no_grad():
            perm_loss = (
                weight.data.norm(p=1, dim=0).sub(weight.data.norm(p=2, dim=0)).mean()
                + (1 - weight.data.norm(p=2, dim=1)).mean()
            )
        if perm_loss < 0.05:
            weight = hard_diff_round(
                weight
            )  # W -> P # once it is very close to permutation, it will be trapped and legalized without any gradients.
        return weight

    def get_ortho_loss(self):
        weight = self.build_weight()
        loss = torch.nn.functional.mse_loss(weight.matmul(weight.t()), self.eye)
        return loss

    def get_perm_loss(self):
        """https://www.math.uci.edu/~jxin/AutoShuffleNet_KDD2020F.pdf"""
        weight = self.build_weight()
        loss = (
            weight.norm(p=1, dim=0).sub(weight.norm(p=2, dim=0)).mean() + (1 - weight.norm(p=2, dim=1)).mean()
        )
        return loss

    def get_alm_perm_loss(self, rho: float = 0.1):
        if self.identity_forward:
            return 0
        ## quadratic tern is also controlled multiplier
        weight = self.build_weight()
        d_weight_r = weight.norm(p=1, dim=0).sub(weight.norm(p=2, dim=0))
        # d_weight_c = weight.norm(p=1, dim=1).sub(weight.norm(p=2, dim=1))
        d_weight_c = 1 - weight.norm(p=2, dim=1)
        loss = self.alm_multiplier[0].dot(d_weight_r + rho / 2 * d_weight_r.square()) + self.alm_multiplier[
            1
        ].dot(d_weight_c + rho / 2 * d_weight_c.square())
        return loss

    def update_alm_multiplier(self, rho: float = 0.1, max_lambda: Optional[float] = None):
        if self.identity_forward:
            return
        with torch.no_grad():
            weight = self.build_weight().detach()
            d_weight_r = weight.norm(p=1, dim=0).sub(weight.norm(p=2, dim=0))
            d_weight_c = weight.norm(p=1, dim=1).sub(weight.norm(p=2, dim=1))
            self.alm_multiplier[0].add_(rho * (d_weight_r + rho / 2 * d_weight_r.square()))
            self.alm_multiplier[1].add_(rho * (d_weight_c + rho / 2 * d_weight_c.square()))
            if max_lambda is not None:
                self.alm_multiplier.data.clamp_max_(max_lambda)

    def get_crossing_loss(self, alg="mse"):
        weight = self.build_weight()
        n = self.n_waveguides
        if alg == "kl":
            return torch.kl_div(weight, self.eye).mean()
        elif alg == "mse":
            return torch.nn.functional.mse_loss(weight, self.eye)

    def _get_num_crossings(self, in_indices):
        res = 0
        for idx, i in enumerate(in_indices):
            for j in range(idx + 1, len(in_indices)):
                if i > in_indices[j]:
                    res += 1
        return res

    def get_num_crossings(self):
        if self.identity_forward:
            return 0
        with torch.no_grad():
            weight = self.build_weight().detach()
            in_indices = weight.max(dim=0)[1].cpu().numpy().tolist()
            return self._get_num_crossings(in_indices)

    def forward(self, x: Tensor) -> Tensor:
        # x[..., n_waveguides] real/complex
        # print("before cr", x.size())
        if self.identity_forward:
            return x
        if self.fast_mode and self.indices is not None:
            return torch.complex(x.real[..., self.indices], x.imag[..., self.indices])

        weight = self.build_weight()
        if self.training:
            weight = weight.to(x.dtype)
            x = x.matmul(weight.t())
        else:  # inference mode always use permutation
            # indices = torch.argmax(weight, dim=1)
            # x = x[..., indices]  # fast permutation
            weight = weight.to(x.dtype)
            x = x.matmul(weight.t())
        # print("after cr", x.size())

        return x

    @property
    def arch_space(self):
        return [1]  # only one selection, this is a differentiable layer

    def count_sample_params(self):
        return 0

    def extra_repr(self) -> str:
        s = f"n_waveguides={self.n_waveguides}, sample_arch={self.sample_arch}, trainable={self.trainable}"
        return s

    def set_cr_noise(self, noise_std: float = 0.0):
        self.cr_noise_std = noise_std

    def check_perm(self, indices):
        return tuple(range(len(indices))) == tuple(sorted(indices.cpu().numpy().tolist()))

    def unitary_projection(self, w, n_step=10, t=0.005, noise_std=0.01):
        w = w.div(t).softmax(dim=-1).round()
        legal_solution = []
        for i in range(n_step):
            u, s, v = w.svd()
            w = u.matmul(v.permute(-1, -2))
            w.add_(torch.randn_like(w) * noise_std)
            w = w.div(t).softmax(dim=-1)
            indices = w.argmax(dim=-1)
            if self.check_perm(indices):
                n_cr = self._get_num_crossings(indices.cpu().numpy().tolist())
                legal_solution.append((n_cr, w.clone().round()))
        legal_solution = sorted(legal_solution, key=lambda x: x[0])
        w = legal_solution[0][1]
        return w

    def fix_arch_solution(self):
        with torch.no_grad():
            weight = self.build_weight().detach().data
            self.indices = torch.argmax(weight, dim=1)
            assert self.check_perm(self.indices), f"{self.indices.cpu().numpy().tolist()}"
            self.fast_mode = True
            self.weight.requires_grad_(False)


class SuperMeshBase(SuperOpticalModule):
    def __init__(self, arch: dict = None, device=torch.device("cuda:0")):
        super().__init__(n_waveguides=arch["n_waveguides"])
        self.arch = arch
        self.device = device

        self.n_front_share_waveguides = arch.get("n_front_share_waveguides", None)
        self.n_front_share_ops = arch.get("n_front_share_ops", None)

        self.n_blocks = arch.get("n_blocks", None)
        assert (
            self.n_blocks % 2 == 0
        ), f"n_blocks in arch should always be an even number, but got {self.n_blocks}"
        self.n_layers_per_block = arch.get("n_layers_per_block", None)
        self.n_front_share_blocks = arch.get("n_front_share_blocks", None)

        self.sample_n_blocks = None

        self.super_layers_all = self.build_super_layers()
        self.fast_mode = False
        self.fast_arch_mask = None

    def build_super_layers(self):
        raise NotImplementedError

    def set_sample_arch(self, sample_arch):
        for k, layer_arch in enumerate(sample_arch[:-1]):
            self.super_layers_all[k].set_sample_arch(layer_arch)
        self.sample_n_blocks = sample_arch[-1]

    def reset_parameters(self) -> None:
        for m in self.super_layers_all:
            m.reset_parameters()

    def forward(self, x: Tensor) -> Tensor:
        for k in range(len(self.super_layers_all)):
            if k < self.sample_n_blocks * self.n_layers_per_block:
                x = self.super_layers_all[k](x)
        return x

    def count_sample_params(self):
        n_params = 0
        for layer_idx, layer in enumerate(self.super_layers_all):
            if layer_idx < self.sample_n_blocks * self.n_layers_per_block:
                n_params += layer.count_sample_params()
        return n_params


class SuperMeshADEPT(SuperMeshBase):
    def build_sampling_coefficients(self):
        self.sampling_coeff = torch.nn.Parameter(torch.zeros(self.n_blocks, 2) + 0.5)
        if self.n_front_share_blocks > 0:
            self.sampling_coeff.data[
                self.n_blocks // 2 - self.n_front_share_blocks // 2 : self.n_blocks // 2, 0
            ] = -100
            self.sampling_coeff.data[
                self.n_blocks // 2 - self.n_front_share_blocks // 2 : self.n_blocks // 2, 1
            ] = 100  # force to choose the block
            self.sampling_coeff.data[-self.n_front_share_blocks // 2 :, 0] = -100  # force to choose the block
            self.sampling_coeff.data[-self.n_front_share_blocks // 2 :, 1] = 100  # force to choose the block

    def set_gumbel_temperature(self, T: float = 5.0):
        self.gumbel_temperature = T

    def build_arch_mask(self, mode="gumbel_soft", batch_size: int = 32):
        logits = self.sampling_coeff
        if mode == "gumbel_hard":
            self.arch_mask = torch.nn.functional.gumbel_softmax(
                logits,
                tau=self.gumbel_temperature,
                hard=True,
                dim=-1,
            )
        if mode == "gumbel_soft":
            self.arch_mask = torch.nn.functional.gumbel_softmax(
                logits,
                tau=self.gumbel_temperature,
                hard=False,
                dim=-1,
            )
        if mode == "gumbel_soft_batch":
            self.arch_mask = torch.nn.functional.gumbel_softmax(
                torch.log_softmax(logits, dim=-1).unsqueeze(0).repeat(batch_size, 1, 1),
                tau=self.gumbel_temperature,
                hard=False,
                dim=-1,
            )
        elif mode == "softmax":
            self.arch_mask = torch.softmax(
                logits / self.gumbel_temperature,
                dim=-1,
            )
        elif mode == "largest":
            logits = torch.cat([logits[..., 0:1] - 100, logits[..., 1:] + 100], dim=-1)
            self.arch_mask = torch.nn.functional.gumbel_softmax(
                torch.log_softmax(logits, dim=-1),
                tau=self.gumbel_temperature,
                hard=True,
                dim=-1,
            )
        elif mode == "smallest":
            if self.n_front_share_blocks > 0:
                logits = logits.view(2, -1, 2)

                logits_small = torch.cat(
                    [
                        logits[:, : -self.n_front_share_blocks // 2, 0:1] + 100,
                        logits[:, : -self.n_front_share_blocks // 2, 1:] - 100,
                    ],
                    dim=-1,
                )
                logits = torch.cat(
                    [logits_small, logits[:, -self.n_front_share_blocks // 2 :, :]], dim=1
                ).view(-1, 2)
            else:
                logits = torch.cat([logits[..., 0:1] + 200, logits[..., 1:] - 200], dim=-1)
            self.arch_mask = torch.nn.functional.gumbel_softmax(
                logits,
                tau=self.gumbel_temperature,
                hard=True,
                dim=-1,
            )
        elif mode == "random":
            logits = torch.ones_like(logits)
            self.arch_mask = torch.nn.functional.gumbel_softmax(
                logits,
                tau=self.gumbel_temperature,
                hard=True,
                dim=-1,
            )

    def build_super_layers(self):
        self.share_ps = self.arch.get("share_ps", "global")
        self.interleave_dc = self.arch.get("interleave_dc", True)
        self.symmetry_cr = self.arch.get("symmetry_cr", False)
        super_layers_all = nn.ModuleList()

        for i in range(self.arch["n_blocks"]):
            super_layers_all.append(
                SuperDCFrontShareLayer(
                    n_waveguides=self.n_waveguides,
                    n_front_share_waveguides=self.n_front_share_waveguides,
                    offset=i % 2 if self.interleave_dc else 0,  # interleaved design space
                    trainable=True,
                    binary=True,
                    device=self.device,
                )
            )
            if i == self.n_blocks - 1:  # pseudo-permutation, which is an identity
                layer = SuperCRLayer(
                    n_waveguides=self.n_waveguides,
                    trainable=False,
                    symmetry=self.symmetry_cr,
                    device=self.device,
                )
                layer.reset_parameters(alg="identity")
                layer.enable_identity_forward()
                super_layers_all.append(layer)
            else:
                super_layers_all.append(
                    SuperCRLayer(
                        n_waveguides=self.n_waveguides,
                        trainable=True,
                        symmetry=self.symmetry_cr,
                        device=self.device,
                    )
                )

        return super_layers_all

    def build_ps_layers(self, grid_dim_x: int, grid_dim_y: int) -> nn.ModuleList:
        ## each CONV or Linear need to explicit build ps layers using this function
        self.share_ps = self.arch.get("share_ps", "global")
        self.interleave_dc = self.arch.get("interleave_dc", True)
        super_ps_layers = nn.ModuleList()
        for i in range(self.arch["n_blocks"]):
            if self.share_ps in {"global", "none"}:
                share_uv = self.share_ps
            elif self.share_ps in {"row_col"}:
                share_uv = "col" if i < self.arch["n_blocks"] // 2 else "row"
            super_ps_layers.append(
                SuperBatchedPSLayer(
                    grid_dim_x=grid_dim_x,
                    grid_dim_y=grid_dim_y,
                    share_uv=share_uv,
                    n_waveguides=self.n_waveguides,
                    n_front_share_waveguides=self.n_front_share_waveguides,
                    trainable=True,
                    device=self.device,
                )
            )
        return super_ps_layers

    def set_identity(self) -> None:
        self.set_identity_cr()

    def set_identity_cr(self) -> None:
        for m in self.super_layers_all:
            if isinstance(m, (SuperCRLayer,)):
                m.reset_parameters(alg="identity")

    def forward(self, x: Tensor, super_ps_layers: nn.ModuleList, first_chunk: bool = True) -> Tensor:
        """super_ps_layers: nn.ModuleList passed from each caller"""
        if first_chunk:
            # first half chunk is always used for V
            start_block, end_block = 0, self.sample_n_blocks // 2
        else:
            # second half chunk is always used for U
            start_block, end_block = self.n_blocks // 2, self.n_blocks // 2 + self.sample_n_blocks // 2

        # re-training with fixed arch
        if self.fast_mode:
            for i in range(start_block, end_block):
                index = self.fast_arch_mask[i]
                if index == 1:
                    if super_ps_layers is not None:
                        x = super_ps_layers[i](x)  # pass through independent phase shifters before each block
                    for j in range(self.n_layers_per_block):
                        layer_idx = i * self.n_layers_per_block + j
                        x = self.super_layers_all[layer_idx](x)

            return x

        # supermesh search stage
        if self.training:
            for i in range(start_block, end_block):
                res = x
                if super_ps_layers is not None:
                    x = super_ps_layers[i](x)  # pass through independent phase shifters before each block

                for j in range(self.n_layers_per_block):
                    layer_idx = i * self.n_layers_per_block + j
                    x = self.super_layers_all[layer_idx](x)

                # residual path to skip this block
                if self.arch_mask.dim() == 2:  # scalar gumbel
                    x = self.arch_mask[i, 0] * res + self.arch_mask[i, 1] * x
                else:
                    # x [bs, ....], mask [bs, ]
                    arch_mask = self.arch_mask[:, i, :].view(-1, *([1] * (x.dim() - 1)), 2)
                    x = arch_mask[..., 0] * res + arch_mask[..., 1] * x
        else:  # inference, validation, test
            arch_mask = torch.nn.functional.gumbel_softmax(
                self.sampling_coeff.data,
                tau=self.gumbel_temperature,
                hard=False,
                dim=-1,
            )
            for i in range(start_block, end_block):
                res = x
                if super_ps_layers is not None:
                    x = super_ps_layers[i](x)  # pass through independent phase shifters before each block

                for j in range(self.n_layers_per_block):
                    layer_idx = i * self.n_layers_per_block + j
                    x = self.super_layers_all[layer_idx](x)

                # residual path to skip this block
                x = arch_mask[i, 0] * res + arch_mask[i, 1] * x

        return x

    @lru_cache(maxsize=16)
    def _build_probe_matrix(self, grid_dim_x: int, grid_dim_y: int):
        if self.share_ps == "global":
            eye_U = eye_V = torch.eye(self.n_waveguides, dtype=torch.cfloat, device=self.device)
        elif self.share_ps == "row_col":
            eye_V = (
                torch.eye(self.n_waveguides, dtype=torch.cfloat, device=self.device)
                .unsqueeze(0)
                .expand(grid_dim_x, -1, -1)
                .permute(1, 0, 2)
                .contiguous()
            )  # [k,q,k]
            eye_U = (
                torch.eye(self.n_waveguides, dtype=torch.cfloat, device=self.device)
                .unsqueeze(0)
                .expand(grid_dim_y, -1, -1)
                .permute(1, 0, 2)
                .contiguous()
            )  # [k,p,k]
        elif self.share_ps == "none":
            eye_V = eye_U = (
                torch.eye(self.n_waveguides, dtype=torch.cfloat, device=self.device)
                .unsqueeze(0)
                .unsqueeze(0)
                .expand(grid_dim_y, grid_dim_x, -1, -1)
                .permute(2, 0, 1, 3)
                .contiguous()
            )  # [k,p,q,k]
        return eye_U, eye_V

    def get_UV(
        self, super_ps_layers: nn.ModuleList, grid_dim_x: int, grid_dim_y: int
    ) -> Tuple[Tensor, Tensor]:
        # return U and V
        eye_U, eye_V = self._build_probe_matrix(grid_dim_x, grid_dim_y)

        # print(self.eye_V.size())
        V = self.forward(eye_V, super_ps_layers, first_chunk=True)  # [k,k] or [k,q,k] or [k,p,q,k]
        if V.dim() == 2:
            # [k,k] -> [1,1,k,k]
            V = V.unsqueeze(0).unsqueeze(0)
        elif V.dim() == 3:
            # [k,q,k] -> [1,q,k,k]
            V = V.transpose(0, 1).unsqueeze(0)
        elif V.dim() == 4:
            # [k,p,q,k] -> [p,q,k,k]
            V = V.permute(1, 2, 0, 3)

        U = self.forward(eye_U, super_ps_layers, first_chunk=False)  # [k,k] or [k,p,k] or [k,p,q,k]
        # logger.info(U[:,0,:].conj().t().matmul(U[:,0,:]))
        if U.dim() == 2:
            # [k,k] -> [1,1,k,k]
            U = U.unsqueeze(0).unsqueeze(0)
        elif U.dim() == 3:
            # [k,p,k] -> [p,1,k,k]
            U = U.transpose(0, 1).unsqueeze(1)
        elif U.dim() == 4:
            # [k,p,q,k] -> [p,q,k,k]
            U = U.permute(1, 2, 0, 3)

        ## re-normalization to control the variance the relaxed U and V
        ## after permutaiton relaxation, U, V might not be unitary
        ## this normalization stabilize the statistics
        ## this has no effects on unitary matrices when training converges
        if not self.fast_mode:
            U = U / U.data.norm(p=2, dim=-1, keepdim=True)  # unit row L2 norm
            V = V / V.data.norm(p=2, dim=-2, keepdim=True)  # unit col L2 norm
        return U, V

    def get_weight_matrix(self, super_ps_layers: nn.ModuleList, sigma: Tensor) -> Tensor:
        # sigma [p, q, k], unique parameters for each caller
        # super_ps_layers, unique parameters for each caller

        U, V = self.get_UV(super_ps_layers, grid_dim_x=sigma.size(1), grid_dim_y=sigma.size(0))

        # U [1,1,k,k] or [p,1,k,k] or [p,q,k,k]
        # V [1,1,k,k] or [1,q,k,k] or [p,q,k,k]
        sv = sigma.unsqueeze(-1).mul(V)  # [p,q,k,1]*[p,q,k,k]->[p,q,k,k]
        weight = U.matmul(sv)  # [p,q,k,k] x [p,q,k,k] -> [p,q,k,k]
        return weight

    def fix_layer_solution(self):
        ## fix DC and CR solution
        for m in self.super_layers_all:
            m.fix_arch_solution()

    def fix_block_solution(self):
        self.fast_arch_mask = self.sampling_coeff.argmax(dim=-1)
        self.sampling_coeff.requires_grad_(False)
        self.fast_mode = True


class SuperMeshMZI(SuperMeshADEPT):
    def build_super_layers(self):
        self.share_ps = self.arch.get("share_ps", "global")
        self.interleave_dc = self.arch.get("interleave_dc", True)
        self.symmetry_cr = self.arch.get("symmetry_cr", False)
        super_layers_all = nn.ModuleList()
        ## force the block number
        self.arch["n_blocks"] = self.n_blocks = 4 * self.n_waveguides
        for i in range(self.arch["n_blocks"]):
            layer = SuperDCFrontShareLayer(
                n_waveguides=self.n_waveguides,
                n_front_share_waveguides=self.n_front_share_waveguides,
                offset=(i // 2) % 2,  # 001100110011...
                trainable=False,
                binary=True,
                device=self.device,
            )
            layer.weight.data.fill_(-1)  # after binarization, all DCs are maintained
            super_layers_all.append(layer)
            layer = SuperCRLayer(
                n_waveguides=self.n_waveguides,
                trainable=False,
                symmetry=self.symmetry_cr,
                device=self.device,
            )
            layer.reset_parameters(alg="identity")
            layer.enable_identity_forward()
            super_layers_all.append(layer)

        return super_layers_all


class SuperMeshButterfly(SuperMeshADEPT):
    def build_super_layers(self):
        self.share_ps = self.arch.get("share_ps", "global")
        self.interleave_dc = self.arch.get("interleave_dc", True)
        self.symmetry_cr = self.arch.get("symmetry_cr", False)
        super_layers_all = nn.ModuleList()
        ## force the block number
        self.arch["n_blocks"] = self.n_blocks = int(2 * np.log2(self.n_waveguides))
        for i in range(self.arch["n_blocks"]):
            layer = SuperDCFrontShareLayer(
                    n_waveguides=self.n_waveguides,
                    n_front_share_waveguides=self.n_front_share_waveguides,
                    offset=0,  # 0000...
                    trainable=False,
                    binary=True,
                    device=self.device,
                )
            layer.weight.data.fill_(-1)
            super_layers_all.append(layer)
            layer = SuperCRLayer(
                n_waveguides=self.n_waveguides,
                trainable=False,
                symmetry=self.symmetry_cr,
                device=self.device,
            )
            if i == self.n_blocks // 2 - 1 or i == self.n_blocks - 1:
                layer.reset_parameters(alg="identity")
                layer.enable_identity_forward()
            else:
                forward = i < self.n_blocks // 2
                if forward:
                    level = i % (self.n_blocks // 2)
                else:
                    level = (self.n_blocks//2-2) - i % (self.n_blocks // 2)
                layer.set_butterfly_forward(forward=forward, level=level)
            super_layers_all.append(layer)

        return super_layers_all


super_layer_name_dict = {
    "ps_dc_cr_adept": SuperMeshADEPT,
    "mzi_clements": SuperMeshMZI,
    "butterfly": SuperMeshButterfly,
}
