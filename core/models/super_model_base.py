"""
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2021-10-07 03:37:23
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2021-10-14 00:27:33
"""

from typing import Dict, List, Optional

import numpy as np
import torch
from pyutils.general import logger
from pyutils.torch_train import set_torch_deterministic
from torch import nn

from .layers import SuperBlockConv2d, SuperBlockLinear
from .layers.super_mesh import SuperCRLayer, SuperDCFrontShareLayer, super_layer_name_dict
from .layers.utils import GradientMask

__all__ = ["SuperModel_CLASS_BASE"]


class SuperModel_CLASS_BASE(nn.Module):
    _conv_linear = (
        SuperBlockConv2d,
        SuperBlockLinear,
    )

    def __init__(
        self,
        *args,
        super_layer_name: str = "ps_dc_cr",
        super_layer_config: Dict = {},
        device=torch.device("cuda:0"),
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.area_multiplier = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.area_aux_variable = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.area = torch.tensor(0)

    def adjust_min_max_blocks(self, arch: Dict = {}):
        ps_weight = arch["device_cost"]["ps_weight"]
        dc_weight = arch["device_cost"]["dc_weight"]
        cr_weight = arch["device_cost"]["cr_weight"]
        upper_bound = arch["device_cost"]["area_upper_bound"]
        ## we assume the min area for each layer consists of k PS, 1 DC, and 0 CR
        n_waveguides = self.block_list[0]
        min_area_per_layer = ps_weight * n_waveguides + dc_weight * 1 + cr_weight * 0
        max_area_per_layer = (
            ps_weight * n_waveguides
            + dc_weight * (n_waveguides // 2)
            + cr_weight * ((n_waveguides // 2) * (n_waveguides // 2 - 1) // 2)
        )
        ## calculate max block
        n_blocks = int(upper_bound / min_area_per_layer)
        n_blocks = n_blocks if n_blocks % 2 == 0 else n_blocks + 1  # cast to next even number
        ## calculate min block
        n_front_share_blocks = int(upper_bound * 0.7 / max_area_per_layer)
        n_front_share_blocks = (
            n_front_share_blocks if n_front_share_blocks % 2 == 0 else n_front_share_blocks - 1
        )
        old_n_blocks = arch["n_blocks"]
        old_n_front_share_blocks = arch["n_front_share_blocks"]
        n_blocks = min(n_blocks, old_n_blocks)
        n_front_share_blocks = min(n_blocks, max(n_front_share_blocks, old_n_front_share_blocks))
        arch["n_blocks"] = n_blocks
        arch["n_front_share_blocks"] = n_front_share_blocks
        logger.info(
            f"Max block number 'n_blocks' is adjusted from {old_n_blocks} to {n_blocks} due to area constraint: A <= {upper_bound}, min block size = {min_area_per_layer}"
        )
        logger.info(
            f"Min block number 'n_front_share_blocks' is adjusted from {old_n_front_share_blocks} to {n_front_share_blocks} due to area constraint: A <= {upper_bound}, max block size = {max_area_per_layer}"
        )

    def build_super_layer(self, name: str, *args, **kwargs):
        ## must be called after build_layers()
        # only one super layer since we need to share DC and CR layers
        self.adjust_min_max_blocks(kwargs["arch"])
        self.super_layer = super_layer_name_dict[name](*args, **kwargs)
        self.super_layer.build_sampling_coefficients()
        for m in self.super_layer.super_layers_all:
            if isinstance(m, SuperCRLayer) and m.weight.requires_grad:
                m.reset_parameters(alg=kwargs["arch"]["cr_layer_init_alg"])

        for m in self.modules():
            ## build independent ps layers for each CONV/Linear layer
            if isinstance(m, (self._conv_linear)):
                m.super_layer = self.super_layer
                m.super_ps_layers = self.super_layer.build_ps_layers(m.grid_dim_x, m.grid_dim_y)

        self._total_trainable_parameters = set([p for p in self.parameters() if p.requires_grad])
        self.partition_parameters()

    def partition_parameters(self, arch_param_list=["theta"]):
        ## collect architecture parameters
        self.arch_params = []
        if "theta" in arch_param_list:
            self.arch_params.append(self.super_layer.sampling_coeff)
        if "perm" in arch_param_list:
            for layer in self.super_layer.super_layers_all:
                if isinstance(layer, (SuperCRLayer,)):
                    if layer.weight.requires_grad:
                        self.arch_params.append(layer.weight)
        if "dc" in arch_param_list:
            for layer in self.super_layer.super_layers_all:
                if isinstance(layer, (SuperDCFrontShareLayer,)):
                    if layer.weight.requires_grad:
                        self.arch_params.append(layer.weight)
        self.weight_params = list(self._total_trainable_parameters - set(self.arch_params))

    def set_super_layer_transfer_matrix(self):
        x = torch.eye(self.super_layer.n_waveguides, device=self.device, dtype=torch.cfloat)
        n_blocks = self.super_layer.n_blocks
        V = self.super_layer.forward(x, start_block=0, end_block=n_blocks // 2)
        U = self.super_layer.forward(x, start_block=n_blocks // 2, end_block=n_blocks)
        for m in self.modules():
            if isinstance(m, self._conv_linear):
                m.set_super_layer_transfer_matrices(U, V)

    def set_sample_arch(self, sample_arch: List) -> None:
        if getattr(self, "super_layer", None):
            self.super_layer.set_sample_arch(sample_arch)

    @property
    def arch_space(self) -> List:
        space = [layer.arch_space for layer in self.super_layer.super_layers_all]
        # for the number of sampled blocks
        space.append(
            list(range(self.super_layer.n_front_share_blocks, self.super_layer.n_blocks + 1, 2))
        )  # n_sample_block must be even number
        return space

    def get_parameters(self, name_list=[]):
        params = []
        for name in name_list:
            if name == "theta":
                params.append(self.super_layer.sampling_coeff)
            elif name == "weight":
                for m in self.modules():
                    if isinstance(m, self._conv_linear):
                        params.append(m.weight)
                        if m.bias is not None:
                            params.append(m.bias)
                    elif isinstance(m, torch.nn.BatchNorm2d):
                        params.append(m.weight)
                        params.append(m.bias)
            elif name == "ps_phi":
                for m in self.modules():
                    if isinstance(m, self._conv_linear):
                        params += [i.weight for i in m.super_ps_layers]
            elif name == "dc_t":
                for layer in self.super_layer.super_layers_all:
                    if isinstance(layer, SuperDCFrontShareLayer) and layer.trainable:
                        params.append(layer.weight)
            elif name == "cr_p":
                for layer in self.super_layer.super_layers_all:
                    if isinstance(layer, SuperCRLayer) and layer.trainable:
                        params.append(layer.weight)
        return params

    def get_perm_loss(self):
        loss = []
        for layer in self.super_layer.super_layers_all:
            if hasattr(layer, "get_perm_loss"):
                loss.append(layer.get_perm_loss().detach().data.item())
        return loss

    def get_alm_perm_loss(self, rho: float = 0.1):
        loss = 0
        for layer in self.super_layer.super_layers_all:
            if hasattr(layer, "get_alm_perm_loss"):
                loss = loss + layer.get_alm_perm_loss(rho=rho)
        return loss

    def update_alm_multiplier(self, rho: float = 0.1, max_lambda: Optional[float] = None):
        for layer in self.super_layer.super_layers_all:
            if hasattr(layer, "update_alm_multiplier"):
                layer.update_alm_multiplier(rho=rho, max_lambda=max_lambda)

    def get_alm_multiplier(self):
        return [
            layer.alm_multiplier.data.mean().item()
            for layer in self.super_layer.super_layers_all
            if hasattr(layer, "alm_multiplier")
        ]

    def _find_first_active_block(self, mask):
        # first active block to be penalized is either in U or V , depends on which unitary has deeper structures
        with torch.no_grad():
            n_blk = int(np.ceil(mask.size(0) / 2))
            blk_U = mask[:n_blk]
            blk_V = mask[n_blk:]
            blk_U_sum = blk_U.sum()
            blk_V_sum = blk_V.sum()
            if blk_U_sum > blk_V_sum:  # penalize first blk in U
                i = 0
                while i < n_blk:
                    if mask[i] > 0.5:  # 1 in boolean mask, larger than 0.5 in soft mask
                        return i
                    i += 1
            elif blk_U_sum < blk_V_sum:  # penalize first blk in V
                i = n_blk
                while i < mask.size(0):
                    if mask[i] > 0.5:
                        return i
                    i += 1
            else:
                if blk_U_sum == 0:  # no active block
                    return None  # no penalty
                i = 0
                while i < n_blk:  # same depth, penalize U
                    if mask[i] > 0.5:
                        return i
                    i += 1

    def get_crossing_density_loss(self, margin=1):
        permutation_list = []
        n_crossings = []
        n_waveguides = self.block_list[0]
        max_crossing_density = (n_waveguides // 2) * (n_waveguides // 2 - 1) // 2
        for layer in self.super_layer.super_layers_all[:-1]:  # remove the last pseudo-permutation
            if isinstance(layer, SuperCRLayer):
                permutation_list.append(layer.build_weight())
                n_crossings.append(layer.get_num_crossings())
        self.num_crossings = n_crossings
        loss = torch.zeros(1, device=self.device)
        eye = torch.eye(n_waveguides, dtype=torch.float, device=self.device)
        for n_cr, p in zip(n_crossings, permutation_list):
            if n_cr > max_crossing_density * margin:
                loss = loss + torch.nn.functional.mse_loss(p, eye)
        return loss

    def get_crossing_loss(
        self,
        alg: str = "mse",
        crossing_weight: float = 1.0,
        arch_mask=None,
        first_active_block_idx: Optional[int] = None,
    ):
        if alg in {"kl", "mse"}:
            loss = []
            arch_mask = arch_mask[:-1]
            for layer in self.super_layer.super_layers_all[:-1]:  # remove the last permutation
                if hasattr(layer, "get_crossing_loss"):
                    loss.append(layer.get_crossing_loss(alg=alg))
            loss = torch.stack(loss).dot(arch_mask)
            return loss
        else:
            raise NotImplementedError(f"Only support alg = (kl, mse), but got {alg}")

    def get_dc_loss(
        self, dc_weight: float = 1.0, arch_mask=None, first_active_block_idx: Optional[int] = None
    ):
        # first active block:
        #   int >=0 : only penalize the first active block
        #   -1: no penalty
        #   None: penalize all
        weight_list = []
        for layer in self.super_layer.super_layers_all:
            if isinstance(layer, SuperDCFrontShareLayer):
                weight_list.append(
                    layer.weight_quantizer(layer.weight).mul(2 / (2 ** 0.5 - 2)).add(2 / (2 - 2 ** 0.5)).sum()
                )  # {sqrt(2)/2, 1} -> {1, 0}
        weight_list = torch.stack(weight_list)
        # arch_mask = self.super_layer.arch_mask[:, 1]
        # arch_mask = GradientMask.apply(arch_mask, first_active_block_idx)

        return weight_list.dot(arch_mask).mul(dc_weight)

    def get_ps_loss(
        self, ps_weight: float = 1.0, arch_mask=None, first_active_block_idx: Optional[int] = None
    ):

        # arch_mask = self.super_layer.arch_mask[:, 1]
        # arch_mask = GradientMask.apply(arch_mask, first_active_block_idx)
        return arch_mask.sum().mul(self.block_list[0] * ps_weight)

    def get_area_bound_loss(
        self,
        ps_weight: float = 1.0,
        dc_weight: float = 1.0,
        cr_weight: float = 1.0,
        upper_bound: float = 100,
        lower_bound: float = 70,
        first_active_block: bool = False,
    ):
        if first_active_block:
            first_active_block_idx = self._find_first_active_block(self.super_layer.arch_mask.data[:, 1])
        else:
            first_active_block_idx = None
        arch_mask = GradientMask.apply(
            self.super_layer.arch_mask[..., 1],
            first_active_block_idx,
            self.super_layer.arch_mask.size(0),  # scale the penalty
        )
        ps_loss = self.get_ps_loss(ps_weight, arch_mask=arch_mask)
        dc_loss = self.get_dc_loss(dc_weight, arch_mask=arch_mask)  # .detach()
        cr_loss = self.get_crossing_loss(alg="mse", crossing_weight=cr_weight, arch_mask=arch_mask)
        with torch.no_grad():
            cr_area_soft = self.get_num_crossings_soft(arch_mask=arch_mask)[1] * cr_weight
        self.area = ps_loss.data + dc_loss.data + cr_area_soft
        area_loss = ps_loss + dc_loss + 100 * cr_loss
        if self.area.item() > upper_bound * 0.95:
            loss = area_loss / (upper_bound * 0.95) - 1  # penalize area violation with a margin
            return loss
        elif self.area.item() < lower_bound * 1.05:
            loss = 1 - area_loss / (lower_bound * 1.05)
            return loss
        else:
            return torch.zeros_like(area_loss)  # accelerate BP

    def update_area_aux_variable(
        self,
        ps_weight: float = 1.0,
        dc_weight: float = 1.0,
        cr_weight: float = 1.0,
        upper_bound: float = 100,
        rho: float = 0.1,
    ):
        with torch.no_grad():
            ps_loss = self.get_ps_loss(ps_weight).detach()
            dc_loss = self.get_dc_loss(dc_weight).detach()
            cr_loss = self.get_crossing_loss(alg="nn", crossing_weight=cr_weight).detach()
            self.updated_area_margin = ps_loss + dc_loss + cr_loss - upper_bound

        self.area_aux_variable.data.fill_(
            max(0, -(self.updated_area_margin + self.area_multiplier / rho).item())
        )

    def update_area_multiplier(self, rho: float = 0.1):
        self.area_multiplier.data += rho * (self.updated_area_margin + self.area_aux_variable)

    def get_area_multiplier(self):
        return self.area_multiplier.data

    def build_sampling_coefficient(self):
        self.super_layer.build_sampling_coefficients()

    def set_gumbel_temperature(self, T: float = 5.0):
        self.super_layer.set_gumbel_temperature(T)

    def build_arch_mask(self, mode="random", batch_size: int = 32):
        self.super_layer.build_arch_mask(mode=mode, batch_size=batch_size)

    def get_num_crossings(self):
        if getattr(self, "num_crossings", None) is not None:
            return self.num_crossings, sum(self.num_crossings)
        n_cr = []
        for layer in self.super_layer.super_layers_all:
            if hasattr(layer, "get_num_crossings"):
                n_cr.append(layer.get_num_crossings())
        total = sum(n_cr)
        return n_cr, total

    def get_num_crossings_soft(self, arch_mask=None):
        n_cr = []
        for layer in self.super_layer.super_layers_all:
            if hasattr(layer, "get_num_crossings"):
                n_cr.append(layer.get_num_crossings())
        total = sum(i * j for i, j in zip(n_cr, arch_mask))
        return n_cr, total

    def get_perm_matrix(self):
        with torch.no_grad():
            return [
                layer.build_weight().detach().data
                for layer in self.super_layer.super_layers_all
                if hasattr(layer, "alm_multiplier")
            ]

    def get_num_dc(self):
        n_dc = [
            int((layer.weight.data < 0).float().sum().item())
            for layer in self.super_layer.super_layers_all
            if isinstance(layer, SuperDCFrontShareLayer)
        ]
        return n_dc, sum(n_dc)

    def reset_parameters(self, random_state: int = None) -> None:
        for name, m in self.named_modules():
            if isinstance(m, self._conv_linear):
                if random_state is not None:
                    # deterministic seed, but different for different layer, and controllable by random_state
                    set_torch_deterministic(random_state + sum(map(ord, name)))
                m.reset_parameters()
            elif isinstance(m, nn.BatchNorm2d) and m.affine:
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def train(self, mode: bool = True):
        super().train(mode)

    def enable_arch_params(self):
        for p in self.arch_params:
            p.requires_grad_(True)

    def freeze_arch_params(self):
        for p in self.arch_params:
            p.requires_grad_(False)

    def enable_weight_params(self):
        for p in self.weight_params:
            p.requires_grad_(True)

    def freeze_weight_params(self):
        for p in self.weight_params:
            p.requires_grad_(False)

    def set_phase_noise(self, noise_std: float = 0.0):
        for m in self.modules():
            if isinstance(m, self._conv_linear):
                for layer in m.super_ps_layers:
                    layer.set_phase_noise(noise_std=noise_std)

    def set_dc_noise(self, noise_std: float = 0.0):
        for m in self.super_layer.super_layers_all:
            if isinstance(m, SuperDCFrontShareLayer):
                m.set_dc_noise(noise_std=noise_std)

    def load_arch_solution(self, checkpoint):
        logger.info(f"Loading architecture solution from {checkpoint} ...")
        state_dict = torch.load(checkpoint, map_location=self.device)
        state_dict = state_dict.get("state_dict", state_dict)
        state_dict_new = {}
        for name, p in state_dict.items():
            if name.startswith("super_layer."):
                state_dict_new[name[12:]] = p
        self.super_layer.load_state_dict(state_dict_new)

    def fix_arch_solution(self):
        logger.info("Fix DC and CR layer solution...")
        self.super_layer.fix_layer_solution()
        logger.info("Fix Block solution...")
        self.super_layer.fix_block_solution()
        self.super_layer.set_gumbel_temperature(0.001)
        self.super_layer.build_arch_mask(mode="softmax")

    def sample_submesh(
        self,
        n_samples: int = 100,
        ps_weight: float = 1.0,
        dc_weight: float = 1.0,
        cr_weight: float = 1.0,
        upper_bound: float = 100,
        lower_bound: float = 80,
    ):
        logger.info("Fix DC and CR layer solution...")
        self.super_layer.fix_layer_solution()
        with torch.no_grad():
            num_crossings = self.get_num_crossings()
            num_dc = self.get_num_dc()
            logger.info(f"num CRs: {num_crossings}")
            logger.info(f"num DCs: {num_dc}")
        logger.info("Search feasible theta...")
        solution = None
        import tqdm

        with torch.no_grad():
            area_list = []
            theta = self.super_layer.sampling_coeff.data.clone()
            distribution = torch.softmax(theta, dim=-1).cpu().numpy()
            n_blocks = distribution.shape[0]
            n_ops = theta.size(-1)
            for i in tqdm.tqdm(range(n_samples)):
                for j in range(n_blocks):
                    op = np.random.choice(n_ops, p=distribution[j, :])
                    self.super_layer.sampling_coeff.data[j] = -1000
                    self.super_layer.sampling_coeff.data[j, op] = 1000
                self.super_layer.set_gumbel_temperature(2 ** (i / n_samples))
                self.super_layer.build_arch_mask(mode="softmax")
                area_loss = (
                    self.get_area_bound_loss(
                        ps_weight=ps_weight,
                        dc_weight=dc_weight,
                        cr_weight=cr_weight,
                        upper_bound=upper_bound,
                        lower_bound=lower_bound,
                        first_active_block=False,
                    )
                    .detach()
                    .data.item()
                )
                area_list.append(self.area.item())
                # if area_loss < 1e-8:  # meet area bound constraints
                if lower_bound <= self.area.item() <= upper_bound:  # meet area bound constraints
                    solution = (
                        self.area.item(),
                        self.super_layer.sampling_coeff.data.argmax(dim=-1).cpu().numpy().tolist(),
                        num_crossings,
                        num_dc,
                    )
                    break

                self.super_layer.sampling_coeff.data.copy_(theta)

            else:
                logger.info(
                    f"No feasible submesh found. Area ranges: [{np.min(area_list), np.max(area_list)}], which violates area constraints [{lower_bound}, {upper_bound}]"
                )
                solution = (None, None, None, None)
        total_cr = int(sum(i * j for i, j in zip(solution[2][0], solution[1])))
        total_dc = int(sum(i * j for i, j in zip(solution[3][0], solution[1])))
        ps_solution = [self.super_layer.n_waveguides] * len(solution[1])
        total_ps = int(sum(i * j for i, j in zip(ps_solution, solution[1])))
        logger.info(f"Found possible solution: \n")
        logger.info(f"\t      Area = {solution[0]:.4f}")
        logger.info(f"\tBlock mask = {solution[1]}")
        logger.info(
            f"\t        CR = {solution[2][0]}, total #CR = {total_cr:4d}, total CR area = {total_cr*cr_weight:.4f}"
        )
        logger.info(
            f"\t        DC = {solution[3][0]}, total #DC = {total_dc:4d}, total DC area = {total_dc*dc_weight:.4f}"
        )
        logger.info(
            f"\t        PS = {ps_solution}, total #PS = {total_ps:4d}, total PS area = {total_ps*ps_weight:.4f}"
        )
        logger.info("Fix arch solution and enable fast mode")
        assert (solution[0] - (total_cr * cr_weight + total_dc * dc_weight + total_ps * ps_weight)) / (
            solution[0]
        ) < 0.01
        self.super_layer.fix_block_solution()

    def check_perm(self):
        with torch.no_grad():
            res = []
            for m in self.super_layer.super_layers_all:
                if isinstance(m, SuperCRLayer):
                    res.append(m.check_perm(m.build_weight().detach().data.argmax(dim=-1)))
            return res

    def sinkhorn_perm(self, n_step=10, t_min=0.1, noise_std=0.01, svd=True, legal_mask=None):
        with torch.no_grad():
            i = 0
            for m in self.super_layer.super_layers_all:
                if isinstance(m, SuperCRLayer):
                    legal = legal_mask[i]
                    if True:  # not legal:
                        w = m.build_weight().data.abs()
                        # logger.info(f"Layer {i}: {w}")
                        # w = sinkhorn(w, n_step=n_step, t_min=t_min, noise_std=0, svd=False)
                        # w = w.div(0.01).softmax(dim=-1)
                        # logger.info(f"Layer {i}: {w}")
                        # logger.info(w.sum(dim=-2))
                        # logger.info(w.sum(dim=-1))
                        # logger.info(w)
                        # w = sinkhorn(w, n_step=n_step, t_min=t_min, noise_std=0.1, svd=True)
                        # w = sinkhorn(w, n_step=n_step, t_min=t_min, noise_std=0, svd=False)
                        w = m.unitary_projection(w, n_step=n_step, t=t_min, noise_std=noise_std)
                        # logger.info(f"Layer {i}: {w}")
                        # logger.info(w)
                        # logger.info(w.sum(dim=-2))
                        # logger.info(w.sum(dim=-1))
                        m.weight.data.copy_(w)
                    i += 1

    def forward(self, x):
        raise NotImplementedError
