'''
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2021-11-15 21:13:04
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2022-02-26 02:25:43
'''
import numpy as np
import torch
from core.models.layers.super_mesh import SuperCRLayer, SuperDCFrontShareLayer, super_layer_name_dict


def set_noise(super_layer, super_ps_layers, phase_noise_std=0, dc_noise_std=0):
    for m in super_layer.super_layers_all:
        if isinstance(m, SuperDCFrontShareLayer):
            m.set_dc_noise(phase_noise_std)
    for m in super_ps_layers:
        m.set_phase_noise(dc_noise_std)


def test_robustness(p=4, q=4, k=8, n_blocks=4):
    device = torch.device("cuda:0")
    x = (
        torch.eye(k, dtype=torch.cfloat, device=device)
        .unsqueeze(0)
        .repeat(q, 1, 1)
        .permute(1, 0, 2)
        .contiguous()
    )
    sigma = torch.ones(p, q, k, device=device)
    # x [bs, q, k]

    arch = dict(
        n_waveguides=k,
        n_front_share_waveguides=k,
        n_front_share_ops=k,
        n_blocks=n_blocks,
        n_layers_per_block=2,
        n_front_share_blocks=2,
        share_ps="row_col",
        interleave_dc=True,
    )
    sample_arch = []
    for _ in range(n_blocks):
        sample_arch.append(k // 2)
        sample_arch.append(1)
    sample_arch.append(n_blocks)
    layer = super_layer_name_dict["ps_dc_cr_adept"](arch, device=device)
    super_ps_layers = layer.build_ps_layers(grid_dim_x=q, grid_dim_y=p)
    for m in super_ps_layers:
        # m.reset_parameters(alg="identity")
        m.reset_parameters(alg="uniform")
    layer.set_sample_arch(sample_arch)
    layer.set_identity_cr()
    for m in layer.super_layers_all:
        if isinstance(m, SuperCRLayer):
            m.reset_parameters(alg="identity")
    layer.build_sampling_coefficients()
    layer.set_gumbel_temperature(0.1)
    layer.set_aux_skip_path(0)
    layer.build_arch_mask(mode="largest")
    set_noise(super_layer=layer, super_ps_layers=super_ps_layers, phase_noise_std=0, dc_noise_std=0)
    weight_gt = layer.get_weight_matrix(super_ps_layers, sigma)
    # print(weight)
    c = 500
    phase_noise_std = 0.02
    dc_noise_std = 0.02
    set_noise(
        super_layer=layer,
        super_ps_layers=super_ps_layers,
        phase_noise_std=phase_noise_std,
        dc_noise_std=dc_noise_std,
    )
    error = []
    for _ in range(c):
        weight = layer.get_weight_matrix(super_ps_layers, sigma)
        e = (weight - weight_gt).norm(p=2).square().div(weight_gt.norm(p=2).square())
        error.append(e.item())

    print(
        f"Average error (p={p}, q={q}, k={k}, n_blocks={n_blocks}, ps_noise={phase_noise_std}, dc_noise={dc_noise_std}): {np.mean(error):.6f} ({np.std(error):.4f})"
    )


if __name__ == "__main__":
    p, q, k = 1, 1, 8
    for n_blocks in [2, 4, 6, 8, 10, 12, 14, 16]:
        test_robustness(p, q, k, n_blocks)
