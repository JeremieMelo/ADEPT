'''
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2021-09-27 23:48:01
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2022-02-26 02:22:52
'''
import torch
from core.models.layers.super_mesh import super_layer_name_dict

def test():
    device=torch.device("cuda:0")
    p, q, k = 2, 2, 4
    x = torch.eye(k, dtype=torch.cfloat, device=device).unsqueeze(0).repeat(q,1,1).permute(1,0,2).contiguous()
    sigma = torch.ones(p,q,k, device=device)
    # x [bs, q, k]

    arch = dict(
        n_waveguides=k,
        n_front_share_waveguides=k,
        n_front_share_ops=k,
        n_blocks=4,
        n_layers_per_block=2,
        n_front_share_blocks=2,
        share_ps="row_col",
        interleave_dc=True,
    )
    sample_arch = [
        k//3,1,
        k//2,1,
        k//2,1,
        k//2,1,
        4
    ]
    layer = super_layer_name_dict["ps_dc_cr"](arch, device=device)
    super_ps_layers = layer.build_ps_layser(grid_dim_x=q, grid_dim_y=p)
    for m in super_ps_layers:
        # m.reset_parameters(alg="identity")
        m.reset_parameters(alg="uniform")
    layer.set_sample_arch(sample_arch)
    print(layer)
    layer.set_identity_cr()
    layer.build_sampling_coefficients()
    layer.set_gumbel_temperature(0.1)
    layer.set_aux_skip_path(0)
    layer.build_arch_mask()
    U,V = layer.get_UV(super_ps_layers, q, p)
    print(U, U.size())
    print(U[0,0].conj().t().matmul(U[0,0]))
    print(V)
    print(V[0,0].conj().t().matmul(V[0,0]))
    weight = layer.get_weight_matrix(super_ps_layers, sigma)
    print(weight)
    weight.sum().backward()
    print(super_ps_layers[0].weight.grad.norm(p=2))
    print(layer.super_layers_all[0].weight.grad.norm(p=2))

    print(layer.super_layers_all[1].weight.grad.norm(p=2))


if __name__ == "__main__":
    test()
