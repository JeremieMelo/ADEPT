'''
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2021-10-14 01:25:36
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2021-10-14 01:25:36
'''

import torch
from core.models.layers.super_conv2d import SuperBlockConv2d
from pyutils.general import print_stat
from core.models.layers.super_mesh import super_layer_name_dict


def test():
    device = torch.device("cuda:0")
    arch = dict(
        n_waveguides=4,
        n_front_share_waveguides=4,
        n_front_share_ops=4,
        n_blocks=4,
        n_layers_per_block=2,
        n_front_share_blocks=4,
        share_ps="row_col",
        interleave_dc=True,
    )
    sample_arch = [2, 1, 2, 1, 2, 1, 2, 1, 4]
    super_layer = super_layer_name_dict["ps_dc_cr"](arch, device=device)

    x = torch.randn(1, 3, 8, 8, device=device).clamp(0)
    layer = SuperBlockConv2d(3, 4, 3, mini_block=4, photodetect=True, super_layer=super_layer)
    super_layer.set_sample_arch(sample_arch)
    super_ps_layers = super_layer.build_ps_layser(grid_dim_x=layer.grid_dim_x, grid_dim_y=layer.grid_dim_y)
    layer.super_ps_layers = super_ps_layers
    y = layer(x)
    print(y)
    print_stat(x)
    print_stat(y)
    y.mean().backward()
    print(layer.weight.grad.norm(p=2))


if __name__ == "__main__":
    test()
