'''
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2021-09-28 05:00:55
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2021-10-14 01:18:34
'''
import torch
from core.models import SuperOCNN


def test():
    device = torch.device("cuda:0")
    x = torch.randn(1, 3, 8, 8, device=device)
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
    sample_arch = [
        2,1,
        2,1,
        2,1,
        2,1,
        4
    ]
    model = SuperOCNN(
        8,
        8,
        in_channels=3,
        num_classes=2,
        kernel_list=[3],
        kernel_size_list=[3],
        stride_list=[1],
        padding_list=[1],
        hidden_list=[],
        block_list=[4, 4],
        photodetect=True,
        super_layer_name="ps_dc_cr",
        super_layer_config=arch,
        device=device,
    ).to(device)
    # model.build_super_layer("ps_dc_cr", arch=arch, device=device)
    model.super_layer.set_sample_arch(sample_arch)
    # model.set_super_layer_transfer_matrix()
    y = model(x)
    print(y)
    y.mean().backward()
    # model.classifier[0].linear.U.real.mean().backward()
    print(model.classifier[0].linear.weight.grad.norm(p=2))


if __name__ == "__main__":
    test()
