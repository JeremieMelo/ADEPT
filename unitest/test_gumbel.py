'''
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2021-11-09 00:36:32
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2021-11-09 00:57:55
'''

import torch

def test_gumbel():
    theta = torch.nn.Parameter(torch.ones(3))
    theta.data[0] = 0.3
    theta.data[1] = 0.3
    theta.data[2] = 0.4
    mask = torch.nn.functional.gumbel_softmax(
                torch.log(theta),
                tau=1,
                hard=False,
                dim=-1,
            )
    # mask = torch.softmax(theta, dim=-1)
    x = torch.randn(4, 4).abs()
    U1 = torch.randn(4, 4).abs()
    U2 = torch.randn(4, 4).abs()
    y1 = U1.matmul(x)
    y2 = U2.matmul(x)
    z = mask[0] * x + mask[1] * y1 + mask[2] * y2

    z.sum().backward()
    print(theta)
    print(mask)
    print(theta.grad)

if __name__ == "__main__":
    test_gumbel()
