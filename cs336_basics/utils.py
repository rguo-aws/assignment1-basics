import torch
from torch.nn import Module
import torch.nn.init as init

from einops import einsum


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    max_val = x.max(dim=dim, keepdim=True).values
    x_offset = x - max_val

    # Step 2: exponentiate
    e_x = torch.exp(x_offset)

    # Step 3: normalize by sum along dimension
    e_x_sum = e_x.sum(dim=dim, keepdim=True)
    res = e_x / e_x_sum
    return res

if __name__ == "__main__":
    x = torch.randn(2, 3, 4)
    print(x)
    softmax(x, dim=2)