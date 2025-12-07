import torch
from torch.nn import Module
import torch.nn.init as init

from einops import einsum
from einops import reduce

import math

class RMSNorm(Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super(RMSNorm, self).__init__()
        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype
        self.weight = torch.nn.Parameter(torch.Tensor(d_model))
        std = math.sqrt(2.0 / d_model)
        init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3.0, b=3.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        x_sq = x ** 2
        x_sq_sum = reduce(x_sq, '... a -> ... 1', 'sum')
        norm_vector =  1.0 / torch.sqrt(x_sq_sum / self.d_model + self.eps)

        y = einsum(x, self.weight, "... d_model, d_model -> ... d_model")

        res = einsum(y, norm_vector, "... d_model, ... sum -> ... d_model")
        return res.to(in_dtype)

if __name__ == "__main__":
    x = torch.tensor([[2, 0, 3], [4, 3, 8]])
    embed = RMSNorm(3)
    output = embed(x)
    #print(output)