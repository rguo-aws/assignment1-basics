import torch
from torch.nn import Module
import torch.nn.init as init

from einops import einsum

import math

class Linear(Module):
    def __init__(self, in_features: int, out_features: int, device =None, dtype = None) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype
        self.weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        std = math.sqrt(2.0 / (in_features + out_features))
        init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3.0, b=3.0)

        #print(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # y = W @ x
        y = einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")
        return y



if __name__ == "__main__":
    x = torch.randn(2, 4)
    linear = Linear(4, 3)
    output = linear(x)
    print(output)