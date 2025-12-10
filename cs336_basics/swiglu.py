import torch
from torch.nn import Module
import torch.nn.init as init

from einops import einsum

import math

class SwiGLU(Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super(SwiGLU, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.device = device
        self.dtype = dtype
        self.w1_weight = torch.nn.Parameter(torch.Tensor(d_ff, d_model))
        self.w2_weight = torch.nn.Parameter(torch.Tensor(d_model, d_ff))
        self.w3_weight = torch.nn.Parameter(torch.Tensor(d_ff, d_model))
        std = math.sqrt(2.0 / d_model)
        init.trunc_normal_(self.w1_weight, mean=0.0, std=std, a=-3.0, b=3.0)
        init.trunc_normal_(self.w2_weight, mean=0.0, std=std, a=-3.0, b=3.0)
        init.trunc_normal_(self.w3_weight, mean=0.0, std=std, a=-3.0, b=3.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        w1_x = einsum(x, self.w1_weight, "... d_model, d_ff d_model -> ... d_ff")
        w1_sigmoid = torch.sigmoid(w1_x) * w1_x
        w3_x = einsum(x, self.w3_weight, "... d_model, d_ff d_model -> ... d_ff")

        element_prod = w1_sigmoid * w3_x

        y = einsum(element_prod, self.w2_weight, "... d_ff, d_model d_ff -> ... d_model")
        return y