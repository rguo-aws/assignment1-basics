import torch
from torch.nn import Module
import torch.nn.init as init

from einops import einsum, repeat

import math

class RotaryPositionalEmbedding(Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super(RotaryPositionalEmbedding, self).__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device

        seq = torch.arange(max_seq_len, dtype=torch.float32)
        dim = torch.arange(d_k // 2, dtype=torch.float32)
        seq = repeat(seq, 'n -> n d', d=d_k // 2)
        dim = repeat(dim, 'd -> n d', n=max_seq_len)
        freqs = seq / (theta ** (2 * dim / d_k))

        sin = torch.sin(freqs)
        cos = torch.cos(freqs)

        # Register buffers so they move with the module and save in state_dict
        self.register_buffer("sin", sin, persistent=False)
        self.register_buffer("cos", cos, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # Gather sin/cos for each token position
        # sin, cos: (max_seq_len, d_k/2)
        sin_pos = self.sin[token_positions]  # shape: (..., seq_len, d_k/2)
        cos_pos = self.cos[token_positions]  # shape: (..., seq_len, d_k/2)

        # Split last dimension into pairs
        x_even = x[..., ::2]  # (..., seq_len, d_k/2)
        x_odd = x[..., 1::2]  # (..., seq_len, d_k/2)

        # Apply rotation for each pair (q0,q1), (q2,q3), ...
        out_even = x_even * cos_pos - x_odd * sin_pos
        out_odd = x_even * sin_pos + x_odd * cos_pos

        # Re-interleave even/odd features
        out = torch.stack((out_even, out_odd), dim=-1)  # (..., seq_len, d_k/2, 2)
        out = out.flatten(-2)  # (..., seq_len, d_k)

        return out


if __name__ == "__main__":
    rope = RotaryPositionalEmbedding(0.2, 6, 10)
    x = torch.randn(10, 6)
    y = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    res = rope.forward(x, y)