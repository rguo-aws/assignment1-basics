import torch
from torch.nn import Module
from torch import Tensor
import torch.nn.init as init

from einops import einsum, rearrange

from cs336_basics.linear import Linear
from cs336_basics.rmsnorm import RMSNorm
from cs336_basics.swiglu import SwiGLU
from cs336_basics.multihead_self_attention import MultiheadSelfAttention

from jaxtyping import Bool, Float, Int


class TransformerBlock(Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta: float = 0.0, device=None,
                 dtype=None) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.device = device
        self.dtype = dtype

        self.ffn = SwiGLU(d_model=d_model, d_ff=d_ff, device=device, dtype=dtype)
        self.norm1 = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.norm2 = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.attn = MultiheadSelfAttention(d_model=d_model, nums_head=num_heads, max_seq_len=max_seq_len,
                                           apply_rope=True, theta=theta, device=device, dtype=dtype)

    def forward(self, x: Float[Tensor, " batch sequence_length d_model"], token_positions: Int[torch.Tensor, "batch seq_len"]) -> Float[
        Tensor, " batch sequence_length d_model"]:
        layer1_path : Float[Tensor, " batch sequence_length d_model"] = self.norm1(x)
        layer1_path : Float[Tensor, " batch sequence_length d_model"] = self.attn(layer1_path, token_positions)

        layer1_res = x + layer1_path

        layer2_path :Float[Tensor, " batch sequence_length d_model"] = self.norm2(layer1_res)
        layer2_path: Float[Tensor, " batch sequence_length d_model"] = self.ffn(layer2_path)

        layer2_res = layer1_res + layer2_path

        return layer2_res

