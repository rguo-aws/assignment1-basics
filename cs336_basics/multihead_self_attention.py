import torch
from torch.nn import Module
import torch.nn.init as init

from einops import einsum, rearrange

from cs336_basics.utils import scaled_dot_product_attention
from cs336_basics.linear import Linear
from cs336_basics.rope import RotaryPositionalEmbedding

from jaxtyping import Bool, Float, Int


class MultiheadSelfAttention(Module):
    def __init__(self, d_model: int, nums_head: int, max_seq_len: int, apply_rope: bool = False, theta: float = 0.0,
                 device=None, dtype=None) -> None:
        super().__init__()
        self.d_model = d_model
        self.nums_head = nums_head
        self.d_k = d_model // nums_head
        self.max_seq_len = max_seq_len
        self.apply_rope = apply_rope
        self.theta = theta
        self.device = device
        self.dtype = dtype

        factory_kwargs = {'device': device, 'dtype': dtype}
        self.q_proj, self.k_proj, self.v_proj, self.o_proj = [Linear(d_model, d_model, **factory_kwargs)
                                                              for _ in range(4)]

        mask = torch.triu(
            torch.ones(self.max_seq_len, self.max_seq_len, dtype=torch.bool, device=device),
            diagonal=1
        )
        mask = ~mask
        self.register_buffer("causal_mask", mask.unsqueeze(0).unsqueeze(0), persistent=False)

        if self.apply_rope:
            self.rope = RotaryPositionalEmbedding(theta, self.d_k, self.max_seq_len, device=self.device)

    def forward(self, x: Float[torch.Tensor, " ... sequence_length d_model"],
                token_positions: Int[torch.Tensor, "batch seq_len"] | None = None) -> Float[
        torch.Tensor, " ... sequence_length d_model"]:
        # q = einsum(in_features, self.q_proj_weight, "... sequence_length d_model, d_model d_model -> ... sequence_length d_model")
        # k = einsum(in_features, self.k_proj_weight, "... sequence_length d_model, d_model d_model -> ... sequence_length d_model")
        # v = einsum(in_features, self.v_proj_weight, "... sequence_length d_model, d_model d_model -> ... sequence_length d_model")
        B, S, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q_transformed = rearrange(q, "... sequence_length (h d_k) -> ... h sequence_length d_k", h=self.nums_head,
                                  d_k=self.d_k)
        k_transformed = rearrange(k, "... sequence_length (h d_k) -> ... h sequence_length d_k", h=self.nums_head,
                                  d_k=self.d_k)
        v_transformed = rearrange(v, "... sequence_length (h d_k) -> ... h sequence_length d_k", h=self.nums_head,
                                  d_k=self.d_k)

        if self.apply_rope:
            q_transformed = self.rope(q_transformed, token_positions)
            k_transformed = self.rope(k_transformed, token_positions)

        attention_dot = scaled_dot_product_attention(q_transformed, k_transformed, v_transformed, self.causal_mask)
        attention_transformed = rearrange(attention_dot, "... h sequence_length d_k -> ... sequence_length (h d_k)")
        # issue found in the line below
        # multihead = einsum(attention_transformed, self.o_proj_weight, "... sequence_length d_model, d_model d_model -> ... sequence_length d_model")
        multihead = self.o_proj(attention_transformed)
        return multihead

    def forward2(self, x: torch.Tensor) -> torch.Tensor:
        B, S, _ = x.shape

        # Project to multi-head Q, K, V
        q, k, v = [rearrange(proj(x), "b s (h d) -> b h s d", h=self.nums_head)
                   for proj in [self.q_proj, self.k_proj, self.v_proj]]

        # Compute attention
        out = scaled_dot_product_attention(q, k, v, mask=self.causal_mask[..., :S, :S])

        # Merge heads and project
        out = rearrange(out, "b h s d -> b s (h d)")
        return self.o_proj(out)


if __name__ == "__main__":
    q = torch.randn(2, 4, 3, 3)
    k = torch.randn(2, 4)
    v = torch.randn(2, 4)
    attention = MultiheadSelfAttention(4, 3)
    output = attention(q, k, v)
    # print(output)
