from typing import List

import torch
from torch.nn import Module
from torch import Tensor
import torch.nn.init as init

from einops import einsum, rearrange

from cs336_basics.linear import Linear
from cs336_basics.rmsnorm import RMSNorm
from cs336_basics.embedding import Embedding
from cs336_basics.transformer_block import TransformerBlock
from cs336_basics.utils import softmax

from jaxtyping import Bool, Float, Int


class TransformerLM(Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta: float, vocab_size: int,
                 context_length: int, num_layer: int, device=None,
                 dtype=None) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.device = device
        self.dtype = dtype

        self.embedding = Embedding(vocab_size, d_model, device=device, dtype=dtype)

        self.transformers: List[TransformerBlock] = [
            TransformerBlock(d_model=d_model, num_heads=num_heads, d_ff=d_ff, max_seq_len=max_seq_len, theta=theta,
                             device=device, dtype=dtype) for _ in range(num_layer)]

        self.final_norm = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.out_embedding = Linear(in_features=d_model, out_features=vocab_size, device=device, dtype=dtype)

    def forward(self, in_indices: Int[Tensor, "batch_size sequence_length"]) -> Float[
        Tensor, "batch_size sequence_length vocab_size"]:
        token_embeddings: Float[Tensor, "batch_size sequence_length d_model"] = self.embedding(in_indices)

        B, S = in_indices.shape

        position_ids = torch.arange(S, device=in_indices.device).unsqueeze(0)
        position_ids = position_ids.expand(B, S)

        transformer_input = token_embeddings

        for transformer_block in self.transformers:
            transformer_input = transformer_block(transformer_input, position_ids)

        output: Float[Tensor, "batch_size sequence_length d_model"] = self.final_norm(transformer_input)
        output = self.out_embedding(output)
        # no need to run softmax at this point
        #output: Float[Tensor, "batch_size sequence_length vocab_size"] = softmax(output, dim=-1)
        return output

    def init_state(self, idx: int, q_proj: Tensor, k_proj: Tensor, v_proj: Tensor, out_proj: Tensor, norm1: Tensor,
                   norm2: Tensor, ffn_w1: Tensor, ffn_w2: Tensor, ffn_w3: Tensor) -> None:
        assert idx < len(self.transformers)
        transformer = self.transformers[idx]
        transformer.load_state_dict({
            "attn.q_proj.weight": q_proj,
            "attn.k_proj.weight": k_proj,
            "attn.v_proj.weight": v_proj,
            "attn.o_proj.weight": out_proj,
            "norm1.weight": norm1,
            "norm2.weight": norm2,
            "ffn.w1_weight": ffn_w1,
            "ffn.w2_weight": ffn_w2,
            "ffn.w3_weight": ffn_w3,
        })
