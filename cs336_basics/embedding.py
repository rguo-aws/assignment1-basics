import torch
from jaxtyping import Float, Int
from torch import Tensor
from torch.nn import Module
import torch.nn.init as init

from einops import einsum

import math

class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype = None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype
        self.weight = torch.nn.Parameter(torch.Tensor(num_embeddings, embedding_dim))
        std = math.sqrt(2.0 / (num_embeddings + embedding_dim))
        init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3.0, b=3.0)

    def forward(self, token_ids: Int[Tensor, "..."]) -> Float[Tensor, "... d_model"]:
        selected = self.weight[token_ids]
        return selected


if __name__ == "__main__":
    x = torch.tensor([[2, 0], [4, 3]])
    embed = Embedding(8, 5)
    output = embed(x)
    print(output)