from typing import Iterable

import torch
from torch import Tensor
from jaxtyping import Float, Int
from torch.nn import Module
import torch.nn.init as init

from einops import einsum

import math


def softmax(x: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    max_val = x.max(dim=dim, keepdim=True).values
    x_offset = x - max_val

    # Step 2: exponentiate
    e_x = torch.exp(x_offset)

    # Step 3: normalize by sum along dimension
    e_x_sum = e_x.sum(dim=dim, keepdim=True)
    res = e_x / e_x_sum
    return res

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    Q_K = einsum(Q, K, " ... queries d_k, ... keys d_k -> ... queries keys")
    Q_K = Q_K / (K.shape[-1] ** 0.5)
    if mask is not None:
        print(f"!!!! Q_K size = {Q_K.size()}")
        mask_v = torch.where(mask, torch.tensor(0.0), torch.tensor(float('-inf')))
        Q_K_masked = mask_v + Q_K
    else:
        Q_K_masked = Q_K
    Q_K_softmax = softmax(Q_K_masked, dim=-1)
    Q_K_V = einsum(Q_K_softmax, V, " ... queries k, ... k d_v -> ... queries d_v")
    return Q_K_V

def cross_entropy_loss(logits: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]) -> Float[Tensor, ""]:
    B, V = logits.shape

    #sm_res : Float[Tensor, " batch_size vocab_size"] = softmax(logits, dim=-1)
    z_max = logits.max(dim=-1, keepdim=True).values
    logsumexp = torch.log(torch.sum(torch.exp(logits - z_max), dim=-1, keepdim=True)) + z_max
    log_softmax = logits - logsumexp

    idx = torch.arange(B)

    loss = -log_softmax[idx, targets]
    return loss.mean()


def learning_rate_schedule(t: int, lr_max: float, lr_min: float, t_w: int, t_c: int) -> float:
    if t < t_w:
        return t / t_w * lr_max
    elif t < t_c:
        return lr_min + 0.5*(1 + math.cos((t - t_w) / (t_c - t_w) * math.pi)) * (lr_max - lr_min)
    else:
        return lr_min

def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float):
    norm_sum = 0
    for param in parameters:
        if param.grad is None:
            continue
        norm_sum += torch.sum(param.grad ** 2)

    total_norm_2 = norm_sum ** 0.5
    if total_norm_2 > max_l2_norm:
        for p in parameters:
            if p.grad is None:
                continue
            p.grad.detach().mul_(max_l2_norm / (total_norm_2 + 1e-6))




if __name__ == "__main__":
    x = torch.randn(2, 3, 4)
    print(x)
    softmax(x, dim=2)