from collections.abc import Callable, Iterable
from typing import Optional, Tuple
import torch
import math

from jaxtyping import Float

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr: float, weight_decay: float, eps: float, betas: Tuple[float, float]):
            if lr < 0:
                raise ValueError(f"Invalid learning rate: {lr}")
            defaults = {"lr": lr,
                        "betas": betas,
                        "eps": eps,
                        "weight_decay": weight_decay}
            super().__init__(params, defaults)


    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            # Get the learning rate.
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue

                #state = self.state[p]  # Get state associated with p.
                #t = state.get("t", 0)  # Get iteration number from the state, or initial value.
                #grad = p.grad.data  # Get the gradient of loss with respect to p.
                #p.data -= lr / math.sqrt(t + 1) * grad  # Update weight tensor in-place.
                #state["t"] = t + 1  # Increment iteration number.

                grad = p.grad
                state = self.state[p]
                t = state.get("t", 1)
                m = state.get("m", torch.zeros_like(p.data))
                v = state.get("v", torch.zeros_like(p.data))
                # m ←β1m + (1−β1)g (Update the first moment estimate)
                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * grad**2
                lr_t = lr * math.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)
                p.data -= lr_t * m / (torch.sqrt(v) + eps)
                p.data -= lr * weight_decay * p.data

                state["t"] = t + 1
                state["m"] = m
                state["v"] = v


        return loss
