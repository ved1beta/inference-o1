import torch
import time

def layernorm_pytorch(x, eps=1e-5, weight=None, bias=None):
    # x: [B, D]
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, unbiased=False, keepdim=True)
    y = (x - mean) / torch.sqrt(var + eps)
    if weight is not None:
        y = y * weight
    if bias is not None:
        y = y + bias
    return y
