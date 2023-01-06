import torch
import torch.nn as nn
import numpy as np
from thop import profile  # https://github.com/Lyken17/pytorch-OpCounter


def count_params(model: nn.Module):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params.item()


def get_macs_and_params(model: nn.Module, input_shape: list):
    model_device = next(model.parameters()).device
    input_ = torch.rand(input_shape, device=model_device)
    macs, params = profile(model, inputs=(input_, ), verbose=False)
    return macs, params
