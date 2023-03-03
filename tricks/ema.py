# https://github.com/mosaicml/composer/blob/dev/composer/algorithms/ema/ema.py

import itertools

import torch

@torch.no_grad()
def compute_ema(model: torch.nn.Module, ema_model: torch.nn.Module, smoothing: float = 0.99):
    model_params = itertools.chain(model.parameters(), model.buffers())
    ema_model_params = itertools.chain(ema_model.parameters(), ema_model.buffers())

    for ema_param, model_param in zip(ema_model_params, model_params):
        model_param = model_param.detach()
        ema_param.copy_(ema_param * smoothing + (1. - smoothing) * model_param)