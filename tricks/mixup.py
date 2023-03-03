# based on: https://github.com/mosaicml/composer/blob/dev/composer/algorithms/mixup/mixup.py

from typing import Optional

import torch
from torch.distributions.beta import Beta
from torch.nn import functional as F

def mix_mixup(
        input: torch.Tensor,
        target: torch.Tensor,
        n_classes: int,
        mixing: Optional[float] = None,
        alpha: float = 0.2):
    
    if mixing is None: mixing = _gen_mixing_coef(alpha)
    # Create permuted versions of x and y in preparation for interpolation
    # Use given indices if there are any.
    permuted_idx = _gen_indices(input.shape[0])
    x_permuted = input[permuted_idx]
    permuted_target = target[permuted_idx]
    # Interpolate between the inputs
    x_mixup = (1 - mixing) * input + mixing * x_permuted
    
    target = F.one_hot(target, num_classes=n_classes)
    permuted_target = F.one_hot(permuted_target, num_classes=n_classes)
    y_mixed = (1 - mixing) * target + mixing * permuted_target
    return x_mixup, y_mixed


def _gen_mixing_coef(alpha: float) -> float:
    """Samples ``max(z, 1-z), z ~ Beta(alpha, alpha)``."""
    # First check if alpha is positive.
    assert alpha >= 0
    # Draw the mixing parameter from a beta distribution.
    # Check here is needed because beta distribution requires alpha > 0
    # but alpha = 0 is fine for mixup.
    if alpha == 0:
        mixing_lambda = 0
    else:
        alpha = torch.tensor([alpha])
        mixing_lambda = Beta(alpha, alpha).sample().item()
    # for symmetric beta distribution, can always use 0 <= lambda <= .5;
    # this way the "main" label is always the original one, which keeps
    # the training accuracy meaningful
    return min(mixing_lambda, 1. - mixing_lambda)


def _gen_indices(num_samples: int) -> torch.Tensor:
    """Generates a random permutation of the batch indices."""
    return torch.randperm(num_samples)