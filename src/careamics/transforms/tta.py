from typing import List

import numpy as np
from torch import Tensor, flip, mean, rot90, stack


class ImageRestorationTTA:
    """Test-time augmentation for image restoration tasks.

    The augmentation is performed using all 90 deg rotations and their flipped version,
    as well as the original image flipped.

    Tensors should be of shape SC(Z)YX
    """

    def __init__(self):
        """Constructor."""

    def forward(self, x: Tensor) -> List[Tensor]:
        """Apply test-time augmentation to the input tensor.

        Parameters
        ----------
        x : Any
            Input tensor, shape SC(Z)YX.

        Returns
        -------
        Any
            List of augmented tensors.
        """
        augmented = [
            x,
            rot90(x, 1, dims=(-2, -1)),
            rot90(x, 2, dims=(-2, -1)),
            rot90(x, 3, dims=(-2, -1)),
        ]
        augmented_flip = augmented.copy()
        for x_ in augmented:
            augmented_flip.append(flip(x_, dims=(-3, -1)))
        return augmented_flip

    def backward(self, x: List[Tensor]) -> np.ndarray:
        """Undo the test-time augmentation.

        Parameters
        ----------
        x : Any
            List of augmented tensors.

        Returns
        -------
        Any
            Original tensor.
        """
        reverse = [
            x[0],
            rot90(x[1], -1, dims=(-2, -1)),
            rot90(x[2], -2, dims=(-2, -1)),
            rot90(x[3], -3, dims=(-2, -1)),
            flip(x[4], dims=(-3, -1)),
            rot90(flip(x[5], dims=(-3, -1)), -1, dims=(-2, -1)),
            rot90(flip(x[6], dims=(-3, -1)), -2, dims=(-2, -1)),
            rot90(flip(x[7], dims=(-3, -1)), -3, dims=(-2, -1)),
        ]
        return mean(stack(reverse), dim=0)
