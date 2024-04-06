"""
Validator functions.

These functions are used to validate dimensions and axes of inputs.
"""
from typing import List

AXES = "STCZYX"


def check_axes_validity(axes: str) -> bool:
    """
    Sanity check on axes.

    The constraints on the axes are the following:
    - must be a combination of 'STCZYX'
    - must not contain duplicates
    - must contain at least 2 contiguous axes: X and Y
    - must contain at most 4 axes
    - cannot contain both S and T axes

    Axes do not need to be in the order 'STCZYX', as this depends on the user data.

    Parameters
    ----------
    axes : str
        Axes to validate.

    Returns
    -------
    bool
        True if axes are valid, False otherwise.
    """
    _axes = axes.upper()

    # Minimum is 2 (XY) and maximum is 4 (TZYX)
    if len(_axes) < 2 or len(_axes) > 6:
        raise ValueError(
            f"Invalid axes {axes}. Must contain at least 2 and at most 6 axes."
        )

    if "YX" not in _axes and "XY" not in _axes:
        raise ValueError(
            f"Invalid axes {axes}. Must contain at least X and Y axes consecutively."
        )

    # all characters must be in REF_AXES = 'STCZYX'
    if not all(s in AXES for s in _axes):
        raise ValueError(f"Invalid axes {axes}. Must be a combination of {AXES}.")

    # check for repeating characters
    for i, s in enumerate(_axes):
        if i != _axes.rfind(s):
            raise ValueError(
                f"Invalid axes {axes}. Cannot contain duplicate axes"
                f" (got multiple {axes[i]})."
            )
        
    return True


def check_tiling_validity(tile_shape: List[int], overlaps: List[int]) -> None:
    """
    Check that the tiling parameters are valid.

    Parameters
    ----------
    tile_shape : List[int]
        Shape of the tiles.
    overlaps : List[int]
        Overlap between tiles.

    Raises
    ------
    ValueError
        If one of the parameters is None.
    ValueError
        If one of the element is zero.
    ValueError
        If one of the element is non-divisible by 2.
    ValueError
        If the number of elements in `overlaps` and `tile_shape` is different.
    ValueError
        If one of the overlaps is larger than the corresponding tile shape.
    """
    # cannot be None
    if tile_shape is None or overlaps is None:
        raise ValueError(
            "Cannot use tiling without specifying `tile_shape` and "
            "`overlaps`, make sure they have been correctly specified."
        )

    # non-zero and divisible by two
    for dims_list in [tile_shape, overlaps]:
        for dim in dims_list:
            if dim < 0:
                raise ValueError(f"Entry must be non-null positive (got {dim}).")

            if dim % 2 != 0:
                raise ValueError(f"Entry must be divisible by 2 (got {dim}).")

    # same length
    if len(overlaps) != len(tile_shape):
        raise ValueError(
            f"Overlaps ({len(overlaps)}) and tile shape ({len(tile_shape)}) must "
            f"have the same number of dimensions."
        )

    # overlaps smaller than tile shape
    for overlap, tile_dim in zip(overlaps, tile_shape):
        if overlap >= tile_dim:
            raise ValueError(
                f"Overlap ({overlap}) must be smaller than tile shape ({tile_dim})."
            )
