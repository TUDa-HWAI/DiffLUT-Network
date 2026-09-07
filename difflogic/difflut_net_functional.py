"""Differentiable operations used by DiffLUT-Net."""


def difflut_net_op(a1, a2, a3, a4, a5, a6, weights):
    """Evaluate the multilinear extension of a six-input LUT."""
    current = weights
    for value in (a1, a2, a3, a4, a5, a6):
        half = current.shape[-1] // 2
        values_for_zero = current[..., :half]
        values_for_one = current[..., half:]

        if half > 1:
            value = value.unsqueeze(-1)
        else:
            values_for_zero = values_for_zero.squeeze(-1)
            values_for_one = values_for_one.squeeze(-1)

        current = values_for_zero * (1.0 - value) + values_for_one * value

    return current
