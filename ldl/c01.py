def compute_output(w, x):
    """
    Compute the output of the perceptron with weights, `w` and vector, `x`.

    Note that the length of the vectors, `w` and `x` must both be identically equal to `n + 1` where `n` is the
    number of actual inputs (excluding the "special", $x_{0}$, component).

    Args:
        w: The weights for the calculation.
        x: The vector whose output is to be calculated. (Note that `$x_{0}$` is identically 1.)

    Returns:
        The computed output of the perceptron.
    """

    z = sum([x[i] * w[i] for i in range(len(w))])
    return -1 if z < 0 else 1
