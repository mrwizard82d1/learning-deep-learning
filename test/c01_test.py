import ldl.c01

import pytest


@pytest.mark.parametrize(
    'x, expected',
    [
        ([1.0, -1.0, -1.0], 1),
        ([1.0, -1.0, 1.0], 1),
        ([1.0, 1.0, -1.0], 1),
        ([1.0, 1.0, 1.0], -1),
    ]
)
def test_compute_output(x, expected):
    assert ldl.c01.compute_output([0.9, -0.6, -0.5], x) == expected
