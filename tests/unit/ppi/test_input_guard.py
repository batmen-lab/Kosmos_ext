"""A matrix torch cannot use is refused in Python, with the reason.

The traceback the user hit ended in `torch.nn.Module._call_impl` and
`Segmentation fault (core dumped)`: the process died inside a C extension, where
there is no error message and no column name. These are the checks that answer
first, in the language of the data.
"""

from __future__ import annotations

import numpy as np
import pytest


def test_a_finite_matrix_passes():
    from kosmos.ppi.trainer import check_matrix

    check_matrix(np.ones((4, 3), dtype=np.float32), what="the matrix")


def test_a_nan_is_refused_with_its_position():
    from kosmos.ppi.trainer import check_matrix

    matrix = np.ones((4, 3), dtype=np.float32)
    matrix[2, 1] = np.nan

    with pytest.raises(ValueError) as caught:
        check_matrix(matrix, what="the evidence matrix from 'cite'")

    message = str(caught.value)
    assert "the evidence matrix from 'cite'" in message
    assert "1 non-finite value" in message
    assert "row 2, column 1" in message


def test_an_infinity_is_refused_too():
    from kosmos.ppi.trainer import check_matrix

    matrix = np.zeros((2, 2), dtype=np.float32)
    matrix[0, 0] = np.inf

    with pytest.raises(ValueError, match="non-finite"):
        check_matrix(matrix, what="the training matrix")


def test_a_value_too_large_for_the_model_is_refused():
    from kosmos.ppi.trainer import check_matrix

    with pytest.raises(ValueError, match="overflows"):
        check_matrix(np.full((2, 2), 1e30, dtype=np.float64), what="the matrix")


def test_an_all_zero_matrix_is_refused():
    """Every column dropped is not a model: it is a run that learns nothing."""
    from kosmos.ppi.trainer import check_matrix

    with pytest.raises(ValueError, match="all zeros"):
        check_matrix(np.zeros((5, 3), dtype=np.float32), what="the matrix")


def test_an_empty_matrix_is_refused():
    from kosmos.ppi.trainer import check_matrix

    with pytest.raises(ValueError, match="empty"):
        check_matrix(np.zeros((0, 3), dtype=np.float32), what="the matrix")
