"""The nearest-neighbor diagnostic scores against the actual training set."""
import numpy as np
import pytest

from scripts.report_similarity_slices import nearest_similarity


def test_identical_molecule_has_similarity_one_and_order_is_preserved():
    values = nearest_similarity(["CCO", "c1ccccc1"], ["c1ccccc1", "CCN", "CCO"])
    assert values[0] == values[2] == pytest.approx(1.)
    assert 0 <= values[1] < 1
    assert np.isfinite(values).all()


def test_invalid_or_empty_training_set_is_rejected():
    with pytest.raises(ValueError, match="without training"):
        nearest_similarity([], ["CCO"])
    with pytest.raises(ValueError, match="unparseable"):
        nearest_similarity(["invalid-molecule"], ["CCO"])
