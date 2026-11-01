"""VHL source weights are bounded and are computed only from supplied rows."""
import numpy as np
import pytest

from scripts.benchmark_vhl_weighting import vhl_source_weights


def test_vhl_sources_are_balanced_and_other_e3s_are_unchanged_relatively():
    e3 = ["VHL", "vhl", "VHL", "VHL", "VHL", "CRBN"]
    source = ["major", "major", "major", "major", "minor", "major"]
    weights = vhl_source_weights(e3, source)
    assert weights.mean() == pytest.approx(1.)
    assert weights[4] > weights[0]
    assert weights[5] > weights[0]
    assert weights[4] / weights[5] == pytest.approx(2.)


def test_missing_or_single_source_does_not_invent_reweighting():
    np.testing.assert_array_equal(vhl_source_weights(["VHL", "CRBN"], ["a", "b"]), [1., 1.])
    with pytest.raises(ValueError, match="aligned"):
        vhl_source_weights(["VHL"], ["a", "b"])
