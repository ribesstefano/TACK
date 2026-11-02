"""Tests for ``tackai.chem_utils.canonicalize_smiles``.

The function standardises as well as canonicalises -- ``FragmentParent``
removes salts and ``Uncharger`` neutralises -- so the whole observable
behaviour is pinned here, not just the canonical spelling.

The guard tests matter more than they look. This standardiser has two sibling
implementations in the programme (``tack_dataset.curation_utils`` in
TACK-Dataset and ``tackos.chem`` in TACKOS). Measured against each other they
agree chemically on all 8,380 parseable curated PROTAC SMILES and on every
salt, charge and multi-fragment case tried; every difference between them was
in these guards. So the guards are the part that drifts.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import pytest

from tackai.chem_utils import canonicalize_smiles

ASPIRIN_KEKULE = "CC(=O)OC1=CC=CC=C1C(=O)O"
ASPIRIN_AROMATIC = "CC(=O)Oc1ccccc1C(=O)O"


# --------------------------------------------------------------------------
# Canonicalization
# --------------------------------------------------------------------------

def test_equivalent_spellings_collapse_to_one_string():
    assert canonicalize_smiles(ASPIRIN_KEKULE) == canonicalize_smiles(ASPIRIN_AROMATIC)


def test_output_is_stable_under_reapplication():
    once = canonicalize_smiles(ASPIRIN_KEKULE)
    assert canonicalize_smiles(once) == once


def test_surrounding_whitespace_is_ignored():
    assert canonicalize_smiles("  CCO\n") == canonicalize_smiles("CCO")


# --------------------------------------------------------------------------
# Standardization (not merely canonicalization)
# --------------------------------------------------------------------------

def test_a_salt_is_reduced_to_its_largest_fragment():
    result = canonicalize_smiles("CC(=O)[O-].[Na+]")
    assert result == "CC(=O)O"


def test_a_formal_charge_is_neutralized():
    assert canonicalize_smiles("C(C(=O)[O-])[NH3+]") == "NCC(=O)O"


def test_a_charge_separated_group_is_left_alone():
    """Uncharger must not 'fix' a nitro group into something else."""
    assert canonicalize_smiles("CC[N+](=O)[O-]") == "CC[N+](=O)[O-]"


# --------------------------------------------------------------------------
# Options
# --------------------------------------------------------------------------

def test_return_mol_gives_back_a_molecule():
    mol = canonicalize_smiles(ASPIRIN_AROMATIC, return_mol=True)
    assert mol is not None
    assert not isinstance(mol, str)
    assert mol.GetNumAtoms() > 0


def test_unique_inchikeys_filters_the_second_sighting():
    seen = {}
    first = canonicalize_smiles(ASPIRIN_KEKULE, unique_inchikeys=seen)
    assert first is not None
    assert len(seen) == 1
    # The same molecule written differently is still the same molecule.
    assert canonicalize_smiles(ASPIRIN_AROMATIC, unique_inchikeys=seen) is None


# --------------------------------------------------------------------------
# Guards: nothing may come back as a non-None falsy value
# --------------------------------------------------------------------------

def test_unparseable_input_returns_none():
    for bad in ("not a molecule", "C(((", "XYZ123", "CN(C)(C)C"):
        assert canonicalize_smiles(bad) is None, bad


def test_missing_input_returns_none():
    for empty in (None, "", float("nan"), 0, []):
        assert canonicalize_smiles(empty) is None, repr(empty)


@pytest.mark.parametrize("blank", ["   ", "\t\n", " \r ", "\n"])
def test_whitespace_only_input_returns_none_not_an_empty_smiles(blank):
    """``Chem.MolFromSmiles('')`` returns an *empty Mol*, not None.

    So stripping and then parsing, without re-checking for emptiness, turns a
    whitespace-only cell into ``''``: falsy but not None, which slips past an
    ``if result is None`` check and reaches a CSV as a blank SMILES on a row
    that otherwise looks curated.
    """
    result = canonicalize_smiles(blank)
    assert result is None, f"{blank!r} produced {result!r}"


def test_no_input_ever_yields_a_falsy_string():
    """The property the above cases are instances of."""
    for value in ("   ", "", "\t", None, float("nan"), "not a molecule"):
        result = canonicalize_smiles(value)
        assert result is None or result, f"{value!r} produced {result!r}"
