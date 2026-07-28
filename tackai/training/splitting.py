"""
On-the-fly data-splitting strategies for PROTAC degradation datasets.

The published TACK dataset ships with precomputed split columns
(``SMILES_Held_Out``, ``SMILES_Scaffold_Cluster``, ``SMILES_Butina_Cluster``).
A raw dataset such as TACK2.0 does not, so this module derives those columns
from the ``SMILES`` structures at load time:

- **Held-out selection** — a diverse test set picked with RDKit's MaxMin
  algorithm (:class:`rdkit.SimDivFilters.rdSimDivPickers.MaxMinPicker`) over the
  Morgan fingerprints of the *unique* compounds. Selecting on unique compounds
  guarantees that no held-out molecule also appears in a training fold.
- **CV grouping** — cluster labels used by ``GroupKFold``, computed directly
  from RDKit (no third-party clustering dependency): generic Bemis-Murcko
  scaffolds for ``group='scaffold'`` and Taylor-Butina clustering for
  ``group='butina'``.

All routines are deterministic: the MaxMin pick is seeded, and the clusterings
are functions of structure alone, so the derived splits are identical across
every model / feature-set experiment run on the same data.
"""
import logging
from functools import lru_cache
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.ML.Cluster import Butina
from rdkit.SimDivFilters.rdSimDivPickers import MaxMinPicker

logger = logging.getLogger(__name__)

# Column names the training pipeline expects (see cross_validation.py and
# tasks.split_held_out). Keep these in sync with those consumers.
HELD_OUT_COLUMN = "SMILES_Held_Out"
GROUP_TO_COLUMN: Dict[str, Optional[str]] = {
    "random": None,
    "scaffold": "SMILES_Scaffold_Cluster",
    "butina": "SMILES_Butina_Cluster",
}

# Fingerprint settings shared by MaxMin selection and Butina clustering.
DEFAULT_FP_RADIUS = 2
DEFAULT_FP_BITS = 2048
# Taylor-Butina distance cutoff (1 - Tanimoto similarity).
DEFAULT_BUTINA_CUTOFF = 0.65
# Fraction of compounds held out, and the fixed seed for the MaxMin pick.
DEFAULT_HELD_OUT_FRAC = 0.10
DEFAULT_HELD_OUT_SEED = 42


def _unique_preserving_order(smiles: Sequence[str]) -> List[str]:
    """Return the distinct SMILES in first-seen order."""
    return list(dict.fromkeys(smiles))


@lru_cache(maxsize=None)
def _morgan_generator(radius: int, n_bits: int):
    """Return a cached RDKit Morgan fingerprint generator."""
    return rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)


@lru_cache(maxsize=None)
def smi2morgan_fp(
    smi: str,
    radius: int = DEFAULT_FP_RADIUS,
    n_bits: int = DEFAULT_FP_BITS,
):
    """Compute the Morgan fingerprint of a SMILES string.

    Results are memoized on ``(smi, radius, n_bits)`` so that the two passes over
    the dataset (MaxMin held-out selection and Taylor-Butina clustering) share a
    single fingerprint per compound instead of recomputing it (see the module
    note on ``group='butina'``).

    Args:
        smi: SMILES string to featurize.
        radius: Morgan fingerprint radius.
        n_bits: Fingerprint length in bits.

    Returns:
        The compound's ``ExplicitBitVect``, or ``None`` if RDKit cannot parse
        ``smi``.
    """
    if not isinstance(smi, str):
        return None
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return _morgan_generator(radius, n_bits).GetFingerprint(mol)


def _fingerprint_map(
    unique_smiles: Sequence[str],
    radius: int = DEFAULT_FP_RADIUS,
    n_bits: int = DEFAULT_FP_BITS,
) -> Dict[str, object]:
    """Compute Morgan fingerprints for each unique SMILES.

    Args:
        unique_smiles: Distinct SMILES strings to featurize.
        radius: Morgan fingerprint radius.
        n_bits: Fingerprint length in bits.

    Returns:
        Mapping from SMILES to its ``ExplicitBitVect``. SMILES that RDKit cannot
        parse are omitted from the mapping.
    """
    fp_map: Dict[str, object] = {}
    n_invalid = 0
    for smi in unique_smiles:
        fp = smi2morgan_fp(smi, radius=radius, n_bits=n_bits)
        if fp is None:
            n_invalid += 1
            continue
        fp_map[smi] = fp
    if n_invalid:
        logger.warning(f"{n_invalid} SMILES could not be parsed and are excluded from fingerprinting.")
    return fp_map


def maxmin_held_out_mask(
    smiles: Sequence[str],
    frac: float = DEFAULT_HELD_OUT_FRAC,
    seed: int = DEFAULT_HELD_OUT_SEED,
    radius: int = DEFAULT_FP_RADIUS,
    n_bits: int = DEFAULT_FP_BITS,
) -> np.ndarray:
    """Flag a diverse held-out subset of rows via the MaxMin algorithm.

    The pick is performed over the *unique* compounds so that every row sharing a
    selected SMILES is held out together (no compound leaks between the held-out
    set and the training folds). ``round(frac * n_unique)`` compounds are picked.

    Args:
        smiles: Per-row SMILES strings (duplicates allowed).
        frac: Fraction of unique compounds to hold out.
        seed: Fixed seed for the MaxMin initial pick (keeps the set reproducible).
        radius: Morgan fingerprint radius.
        n_bits: Fingerprint length in bits.

    Returns:
        Boolean array aligned to ``smiles``; ``True`` marks held-out rows.
    """
    smiles = list(smiles)
    unique = _unique_preserving_order(smiles)
    fp_map = _fingerprint_map(unique, radius=radius, n_bits=n_bits)
    valid = [s for s in unique if s in fp_map]

    if not valid:
        logger.warning("No valid compounds to select a held-out set from; holding out nothing.")
        return np.zeros(len(smiles), dtype=bool)

    fps = [fp_map[s] for s in valid]
    pick_size = min(len(valid), max(1, round(frac * len(valid))))

    picker = MaxMinPicker()
    picked_idx = list(picker.LazyBitVectorPick(fps, len(fps), pick_size, seed=seed))
    held_out_smiles = {valid[i] for i in picked_idx}

    mask = np.array([s in held_out_smiles for s in smiles], dtype=bool)
    logger.info(
        f"MaxMin held-out: {len(held_out_smiles)}/{len(valid)} unique compounds "
        f"({mask.sum()}/{len(smiles)} rows, {mask.mean():.1%}), seed={seed}."
    )
    return mask


@lru_cache(maxsize=None)
def carbon_scaffold_smiles(smi: str) -> Optional[str]:
    """Return the canonical SMILES of a compound's carbon-only scaffold.

    The scaffold is the generic Bemis-Murcko framework: the ring systems and
    their connecting linkers, with every atom reduced to carbon and every bond
    to a single bond (RDKit ``MakeScaffoldGeneric``). Reducing to a pure-carbon
    skeleton groups compounds that share the same topological framework
    regardless of heteroatom substitution.

    Args:
        smi: SMILES string of the compound.

    Returns:
        Canonical SMILES of the carbon-only scaffold, or ``None`` if RDKit
        cannot parse ``smi`` or derive its scaffold.
    """
    if not isinstance(smi, str):
        return None
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        generic = MurckoScaffold.MakeScaffoldGeneric(scaffold)
    except Exception:
        # Acyclic molecules yield an empty scaffold; malformed frameworks can
        # raise on genericization. Either way there is no framework to group on.
        return None
    smiles = Chem.MolToSmiles(generic)
    return smiles or None


def bemis_murcko_clusters(smiles: Sequence[str]) -> np.ndarray:
    """Assign a Bemis-Murcko scaffold cluster id to each row.

    Compounds sharing the same carbon-only scaffold (see
    :func:`carbon_scaffold_smiles`) get the same cluster id; identical SMILES
    therefore always share a cluster. SMILES that RDKit cannot parse, and
    compounds with no ring framework (acyclic molecules), each receive their own
    singleton cluster so they never group unrelated rows together.

    Args:
        smiles: Per-row SMILES strings (duplicates allowed).

    Returns:
        Integer cluster-id array aligned to ``smiles``.
    """
    smiles = list(smiles)
    unique = _unique_preserving_order(smiles)

    # Map each distinct scaffold SMILES to a stable integer id, in first-seen
    # order so the labelling is deterministic.
    scaffold_to_id: Dict[str, int] = {}
    cluster_of: Dict[str, int] = {}
    next_id = 0
    n_singletons = 0
    for smi in unique:
        scaffold = carbon_scaffold_smiles(smi)
        if scaffold is None:
            cluster_of[smi] = next_id
            next_id += 1
            n_singletons += 1
            continue
        if scaffold not in scaffold_to_id:
            scaffold_to_id[scaffold] = next_id
            next_id += 1
        cluster_of[smi] = scaffold_to_id[scaffold]

    logger.info(
        f"Bemis-Murcko: {len(scaffold_to_id)} scaffold clusters over "
        f"{len(unique)} unique compounds ({n_singletons} unparseable/acyclic "
        "singletons)."
    )
    return np.array([cluster_of[s] for s in smiles], dtype=int)


def _taylor_butina_labels(fps: Sequence[object], cutoff: float) -> List[int]:
    """Cluster fingerprints with the Taylor-Butina algorithm.

    Builds the condensed lower-triangular Tanimoto *distance* matrix (1 - Tc)
    and delegates the sphere-exclusion clustering to RDKit's
    :func:`rdkit.ML.Cluster.Butina.ClusterData`. Clusters are returned largest
    first (RDKit's ordering), and each is assigned a contiguous integer id.

    Args:
        fps: Fingerprints (``ExplicitBitVect``) to cluster.
        cutoff: Taylor-Butina distance cutoff (1 - Tanimoto similarity); points
            within this distance of a cluster centroid join that cluster.

    Returns:
        A list of integer cluster ids aligned to ``fps``.
    """
    n = len(fps)
    # Condensed distance matrix: for i>0, distances to all j<i, row-major.
    dists: List[float] = []
    for i in range(1, n):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], list(fps[:i]))
        dists.extend(1.0 - s for s in sims)

    clusters = Butina.ClusterData(dists, n, cutoff, isDistData=True)

    labels = [0] * n
    for cluster_id, members in enumerate(clusters):
        for idx in members:
            labels[idx] = cluster_id
    return labels


def butina_clusters(
    smiles: Sequence[str],
    cutoff: float = DEFAULT_BUTINA_CUTOFF,
    radius: int = DEFAULT_FP_RADIUS,
    n_bits: int = DEFAULT_FP_BITS,
) -> np.ndarray:
    """Assign a Taylor-Butina cluster id to each row.

    Clustering runs on the unique valid compounds and is mapped back to rows.
    Unparseable SMILES each receive their own singleton cluster so they never
    group spurious rows together.

    Args:
        smiles: Per-row SMILES strings (duplicates allowed).
        cutoff: Taylor-Butina distance cutoff (1 - Tanimoto similarity).
        radius: Morgan fingerprint radius.
        n_bits: Fingerprint length in bits.

    Returns:
        Integer cluster-id array aligned to ``smiles``.
    """
    smiles = list(smiles)
    unique = _unique_preserving_order(smiles)
    fp_map = _fingerprint_map(unique, radius=radius, n_bits=n_bits)
    valid = [s for s in unique if s in fp_map]

    cluster_of: Dict[str, int] = {}
    if valid:
        fps = [fp_map[s] for s in valid]
        ids = _taylor_butina_labels(fps, cutoff=cutoff)
        cluster_of = {smi: int(cid) for smi, cid in zip(valid, ids)}

    # Give each unparseable SMILES its own cluster id after the valid ones.
    next_id = max(cluster_of.values(), default=-1) + 1
    for smi in unique:
        if smi not in cluster_of:
            cluster_of[smi] = next_id
            next_id += 1

    n_clusters = len(set(cluster_of.values()))
    logger.info(f"Taylor-Butina: {n_clusters} clusters over {len(unique)} unique compounds (cutoff={cutoff}).")
    return np.array([cluster_of[s] for s in smiles], dtype=int)


def assign_held_out_column(
    df: pd.DataFrame,
    smiles_col: str = "SMILES",
    frac: float = DEFAULT_HELD_OUT_FRAC,
    seed: int = DEFAULT_HELD_OUT_SEED,
    out_col: str = HELD_OUT_COLUMN,
) -> pd.DataFrame:
    """Add a boolean held-out column, computed with MaxMin, if it is absent.

    A dataset that already carries ``out_col`` (e.g. the published TACK dataset)
    is returned untouched so precomputed splits are preserved.

    Args:
        df: Input dataframe containing ``smiles_col``.
        smiles_col: Name of the SMILES column.
        frac: Fraction of unique compounds to hold out.
        seed: Fixed seed for the MaxMin pick.
        out_col: Name of the boolean column to create.

    Returns:
        A dataframe with ``out_col`` present.
    """
    if out_col in df.columns:
        logger.debug(f"'{out_col}' already present; skipping MaxMin held-out selection.")
        return df
    df = df.copy()
    df[out_col] = maxmin_held_out_mask(df[smiles_col].tolist(), frac=frac, seed=seed)
    return df


def assign_group_column(
    df: pd.DataFrame,
    group: str,
    smiles_col: str = "SMILES",
    butina_cutoff: float = DEFAULT_BUTINA_CUTOFF,
) -> pd.DataFrame:
    """Add the CV grouping column required by ``group``, if it is absent.

    ``group='random'`` needs no grouping column and is a no-op. A dataset that
    already carries the target column is returned untouched.

    Args:
        df: Input dataframe containing ``smiles_col``.
        group: CV grouping strategy ('random', 'scaffold', or 'butina').
        smiles_col: Name of the SMILES column.
        butina_cutoff: Distance cutoff forwarded to Taylor-Butina clustering.

    Returns:
        A dataframe with the grouping column present (or unchanged for 'random').

    Raises:
        ValueError: If ``group`` is not a recognized strategy.
    """
    if group not in GROUP_TO_COLUMN:
        raise ValueError(f"Unknown group '{group}'; expected one of {sorted(GROUP_TO_COLUMN)}.")

    out_col = GROUP_TO_COLUMN[group]
    if out_col is None or out_col in df.columns:
        return df

    df = df.copy()
    if group == "scaffold":
        df[out_col] = bemis_murcko_clusters(df[smiles_col].tolist())
    elif group == "butina":
        df[out_col] = butina_clusters(df[smiles_col].tolist(), cutoff=butina_cutoff)
    return df
