"""Reproducible grouped partitions used by the surrogate research experiments.

All endpoint heads share the same partition assignment. Molecular identities
connect scaffold groups so that alternate SMILES, salts and stereo variants
cannot silently cross a partition through differing scaffold calculations.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupShuffleSplit


def sha256(path):
    """Hash a file without loading the complete artifact into memory."""
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def identity_groups(scaffolds, canonical, connectivity):
    """Return connected components of scaffold and nonempty molecular identities."""
    parent = list(range(len(scaffolds)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    seen = {}
    for i, values in enumerate(zip(scaffolds, canonical, connectivity)):
        for kind, value in enumerate(values):
            if value is None or str(value).strip().lower() in {"", "nan", "none"}:
                continue
            key = kind, str(value).strip()
            if key in seen:
                parent[root(i)] = root(seen[key])
            else:
                seen[key] = i
    roots = [root(i) for i in range(len(scaffolds))]
    mapping = {v: i for i, v in enumerate(sorted(set(roots)))}
    return np.array([mapping[v] for v in roots])


def assign_partitions(groups, seed=20261030):
    """Reserve 10% of groups each for confirmation, calibration and stacking.

    Fractions describe groups, not rows. Actual row counts are reported by the
    preparation script, since chemical series can differ greatly in size.
    """
    groups = np.asarray(groups)
    if len(np.unique(groups)) < 40:
        raise ValueError("Need at least 40 molecular/scaffold groups for research partitions")
    roles = np.full(len(groups), "search", dtype="U16")
    pool = np.arange(len(groups))
    for k, (name, fraction) in enumerate([
            ("confirmation", .1), ("calibration", 1 / 9), ("stacking", 1 / 8)]):
        splitter = GroupShuffleSplit(n_splits=1, test_size=fraction, random_state=seed + k)
        train, held = next(splitter.split(pool, groups=groups[pool]))
        roles[pool[held]] = name
        pool = pool[train]
    for g in np.unique(groups):
        if len(set(roles[groups == g])) != 1:
            raise AssertionError("group crossed research partitions")
    return roles


def write_once_json(path, content):
    """Create a record or refuse to reuse a path with different provenance."""
    path = Path(path)
    payload = json.dumps(content, sort_keys=True, indent=2, allow_nan=False) + "\n"
    if path.exists() and path.read_text() != payload:
        raise ValueError(f"Provenance mismatch at {path}; use a new experiment directory")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload)
