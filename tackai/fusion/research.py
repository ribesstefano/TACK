"""Reproducible grouped partitions used by the surrogate research experiments.

All endpoint heads share the same partition assignment. Molecular identities
connect scaffold groups so that alternate SMILES, salts and stereo variants
cannot silently cross a partition through differing scaffold calculations.
"""
import hashlib
import json
import zipfile
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


def read_npz_rows(path, key, indices):
    """Read selected offsets from a numeric array in an uncompressed NPZ file.

    Research snapshots are written with ``np.savez``. Refusing compressed or
    object arrays avoids materializing reserved label partitions while reading
    a permitted subset. One- and two-dimensional C-order arrays are supported;
    returned values follow the requested index order, including duplicates.
    """
    indices = np.asarray(indices, dtype=np.int64)
    if indices.ndim != 1:
        raise ValueError("Label row indices must be one-dimensional")
    with zipfile.ZipFile(path) as archive:
        try:
            info = archive.getinfo(f"{key}.npy")
        except KeyError as exc:
            raise ValueError(f"label array {key!r} is absent from {path}") from exc
        if info.compress_type != zipfile.ZIP_STORED:
            raise ValueError("label arrays must be uncompressed for role-limited reads")
        with archive.open(info) as stream:
            version = np.lib.format.read_magic(stream)
            if version == (1, 0):
                shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
            elif version == (2, 0):
                shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream)
            else:
                raise ValueError(f"unsupported label NPY header version: {version}")
            if len(shape) not in (1, 2) or fortran or dtype.hasobject or dtype.kind not in "biuf":
                raise ValueError("label arrays must be one- or two-dimensional C-order numeric values")
            if np.any(indices < 0) or np.any(indices >= shape[0]):
                raise ValueError("requested label row lies outside the snapshot")
            offset = stream.tell()
            width = int(np.prod(shape[1:], dtype=np.int64))
            row_bytes = width * dtype.itemsize
            values = np.empty((indices.size, *shape[1:]), dtype=dtype)
            for i, row in enumerate(indices):
                stream.seek(offset + int(row) * row_bytes)
                raw = stream.read(row_bytes)
                if len(raw) != row_bytes:
                    raise ValueError("truncated label array in snapshot")
                values[i] = np.frombuffer(raw, dtype=dtype, count=width).reshape(shape[1:])
    return values


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
