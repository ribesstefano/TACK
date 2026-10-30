"""Versioned, inference-safe specifications for molecular feature encoders.

The existing :class:`MolEncoder` defaults remain unchanged. This module wraps it
with a serializable description of every molecular input choice and adds count
Morgan fingerprints as an opt-in alternative.
"""

import json
from dataclasses import asdict, dataclass
from typing import List, Optional, Tuple

import numpy as np
from rdkit import Chem, RDLogger, rdBase
from rdkit.Chem import Graphs, rdFingerprintGenerator
from rdkit.ML.InfoTheory import entropy

from tackai.chem_utils import canonicalize_smiles
from tackai.fusion.mol_encoder import (
    DESC_LIMIT,
    DESCRIPTOR_NAMES,
    SENTINEL,
    MolEncoder,
    _select_descriptors,
)

RDLogger.DisableLog("rdApp.*")

SPEC_SCHEMA_VERSION = 1
CANONICALIZATION_VERSION = "fragment-parent-uncharge-canonical-isomeric-v1"


@dataclass(frozen=True)
class MolecularFeatureSpec:
    """Complete molecular feature contract saved alongside a fitted model.

    `descriptor_names` is the exact output order, which follows the installed
    RDKit descriptor list after selection. `rdkit_version` is pinned because
    both fingerprint implementation and descriptor definitions can change.
    """

    radius: int = 16
    size: int = 1024
    count: bool = False
    chirality: bool = True
    descriptor_names: Tuple[str, ...] = tuple(DESCRIPTOR_NAMES)
    canonicalization_version: str = CANONICALIZATION_VERSION
    rdkit_version: str = rdBase.rdkitVersion
    schema_version: int = SPEC_SCHEMA_VERSION

    def __post_init__(self):
        if self.radius < 0:
            raise ValueError("radius must be non-negative")
        if self.size <= 0:
            raise ValueError("size must be positive")
        if not self.chirality:
            raise ValueError("TACK molecular features require chirality=True")
        if self.schema_version != SPEC_SCHEMA_VERSION:
            raise ValueError("unsupported molecular spec schema version: {}".format(self.schema_version))
        if self.canonicalization_version != CANONICALIZATION_VERSION:
            raise ValueError("unsupported canonicalization version: {}".format(self.canonicalization_version))
        if not isinstance(self.descriptor_names, tuple):
            object.__setattr__(self, "descriptor_names", tuple(self.descriptor_names))
        selected = tuple(name for name, _ in _select_descriptors(self.descriptor_names))
        if selected != self.descriptor_names:
            raise ValueError("descriptor_names must exactly match the current RDKit descriptor order")

    @classmethod
    def preset(cls, name: str) -> "MolecularFeatureSpec":
        """Build a named, versioned preset.

        Presets are `default`, `morgan_r{2,3}_{2048,4096}` and
        `count_r2_{2048,4096}`. All use the established descriptor list.
        """
        if name == "default":
            return cls()
        presets = {
            "morgan_r2_2048": (2, 2048, False),
            "morgan_r2_4096": (2, 4096, False),
            "morgan_r3_2048": (3, 2048, False),
            "morgan_r3_4096": (3, 4096, False),
            "count_r2_2048": (2, 2048, True),
            "count_r2_4096": (2, 4096, True),
        }
        try:
            radius, size, count = presets[name]
        except KeyError as exc:
            raise ValueError("unknown molecular feature preset: {}".format(name)) from exc
        return cls(radius=radius, size=size, count=count)

    def to_dict(self) -> dict:
        values = asdict(self)
        values["descriptor_names"] = list(self.descriptor_names)
        return values

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, indent=2)

    @classmethod
    def from_dict(cls, values: dict, validate_runtime: bool = True) -> "MolecularFeatureSpec":
        """Load a spec, rejecting schema, canonicalization and descriptor drift.

        With `validate_runtime=True` (the inference default), the spec also has
        to name the currently installed RDKit version.
        """
        expected = {
            "radius", "size", "count", "chirality", "descriptor_names",
            "canonicalization_version", "rdkit_version", "schema_version",
        }
        if set(values) != expected:
            raise ValueError("molecular spec fields differ: expected {}, got {}".format(
                sorted(expected), sorted(values)))
        data = dict(values)
        data["descriptor_names"] = tuple(data["descriptor_names"])
        spec = cls(**data)
        if validate_runtime and spec.rdkit_version != rdBase.rdkitVersion:
            raise ValueError("RDKit version mismatch: spec={}, runtime={}".format(
                spec.rdkit_version, rdBase.rdkitVersion))
        return spec

    @classmethod
    def from_json(cls, value: str, validate_runtime: bool = True) -> "MolecularFeatureSpec":
        try:
            data = json.loads(value)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("invalid molecular feature spec JSON") from exc
        if not isinstance(data, dict):
            raise ValueError("molecular feature spec JSON must contain an object")
        return cls.from_dict(data, validate_runtime=validate_runtime)


def _featurize_count_chunk(smiles_list, share_ipc, radius, fp_size, dlist):
    """MolEncoder-compatible chunk worker returning Morgan count fingerprints."""
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=radius, fpSize=fp_size, includeChirality=True)
    out = []
    for smiles in smiles_list:
        mol = canonicalize_smiles(smiles, return_mol=True) if isinstance(smiles, str) and smiles.strip() else None
        if mol is None:
            out.append(None)
            continue
        fp = generator.GetCountFingerprintAsNumPy(mol).astype(np.float32)
        shared = None
        values = []
        for name, fn in dlist:
            try:
                if share_ipc and name in ("Ipc", "AvgIpc"):
                    if shared is None:
                        dmat = Chem.GetDistanceMatrix(mol, 0)
                        cpoly = abs(Graphs.CharacteristicPolynomial(mol, np.equal(dmat, 1)))
                        ent = entropy.InfoEntropy(cpoly)
                        shared = (sum(cpoly) * ent, ent)
                    value = shared[0] if name == "Ipc" else shared[1]
                else:
                    value = fn(mol)
            except Exception:
                value = np.nan
            if value is None or np.isnan(value) or np.isinf(value) or abs(value) > DESC_LIMIT:
                value = SENTINEL
            values.append(value)
        out.append((fp, np.asarray(values, dtype=np.float32)))
    return out


def _featurize_count_chunk_named(smiles_list, share_ipc, radius, fp_size, names):
    """Worker entry point matching `MolEncoder`'s pickle-safe name resolution."""
    return _featurize_count_chunk(smiles_list, share_ipc, radius, fp_size,
                                  _select_descriptors(names))


class MolecularFeatureEncoder(MolEncoder):
    """`MolEncoder` with an explicit spec and optional count Morgan fingerprints.

    The output signature remains ``(fingerprint, descriptors, ok)``. Binary
    presets use the existing `MolEncoder` implementation, preserving its
    bit-exact default. Count presets override only `_compute` and retain the
    parent class's de-duplication, cache and input-order behavior.
    """

    def __init__(self, spec: Optional[MolecularFeatureSpec] = None,
                 use_cache: bool = True, n_workers: int = 1,
                 share_ipc: bool = True):
        self.spec = spec or MolecularFeatureSpec()
        if not isinstance(self.spec, MolecularFeatureSpec):
            raise TypeError("spec must be a MolecularFeatureSpec")
        if self.spec.rdkit_version != rdBase.rdkitVersion:
            raise ValueError("RDKit version mismatch: spec={}, runtime={}".format(
                self.spec.rdkit_version, rdBase.rdkitVersion))
        super().__init__(radius=self.spec.radius, fp_size=self.spec.size,
                         descriptors=self.spec.descriptor_names, share_ipc=share_ipc,
                         use_cache=use_cache, n_workers=n_workers)
        if tuple(self.descriptors) != self.spec.descriptor_names:
            raise ValueError("encoder descriptor order does not match molecular spec")

    @classmethod
    def from_json(cls, value: str, **kwargs) -> "MolecularFeatureEncoder":
        return cls(MolecularFeatureSpec.from_json(value), **kwargs)

    def _compute(self, uniq: List[str]):
        if not self.spec.count:
            return super()._compute(uniq)
        if self.n_workers <= 1 or len(uniq) < 2 * self.n_workers:
            return _featurize_count_chunk(uniq, self.share_ipc, self.radius,
                                          self.fp_size, self._dlist)
        n_chunks = min(len(uniq), self.n_workers * 4)
        chunks = [uniq[i::n_chunks] for i in range(n_chunks)]
        futures = [self._pool().submit(_featurize_count_chunk_named, chunk, self.share_ipc,
                                       self.radius, self.fp_size, self.descriptors)
                   for chunk in chunks]
        results = [future.result() for future in futures]
        out = [None] * len(uniq)
        for i, chunk_result in enumerate(results):
            out[i::n_chunks] = chunk_result
        return out
