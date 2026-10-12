"""On-the-fly molecular features, bit-exact with :class:`~tackai.data.embeddings.mol_embeddings.MolEmbedding`.

Inference featurises molecules as they arrive rather than reading a cache, so a screening
loop can score molecules that have never been seen. The values must nevertheless match the
cached features the models were trained on exactly, which is what
``test_bit_exact_with_tackai_mol_embedding`` pins down: same Morgan generator settings, the
same 217 descriptors in the same order, and the same ``-1`` sentinel for values RDKit cannot
produce or that overflow.
"""
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, Graphs, rdFingerprintGenerator
from rdkit.ML.InfoTheory import entropy

from tackai.chem_utils import canonicalize_smiles

RDLogger.DisableLog("rdApp.*")

FP_RADIUS = 16
FP_SIZE = 1024
# NOTE: "fr_Nhpyrrole" and "fr_Ar_NH" are actually the same, so we drop one
DESCRIPTOR_NAMES: List[str] = [name for name, _ in Descriptors._descList if name != "fr_Ar_NH"]
N_DESC = len(DESCRIPTOR_NAMES)
#: The molecular blocks of the design matrix, in column order.
MOL_BLOCKS = ("fingerprint", "descriptors")
SENTINEL = -1.0          # tackai's stand-in for a descriptor it cannot compute
DESC_LIMIT = 1e20        # above this magnitude a descriptor becomes the sentinel


def _select_descriptors(names: Optional[Sequence[str]]) -> List[Tuple[str, Callable]]:
    """Filter ``Descriptors._descList`` down to the requested descriptors.

    Args:
        names: RDKit descriptor names, or ``None`` for no descriptors at all.

    Returns:
        The matching ``(name, function)`` pairs in RDKit's order, whatever the order of ``names``.

    Raises:
        ValueError: If a name is not an RDKit descriptor.
    """
    if names is None:
        return []
    wanted = set(names)
    unknown = wanted - {name for name, _ in Descriptors._descList}   # not DESCRIPTOR_NAMES: a
    # descriptor left out of the default must still be requestable
    if unknown:
        raise ValueError(f"unknown RDKit descriptor(s): {sorted(unknown)}")
    return [(name, fn) for name, fn in Descriptors._descList if name in wanted]


def _featurize_chunk(smiles_list: Sequence[str], share_ipc: bool, radius: int, fp_size: int,
                     dlist: Sequence[Tuple[str, Callable]]) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Features of a list of unique SMILES.

    Args:
        smiles_list: SMILES strings to featurise.
        share_ipc: Compute ``Ipc`` and ``AvgIpc`` from one characteristic polynomial.
        radius: Morgan fingerprint radius.
        fp_size: Morgan fingerprint length.
        dlist: ``(name, function)`` descriptor pairs to compute, in output order; empty for
            fingerprints only.

    Returns:
        One ``(fingerprint, descriptors)`` pair per input, or ``None`` for an invalid SMILES.
    """
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=fp_size,
                                                    includeChirality=True)
    out = []
    for s in smiles_list:
        mol = canonicalize_smiles(s, return_mol=True) if isinstance(s, str) and s.strip() else None
        if mol is None:
            out.append(None)
            continue
        fp = gen.GetFingerprintAsNumPy(mol).astype(np.float32)
        shared = None
        vals = []
        for name, fn in dlist:
            try:
                if share_ipc and name in ("Ipc", "AvgIpc"):
                    if shared is None:          # one characteristic polynomial feeds both
                        dmat = Chem.GetDistanceMatrix(mol, 0)
                        cpoly = abs(Graphs.CharacteristicPolynomial(mol, np.equal(dmat, 1)))
                        ent = entropy.InfoEntropy(cpoly)
                        shared = (sum(cpoly) * ent, ent)
                    val = shared[0] if name == "Ipc" else shared[1]
                else:
                    val = fn(mol)
            except Exception:
                val = np.nan
            if val is None or np.isnan(val) or np.isinf(val) or abs(val) > DESC_LIMIT:
                val = SENTINEL
            vals.append(val)
        out.append((fp, np.array(vals, dtype=np.float32)))
    return out


def _featurize_chunk_named(smiles_list: Sequence[str], share_ipc: bool, radius: int,
                           fp_size: int, names: Sequence[str]):
    """Worker entry point: RDKit descriptor functions do not all pickle, so ship their names.

    Args:
        smiles_list: SMILES strings to featurise.
        share_ipc: See :func:`_featurize_chunk`.
        radius: Morgan fingerprint radius.
        fp_size: Morgan fingerprint length.
        names: Descriptor names, already filtered; re-resolved here against ``_descList``.

    Returns:
        What :func:`_featurize_chunk` returns.
    """
    return _featurize_chunk(smiles_list, share_ipc, radius, fp_size, _select_descriptors(names))


class MolEncoder:
    """Morgan bits and RDKit descriptors computed on demand, de-duplicated and memoised.

    Args:
        radius: Morgan fingerprint radius.
        fp_size: Morgan fingerprint length.
        descriptors: Names of the RDKit descriptors to compute (default: all of
            :data:`DESCRIPTOR_NAMES`), or ``None`` for Morgan fingerprints only. They are
            filtered out of ``Descriptors._descList``, so columns follow RDKit's order, not
            the order given here.
        share_ipc: Compute ``Ipc`` and ``AvgIpc`` from one characteristic polynomial (the
            same values, roughly half the time: it is the most expensive descriptor).
        use_cache: Keep every featurised SMILES in memory, since a generative loop
            might propose the same molecule many times.
        n_workers: Worker processes (1 = in-process).
    """

    def __init__(self, radius: int = FP_RADIUS, fp_size: int = FP_SIZE,
                 descriptors: Optional[Sequence[str]] = DESCRIPTOR_NAMES,
                 share_ipc: bool = True, use_cache: bool = True, n_workers: int = 1):
        self.radius, self.fp_size = radius, fp_size
        self._dlist = _select_descriptors(descriptors)
        self.descriptors: List[str] = [name for name, _ in self._dlist]
        self.share_ipc, self.use_cache, self.n_workers = share_ipc, use_cache, n_workers
        self.cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self._executor = None
        self.stats = {"computed": 0, "cache_hits": 0, "invalid": 0}

    @property
    def dims(self) -> Dict[str, int]:
        """Width of each molecular block: the fingerprint length and the descriptor count."""
        return {"fingerprint": self.fp_size, "descriptors": len(self.descriptors)}

    def _pool(self):
        if self._executor is None:
            from joblib.externals.loky import ProcessPoolExecutor
            self._executor = ProcessPoolExecutor(max_workers=self.n_workers, timeout=3600)
        return self._executor

    def close(self) -> None:
        """Shut the worker pool down, if one was started."""
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

    def _compute(self, uniq: List[str]) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Featurise unique SMILES, in-process or spread over the worker pool."""
        if self.n_workers <= 1 or len(uniq) < 2 * self.n_workers:
            return _featurize_chunk(uniq, self.share_ipc, self.radius, self.fp_size,
                                    self._dlist)
        n_chunks = min(len(uniq), self.n_workers * 4)
        chunks = [uniq[i::n_chunks] for i in range(n_chunks)]   # interleaved: balances big molecules
        futures = [self._pool().submit(_featurize_chunk_named, c, self.share_ipc, self.radius,
                                       self.fp_size, self.descriptors) for c in chunks]
        results = [f.result() for f in futures]
        out: List = [None] * len(uniq)
        for i, chunk_result in enumerate(results):
            out[i::n_chunks] = chunk_result
        return out

    def featurize(self, smiles: Sequence[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Features of every SMILES, in input order.

        Args:
            smiles: SMILES strings; repeats are computed once.

        Returns:
            ``(fingerprint (n, fp_size), descriptors (n, len(descriptors)), ok (n,))``. An unparseable
            SMILES yields zero rows and ``ok=False`` rather than raising, so one bad molecule
            cannot abort a screening batch.
        """
        smiles = list(smiles)
        if not smiles:
            return (np.empty((0, self.fp_size), dtype=np.float32),
                    np.empty((0, len(self.descriptors)), dtype=np.float32),
                    np.empty(0, dtype=bool))

        uniq = list(dict.fromkeys(smiles))
        todo = [s for s in uniq if not (self.use_cache and s in self.cache)]
        self.stats["cache_hits"] += len(uniq) - len(todo)
        if todo:
            for s, res in zip(todo, self._compute(todo)):
                if res is None:
                    self.stats["invalid"] += 1
                else:
                    self.stats["computed"] += 1
                if self.use_cache:
                    self.cache[s] = res
                else:
                    self.cache.setdefault(s, res)

        fp = np.zeros((len(smiles), self.fp_size), dtype=np.float32)
        desc = np.zeros((len(smiles), len(self.descriptors)), dtype=np.float32)
        ok = np.zeros(len(smiles), dtype=bool)
        for i, s in enumerate(smiles):
            res = self.cache.get(s)
            if res is not None:
                fp[i], desc[i] = res
                ok[i] = True
        if not self.use_cache:
            self.cache.clear()
        return fp, desc, ok
