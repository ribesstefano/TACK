"""On-the-fly molecular features, bit-exact with :class:`~tackai.data.embeddings.mol_embeddings.MolEmbedding`.

Inference featurises molecules as they arrive rather than reading a cache, so a screening
loop can score molecules that have never been seen. The values must nevertheless match the
cached features the models were trained on exactly, which is what
``test_bit_exact_with_tackai_mol_embedding`` pins down: same Morgan generator settings, the
same 217 descriptors in the same order, and the same ``-1`` sentinel for values RDKit cannot
produce or that overflow.
"""
from typing import Dict, List, Sequence, Tuple

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, Graphs
from rdkit.ML.InfoTheory import entropy

RDLogger.DisableLog("rdApp.*")

FP_RADIUS = 16
FP_SIZE = 1024
DESCRIPTOR_NAMES: List[str] = [name for name, _ in Descriptors._descList]
N_DESC = len(DESCRIPTOR_NAMES)
SENTINEL = -1.0          # tackai's stand-in for a descriptor it cannot compute
DESC_LIMIT = 1e20        # above this magnitude a descriptor becomes the sentinel


def _featurize_chunk(smiles_list: Sequence[str], share_ipc: bool, radius: int,
                     fp_size: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Features of a list of unique SMILES.

    Args:
        smiles_list: SMILES strings to featurise.
        share_ipc: Compute ``Ipc`` and ``AvgIpc`` from one characteristic polynomial.
        radius: Morgan fingerprint radius.
        fp_size: Morgan fingerprint length.

    Returns:
        One ``(fingerprint, descriptors)`` pair per input, or ``None`` for an invalid SMILES.
    """
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Descriptors, Graphs, rdFingerprintGenerator
    from rdkit.ML.InfoTheory import entropy
    RDLogger.DisableLog("rdApp.*")

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=fp_size,
                                                    includeChirality=True)
    dlist = Descriptors._descList
    out = []
    for s in smiles_list:
        mol = Chem.MolFromSmiles(s) if isinstance(s, str) and s.strip() else None
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


class MolFeaturizer:
    """Morgan bits and RDKit descriptors computed on demand, de-duplicated and memoised.

    Args:
        radius: Morgan fingerprint radius.
        fp_size: Morgan fingerprint length.
        share_ipc: Compute ``Ipc`` and ``AvgIpc`` from one characteristic polynomial (the
            same values, roughly half the time: it is the most expensive descriptor).
        use_cache: Keep every featurised SMILES in memory, since a generative loop proposes
            the same molecule many times.
        n_workers: Worker processes (1 = in-process).
    """

    def __init__(self, radius: int = FP_RADIUS, fp_size: int = FP_SIZE, share_ipc: bool = True,
                 use_cache: bool = True, n_workers: int = 1):
        self.radius, self.fp_size = radius, fp_size
        self.share_ipc, self.use_cache, self.n_workers = share_ipc, use_cache, n_workers
        self.cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self._executor = None
        self.stats = {"computed": 0, "cache_hits": 0, "invalid": 0}

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
            return _featurize_chunk(uniq, self.share_ipc, self.radius, self.fp_size)
        n_chunks = min(len(uniq), self.n_workers * 4)
        chunks = [uniq[i::n_chunks] for i in range(n_chunks)]   # interleaved: balances big molecules
        futures = [self._pool().submit(_featurize_chunk, c, self.share_ipc, self.radius,
                                       self.fp_size) for c in chunks]
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
            ``(fingerprint (n, fp_size), descriptors (n, 217), ok (n,))``. An unparseable
            SMILES yields zero rows and ``ok=False`` rather than raising, so one bad molecule
            cannot abort a screening batch.
        """
        smiles = list(smiles)
        if not smiles:
            return (np.empty((0, self.fp_size), dtype=np.float32),
                    np.empty((0, N_DESC), dtype=np.float32),
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
        desc = np.zeros((len(smiles), N_DESC), dtype=np.float32)
        ok = np.zeros(len(smiles), dtype=bool)
        for i, s in enumerate(smiles):
            res = self.cache.get(s)
            if res is not None:
                fp[i], desc[i] = res
                ok[i] = True
        if not self.use_cache:
            self.cache.clear()
        return fp, desc, ok
