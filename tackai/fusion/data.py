"""One data handler for training and inference.

:class:`FusionData` owns the block layout: it turns the curated CSVs into a design matrix
with scaffold-grouped cross-validation splits for training, and turns inference records into
rows of the same matrix. Context blocks come from the cached embedding tables and are cached
again per table-content hash; molecular features are always computed on the fly, so a
screening loop can score molecules that did not exist when the models were fitted.
"""
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import StratifiedGroupKFold

from tackai.data.utils import get_cache_dir
from tackai.fusion.blocks import BLOCK_ORDER, block_index
from tackai.fusion.context import CONTEXT_BLOCKS, ContextEncoder
from tackai.fusion.features import FP_RADIUS, FP_SIZE, MolFeaturizer

RDLogger.DisableLog("rdApp.*")

DMAX_THR = 0.80          # Dmax fraction above which a degrader counts as active
PDC50_THR = 6.0          # pDC50 above which a degrader counts as active (DC50 < 1 uM)
TASKS = ("dmax", "pdc50", "activity")
TASK_TYPES = {"dmax": "regression", "pdc50": "regression", "activity": "binary"}
TASK_LABELS = {"dmax": "Dmax (fraction)", "pdc50": "pDC50", "activity": "P(active)"}
#: Range each target can physically take; ``None`` means unbounded.
TASK_SUPPORT = {"dmax": (0.0, 1.0), "pdc50": None, "activity": (0.0, 1.0)}

N_STRAT_BINS = 5
REPEAT_SEEDS = [1000 + r for r in range(5)]
CACHE_VERSION = "v1"

MOL_BLOCKS_IN_ORDER = ("fingerprint", "descriptors")


def make_targets(dmax_pct, dc50_nM) -> Dict[str, np.ndarray]:
    """Build the three task targets from raw Dmax [%] and DC50 [nM].

    Args:
        dmax_pct: Dmax in percent (NaN if unknown).
        dc50_nM: DC50 in nM (NaN if unknown).

    Returns:
        Dict with float arrays ``dmax`` (on a [0, 1] scale), ``pdc50`` and ``activity``
        (in {0, 1}); NaN marks an undefined target. A single known measurement can still
        decide *inactive*, but never *active*, which needs both.
    """
    dmax_pct = np.asarray(dmax_pct, dtype=float)
    dc50_nM = np.asarray(dc50_nM, dtype=float)
    dmax = dmax_pct / 100.0
    pdc50 = np.full_like(dc50_nM, np.nan)
    ok = dc50_nM > 0                                  # NaN compares False
    pdc50[ok] = 9.0 - np.log10(dc50_nM[ok])           # == -log10(DC50[nM] * 1e-9)

    d_known, p_known = ~np.isnan(dmax), ~np.isnan(pdc50)
    activity = np.full(len(dmax), np.nan)
    both = d_known & p_known
    activity[both] = ((dmax[both] > DMAX_THR) & (pdc50[both] > PDC50_THR)).astype(float)
    only_d, only_p = d_known & ~p_known, p_known & ~d_known
    activity[only_d & (dmax < DMAX_THR)] = 0.0
    activity[only_p & (pdc50 < PDC50_THR)] = 0.0
    return {"dmax": dmax, "pdc50": pdc50, "activity": activity}


def scaffold_groups(smiles: Sequence[str], verbose: bool = False) -> Tuple[np.ndarray, dict]:
    """Integer group id per row from generic Murcko scaffolds.

    Args:
        smiles: SMILES strings.
        verbose: Print how many molecules fell back to a group of their own.

    Returns:
        ``(groups, info)``: group ids ``0..G-1`` and a dict with the scaffold strings and
        failure counts. A molecule with no ring system (or one RDKit cannot sanitise) gets a
        unique group, so it can never leak between train and test.
    """
    smiles = list(smiles)
    scaf: Dict[str, str] = {}
    for s in dict.fromkeys(smiles):
        try:
            core = MurckoScaffold.GetScaffoldForMol(Chem.MolFromSmiles(s))
            scaf[s] = Chem.MolToSmiles(MurckoScaffold.MakeScaffoldGeneric(core))
        except Exception:                      # sanitisation error / unparseable molecule
            scaf[s] = ""
        if scaf[s] == "":
            scaf[s] = f"__failed__{s}"
    ids = {k: i for i, k in enumerate(dict.fromkeys(scaf[s] for s in smiles))}
    groups = np.array([ids[scaf[s]] for s in smiles], dtype=int)
    info = {"scaffold": scaf,
            "n_failed_molecules": sum(v.startswith("__failed__") for v in scaf.values()),
            "n_failed_rows": int(sum(scaf[s].startswith("__failed__") for s in smiles))}
    if verbose:
        print(f"scaffold failures: {info['n_failed_molecules']} distinct molecules, "
              f"{info['n_failed_rows']} rows -> each got its own group")
    return groups, info


def _strat_labels(y, n_bins: int = N_STRAT_BINS) -> np.ndarray:
    """Stratification variable: quantile bins of a continuous target, labels if few values."""
    y = np.asarray(y)
    if len(np.unique(y)) <= n_bins:
        return np.unique(y, return_inverse=True)[1]
    return pd.qcut(pd.Series(y).rank(method="first"), n_bins, labels=False).to_numpy()


def make_splits(y, groups, n_repeats: int = 5, n_folds: int = 5,
                seeds: Optional[Sequence[int]] = None) -> List[List[Tuple[np.ndarray, np.ndarray]]]:
    """``n_repeats`` x ``n_folds`` ``(train_idx, test_idx)`` pairs that never split a group.

    Args:
        y: Target values, used only for stratification.
        groups: Scaffold group id per row.
        n_repeats: Number of repeats (each with its own seed).
        n_folds: Folds per repeat.
        seeds: Seeds per repeat (default: :data:`REPEAT_SEEDS`).

    Returns:
        ``splits[repeat][fold] = (train_idx, test_idx)``.
    """
    seeds = list(seeds if seeds is not None else REPEAT_SEEDS)[:n_repeats]
    strat = _strat_labels(y)
    out = []
    for seed in seeds:
        cv = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        out.append([(tr, te) for tr, te in cv.split(np.zeros(len(y)), strat, groups)])
    return out


def _first_valid(*series: pd.Series) -> pd.Series:
    """First non-blank value per row across several columns."""
    out = series[0].copy()
    for s in series[1:]:
        out = out.where(out.notna() & (out.astype(str).str.strip() != ""), s)
    return out


def build_table(files: Sequence[Union[str, Path]]) -> pd.DataFrame:
    """Read and harmonise the curated CSVs into one row-per-measurement table.

    Handles both layouts in ``data/yaochen`` (AutoTPD+ and TACKv2), drops rows without a
    SMILES, and fills a missing POI sequence from the most frequent sequence recorded for the
    same UniProt entry. No de-duplication: repeated measurements are repeated rows.

    Args:
        files: CSV paths.

    Returns:
        Frame with columns ``source``, ``smiles``, ``e3_raw``, ``e3_seq``, ``poi_uniprot``,
        ``poi_seq``, ``cell_key``, ``assay_raw``, ``assay_time``, ``dmax_pct``, ``dc50_nM``.
    """
    parts = []
    for f in files:
        f = Path(f)
        raw = pd.read_csv(f, low_memory=False)
        units = set(raw["DC50_units"].dropna().unique())
        if not units <= {"nM"}:
            raise ValueError(f"{f.name}: unexpected DC50 units {units}")
        if "Cell_Line_STD_Accession" in raw:                       # AutoTPD+ layout
            cell_key = _first_valid(raw["Cell_Line_STD_Accession"], raw["Cell_Line_STD"],
                                    raw["Cell_Line"])
        else:                                                      # TACKv2 layout
            cell_key = _first_valid(raw["Cell_Line_ID"], raw["Cell_Line"])
        e3_raw = (_first_valid(raw["Recruiter"], raw["Recruiter_Gene"])
                  if "Recruiter_Gene" in raw else raw["Recruiter"])
        hours = raw[["DC50_h", "Dmax_h"]].astype(float)
        parts.append(pd.DataFrame({
            "source": f.stem, "smiles": raw["SMILES"], "e3_raw": e3_raw,
            "e3_seq": raw["Recruiter_Sequence"],
            "poi_uniprot": raw["Degradation_Target_Uniprot"],
            "poi_seq": raw["Degradation_Target_Sequence"],
            "cell_key": cell_key, "assay_raw": raw["Assay"],
            "assay_time": hours.mean(axis=1, skipna=True),   # DC50_h and Dmax_h agree in ~98% of rows
            "dmax_pct": raw["Dmax"].astype(float), "dc50_nM": raw["DC50"].astype(float)}))

    tab = pd.concat(parts, ignore_index=True)
    tab = tab[tab["smiles"].notna() & (tab["smiles"].astype(str).str.strip() != "")]
    tab = tab.reset_index(drop=True)
    have = tab.dropna(subset=["poi_uniprot", "poi_seq"])
    if len(have):
        uni2seq = have.groupby("poi_uniprot")["poi_seq"].agg(lambda s: s.value_counts().index[0])
        miss = tab["poi_seq"].isna() & tab["poi_uniprot"].isin(uni2seq.index)
        tab.loc[miss, "poi_seq"] = tab.loc[miss, "poi_uniprot"].map(uni2seq)
    return tab


class FusionData:
    """Design matrix, targets and splits for training; a row encoder for inference.

    Args:
        table: Harmonised measurement table (see :func:`build_table`); may be ``None`` for an
            encoder-only instance used purely for inference.
        encoder: Context embedding lookup (default: a fresh :class:`ContextEncoder`).
        featurizer: Molecular featuriser (default: a fresh :class:`MolFeaturizer`).
    """

    def __init__(self, table: Optional[pd.DataFrame] = None,
                 encoder: Optional[ContextEncoder] = None,
                 featurizer: Optional[MolFeaturizer] = None):
        self.table = table
        self.encoder = encoder or ContextEncoder()
        self.featurizer = featurizer or MolFeaturizer()
        self.dims = {"fingerprint": self.featurizer.fp_size, "descriptors": 217,
                     "e3": self.encoder.dim("e3"), "cell": self.encoder.dim("cell"),
                     "poi": self.encoder.dim("poi"), "assay": self.encoder.dim("assay"),
                     "assay_time": 1}
        self.index = block_index(self.dims)
        self.dropped: Dict[str, int] = {"poi": 0, "e3": 0, "cell": 0, "total": 0,
                                        "read": 0 if table is None else len(table)}
        self._X: Optional[np.ndarray] = None
        self._groups: Optional[np.ndarray] = None
        self._targets: Optional[Dict[str, np.ndarray]] = None
        self._splits: Dict[tuple, list] = {}

    # ---------------------------------------------------------------- construction

    @classmethod
    def from_csv(cls, files: Sequence[Union[str, Path]], *,
                 encoder: Optional[ContextEncoder] = None,
                 featurizer: Optional[MolFeaturizer] = None,
                 cache: bool = True, on_missing: str = "drop",
                 verbose: bool = False) -> "FusionData":
        """Build the design matrix from curated CSVs.

        Some measurements cannot be encoded: a POI sequence the cached table does not hold, a
        cell line outside Cellosaurus, a row with no ligase sequence at all. On the two
        development CSVs that is about 4.6% of rows. Dropping them with a report is the
        default, because a training table that refuses to build is worse than one that is
        4.6% smaller; inference (:meth:`encode`) still raises, so a typo in a screening
        request is never silently replaced.

        Args:
            files: CSV paths.
            encoder: Context embedding lookup.
            featurizer: Molecular featuriser.
            cache: Reuse (and write) the per-block npy cache under
                ``TACKAI_CACHE/fusion_blocks/<hash>/``.
            on_missing: ``"drop"`` to discard rows whose context is not in the cache (and
                record them in :attr:`dropped`), or ``"raise"`` to fail on the first one.
            verbose: Print how many rows were read, dropped and kept.

        Returns:
            A populated :class:`FusionData`.
        """
        if on_missing not in {"drop", "raise"}:
            raise ValueError(f"on_missing must be 'drop' or 'raise', got {on_missing!r}")
        data = cls(build_table(files), encoder=encoder, featurizer=featurizer)
        if on_missing == "drop":
            data._drop_unencodable(verbose=verbose)
        data._build_matrix(cache=cache)
        return data

    def _drop_unencodable(self, verbose: bool = False) -> None:
        """Remove rows whose context the cached tables cannot encode, counting them by block."""
        keep = np.ones(len(self.table), dtype=bool)
        counts = {}
        for block, column in (("poi", "poi_seq"), ("e3", "e3_seq"), ("cell", "cell_key")):
            ok = self.encoder.encodable(block, self.table[column])
            counts[block] = int((~ok).sum())
            keep &= ok
        self.dropped = {**counts, "total": int((~keep).sum()), "read": len(self.table)}
        if verbose:
            print(f"table: {self.dropped['read']} rows read, {self.dropped['total']} dropped "
                  f"(context not in the cache: " +
                  ", ".join(f"{b} {counts[b]}" for b in counts) +
                  f") -> {int(keep.sum())} kept")
        self.table = self.table[keep].reset_index(drop=True)

    def _cache_dir(self) -> Path:
        """Directory holding this table's cached blocks, keyed by content (never a label)."""
        cols = ["smiles", "e3_seq", "poi_seq", "cell_key", "assay_raw", "assay_time"]
        h = hashlib.sha1(json.dumps([CACHE_VERSION, self.encoder.protein_space,
                                     self.featurizer.radius, self.featurizer.fp_size,
                                     self.featurizer.share_ipc,
                                     sorted(self.dims.items())]).encode())
        h.update(self.table[cols].astype(str).to_csv(index=False).encode())
        return Path(get_cache_dir()) / "fusion_blocks" / h.hexdigest()[:12]

    def _build_matrix(self, cache: bool = True) -> None:
        """Fill ``X``, the scaffold groups and the targets, using the block cache if allowed."""
        directory = self._cache_dir()
        if cache and all((directory / f"{b}.npy").exists() for b in BLOCK_ORDER):
            blocks = {b: np.load(directory / f"{b}.npy") for b in BLOCK_ORDER}
        else:
            blocks = self._compute_blocks()
            if cache:
                directory.mkdir(parents=True, exist_ok=True)
                for b, arr in blocks.items():
                    np.save(directory / f"{b}.npy", arr)
                (directory / "manifest.json").write_text(json.dumps({
                    "n_rows": len(self.table), "dims": self.dims,
                    "protein_space": self.encoder.protein_space,
                    "fingerprint": [self.featurizer.radius, self.featurizer.fp_size],
                    "share_ipc": self.featurizer.share_ipc,
                    "version": CACHE_VERSION}, indent=1))
        self._X = np.concatenate([blocks[b] for b in BLOCK_ORDER], axis=1).astype(np.float32)
        self._groups, self.scaffold_info = scaffold_groups(self.table["smiles"])
        self._targets = make_targets(self.table["dmax_pct"], self.table["dc50_nM"])

    def _compute_blocks(self) -> Dict[str, np.ndarray]:
        """Featurise the molecules and look up every context block for the whole table."""
        fp, desc, ok = self.featurizer.featurize(self.table["smiles"].tolist())
        if not ok.all():
            raise ValueError(f"{(~ok).sum()} row(s) have a SMILES RDKit cannot parse; "
                             "clean the table before building the design matrix")
        blocks = {"fingerprint": fp, "descriptors": desc}
        blocks["e3"] = self.encoder.encode("e3", self.table["e3_seq"])
        blocks["cell"] = self.encoder.encode("cell", self.table["cell_key"])
        blocks["poi"] = self.encoder.encode("poi", self.table["poi_seq"])
        blocks["assay"] = self.encoder.encode("assay", self.table["assay_raw"])
        blocks["assay_time"] = self.table[["assay_time"]].to_numpy(np.float32)
        return blocks

    # ---------------------------------------------------------------- training views

    @property
    def X(self) -> np.ndarray:
        """The design matrix of the whole table."""
        if self._X is None:
            raise ValueError("this FusionData has no table; use from_csv() for training data")
        return self._X

    @property
    def groups(self) -> np.ndarray:
        """Scaffold group id per row."""
        if self._groups is None:
            raise ValueError("this FusionData has no table; use from_csv() for training data")
        return self._groups

    @property
    def smiles(self) -> np.ndarray:
        """SMILES per row."""
        return self.table["smiles"].to_numpy(object)

    @property
    def n_columns(self) -> int:
        """Width of the design matrix."""
        return sum(self.dims.values())

    @property
    def context_columns(self) -> np.ndarray:
        """Columns holding the biological context blocks."""
        return np.concatenate([self.index[b] for b in BLOCK_ORDER if b in CONTEXT_BLOCKS])

    @property
    def mol_columns(self) -> np.ndarray:
        """Columns holding the molecular blocks."""
        return np.concatenate([self.index[b] for b in MOL_BLOCKS_IN_ORDER])

    def target(self, task: str) -> np.ndarray:
        """Target values of a task for every row (NaN where undefined)."""
        if task not in TASKS:
            raise ValueError(f"task must be one of {TASKS}, got {task!r}")
        if self._targets is None:
            raise ValueError("this FusionData has no table; use from_csv() for training data")
        return self._targets[task]

    def task_rows(self, task: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Rows with a defined target for a task.

        Args:
            task: ``"dmax"``, ``"pdc50"`` or ``"activity"``.

        Returns:
            ``(row_indices, X, y, groups)`` restricted to finite targets.
        """
        y_all = self.target(task)
        idx = np.flatnonzero(np.isfinite(y_all))
        return idx, self.X[idx], y_all[idx], self.groups[idx]

    def splits(self, task: str, n_repeats: int = 5, n_folds: int = 5,
               seeds: Optional[Sequence[int]] = None) -> List[List[Tuple[np.ndarray, np.ndarray]]]:
        """Scaffold-grouped CV splits of a task's rows, generated once and kept.

        Args:
            task: Task name.
            n_repeats: Number of repeats.
            n_folds: Folds per repeat.
            seeds: Seeds per repeat.

        Returns:
            ``splits[repeat][fold] = (train_idx, test_idx)``, indexing the task's own rows.
        """
        key = (task, n_repeats, n_folds, tuple(seeds) if seeds is not None else None)
        if key not in self._splits:
            _, _, y, g = self.task_rows(task)
            self._splits[key] = make_splits(y, g, n_repeats, n_folds, seeds)
        return self._splits[key]

    # ---------------------------------------------------------------- inference

    def encode(self, records: Union[Sequence[dict], pd.DataFrame]) -> np.ndarray:
        """Encode inference records into rows of the design matrix.

        Molecular features are computed on the fly; context blocks are looked up in the
        cached tables.

        Args:
            records: Dicts (or a DataFrame) with ``smiles``, ``poi_seq``, ``e3_seq``,
                ``cell_id``, ``assay`` and optionally ``assay_time``.

        Returns:
            Array of shape ``(len(records), n_columns)``, ``float32``.
        """
        if isinstance(records, pd.DataFrame):
            records = records.to_dict("records")
        records = list(records)
        if not records:
            return np.empty((0, self.n_columns), dtype=np.float32)

        fp, desc, _ = self.featurizer.featurize([r.get("smiles") for r in records])
        blocks = {"fingerprint": fp, "descriptors": desc}
        blocks["e3"] = self.encoder.encode("e3", [r.get("e3_seq") for r in records])
        blocks["cell"] = self.encoder.encode(
            "cell", [r.get("cell_id", r.get("cell_key")) for r in records])
        blocks["poi"] = self.encoder.encode("poi", [r.get("poi_seq") for r in records])
        blocks["assay"] = self.encoder.encode("assay", [r.get("assay") for r in records])
        blocks["assay_time"] = np.array(
            [[np.nan if r.get("assay_time") is None else float(r["assay_time"])] for r in records],
            dtype=np.float32)
        return np.concatenate([blocks[b] for b in BLOCK_ORDER], axis=1).astype(np.float32)

    def encode_context(self, record: dict) -> np.ndarray:
        """Encode one experimental context, for reuse across many molecules.

        Args:
            record: Mapping with ``poi_seq``, ``e3_seq``, ``cell_id``, ``assay`` and
                optionally ``assay_time``.

        Returns:
            Array of shape ``(1, len(context_columns))`` in block order.
        """
        blocks = self.encoder.encode_context(record)
        return np.concatenate([blocks[b] for b in BLOCK_ORDER if b in CONTEXT_BLOCKS],
                              axis=1).astype(np.float32)

    def assemble(self, context_row: np.ndarray, smiles: Sequence[str]) -> np.ndarray:
        """Combine one encoded context with freshly featurised molecules.

        Args:
            context_row: Output of :meth:`encode_context`.
            smiles: SMILES strings to featurise on the fly.

        Returns:
            Array of shape ``(len(smiles), n_columns)``, ``float32``.
        """
        smiles = list(smiles)
        X = np.empty((len(smiles), self.n_columns), dtype=np.float32)
        if not smiles:
            return X
        fp, desc, _ = self.featurizer.featurize(smiles)
        X[:, self.index["fingerprint"]] = fp
        X[:, self.index["descriptors"]] = desc
        X[:, self.context_columns] = np.asarray(context_row, dtype=np.float32)
        return X
