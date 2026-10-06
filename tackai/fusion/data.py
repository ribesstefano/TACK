"""One data handler for training and inference.

:class:`FusionData` owns the block layout: it turns the curated CSVs into a design matrix
with scaffold-grouped cross-validation splits for training, and turns inference records into
rows of the same matrix. Context blocks come from the cached embedding tables and are cached
again per table-content hash; molecular features are always computed on the fly, so a
screening loop can score molecules that did not exist when the models were fitted.
"""
import hashlib
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import StratifiedGroupKFold

from tackai.data.utils import get_cache_dir
from tackai.fusion.context import (CONTEXT_BLOCKS, DEFAULT_CONTEXT_REPO, SEQUENCE_BLOCKS,
                                   ContextEncoder, assay_time_or_default)
from tackai.fusion.mol_encoder import (DESCRIPTOR_NAMES, FP_RADIUS, FP_SIZE, MOL_BLOCKS,
                                       MolEncoder)

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
CACHE_VERSION = "v2"   # v2: a missing assay_time is stored as ASSAY_TIME_DEFAULT, not NaN

#: Every block of the design matrix, in column order: molecular first, then the context.
BLOCK_ORDER = [*MOL_BLOCKS, *CONTEXT_BLOCKS]


def block_index(dims: Dict[str, int]) -> Dict[str, np.ndarray]:
    """Column indices of every block, laid out contiguously in :data:`BLOCK_ORDER`.

    Args:
        dims: Mapping of block name to its width.

    Returns:
        Mapping of block name to an integer array of column indices.
    """
    out, start = {}, 0
    for b in BLOCK_ORDER:
        out[b] = np.arange(start, start + dims[b])
        start += dims[b]
    return out


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


def _file_sha256(path: Path) -> str:
    """Streamed sha256 of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_table(files: Sequence[Union[str, Path]]) -> pd.DataFrame:
    """Read and harmonise the curated CSVs into one row-per-measurement table.

    Handles both layouts in AutoTPD+ and TACKv2, drops rows without a
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
            "dmax_pct": raw["Dmax"].astype(float).clip(0, 100),
            "dc50_nM": raw["DC50"].astype(float)}))

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
        featurizer: Molecular featuriser (default: a fresh :class:`MolEncoder`).
        descriptors: Names of the RDKit descriptors to use (default: all of
            :data:`DESCRIPTOR_NAMES`), or ``None`` for Morgan fingerprints only. Only applies
            to the default featuriser; combining it with ``featurizer`` is an error.
    """

    def __init__(self, table: Optional[pd.DataFrame] = None,
                 encoder: Optional[ContextEncoder] = None,
                 featurizer: Optional[MolEncoder] = None,
                 descriptors: Optional[Sequence[str]] = DESCRIPTOR_NAMES):
        if featurizer is not None and descriptors is not DESCRIPTOR_NAMES:
            raise ValueError("pass either featurizer or descriptors, not both: a supplied "
                             "featurizer already fixes its own descriptors")
        self.table = table
        self.encoder = encoder or ContextEncoder()
        self.featurizer = featurizer or MolEncoder(descriptors=descriptors)
        # Nothing about the layout is assumed: the featuriser reports the molecular widths and
        # the encoder discovers the context widths from the cached tables.
        self.dims = {**self.featurizer.dims, **self.encoder.dims}
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
                 featurizer: Optional[MolEncoder] = None,
                 descriptors: Optional[Sequence[str]] = DESCRIPTOR_NAMES,
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
            descriptors: RDKit descriptor names for the default featuriser, or ``None`` for
                fingerprints only (see :class:`FusionData`).
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
        data = cls(build_table(files), encoder=encoder, featurizer=featurizer,
                   descriptors=descriptors)
        if on_missing == "drop":
            data._drop_unencodable(verbose=verbose)
        data._build_matrix(cache=cache)
        return data

    @classmethod
    def from_pretrained(cls, repo_id: Union[str, Path] = DEFAULT_CONTEXT_REPO, *,
                        revision: Optional[str] = None,
                        token: Optional[str] = None,
                        cache_dir: Optional[Union[str, Path]] = None,
                        force_download: bool = False,
                        protein_space: str = "per_block",
                        featurizer: Optional[MolEncoder] = None,
                        descriptors: Optional[Sequence[str]] = DESCRIPTOR_NAMES) -> "FusionData":
        """Build an encoder-only FusionData from the published context embedding tables.

        Downloads (or reuses a local directory of) the context tables named in
        :data:`~tackai.fusion.context.CONTEXT_FILES`, installs them into the local cache, and
        returns a :class:`FusionData` with no table — ready for :meth:`encode`,
        :meth:`encode_context` and :meth:`assemble`, but not for :attr:`X`, :attr:`groups` or
        :meth:`target` (use :meth:`from_csv` for a training table).

        Args:
            repo_id: A Hugging Face Hub dataset repo id, or a local directory written by
                ``scripts/publish_fusion_context.py``'s ``stage`` phase (default:
                :data:`~tackai.fusion.context.DEFAULT_CONTEXT_REPO`).
            revision: Hub revision, for a repo id.
            token: Hub token, for a private repo.
            cache_dir: Directory to install the tables into (default: ``TACKAI_CACHE`` via
                :func:`get_cache_dir`). Created if it does not exist.
            force_download: Overwrite a cached file whose content differs from the published
                one, instead of raising. A mismatch almost always means the cache already
                holds tables from a different, locally refitted PCA.
            protein_space: Passed to :class:`ContextEncoder`.
            featurizer: Molecular featuriser (default: a fresh :class:`MolEncoder`).
            descriptors: RDKit descriptor names for the default featuriser, or ``None`` for
                fingerprints only.

        Returns:
            A :class:`FusionData` with ``table=None``.

        Raises:
            FileNotFoundError: If the resolved source holds no ``manifest.json``.
            ValueError: If a cached file's content differs from the manifest's record of it
                (and ``force_download`` is false), or if the installed tables' widths
                disagree with the manifest's ``block_dims``.
        """
        source = Path(repo_id)
        if not source.exists():
            try:
                from huggingface_hub import snapshot_download
            except ImportError as e:
                raise ImportError(
                    "huggingface_hub is required to download from the Hub; install it, or "
                    "pass a local directory to from_pretrained() instead."
                ) from e
            source = Path(snapshot_download(repo_id=str(repo_id), repo_type="dataset",
                                            revision=revision, token=token))

        manifest_path = source / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"no manifest.json found in {source}; this does not look like a context "
                "embedding repo published by scripts/publish_fusion_context.py"
            )
        manifest = json.loads(manifest_path.read_text())

        target = Path(cache_dir) if cache_dir is not None else Path(get_cache_dir())
        target.mkdir(parents=True, exist_ok=True)
        for filename, entry in manifest["files"].items():
            dest = target / filename
            if dest.exists():
                dest_hash = _file_sha256(dest)
                if dest_hash == entry["sha256"]:
                    continue
                if not force_download:
                    raise ValueError(
                        f"{filename} already exists in {target} with different content "
                        f"(cached sha256 {dest_hash[:12]}…, published sha256 "
                        f"{entry['sha256'][:12]}…). This usually means the cache holds "
                        "tables from a different, locally refitted PCA. Pass "
                        "force_download=True to overwrite, or point cache_dir elsewhere."
                    )
            shutil.copy2(source / filename, dest)

        encoder = ContextEncoder(cache_dir=target, protein_space=protein_space)
        data = cls(table=None, encoder=encoder, featurizer=featurizer, descriptors=descriptors)

        expected = dict(manifest.get("block_dims", {}))
        combined_dim = manifest.get("combined_dim")
        if protein_space == "combined" and combined_dim is not None:
            # poi/e3 now read the shared combined table instead of their own per-block one
            # (ContextEncoder.files), so the per-block widths staged into block_dims are the
            # wrong thing to check them against.
            for seq_block in SEQUENCE_BLOCKS:
                expected[seq_block] = combined_dim
        bad = {b: (expected[b], data.dims[b]) for b in expected
              if b in data.dims and expected[b] != data.dims[b]}
        if bad:
            detail = ", ".join(f"{b}: published with {e}, the cache now has {a}"
                               for b, (e, a) in sorted(bad.items()))
            raise ValueError(
                f"block layout mismatch after installing the published context tables "
                f"({detail}). The cache already held tables for one or more blocks from a "
                "different PCA fit; clear them from the cache or point cache_dir elsewhere."
            )
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
                                     self.featurizer.share_ipc, self.featurizer.descriptors,
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
                    "descriptors": self.featurizer.descriptors,
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
        blocks["assay_time"] = np.array([[assay_time_or_default(t)] for t in self.table["assay_time"]],
                                        dtype=np.float32)
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
    def blocks_indexes(self) -> Dict[str, np.ndarray]:
        """Column indices of every block, as the ``blocks`` argument of an estimator.

        Built from the featuriser's and the encoder's own widths, so it follows this data's
        layout whatever it is. Pass it explicitly: ``GPInteraction(blocks=data.blocks_indexes)``.

        Returns:
            A fresh mapping; changing it does not change this data's layout.
        """
        return {b: idx.copy() for b, idx in self.index.items()}

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
        return np.concatenate([self.index[b] for b in MOL_BLOCKS])

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

    def encode(self, records: Union[Sequence[dict], pd.DataFrame], return_ok: bool = False):
        """Encode inference records into rows of the design matrix.

        Molecular features are computed on the fly; context blocks are looked up in the
        cached tables.

        Args:
            records: Dicts (or a DataFrame) with ``smiles``, ``poi_seq``, ``e3_seq``,
                ``cell_id``, ``assay`` and optionally ``assay_time`` (default 24 h).
            return_ok: Also return which rows' SMILES RDKit could parse, so a caller that
                needs it does not featurise a second time to find out.

        Returns:
            Array of shape ``(len(records), n_columns)``, ``float32``; with ``return_ok``,
            ``(array, ok)``.
        """
        if isinstance(records, pd.DataFrame):
            records = records.to_dict("records")
        records = list(records)
        if not records:
            empty = np.empty((0, self.n_columns), dtype=np.float32)
            return (empty, np.empty(0, dtype=bool)) if return_ok else empty

        fp, desc, ok = self.featurizer.featurize([r.get("smiles") for r in records])
        blocks = {"fingerprint": fp, "descriptors": desc}
        blocks["e3"] = self.encoder.encode("e3", [r.get("e3_seq") for r in records])
        blocks["cell"] = self.encoder.encode(
            "cell", [r.get("cell_id", r.get("cell_key")) for r in records])
        blocks["poi"] = self.encoder.encode("poi", [r.get("poi_seq") for r in records])
        blocks["assay"] = self.encoder.encode("assay", [r.get("assay") for r in records])
        blocks["assay_time"] = np.array([[assay_time_or_default(r.get("assay_time"))]
                                         for r in records], dtype=np.float32)
        X = np.concatenate([blocks[b] for b in BLOCK_ORDER], axis=1).astype(np.float32)
        return (X, ok) if return_ok else X

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
        if not smiles:
            return np.empty((0, self.n_columns), dtype=np.float32)
        fp, desc, _ = self.featurizer.featurize(smiles)
        return self.assemble_features(context_row, fp, desc)

    def assemble_features(self, context_row: np.ndarray, fp: np.ndarray,
                          desc: np.ndarray) -> np.ndarray:
        """Combine one encoded context with molecular features that are already computed.

        Args:
            context_row: Output of :meth:`encode_context`.
            fp: Fingerprints, shape ``(n, fp_size)``.
            desc: Descriptors, shape ``(n, n_descriptors)``.

        Returns:
            Array of shape ``(n, n_columns)``, ``float32``.
        """
        X = np.empty((len(fp), self.n_columns), dtype=np.float32)
        X[:, self.index["fingerprint"]] = fp
        X[:, self.index["descriptors"]] = desc
        X[:, self.context_columns] = np.asarray(context_row, dtype=np.float32)
        return X
