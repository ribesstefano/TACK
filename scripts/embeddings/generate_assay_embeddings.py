"""
Regenerate the cached assay-type text embeddings, the generation half of
``notebooks/assay_embeddings.ipynb`` with the analysis left out.

Each assay type has three short written descriptions. They are embedded with a
sentence-transformer, pooled into one unit vector per type, PCA-compressed, and written to
``TACKAI_CACHE`` as the three npz files :mod:`tackai.fusion.context` reads.

**One vocabulary**: :data:`tackai.fusion.context.ASSAY_VOCAB`, the finer of the two the
notebook wrote (the 14-category "closed" scheme is gone). It owns both halves of the rule —
the keys and the patterns that select them — so the canonisation the surrogates apply at
inference is the same one that decided what is in this table. The vectors depend on the
*descriptions* only; the dataset no longer decides the vocabulary, and is read here purely to
report coverage.

**To add a category**, extend ``ASSAY_VOCAB`` and ``ASSAY_RULES`` in
:mod:`tackai.fusion.context`, add its descriptions to :data:`ASSAY_DESCRIPTIONS` below, and
rerun this script. It refuses to start while a key has no descriptions, and the vocabulary
size is part of the file name (``vocab=open<N>``), so a new category lands the table under a
name no release has used — update ``CONTEXT_FILES`` to match, as the script tells you to.

Coverage is read from the public ``ailab-bio/AutoTPDplus`` dataset on the Hugging Face Hub,
config ``autotpdplus-protac``; see ``scripts/embeddings/_dataset.py``.

    python scripts/embeddings/generate_assay_embeddings.py
    python scripts/embeddings/generate_assay_embeddings.py --dry-run  # report, write nothing

``--verify`` recomputes the tables and compares them against the files already in the cache
instead of overwriting, which is how you check this script still reproduces the published
embeddings. Expect agreement to about 1e-7 on the full tables rather than bit-equality: the
sentence-transformer forward pass is float32 and not bit-reproducible across library versions
or thread counts, and the PCA projection amplifies that drift to ~1e-5. Hence the default
``--atol``. A *logic* difference shows up orders of magnitude larger than this, or as a
difference in the key set.

Because that drift would change the published file's checksum for no change in meaning,
regenerate the assay cache only when the vocabulary or a description actually changes —
folding a new abbreviation into an existing type (``wb`` -> western blot) does not, since the
vectors depend on the descriptions and not on the data. Adding a *key*, however, refits the
PCA over one more row and so moves every compressed vector, not only the new one.

Author: Stefano Ribes
"""
import os
import sys

# torch and xgboost each bundle their own libomp and segfault when both run multi-threaded in
# one process. Must be set before numpy/torch are imported.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import json
import logging
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _dataset import add_source_arguments, load_env, load_frames, source_record

REPO_ROOT = Path(__file__).resolve().parent.parent
load_env(REPO_ROOT)

from tackai.data.utils import get_cache_dir
from tackai.data.embeddings.utils import EmbeddingMixin
from tackai.fusion.context import ASSAY_RULES, ASSAY_VOCAB, CONTEXT_FILES, normalize_assay

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

MODEL_NAME = "sentence-transformers/all-mpnet-base-v1"   # same encoder as the cell-line embeddings
MODEL_TAG = MODEL_NAME.split("/")[-1]
N_DESCRIPTIONS = 3        # short descriptions written per canonical assay
PCA_DIMS = 8              # size of the compressed embedding cached next to the full one
ASSAY_COLUMNS = ["Assay", "Assay_STD"]   # Assay_STD is the upstream, coarser standardisation
DESCRIPTIONS_FILE = "assay_descriptions.json"


#: ``ASSAY_VOCAB`` key -> the descriptions embedded and pooled into that key's vector. Keyed by
#: the vocabulary key itself: one key, one concept, one set of descriptions. The text of the
#: types the notebook already wrote is unchanged, so their full-table vectors are unchanged too.
ASSAY_DESCRIPTIONS = {
    "western blot": [
        "Western blot: proteins from cell lysate are separated by gel electrophoresis, transferred to a membrane and detected with a specific antibody.",
        "Western blot with band densitometry measures how much target protein remains after degrader treatment.",
        "Western blotting is a low-throughput, antibody-based, semi-quantitative readout of protein abundance in cell lysates.",
    ],
    "in-cell immunoassay": [
        "In-Cell Western: a microplate immunofluorescence assay that quantifies a target protein inside fixed cells using fluorescently labeled antibodies.",
        "In the In-Cell Western, cells are fixed and permeabilized in the wells, probed with near-infrared antibodies and scanned without any gel.",
        "In-Cell Western is a higher-throughput, in-situ alternative to western blot for measuring protein degradation in cells.",
    ],
    "capillary immunoassay": [
        "Capillary immunoassay (Simple Western, Wes, Jess): proteins are separated by size and detected with antibodies inside a capillary.",
        "The capillary immunoassay is a fully automated, capillary-based western blot with chemiluminescent or fluorescent detection.",
        "Capillary immunoassays quantify target protein from very small lysate volumes with better reproducibility than a gel-based western blot.",
    ],
    "hibit": [
        "HiBiT assay: the target protein carries an 11-amino-acid HiBiT tag that binds LgBiT to reconstitute active NanoLuc luciferase.",
        "In the HiBiT assay, the luminescence of the reconstituted HiBiT-LgBiT enzyme is proportional to the amount of tagged protein, in lysed or live cells.",
        "The HiBiT assay is a sensitive, plate-based luminescent readout of degradation for endogenously tagged proteins.",
    ],
    "nanoluc reporter": [
        "NanoLuc reporter assay: the target protein is fused to NanoLuc luciferase, so degradation lowers the luminescent signal.",
        "A NanoLuc fusion reporter's luminescence reflects protein level and is often normalized to a second luciferase as internal control.",
        "The NanoLuc reporter assay is a plate-based luminescence readout, endpoint or kinetic, of reporter-tagged target degradation in living cells.",
    ],
    "htrf": [
        "HTRF (homogeneous time-resolved fluorescence): a time-resolved FRET technology in which two antibodies bound to the target protein bring a lanthanide donor and a fluorescent acceptor together.",
        "HTRF is a no-wash, plate-based immunoassay in cell lysate where the FRET signal ratio is proportional to the target protein level.",
        "HTRF gives high-throughput antibody-pair detection of protein degradation using a lanthanide donor and a fluorescent acceptor.",
    ],
    "elisa": [
        "ELISA: an enzyme-linked immunosorbent assay captures the target protein with an antibody and detects it with an enzymatic colorimetric readout.",
        "ELISA is a plate-based sandwich immunoassay that quantifies protein concentration in cell lysate.",
        "ELISA gives an antibody-based, quantitative, medium-throughput measurement of the target protein remaining after degrader treatment.",
    ],
    "msd": [
        "MSD: Meso Scale Discovery electrochemiluminescence immunoassay run in multi-spot microplates.",
        "In the MSD assay the target protein is captured on an electrode surface and detected with a SULFO-TAG labeled antibody that emits light on electrical stimulation.",
        "MSD is a sensitive, wide-dynamic-range, plate-based immunoassay for quantifying protein degradation in lysates.",
    ],
    "flow cytometry": [
        "Flow cytometry: single cells in suspension are analyzed for fluorescence as they stream past a laser.",
        "Flow cytometry measures target protein level per cell with an antibody stain or a fluorescent protein reporter such as GFP, often as a ratio to mCherry.",
        "Flow cytometry (FACS) is a single-cell quantification of degradation that reports the whole cell population distribution.",
    ],
    "high-content imaging": [
        "High-content imaging: automated microscopy of multi-well plates followed by image analysis that extracts fluorescence features per cell.",
        "High-content platforms such as the Operetta quantify the target protein signal in thousands of cells across a dose series.",
        "High-content imaging is a quantitative, high-throughput readout of protein degradation with single-cell and subcellular resolution.",
    ],
    "fluorescence imaging": [
        "Fluorescence imaging: protein levels are visualized by immunofluorescence or a fluorescent protein fusion under a microscope.",
        "In fluorescence imaging, cells are stained with a fluorescent antibody and the signal intensity is quantified from microscopy images.",
        "Fluorescence imaging reads out protein abundance and localization by microscopy, and is less automated than high-content screening.",
    ],
    "mass spectrometry": [
        "Mass spectrometry proteomics: peptides from cell lysate are identified and quantified by liquid chromatography tandem mass spectrometry.",
        "Mass spectrometry proteomics gives an unbiased, proteome-wide measurement of protein abundance changes after degrader treatment, for example with isobaric labeling.",
        "Mass spectrometry reveals degradation selectivity across thousands of proteins without target-specific antibodies.",
    ],
    "other": [
        "Other assay: an uncommon or non-standard assay used to read out protein degradation.",
        "Other assay: a miscellaneous readout, such as a fluorescent reporter or a cell viability assay, that does not belong to a standard assay class.",
        "Other assay: some degradation or cellular activity measurement method not covered by the main assay classes.",
    ],
    "unknown": [
        "Unknown assay: the assay method is unknown.",
        "Unknown assay: no assay type was reported for this measurement.",
        "Unknown assay: the measurement technique is unspecified and the experimental readout is missing from the source record.",
    ],
    "in-cell immunoassay + western blot": [
        "In-Cell Western plus western blot: an in-cell immunoassay in microplates, confirmed by western blot.",
        "In-Cell Western plus western blot: degradation is quantified in fixed cells in the wells and then validated by gel-based immunoblotting.",
        "In-Cell Western plus western blot combines two orthogonal antibody-based assays, a plate-based in-cell immunofluorescence readout and a western blot.",
    ],
    "elisa + western blot": [
        "ELISA plus western blot: ELISA for quantification, confirmed by western blot.",
        "ELISA plus western blot: target protein is measured with a plate-based sandwich immunoassay and validated by gel-based immunoblotting.",
        "ELISA plus western blot combines two orthogonal antibody-based assays: an ELISA and a western blot.",
    ],
    "dot blot": [
        "Dot blot: lysate is spotted directly onto a membrane, without electrophoretic separation, and probed with an antibody.",
        "A dot blot is a simple, semi-quantitative antibody readout of total protein level in a sample.",
        "A dot blot is faster than a western blot but gives no information on protein size.",
    ],
    "egfp/mcherry reporter assay": [
        "EGFP/mCherry reporter assay: the target is fused to EGFP while mCherry, expressed from the same construct, serves as an internal control.",
        "In the EGFP/mCherry reporter assay, degradation appears as a drop in the EGFP to mCherry fluorescence ratio, read by flow cytometry or imaging.",
        "The EGFP/mCherry reporter is a ratiometric fluorescent protein reporter in living cells that corrects for expression level and cell number.",
    ],
    "fluorescent reporter": [
        "Fluorescent reporter assay: the target is fused to a fluorescent protein, so degradation lowers the cellular fluorescence.",
        "In a fluorescent reporter assay, the fluorescence intensity of a GFP-tagged protein is measured in living cells with a plate reader or a microscope.",
        "A fluorescent reporter is a live-cell, antibody-free readout of target protein level through a fluorescent protein tag.",
    ],
    "incell hunter": [
        "InCELL Hunter: an enzyme fragment complementation assay in which a small tag on the target protein produces a chemiluminescent signal.",
        "In the InCELL Hunter assay, the signal from the complemented beta-galactosidase fragments is proportional to the tagged protein level, read after cell lysis with the detection reagent.",
        "InCELL Hunter is a homogeneous, plate-based, antibody-free measurement of protein degradation.",
    ],
    "fluoresyn nanobody biosensor (gfp immunostaining)": [
        "Nanobody biosensor assay: a fluorescent nanobody-based probe detects a GFP-tagged target protein by immunostaining.",
        "In the nanobody biosensor assay, the fluorescence of the GFP-binding probe reports how much tagged target remains in the cells.",
        "The nanobody biosensor assay is an imaging-based, GFP immunostaining readout of target protein abundance.",
    ],
    "viability assay": [
        "Cell viability assay: a colorimetric or luminescent readout of metabolically active cells, such as MTS, CCK-8 or CellTiter-Glo.",
        "A cell viability assay measures cell proliferation or cytotoxicity rather than the abundance of one specific protein.",
        "A cell viability assay is a cell-health measurement used alongside degradation assays to gauge the antiproliferative effect of a compound.",
    ],
    # ---- added for the AutoTPD+ living release; see ASSAY_RULES for what reaches each ----
    "alphalisa": [
        "AlphaLISA: a bead-based amplified luminescent proximity homogeneous assay in which donor and acceptor beads brought together by antibodies bound to the target protein emit light on excitation.",
        "In the AlphaLISA assay the no-wash luminescent signal from cell lysate is proportional to the amount of target protein remaining after degrader treatment.",
        "AlphaLISA and AlphaLISA SureFire Ultra are homogeneous, plate-based immunoassays for quantifying protein degradation without electrophoresis.",
    ],
    "nanobit": [
        "NanoBiT: a split luciferase complementation assay in which the target protein carries one NanoLuc fragment that reconstitutes an active enzyme with its complementary subunit.",
        "In the NanoBiT assay the luminescence of the complemented split NanoLuc reports how much tagged target protein is present in live cells.",
        "NanoBiT is a plate-based, split-reporter luminescent readout of protein level and protein-protein proximity in living cells.",
    ],
    "luciferase reporter": [
        "Luciferase reporter assay: the target protein is fused to a luciferase, so degradation lowers the luminescent signal of the lysate or the live cells.",
        "In a firefly or Renilla luciferase reporter assay the measured luminescence is proportional to the amount of reporter-tagged protein, often normalized to a second luciferase.",
        "A luciferase reporter assay is a plate-based luminescence readout of tagged target abundance, distinct from the smaller and brighter NanoLuc reporters.",
    ],
    "gel densitometry": [
        "Gel densitometry: protein bands separated by SDS-PAGE are stained, for example with Coomassie, and quantified by the optical density of each band.",
        "In a gel densitometry readout the amount of target protein is measured directly from band intensity, without any antibody.",
        "Gel densitometry is a label-free, low-throughput quantification of protein abundance from a stained polyacrylamide gel.",
    ],
}



class AssayEmbedding(EmbeddingMixin):
    """Store for the precomputed assay-type vectors; saves and loads like the other TACK
    embeddings. An assay type absent from the table raises rather than being embedded on the
    fly — the vocabulary is closed by ``ASSAY_VOCAB``."""

    def _transform_batch(self, keys):
        raise KeyError(f"no cached embedding for {keys}")


def load_raw_assays(frames) -> pd.DataFrame:
    """Collect the assay columns of every source into one long frame.

    Args:
        frames: Curated tables as DataFrames (see ``scripts/embeddings/_dataset.py``).

    Returns:
        DataFrame with columns ``source``, ``column`` (``Assay`` or ``Assay_STD``) and ``raw``
        (the string, ``None`` when missing).
    """
    parts = []
    for i, frame in enumerate(frames):
        label = str(frame.attrs.get("name", f"frame{i}"))
        for col in [c for c in ASSAY_COLUMNS if c in frame.columns]:
            # pandas may hand over pd.NA; the rule sets expect None/NaN.
            raw = frame[col].astype(object).where(frame[col].notna(), None)
            parts.append(pd.DataFrame({"source": label, "column": col, "raw": raw}))
    if not parts:
        raise ValueError(f"no source has an {' or '.join(ASSAY_COLUMNS)} column")
    return pd.concat(parts, ignore_index=True)


def check_descriptions() -> None:
    """Refuse to run unless every vocabulary key has exactly ``N_DESCRIPTIONS`` descriptions.

    This is the guard that makes the vocabulary extensible: a key added to ``ASSAY_VOCAB``
    without descriptions would otherwise be written as a missing or borrowed vector.

    Raises:
        SystemExit: If a key has no descriptions, the wrong number of them, or descriptions
            exist for a key the vocabulary does not hold.
    """
    missing = [k for k in ASSAY_VOCAB if k not in ASSAY_DESCRIPTIONS]
    orphans = [k for k in ASSAY_DESCRIPTIONS if k not in ASSAY_VOCAB]
    wrong = {k: len(ASSAY_DESCRIPTIONS[k]) for k in ASSAY_VOCAB
             if k in ASSAY_DESCRIPTIONS and len(ASSAY_DESCRIPTIONS[k]) != N_DESCRIPTIONS}
    problems = []
    if missing:
        problems.append(f"no descriptions for {missing}: add {N_DESCRIPTIONS} to "
                        f"ASSAY_DESCRIPTIONS in this script")
    if orphans:
        problems.append(f"descriptions for {orphans}, which are not in ASSAY_VOCAB: add them to "
                        f"tackai.fusion.context.ASSAY_VOCAB or drop the descriptions")
    if wrong:
        problems.append(f"expected {N_DESCRIPTIONS} descriptions per key, got {wrong}")
    if problems:
        raise SystemExit("assay vocabulary and descriptions disagree -- " + "; ".join(problems))


def report_coverage(raw_df: pd.DataFrame) -> Dict[str, Dict[str, int]]:
    """Log how the dataset's assay strings distribute over the vocabulary, and return the counts.

    The vocabulary is fixed, so this does not decide what is written — it is the signal for
    *whether a category is worth adding*: a raw string landing in ``"other"`` is a real
    readout this vocabulary has no key for.

    Args:
        raw_df: Long frame from :func:`load_raw_assays`.

    Returns:
        ``column -> {assay key: row count}``.
    """
    counts = {}
    for col in sorted(raw_df["column"].unique()):
        sub = raw_df[raw_df["column"] == col]
        keys = sub["raw"].map(normalize_assay)
        counts[col] = keys.value_counts().to_dict()
        logger.info("  %s: %d rows over %d of %d keys", col, len(sub), keys.nunique(),
                    len(ASSAY_VOCAB))
        unused = [k for k in ASSAY_VOCAB if k not in set(keys)]
        if unused:
            logger.info("    never used by this dataset: %s", ", ".join(unused))
        for bucket in ("other", "unknown"):
            texts = sub.loc[(keys == bucket) & sub["raw"].notna(), "raw"]
            if not len(texts):
                continue
            top = texts.value_counts()
            logger.info("    -> %r: %d rows, %d distinct strings; most common: %s",
                        bucket, int((keys == bucket).sum()), top.size,
                        "; ".join(f"{t!r} ({n})" for t, n in top.head(3).items()))
    return counts


def encode_descriptions():
    """Embed the descriptions of every vocabulary key and pool each key into one unit vector.

    Returns:
        ``(key -> vector, embedding dim, the encoder's own token pooling mode)``.
    """
    from sentence_transformers import SentenceTransformer, models as st_models

    model = SentenceTransformer(MODEL_NAME, device="cpu")
    pooling = next(m for m in model if isinstance(m, st_models.Pooling)).pooling_mode
    keys = list(ASSAY_VOCAB)
    sentences = [d for k in keys for d in ASSAY_DESCRIPTIONS[k]]
    embs = model.encode(sentences, batch_size=32, convert_to_numpy=True,
                        normalize_embeddings=True, show_progress_bar=False).astype(np.float32)
    embs = embs.reshape(len(keys), N_DESCRIPTIONS, -1)
    # Pool the unit-norm description vectors into a single unit vector: re-normalised mean.
    key_emb = {k: embs[i].mean(axis=0) / np.linalg.norm(embs[i].mean(axis=0))
               for i, k in enumerate(keys)}
    return key_emb, model.get_embedding_dimension(), pooling


def stem_of(vocab_size: int, pooling: str) -> str:
    # "open" is kept in the name for continuity with the published tables, which encoded the
    # scheme alongside its size; it is the only scheme now.
    return f"vocab=open{vocab_size}_model={MODEL_TAG}_pooling={pooling}"


def cache_files(vocab_size: int, pooling: str) -> dict:
    stem = stem_of(vocab_size, pooling)
    return {"full": f"assay_embeddings_{stem}.npz",
            "pca": f"assay_embeddings_{stem}_pca{PCA_DIMS}.npz",
            "pca_model": f"assay_pca_{stem}_pca{PCA_DIMS}.npz"}


def compare_to_cached(path: Path, table: dict, atol: float) -> str:
    """One-line verdict comparing a freshly built table against the npz already on disk."""
    if not path.exists():
        return f"MISSING   {path.name} is not in the cache"
    with np.load(path, allow_pickle=False) as cached:
        cached_keys, new_keys = set(cached.keys()), set(table)
        if cached_keys != new_keys:
            only_cached, only_new = sorted(cached_keys - new_keys), sorted(new_keys - cached_keys)
            return (f"DIFFERS   {path.name}: keys differ "
                    f"(only cached: {only_cached[:4]}, only new: {only_new[:4]})")
        worst = max(float(np.abs(np.asarray(cached[k]) - table[k]).max()) for k in new_keys)
    verdict = "OK" if worst <= atol else "DIFFERS"
    return f"{verdict:9s} {path.name}: {len(table)} keys, max abs diff {worst:.3e}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate the cached assay-type embeddings (generation only, no analysis).")
    add_source_arguments(parser)
    parser.add_argument("--cache-dir", default=None,
                        help="Directory to write the npz files to (default: TACKAI_CACHE).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report the vocabulary, coverage and file names, write nothing.")
    parser.add_argument("--verify", action="store_true",
                        help="Compare the rebuilt tables against the cached ones instead of "
                             "overwriting; exits non-zero on any difference.")
    parser.add_argument("--atol", type=float, default=1e-4,
                        help="Absolute tolerance used by --verify. The default (1e-4) is set "
                             "above the float32 drift of a non-deterministic transformer "
                             "forward pass as amplified by the PCA projection (~1e-5), so a "
                             "difference it flags is a real one.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    check_descriptions()
    cache_dir = Path(args.cache_dir) if args.cache_dir else Path(get_cache_dir())
    vocab = list(ASSAY_VOCAB)
    frames = load_frames(args)

    raw_df = load_raw_assays(frames)
    logger.info("vocabulary: %d keys, %d rules; coverage over %d source(s), "
                "%d distinct raw strings",
                len(vocab), len(ASSAY_RULES), raw_df["source"].nunique(),
                raw_df["raw"].nunique())
    counts = report_coverage(raw_df)

    key_emb, emb_dim, pooling = encode_descriptions()
    logger.info("%s: %d keys embedded (dim %d, token pooling %s)",
                MODEL_TAG, len(key_emb), emb_dim, pooling)

    table = {key: key_emb[key] for key in vocab}
    matrix = np.stack([table[k] for k in vocab])
    # Deterministic solver, so the same vocabulary always yields the same cache.
    pca = PCA(n_components=PCA_DIMS, svd_solver="full").fit(matrix)
    reduced = dict(zip(vocab, pca.transform(matrix).astype(np.float32)))
    files = cache_files(len(vocab), pooling)
    logger.info("%d keys -> PCA %d dims (%.1f%% variance)",
                len(vocab), PCA_DIMS, 100 * pca.explained_variance_ratio_.sum())
    if files["pca"] != CONTEXT_FILES["assay"]["table"]:
        logger.warning("  !! this vocabulary writes %s, but tackai.fusion.context.CONTEXT_FILES "
                       "names %s. Update CONTEXT_FILES (and republish the context repo), or the "
                       "surrogates will keep reading the old table.",
                       files["pca"], CONTEXT_FILES["assay"]["table"])

    status = 0
    if args.verify:
        for kind, built in (("full", table), ("pca", reduced)):
            verdict = compare_to_cached(cache_dir / files[kind], built, args.atol)
            logger.info("  %s", verdict)
            status |= int(not verdict.startswith("OK"))
        return status
    if args.dry_run:
        logger.info("would write %s", ", ".join(files.values()))
        return 0

    written = []
    AssayEmbedding(embeddings=table, filename=files["full"], cache_dir=cache_dir).save()
    AssayEmbedding(embeddings=reduced, filename=files["pca"], cache_dir=cache_dir).save()
    np.savez(cache_dir / files["pca_model"],
             mean_=pca.mean_.astype(np.float32),
             components_=pca.components_.astype(np.float32),
             explained_variance_ratio_=pca.explained_variance_ratio_.astype(np.float32))
    written.extend(files.values())

    meta = {"script": "scripts/embeddings/generate_assay_embeddings.py",
            "notebook": "notebooks/assay_embeddings.ipynb", "model": MODEL_NAME,
            "pooling": {"per_description": pooling,
                        "across_descriptions": "mean of unit vectors, re-normalised"},
            "n_descriptions": N_DESCRIPTIONS, "embedding_dim": emb_dim, "pca_dims": PCA_DIMS,
            "source": {"columns": ASSAY_COLUMNS, **source_record(args)},
            "canonizer": "tackai.fusion.context.normalize_assay",
            "vocab": vocab,
            "rules": [list(r) for r in ASSAY_RULES],
            "coverage": counts,
            "descriptions": ASSAY_DESCRIPTIONS,
            "files": files}
    (cache_dir / DESCRIPTIONS_FILE).write_text(json.dumps(meta, indent=1, ensure_ascii=False))
    written.append(DESCRIPTIONS_FILE)
    for name in written:
        logger.info("%9.1f KB  %s", (cache_dir / name).stat().st_size / 1024, name)
    return status


if __name__ == "__main__":
    sys.exit(main())
