"""
Regenerate the cached assay-type text embeddings, the generation half of
``notebooks/assay_embeddings.ipynb`` with the analysis left out.

Each canonical assay type has three short written descriptions. They are embedded with a
sentence-transformer, pooled into one unit vector per type, PCA-compressed, and written to
``TACKAI_CACHE`` as the three npz files :mod:`tackai.fusion.context` reads. The vectors
therefore depend on the *descriptions*, not on the data: the CSVs only decide which keys make
up the open vocabulary, and so what the published file name says (``vocab=open<N>``).

Two vocabularies are written, as in the notebook: ``closed`` (14 fixed categories) and
``open`` (data-driven, finer, the one production uses). ``normalize_assay`` is imported from
:mod:`tackai.fusion.context` rather than copied, so the rule that canonises raw text has one
definition shared with the surrogates that consume these tables.

    python scripts/generate_assay_embeddings.py \
        --csv data/AutoTPD-TPDdb.csv data/auto_tpd_plus/*.csv

    python scripts/generate_assay_embeddings.py --csv ... --dry-run   # report, write nothing

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
vectors depend on the descriptions and not on the data.

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
import re
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import find_dotenv, load_dotenv
from sklearn.decomposition import PCA

REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(REPO_ROOT / ".env")
load_dotenv(find_dotenv(usecwd=True))

from tackai.data.utils import get_cache_dir
from tackai.data.embeddings.utils import EmbeddingMixin
from tackai.fusion.context import normalize_assay

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

MODEL_NAME = "sentence-transformers/all-mpnet-base-v1"   # same encoder as the cell-line embeddings
MODEL_TAG = MODEL_NAME.split("/")[-1]
N_DESCRIPTIONS = 3        # short descriptions written per canonical assay
PCA_DIMS = 8              # size of the compressed embedding cached next to the full one
ASSAY_COLUMNS = ["Assay", "Assay_STD"]   # Assay_STD is the upstream, coarser standardisation
SCHEMES = ("closed", "open")
DESCRIPTIONS_FILE = "assay_descriptions.json"


# ===================== ported verbatim from notebooks/assay_embeddings.ipynb =====================
ASSAY_VOCAB = ["western_blot", "in_cell_western", "capillary_immunoassay", "hibit", "nanoluc_reporter", "htrf",
               "elisa", "msd", "flow_cytometry", "high_content", "imaging", "mass_spec", "other", "unknown"]
# ordered: the first matching rule wins
_ASSAY_RULES = [("in_cell_western", r"in[- ]?cell western|incell western|in-cell elisa|cytoblot"),
                ("capillary_immunoassay", r"capillary|simple western|jess|\bwes\b"),
                ("hibit", r"hibit"),
                ("nanoluc_reporter", r"nanoluc|nano-glo|luciferase|luminescen"),
                ("htrf", r"htrf"),
                ("flow_cytometry", r"flow cytometry|facs"),
                ("high_content", r"high[- ]content|operetta|incell hunter"),
                ("western_blot", r"western|immunoblot|dot blot|licor"),
                ("elisa", r"elisa"),
                ("msd", r"\bmsd\b"),
                ("imaging", r"immunofluorescence|imaging|microscop"),
                ("mass_spec", r"mass spec|proteomic|lc-ms")]

def _missing(v):
    return v is None or (isinstance(v, float) and np.isnan(v)) or (isinstance(v, str) and not v.strip())

def canonical_assay(raw):
    """Map free-text assay descriptions to a category of ``ASSAY_VOCAB`` (first matching rule wins)."""
    if _missing(raw):
        return "unknown"
    s = str(raw).strip().lower()
    return next((name for name, pat in _ASSAY_RULES if re.search(pat, s)), "other")

CLOSED_VOCAB = list(ASSAY_VOCAB)

ASSAY_DESCRIPTIONS = {
    "western_blot": [
        "Western blot: proteins from cell lysate are separated by gel electrophoresis, transferred to a membrane and detected with a specific antibody.",
        "Western blot with band densitometry measures how much target protein remains after degrader treatment.",
        "Western blotting is a low-throughput, antibody-based, semi-quantitative readout of protein abundance in cell lysates.",
    ],
    "in_cell_western": [
        "In-Cell Western: a microplate immunofluorescence assay that quantifies a target protein inside fixed cells using fluorescently labeled antibodies.",
        "In the In-Cell Western, cells are fixed and permeabilized in the wells, probed with near-infrared antibodies and scanned without any gel.",
        "In-Cell Western is a higher-throughput, in-situ alternative to western blot for measuring protein degradation in cells.",
    ],
    "capillary_immunoassay": [
        "Capillary immunoassay (Simple Western, Wes, Jess): proteins are separated by size and detected with antibodies inside a capillary.",
        "The capillary immunoassay is a fully automated, capillary-based western blot with chemiluminescent or fluorescent detection.",
        "Capillary immunoassays quantify target protein from very small lysate volumes with better reproducibility than a gel-based western blot.",
    ],
    "hibit": [
        "HiBiT assay: the target protein carries an 11-amino-acid HiBiT tag that binds LgBiT to reconstitute active NanoLuc luciferase.",
        "In the HiBiT assay, the luminescence of the reconstituted HiBiT-LgBiT enzyme is proportional to the amount of tagged protein, in lysed or live cells.",
        "The HiBiT assay is a sensitive, plate-based luminescent readout of degradation for endogenously tagged proteins.",
    ],
    "nanoluc_reporter": [
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
    "flow_cytometry": [
        "Flow cytometry: single cells in suspension are analyzed for fluorescence as they stream past a laser.",
        "Flow cytometry measures target protein level per cell with an antibody stain or a fluorescent protein reporter such as GFP, often as a ratio to mCherry.",
        "Flow cytometry (FACS) is a single-cell quantification of degradation that reports the whole cell population distribution.",
    ],
    "high_content": [
        "High-content imaging: automated microscopy of multi-well plates followed by image analysis that extracts fluorescence features per cell.",
        "High-content platforms such as the Operetta quantify the target protein signal in thousands of cells across a dose series.",
        "High-content imaging is a quantitative, high-throughput readout of protein degradation with single-cell and subcellular resolution.",
    ],
    "imaging": [
        "Fluorescence imaging: protein levels are visualized by immunofluorescence or a fluorescent protein fusion under a microscope.",
        "In fluorescence imaging, cells are stained with a fluorescent antibody and the signal intensity is quantified from microscopy images.",
        "Fluorescence imaging reads out protein abundance and localization by microscopy, and is less automated than high-content screening.",
    ],
    "mass_spec": [
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
    # ---- concepts that exist only in the open vocabulary ----
    "in_cell_plus_western_blot": [
        "In-Cell Western plus western blot: an in-cell immunoassay in microplates, confirmed by western blot.",
        "In-Cell Western plus western blot: degradation is quantified in fixed cells in the wells and then validated by gel-based immunoblotting.",
        "In-Cell Western plus western blot combines two orthogonal antibody-based assays, a plate-based in-cell immunofluorescence readout and a western blot.",
    ],
    "elisa_plus_western_blot": [
        "ELISA plus western blot: ELISA for quantification, confirmed by western blot.",
        "ELISA plus western blot: target protein is measured with a plate-based sandwich immunoassay and validated by gel-based immunoblotting.",
        "ELISA plus western blot combines two orthogonal antibody-based assays: an ELISA and a western blot.",
    ],
    "dot_blot": [
        "Dot blot: lysate is spotted directly onto a membrane, without electrophoretic separation, and probed with an antibody.",
        "A dot blot is a simple, semi-quantitative antibody readout of total protein level in a sample.",
        "A dot blot is faster than a western blot but gives no information on protein size.",
    ],
    "egfp_mcherry_reporter": [
        "EGFP/mCherry reporter assay: the target is fused to EGFP while mCherry, expressed from the same construct, serves as an internal control.",
        "In the EGFP/mCherry reporter assay, degradation appears as a drop in the EGFP to mCherry fluorescence ratio, read by flow cytometry or imaging.",
        "The EGFP/mCherry reporter is a ratiometric fluorescent protein reporter in living cells that corrects for expression level and cell number.",
    ],
    "fluorescent_reporter": [
        "Fluorescent reporter assay: the target is fused to a fluorescent protein, so degradation lowers the cellular fluorescence.",
        "In a fluorescent reporter assay, the fluorescence intensity of a GFP-tagged protein is measured in living cells with a plate reader or a microscope.",
        "A fluorescent reporter is a live-cell, antibody-free readout of target protein level through a fluorescent protein tag.",
    ],
    "incell_hunter": [
        "InCELL Hunter: an enzyme fragment complementation assay in which a small tag on the target protein produces a chemiluminescent signal.",
        "In the InCELL Hunter assay, the signal from the complemented beta-galactosidase fragments is proportional to the tagged protein level, read after cell lysis with the detection reagent.",
        "InCELL Hunter is a homogeneous, plate-based, antibody-free measurement of protein degradation.",
    ],
    "nanobody_biosensor": [
        "Nanobody biosensor assay: a fluorescent nanobody-based probe detects a GFP-tagged target protein by immunostaining.",
        "In the nanobody biosensor assay, the fluorescence of the GFP-binding probe reports how much tagged target remains in the cells.",
        "The nanobody biosensor assay is an imaging-based, GFP immunostaining readout of target protein abundance.",
    ],
    "viability": [
        "Cell viability assay: a colorimetric or luminescent readout of metabolically active cells, such as MTS, CCK-8 or CellTiter-Glo.",
        "A cell viability assay measures cell proliferation or cytotoxicity rather than the abundance of one specific protein.",
        "A cell viability assay is a cell-health measurement used alongside degradation assays to gauge the antiproliferative effect of a compound.",
    ],
}

# canonical key -> concept. Closed keys are their own concept; open keys point at the closest concept.
CONCEPT_OF = {
    "closed": {k: k for k in CLOSED_VOCAB},
    "open": {
        "western blot": "western_blot", "htrf": "htrf", "hibit": "hibit", "capillary immunoassay": "capillary_immunoassay",
        "elisa": "elisa", "in-cell immunoassay": "in_cell_western", "fluorescence imaging": "imaging",
        "high-content imaging": "high_content", "flow cytometry": "flow_cytometry", "nanoluc reporter": "nanoluc_reporter",
        "msd": "msd", "in-cell immunoassay + western blot": "in_cell_plus_western_blot",
        "egfp/mcherry reporter assay": "egfp_mcherry_reporter", "incell hunter": "incell_hunter", "dot blot": "dot_blot",
        "fluoresyn nanobody biosensor (gfp immunostaining)": "nanobody_biosensor", "viability assay": "viability",
        "elisa + western blot": "elisa_plus_western_blot", "fluorescent reporter": "fluorescent_reporter", "unknown": "unknown",
    },
}

# ================================= end of the ported blocks ======================================


class AssayEmbedding(EmbeddingMixin):
    """Store for the precomputed assay-type vectors; saves and loads like the other TACK
    embeddings. An assay type absent from the table raises rather than being embedded on the
    fly — the vocabulary is closed by ``CONCEPT_OF``."""

    def _transform_batch(self, keys):
        raise KeyError(f"no cached embedding for {keys}")


def load_raw_assays(files) -> pd.DataFrame:
    """Read the assay columns of every CSV into one long frame.

    Args:
        files: CSV paths.

    Returns:
        DataFrame with columns ``file``, ``column`` (``Assay`` or ``Assay_STD``) and ``raw``
        (the string, ``None`` when missing).
    """
    parts = []
    for path in files:
        df = pd.read_csv(path, usecols=lambda c: c in ASSAY_COLUMNS, low_memory=False)
        for col in df.columns:
            # pandas may hand over pd.NA; the rule sets expect None/NaN.
            raw = df[col].astype(object).where(df[col].notna(), None)
            parts.append(pd.DataFrame({"file": Path(path).name, "column": col, "raw": raw}))
    if not parts:
        raise ValueError(f"none of {[str(f) for f in files]} has an {' or '.join(ASSAY_COLUMNS)} column")
    return pd.concat(parts, ignore_index=True)


def build_vocabularies(raw_df: pd.DataFrame):
    """The closed and open vocabularies, and the per-key row counts behind the open ordering.

    The open vocabulary is ordered by how many ``Assay`` rows use each key, most frequent
    first, with ``"unknown"`` last — the order the published file was written in.

    Raises:
        ValueError: If the data yields an open key that ``CONCEPT_OF`` has no concept for.
            Such a key cannot be embedded (there is no description to embed), so the right
            fix is a new rule in :func:`~tackai.fusion.context.normalize_assay` folding it
            into an existing type, or a new concept with its three descriptions here.
    """
    closed = raw_df["raw"].map(canonical_assay)
    open_ = raw_df["raw"].map(normalize_assay)
    is_assay = raw_df["column"] == "Assay"
    counts = {"closed": closed[is_assay].value_counts(), "open": open_[is_assay].value_counts()}
    open_vocab = (sorted(set(open_) - {"unknown"}, key=lambda k: (-counts["open"].get(k, 0), k))
                  + ["unknown"])
    unknown_concepts = [k for k in open_vocab if k not in CONCEPT_OF["open"]]
    if unknown_concepts:
        raise ValueError(
            f"the data yields open assay key(s) with no concept to embed: {unknown_concepts}. "
            f"Add a rule to normalize_assay() folding them into one of "
            f"{sorted(set(CONCEPT_OF['open'].values()))}, or add a new concept with "
            f"{N_DESCRIPTIONS} descriptions to ASSAY_DESCRIPTIONS.")
    return {"closed": list(CLOSED_VOCAB), "open": open_vocab}, counts


def encode_descriptions():
    """Embed the three descriptions of every concept and pool each concept into one unit vector.

    Returns:
        ``(concept -> vector, embedding dim, the encoder's own token pooling mode)``.
    """
    from sentence_transformers import SentenceTransformer, models as st_models

    model = SentenceTransformer(MODEL_NAME, device="cpu")
    pooling = next(m for m in model if isinstance(m, st_models.Pooling)).pooling_mode
    concepts = list(ASSAY_DESCRIPTIONS)
    sentences = [d for c in concepts for d in ASSAY_DESCRIPTIONS[c]]
    embs = model.encode(sentences, batch_size=32, convert_to_numpy=True,
                        normalize_embeddings=True, show_progress_bar=False).astype(np.float32)
    embs = embs.reshape(len(concepts), N_DESCRIPTIONS, -1)
    # Pool the unit-norm description vectors into a single unit vector: re-normalised mean.
    concept_emb = {c: embs[i].mean(axis=0) / np.linalg.norm(embs[i].mean(axis=0))
                   for i, c in enumerate(concepts)}
    return concept_emb, model.get_embedding_dimension(), pooling


def stem_of(scheme: str, vocab_size: int, pooling: str) -> str:
    return f"vocab={scheme}{vocab_size}_model={MODEL_TAG}_pooling={pooling}"


def cache_files(scheme: str, vocab_size: int, pooling: str) -> dict:
    stem = stem_of(scheme, vocab_size, pooling)
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
    parser.add_argument("--csv", nargs="+", required=True, type=Path,
                        help="Curated CSV(s) whose Assay/Assay_STD columns define the open vocabulary.")
    parser.add_argument("--cache-dir", default=None,
                        help="Directory to write the npz files to (default: TACKAI_CACHE).")
    parser.add_argument("--schemes", nargs="+", default=list(SCHEMES), choices=list(SCHEMES),
                        help="Vocabulary schemes to write (default: both).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report the vocabularies and file names, write nothing.")
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
    cache_dir = Path(args.cache_dir) if args.cache_dir else Path(get_cache_dir())
    missing = [str(p) for p in args.csv if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(f"CSV(s) not found: {missing}")

    raw_df = load_raw_assays(args.csv)
    vocabs, counts = build_vocabularies(raw_df)
    logger.info("%d CSV(s), %d distinct raw strings, %d Assay rows",
                raw_df["file"].nunique(), raw_df["raw"].nunique(),
                int((raw_df["column"] == "Assay").sum()))
    for scheme in args.schemes:
        logger.info("  %s vocabulary: %d types", scheme, len(vocabs[scheme]))

    concept_emb, emb_dim, pooling = encode_descriptions()
    logger.info("%s: %d concepts embedded (dim %d, token pooling %s)",
                MODEL_TAG, len(concept_emb), emb_dim, pooling)

    status = 0
    written, files_by_scheme = [], {}
    for scheme in args.schemes:
        vocab = vocabs[scheme]
        table = {key: concept_emb[CONCEPT_OF[scheme][key]] for key in vocab}
        matrix = np.stack([table[k] for k in vocab])
        # Deterministic solver, so the same vocabulary always yields the same cache.
        pca = PCA(n_components=PCA_DIMS, svd_solver="full").fit(matrix)
        reduced = dict(zip(vocab, pca.transform(matrix).astype(np.float32)))
        files = cache_files(scheme, len(vocab), pooling)
        files_by_scheme[scheme] = files

        if args.verify:
            for kind, built in (("full", table), ("pca", reduced)):
                verdict = compare_to_cached(cache_dir / files[kind], built, args.atol)
                logger.info("  %s", verdict)
                status |= int(not verdict.startswith("OK"))
            continue
        if args.dry_run:
            logger.info("  would write %s", ", ".join(files.values()))
            continue

        AssayEmbedding(embeddings=table, filename=files["full"], cache_dir=cache_dir).save()
        AssayEmbedding(embeddings=reduced, filename=files["pca"], cache_dir=cache_dir).save()
        np.savez(cache_dir / files["pca_model"],
                 mean_=pca.mean_.astype(np.float32),
                 components_=pca.components_.astype(np.float32),
                 explained_variance_ratio_=pca.explained_variance_ratio_.astype(np.float32))
        written.extend(files.values())
        logger.info("  %s: %d types -> PCA %d dims (%.1f%% variance)",
                    scheme, len(vocab), PCA_DIMS, 100 * pca.explained_variance_ratio_.sum())

    if not args.verify and not args.dry_run:
        meta = {"script": "scripts/generate_assay_embeddings.py",
                "notebook": "notebooks/assay_embeddings.ipynb", "model": MODEL_NAME,
                "pooling": {"per_description": pooling,
                            "across_descriptions": "mean of unit vectors, re-normalised"},
                "n_descriptions": N_DESCRIPTIONS, "embedding_dim": emb_dim, "pca_dims": PCA_DIMS,
                "source": {"columns": ASSAY_COLUMNS,
                           "files": sorted(str(p) for p in args.csv)},
                "canonizers": {"closed": "canonical_assay (this script)",
                               "open": "tackai.fusion.context.normalize_assay"},
                "vocab": {s: vocabs[s] for s in args.schemes},
                "concept_of": {s: CONCEPT_OF[s] for s in args.schemes},
                "descriptions": ASSAY_DESCRIPTIONS,
                "files": files_by_scheme}
        (cache_dir / DESCRIPTIONS_FILE).write_text(json.dumps(meta, indent=1, ensure_ascii=False))
        written.append(DESCRIPTIONS_FILE)
        for name in written:
            logger.info("%9.1f KB  %s", (cache_dir / name).stat().st_size / 1024, name)
    return status


if __name__ == "__main__":
    sys.exit(main())
