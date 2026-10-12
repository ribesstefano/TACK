"""Gradio front end for the TACK fast ensembles, deployed as the ailab-bio/TACK-Demo Space.

The Space is a thin shell over the same three objects the tutorial notebook uses — one
:class:`~tackai.fusion.data.FusionData` context encoder and one
:class:`~tackai.fusion.xgb_ensemble.XGBStackedEnsemble` per task — so a prediction made here
and one made in a notebook agree exactly. All three load once at import time, which is what
the Space's cold start pays for (~100 MB of tables and model files); every request after that
is a table lookup plus a fingerprint.

Two things the library deliberately leaves strict are softened *for the UI only*, never by
touching the encoder: POI, E3 and cell line are offered as dropdowns built from
``vocab.json`` (see ``build_vocab.py``), so a user cannot ask for a key the cached tables
lack, and a sequence pasted by hand is checked with ``ContextEncoder.encodable`` before
``predict`` runs rather than being allowed to raise.
"""
import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")  # must precede the first xgboost import

import gradio as gr
import numpy as np
import pandas as pd

from tackai.fusion import FusionData, XGBStackedEnsemble

CONTEXT_REPO = "ailab-bio/TACK-Context-Encoding"
ENSEMBLE_REPO = "ailab-bio/TACK-Fast-Ensemble"
TASKS = ("dmax", "pdc50", "activity")
REPO_URL = "https://github.com/ribesstefano/TACK"

HERE = Path(__file__).resolve().parent

# ----------------------------------------------------------------------------------------
# Models and vocabulary
# ----------------------------------------------------------------------------------------

context = FusionData.from_pretrained(CONTEXT_REPO)
ensembles = {task: XGBStackedEnsemble.from_pretrained(ENSEMBLE_REPO, subfolder=task)
             for task in TASKS}

VOCAB = json.loads((HERE / "vocab.json").read_text())

PASTE_OPTION = "— paste a sequence below —"
POI_SEQUENCES = {entry["label"]: entry["sequence"] for entry in VOCAB["poi"]}
E3_SEQUENCES = {entry["label"]: entry["sequence"] for entry in VOCAB["e3"]}
CELL_ACCESSIONS = {entry["label"]: entry["accession"] for entry in VOCAB["cell"]}
CELL_UNKNOWN = "Unknown / not reported"

POI_CHOICES = sorted(POI_SEQUENCES) + [PASTE_OPTION]
E3_CHOICES = sorted(E3_SEQUENCES) + [PASTE_OPTION]
CELL_CHOICES = [CELL_UNKNOWN] + sorted(CELL_ACCESSIONS)
ASSAY_CHOICES = [a for a in VOCAB["assay"] if a != "unknown"] + ["unknown"]


def _label_for(mapping: dict, needle: str, fallback: str) -> str:
    """The dropdown label whose value is ``needle``, or ``fallback`` when there is none."""
    return next((label for label, value in mapping.items() if value == needle), fallback)


def _default(choices: list, *preferred: str) -> str:
    """The first of ``preferred`` present in ``choices``, else the first choice."""
    return next((p for p in preferred if p in choices), choices[0])


DEFAULT_POI = _default(POI_CHOICES, "BRD4 (O60885)")
DEFAULT_E3 = _default(E3_CHOICES, "VHL (P40337)", "CRBN (Q96SW2)")
DEFAULT_CELL = _default(CELL_CHOICES, "HeLa (CVCL_0030)")
DEFAULT_ASSAY = _default(ASSAY_CHOICES, "western blot")

# ----------------------------------------------------------------------------------------
# Context assembly
# ----------------------------------------------------------------------------------------


def resolve_context(poi_label: str, poi_paste: str, e3_label: str, e3_paste: str,
                    cell_label: str, assay: str, assay_time: float) -> dict:
    """Turn the UI's labels into an inference record's context fields.

    Args:
        poi_label: Selected target label, or :data:`PASTE_OPTION`.
        poi_paste: Raw amino-acid sequence, used only when pasting was selected.
        e3_label: Selected E3 ligase label, or :data:`PASTE_OPTION`.
        e3_paste: Raw amino-acid sequence, used only when pasting was selected.
        cell_label: Selected cell line label, or :data:`CELL_UNKNOWN`.
        assay: Assay type; free text is canonised downstream.
        assay_time: Treatment time in hours.

    Returns:
        A dict with ``poi_seq``, ``e3_seq``, ``cell_id``, ``assay`` and ``assay_time``.

    Raises:
        gradio.Error: A pasted sequence is empty, or is not one the cached tables hold.
            The check runs here so the user sees an explanation instead of the library's
            ``KeyError`` — an unseen sequence genuinely cannot be embedded on the fly.
    """
    sequences = {}
    for block, label, pasted, chooser in (("poi", poi_label, poi_paste, POI_SEQUENCES),
                                          ("e3", e3_label, e3_paste, E3_SEQUENCES)):
        if label != PASTE_OPTION:
            sequences[block] = chooser[label]
            continue
        sequence = "".join((pasted or "").split()).upper()
        if not sequence:
            raise gr.Error(f"Paste a {block.upper()} sequence, or pick one from the list.")
        if not context.encoder.encodable(block, [sequence])[0]:
            raise gr.Error(
                f"That {block.upper()} sequence is not in the published context tables, and "
                "an unseen sequence cannot be embedded on the fly — the ESM poolings behind "
                "these blocks are fitted over a whole vocabulary. Pick a listed protein, or "
                "register the embedding yourself with the Python API "
                "(`encoder.register_sequence`)."
            )
        sequences[block] = sequence

    return {
        "poi_seq": sequences["poi"],
        "e3_seq": sequences["e3"],
        # A blank accession is the table's documented "unknown" vector; an unrecognised one
        # would raise, which is why the dropdown offers no way to type one.
        "cell_id": "" if cell_label == CELL_UNKNOWN else CELL_ACCESSIONS[cell_label],
        "assay": assay,
        "assay_time": float(assay_time),
    }


def predict_all(X: np.ndarray) -> dict:
    """Run every task ensemble over one design matrix."""
    return {task: ensembles[task].predict(X) for task in TASKS}


def dc50_nM(pdc50: float) -> float:
    """DC50 in nanomolar from pDC50 = -log10(DC50 / M)."""
    return float(10 ** (9.0 - pdc50))


# ----------------------------------------------------------------------------------------
# Single prediction
# ----------------------------------------------------------------------------------------

SINGLE_COLUMNS = ["Prediction", "Value", "Uncertainty", "90% interval"]


def predict_single(smiles: str, poi_label: str, poi_paste: str, e3_label: str,
                   e3_paste: str, cell_label: str, assay: str, assay_time: float):
    """Score one molecule in one context.

    Returns:
        ``(table, notes)`` — a four-column DataFrame, one row per task, and a Markdown
        string restating the context the numbers belong to.
    """
    smiles = (smiles or "").strip()
    if not smiles:
        raise gr.Error("Enter a SMILES string.")

    record = resolve_context(poi_label, poi_paste, e3_label, e3_paste,
                             cell_label, assay, assay_time)
    X, ok = context.transform([{**record, "smiles": smiles}], return_ok=True)
    if not ok[0]:
        raise gr.Error("RDKit could not parse that SMILES.")

    out = predict_all(X)
    dmax, pdc50, activity = out["dmax"], out["pdc50"], out["activity"]

    def interval(prediction: dict, fmt: str, lo: float = -np.inf,
                 hi: float = np.inf) -> str:
        """Conformal interval as text, clipped to the task's support."""
        if "lower" not in prediction:
            return "not calibrated"
        low = float(np.clip(prediction["lower"][0], lo, hi))
        high = float(np.clip(prediction["upper"][0], lo, hi))
        return f"{low:{fmt}} – {high:{fmt}}"

    rows = [
        ("Dmax (fraction degraded)",
         f"{dmax['mean'][0]:.2f}",
         f"± {dmax['std'][0]:.2f}",
         interval(dmax, ".2f", 0.0, 1.0)),
        ("pDC50 (−log₁₀ DC50/M)",
         f"{pdc50['mean'][0]:.2f}",
         f"± {pdc50['std'][0]:.2f}",
         interval(pdc50, ".2f")),
        ("DC50 (nM, from pDC50)",
         f"{dc50_nM(pdc50['mean'][0]):,.1f}",
         "—",
         "—"),
        ("P(active): Dmax > 0.80 and DC50 < 1 µM",
         f"{activity['proba'][0]:.2f}",
         f"{activity['entropy_total'][0]:.2f} nats entropy",
         "—"),
    ]
    table = pd.DataFrame(rows, columns=SINGLE_COLUMNS)

    notes = (
        f"**Context scored** — {poi_label} degraded via {e3_label} in "
        f"{cell_label}, measured by *{assay}* at {assay_time:g} h.\n\n"
        "A wide interval or a `P(active)` near 0.5 is the ensemble saying the measurement "
        "is worth making; it is not a refusal to answer."
    )
    return table, notes


# ----------------------------------------------------------------------------------------
# Batch screening
# ----------------------------------------------------------------------------------------

BATCH_COLUMNS = ["name", "smiles", "Dmax", "Dmax ±", "pDC50", "pDC50 ±", "DC50 (nM)",
                 "P(active)"]


def parse_batch(text: str):
    """Split pasted lines into names and SMILES.

    Each non-empty line is either ``SMILES`` or ``name,SMILES`` (or ``name<TAB>SMILES``);
    a line whose first field is unnamed gets the index as its name. A leading header line
    is dropped.

    Returns:
        ``(names, smiles)``, two equal-length lists.
    """
    names, smiles = [], []
    for index, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line or line.lower().startswith(("smiles", "name,", "name\t")):
            continue
        parts = [p.strip() for p in (line.split("\t") if "\t" in line else line.split(","))]
        if len(parts) >= 2 and parts[1]:
            names.append(parts[0])
            smiles.append(parts[1])
        else:
            names.append(str(index))
            smiles.append(parts[0])
    return names, smiles


def predict_batch(text: str, poi_label: str, poi_paste: str, e3_label: str, e3_paste: str,
                  cell_label: str, assay: str, assay_time: float):
    """Score many molecules in one fixed context, ranked by predicted potency.

    The context is encoded once and re-used for every molecule, so the per-molecule cost is
    just the fingerprint and the descriptors — the same trick the tutorial's screening
    section uses.

    Returns:
        ``(table, csv_path, notes)`` — the ranked DataFrame, a CSV of it for download, and
        a Markdown string naming anything RDKit rejected.
    """
    names, smiles = parse_batch(text or "")
    if not smiles:
        raise gr.Error("Paste at least one SMILES, one per line.")
    if len(smiles) > 500:
        raise gr.Error(f"{len(smiles)} molecules is above this demo's 500-row limit; "
                       "use the Python API for larger screens.")

    record = resolve_context(poi_label, poi_paste, e3_label, e3_paste,
                             cell_label, assay, assay_time)
    fp, desc, ok = context.featurizer.featurize(smiles)
    if not ok.any():
        raise gr.Error("RDKit could not parse any of those SMILES.")
    X = context.assemble_features(context.transform_context(record), fp, desc)

    out = predict_all(X)
    table = pd.DataFrame({
        "name": names,
        "smiles": smiles,
        "Dmax": out["dmax"]["mean"].round(2),
        "Dmax ±": out["dmax"]["std"].round(2),
        "pDC50": out["pdc50"]["mean"].round(2),
        "pDC50 ±": out["pdc50"]["std"].round(2),
        "DC50 (nM)": [round(dc50_nM(p), 1) for p in out["pdc50"]["mean"]],
        "P(active)": out["activity"]["proba"].round(2),
    })[BATCH_COLUMNS]
    table = table[ok].sort_values("pDC50", ascending=False).reset_index(drop=True)

    csv_path = Path(tempfile.gettempdir()) / "tack_predictions.csv"
    table.to_csv(csv_path, index=False)

    rejected = [name for name, good in zip(names, ok) if not good]
    notes = (f"Scored {len(table)} molecule(s) in one context. "
             "Ranking is what to trust here, more than the absolute values.")
    if rejected:
        notes += f"\n\n**Skipped — RDKit could not parse:** {', '.join(rejected)}."
    return table, str(csv_path), notes


# ----------------------------------------------------------------------------------------
# Examples, drawn from the held-out literature set in the tutorial notebook
# ----------------------------------------------------------------------------------------

ACBI2 = ("Cc1ncsc1-c1ccc([C@H](CC[C@H](C)CCN2CCC(c3ccc4c(c3)n(C3CCCC3)c3nc(=O)c5c(Br)cccc5n43)"
         "CC2)NC(=O)[C@@H]2C[C@@H](O)CN2C(=O)[C@@H](NC(=O)C2(F)CC2)C(C)(C)C)cc1")
MZ1 = ("Cc1sc2c(c1C)C(c1ccc(Cl)cc1)=N[C@@H](CC(=O)Nc1ccc(OCCOCCOCCOCCNc3cccc4c3C(=O)N"
       "(C3CCC(=O)NC3=O)C4=O)cc1)c1nnc(C)n1-2")
MS67 = ("Cc1ncsc1-c1ccc([C@H](C)NC(=O)[C@@H]2C[C@@H](O)CN2C(=O)[C@@H](NC(=O)CNC(=O)c2ccc(F)c"
        "(-c3ccc(N4C[C@@H](C)N(C)[C@@H](C)C4)c(NC(=O)c4c[nH]c(=O)cc4C(F)(F)F)c3)c2)C(C)(C)C)cc1")

#: (smiles, poi, e3, cell, assay, hours) with the measured value in the comment.
SINGLE_EXAMPLES = [
    # compound 11, DC50 0.63 nM / Dmax 97%, doi:10.1021/acs.jmedchem.6b01912
    [MZ1, "BRD4 (O60885)", "CRBN (Q96SW2)", "HeLa (CVCL_0030)", "western blot", 24],
    # ACBI2, DC50 1.0 nM / Dmax 81%, doi:10.1038/s41467-022-33430-6
    [ACBI2, "SMARCA2 (P51531)", "VHL (P40337)", "RKO (CVCL_0504)",
     "capillary immunoassay", 18],
    # MS67, DC50 3.7 nM / Dmax 94%, doi:10.1126/scitranslmed.abj1578
    [MS67, "WDR5 (P61964)", "VHL (P40337)", "MV4-11 (CVCL_0064)", "western blot", 18],
]

#: Four of the ACBI2 paper's congeneric SMARCA2 degraders; measured DC50 spans 1 nM to 10 µM.
BATCH_EXAMPLE = "\n".join([
    f"ACBI2,{ACBI2}",
    "11,Cc1ncsc1-c1ccc([C@H](CCCCCN2CCC(c3ccc4c(c3)n(C3CCCC3)c3nc(=O)c5c(Br)cccc5n43)CC2)NC(=O)"
    "[C@@H]2C[C@@H](O)CN2C(=O)[C@@H](NC(=O)C2(F)CC2)C(C)(C)C)cc1",
    "23,Cc1ncsc1-c1ccc(CNC(=O)[C@@H]2C[C@@H](O)CN2C(=O)[C@@H](NC(=O)C2(F)CC2)C(C)(C)C)c"
    "(OCCCCN2CCC(c3ccc4c(c3)n(C3CCCC3)c3nc(=O)c5c(Br)cccc5n43)CC2)c1",
    "7,Cc1ncsc1-c1ccc([C@H](CCCN2CCC(c3ccc4c(c3)n(C3CCCC3)c3nc(=O)c5c(Br)cccc5n43)CC2)NC(=O)"
    "[C@@H]2C[C@H](O)CN2C(=O)[C@@H](NC(=O)C2(F)CC2)C(C)(C)C)cc1",
])

# ----------------------------------------------------------------------------------------
# Interface
# ----------------------------------------------------------------------------------------

DISCLAIMER = """
> ## ⚠️ Work in progress — do not use for decisions
>
> This Space is an **unreviewed preview** of the TACK fast ensembles, published for
> feedback while the work is still in development. Specifically:
>
> * These are **not** the models from the TACK paper. They are trained on an **unpublished,
>   extended curation** of the dataset, and neither the models nor that curation have been
>   peer reviewed.
> * The accuracy ceiling in 5×5 scaffold-grouped cross-validation is **R² ≈ 0.42 (Dmax),
>   R² ≈ 0.55 (pDC50), ROC-AUC ≈ 0.87 (activity)**. Treat a single number as a hypothesis
>   to test, not a measurement.
> * Nothing checks that your molecule is PROTAC-like — any SMILES gets a score.
> * Predictions, intervals and the interface itself may change without notice.
>
> Not for clinical, diagnostic, regulatory or safety-critical use.
"""

HEADER = f"""
# TACK — fast PROTAC degradation predictions

Score a PROTAC in a specific experimental context — target protein, E3 ligase, cell line,
assay and treatment time — and get **Dmax**, **pDC50** and **P(active)**, each with a
calibrated uncertainty. CPU only, milliseconds per molecule.

[Code]({REPO_URL}) · [Models](https://huggingface.co/{ENSEMBLE_REPO}) ·
[Context tables](https://huggingface.co/datasets/{CONTEXT_REPO}) ·
[Tutorial notebook]({REPO_URL}/blob/main/notebooks/fast_TACK_models_tutorial.ipynb)
"""

ABOUT = f"""
## What the three numbers mean

| Task | Output | Meaning |
| --- | --- | --- |
| `dmax` | fraction in [0, 1] | maximum degradation achieved |
| `pdc50` | −log₁₀(DC50 / M) | potency; 9.0 means DC50 = 1 nM |
| `activity` | probability | P(Dmax > 0.80 **and** DC50 < 1 µM) |

Each prediction carries two uncertainties that are reported together as `±`: an
**aleatoric** part the ensemble members agree on, and an **epistemic** part that is their
disagreement, which shrinks where the training data covers your context. The 90% interval is
**conformal** — calibrated to cover the truth 9 times out of 10 on data exchangeable with
the calibration set, which held-out chemistry often is not.

## Known limitations

* **Closed vocabularies.** Target, E3 ligase and cell line must be ones the published
  context tables hold, which is why they are dropdowns. An unseen protein sequence cannot be
  embedded on the fly; register its 640-d ESM-2 embedding through the Python API instead.
* **One timepoint.** `assay_time` is a single number, so there is no kinetics — the same
  PROTAC has a different Dmax at 4 h and at 24 h, but nothing here models the curve.
* **Censored measurements** in training were taken at face value (`DC50 > 10 µM` → 10 µM),
  operator columns ignored, repeated measurements not de-duplicated.
* **No applicability domain gate.** The uncertainty widens outside the training chemistry
  but nothing refuses to answer.

## Running it yourself

```python
from tackai.fusion import FusionData, XGBStackedEnsemble

context = FusionData.from_pretrained("{CONTEXT_REPO}")
model = XGBStackedEnsemble.from_pretrained("{ENSEMBLE_REPO}", subfolder="dmax")

X = context.transform([{{
    "smiles": "...", "poi_seq": "MKAP...", "e3_seq": "MAGE...",
    "cell_id": "CVCL_0333", "assay": "western blot", "assay_time": 72.0,
}}])
model.predict(X)
```

`pip install git+{REPO_URL}.git` needs no `torch` and no GPU. `FusionData.help()` and
`XGBStackedEnsemble.help(show_extras=True)` document every field and output key.

## Vocabulary in this demo

{len(POI_CHOICES) - 1} targets, {len(E3_CHOICES) - 1} E3 ligases,
{len(CELL_ACCESSIONS)} cell lines and {len(ASSAY_CHOICES)} assay types, built by
`demo/build_vocab.py` as the intersection of the curated dataset's names with the published
context tables. Proteins and cell lines the tables hold but the curation never named are
reachable through the Python API, and through the *paste a sequence* option here.
"""


def context_controls():
    """The shared experimental-context panel. Returns the components in record order."""
    with gr.Row():
        poi = gr.Dropdown(POI_CHOICES, value=DEFAULT_POI, label="Target protein (POI)",
                          info="Gene symbol and UniProt accession", filterable=True)
        e3 = gr.Dropdown(E3_CHOICES, value=DEFAULT_E3, label="E3 ligase",
                         filterable=True)
    with gr.Row():
        poi_paste = gr.Textbox(label="POI sequence", lines=2, visible=False,
                               placeholder="MKAP… (must already be in the context tables)")
        e3_paste = gr.Textbox(label="E3 sequence", lines=2, visible=False,
                              placeholder="MAGE… (must already be in the context tables)")
    with gr.Row():
        cell = gr.Dropdown(CELL_CHOICES, value=DEFAULT_CELL, label="Cell line",
                           info="Cellosaurus accession", filterable=True)
        assay = gr.Dropdown(ASSAY_CHOICES, value=DEFAULT_ASSAY, label="Assay",
                            allow_custom_value=True,
                            info="Free text is canonised; an unknown type falls back")
        hours = gr.Number(value=24, label="Treatment time (h)", minimum=0.5, maximum=240,
                          step=1)

    # The paste boxes only exist for the handful of table entries the curation never named.
    poi.change(lambda choice: gr.update(visible=choice == PASTE_OPTION),
               inputs=poi, outputs=poi_paste)
    e3.change(lambda choice: gr.update(visible=choice == PASTE_OPTION),
              inputs=e3, outputs=e3_paste)
    return poi, poi_paste, e3, e3_paste, cell, assay, hours


with gr.Blocks(title="TACK — PROTAC degradation predictions") as demo:
    gr.Markdown(DISCLAIMER)
    gr.Markdown(HEADER)

    with gr.Accordion("Experimental context", open=True):
        controls = context_controls()

    with gr.Tabs():
        with gr.Tab("One molecule"):
            smiles_box = gr.Textbox(label="SMILES", lines=3,
                                    placeholder="Paste a PROTAC SMILES string")
            run_single = gr.Button("Predict", variant="primary")
            single_table = gr.Dataframe(headers=SINGLE_COLUMNS, label="Predictions",
                                        interactive=False, wrap=True)
            single_notes = gr.Markdown()

            gr.Examples(
                examples=SINGLE_EXAMPLES,
                inputs=[smiles_box, controls[0], controls[2], controls[4], controls[5],
                        controls[6]],
                label="Held-out literature measurements (none of them in the training data)",
            )
            run_single.click(predict_single, inputs=[smiles_box, *controls],
                             outputs=[single_table, single_notes])

        with gr.Tab("Screen a series"):
            gr.Markdown("One molecule per line, as `SMILES` or `name,SMILES`. The context "
                        "above is encoded once and reused, so only the molecules cost "
                        "anything. Up to 500 rows.")
            batch_box = gr.Textbox(label="SMILES list", lines=8,
                                   placeholder="ACBI2,Cc1ncsc1-…\ncompound-2,Cc1ncsc1-…")
            run_batch = gr.Button("Predict", variant="primary")
            batch_table = gr.Dataframe(headers=BATCH_COLUMNS,
                                       label="Ranked by predicted pDC50",
                                       interactive=False, wrap=True)
            batch_notes = gr.Markdown()
            batch_csv = gr.File(label="Download predictions (CSV)")

            gr.Examples(
                examples=[[BATCH_EXAMPLE]],
                inputs=[batch_box],
                label="A congeneric SMARCA2 series (measured DC50 spans 1 nM to 10 µM)",
            )
            run_batch.click(predict_batch, inputs=[batch_box, *controls],
                            outputs=[batch_table, batch_csv, batch_notes])

        with gr.Tab("About"):
            gr.Markdown(ABOUT)


if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
