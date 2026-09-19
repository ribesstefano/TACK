# Loading the TACK Dataset

The curated TACK dataset itself is published on the Hugging Face Hub as
[`ailab-bio/TACK`](https://huggingface.co/datasets/ailab-bio/TACK) and loads
through the standard 🤗 `datasets` library — no `tackai`-specific loader
needed for read access to the raw data.

```python
from datasets import load_dataset

dmax_ds = load_dataset("ailab-bio/TACK", "Dmax", split="train")
dc50_ds = load_dataset("ailab-bio/TACK", "DC50", split="train")
multitask_ds = load_dataset("ailab-bio/TACK", "multitask", split="train")

df = dc50_ds.to_pandas()
```

## Configs

| Config | Rows | Columns | Contents |
|---|---:|---:|---|
| `default` | 6.6K | 32 | Every curated row (DC50 and/or Dmax may be present). |
| `DC50` | 4.2K | 32 | Rows with a DC50 readout. |
| `Dmax` | 2.4K | 32 | Rows with a Dmax readout. |
| `multitask` | 1.6K | 43 | Rows with both DC50 and Dmax, for multi-task training. |

Column semantics (compound SMILES, POI/recruiter identity and sequence,
assay context, DC50/Dmax value + censoring metadata, provenance) are
documented in full in the
[repository README](https://github.com/ribesstefano/TACK#dataset-structure).

## Using dataset rows as `predict()` input

Rows from this dataset map directly onto the dict form `predict()` expects
(same column names — `SMILES`, `POI_Name`, `POI_Sequence`, `Ligase_Name`,
`Cell_Line_ID`, `Assay`, `Assay_Time`, ...), which is what makes it
convenient to round-trip: load rows, predict, compare against the ground
truth.

```python
from tackai.ensemble_predictor import EnsemblePredictor, SampleInput

predictor = EnsemblePredictor.from_pretrained(
    "ailab-bio/TACK-ensembles", subfolder="dc50_caruana",
)

test_df = load_dataset("ailab-bio/TACK", "DC50", split="train").to_pandas()
test_df["Cell_Line_ID"] = test_df["Cell_Line_ID"].fillna("Unknown cell line.")
test_df["Assay"] = test_df["Assay"].fillna("Unknown")

samples = [
    SampleInput(
        smiles=row["SMILES"],
        poi_name=row["POI_Name"],
        poi_sequence=row["POI_Sequence"],
        ligase_name=row["Ligase_Name"],
        cell_line=row["Cell_Line_ID"],
        treatment_time=row["Assay_Time"],
        assay_type=row["Assay"],
    )
    for _, row in test_df.head(100).iterrows()
]

batch_results = predictor.predict(samples, tasks=["dc50"])
predicted = [r["dc50"].weighted_mean[0] for r in batch_results]
```

This is exactly the pattern used by
[`test/test_ensemble.py`](https://github.com/ribesstefano/TACK/blob/main/test/test_ensemble.py),
which validates a pretrained ensemble's MAE/CI-coverage against held-out
`ailab-bio/TACK` rows end-to-end.

!!! note "This is the only dataset config used by `EnsemblePredictor`"
    Loading `ailab-bio/TACK` is independent of `EnsemblePredictor` — you can
    use the dataset for your own analysis without ever instantiating a
    predictor. Training new models from this dataset (nested cross-validation,
    scaffold/Butina splitting, held-out set construction) is a separate,
    broader topic covered in the main repository README and `CLAUDE.md`, not
    here.
