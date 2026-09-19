# Quickstart

This walks through the shortest path from "installed `tackai`" to a
prediction, using the same compound/target/ligase example as the
[tutorial notebook](../tutorial/ensemble_predictor_tutorial.ipynb).

## 1. Load an ensemble from the Hub

```python
from tackai.ensemble_predictor import EnsemblePredictor, SampleInput

predictor = EnsemblePredictor.from_pretrained(
    "ailab-bio/TACK-ensembles", subfolder="bin_caruana",
)
print(predictor)
```

The first call downloads the `bin_caruana` checkpoints from
[`ailab-bio/TACK-ensembles`](https://huggingface.co/ailab-bio/TACK-ensembles)
and the shared embedding/lookup caches they need from
[`ailab-bio/TACK-cache`](https://huggingface.co/datasets/ailab-bio/TACK-cache),
then caches both locally — subsequent calls are instant. See
[Loading a Pretrained Ensemble](../guide/loading-models.md) for what's
happening under the hood.

## 2. Describe a sample

Every model in this ensemble was trained with full POI/E3 amino-acid
sequences, so `POI_Sequence` and `Ligase_Sequence` are required inputs here
(not just the gene names) — see
[Making Predictions](../guide/making-predictions.md) for which fields are
required vs. optional in general.

??? example "AR (POI) and CRBN (E3 ligase) sequences used below"
    ```python
    AR_SEQ = (
        'MEVQLGLGRVYPRPPSKTYRGAFQNLFQSVREVIQNPGPRHPEAASAAPPGASLLLLQQQQQQQQQQQQQQQQQQQQQQ'
        'QETSPRQQQQQQGEDGSPQAHRRGPTGYLVLDEEQQPSQPQSALECHPERGCVPEPGAAVAASKGLPQQLPAPPDEDDS'
        'AAPSTLSLLGPTFPGLSSCSADLKDILSEASTMQLLQQQQQEAVSEGSSSGRAREASGAPTSSKDNYLGGTSTISDNAK'
        'ELCKAVSVSMGLGVEALEHLSPGEQLRGDCMYAPLLGVPPAVRPTPCAPLAECKGSLLDDSAGKSTEDTAEYSPFKGGY'
        'TKGLEGESLGCSGSAAAGSSGTLELPSTLSLYKSGALDEAAAYQSRDYYNFPLALAGPPPPPPPPHPHARIKLENPLDY'
        'GSAWAAAAAQCRYGDLASLHGAGAAGPGSGSPSAAASSSWHTLFTAEEGQLYGPCGGGGGGGGGGGGGGGGGGGGGGGE'
        'AGAVAPYGYTRPPQGLAGQESDFTAPDVWYPGGMVSRVPYPSPTCVKSEMGPWMDSYSGPYGDMRLETARDHVLPIDYY'
        'FPPQKTCLICGDEASGCHYGALTCGSCKVFFKRAAEGKQKYLCASRNDCTIDKFRRKNCPSCRLRKCYEAGMTLGARKL'
        'KKLGNLKLQEEGEASSTTSPTEETTQKLTVSHIEGYECQPIFLNVLEAIEPGVVCAGHDNNQPDSFAALLSSLNELGER'
        'QLVHVVKWAKALPGFRNLHVDDQMAVIQYSWMGLMVFAMGWRSFTNVNSRMLYFAPDLVFNEYRMHKSRMYSQCVRMRH'
        'LSQEFGWLQITPQEFLCMKALLLFSIIPVDGLKNQKFFDELRMNYIKELDRIIACKRKNPTSCSRRFYQLTKLLDSVQP'
        'IARELHQFTFDLLIKSHMVSVDFPEMMAEIISVQVPKILSGKVKPIYFHTQ'
    )
    E3_SEQ = (
        'MAGEGDQQDAAHNMGNHLPLLPAESEEEDEMEVEDQDSKEAKKPNIINFDTSLPTSHTYLGADMEEFHGRTLHDDDSCQ'
        'VIPVLPQVMMILIPGQTLPLQLFHPQEVSMVRNLIQKDRTFAVLAYSNVQEREAQFGTTAEIYAYREEQDFGIEIVKVK'
        'AIGRQRFKVLELRTQSDGIQQAKVQILPECVLPSTMSAVQLESLNKCQIFPSKPVSREDQCSYKWWQKYQKRKFHCANL'
        'TSWPRWLYSLYDAETLMDRIKKQLREWDENLKDDSLPSNPIDFSYRVAACLPIDDVLRIQLLKIGSAIQRLRCELDIMN'
        'KCTSLCCKQCQETEITTKNEIFSLSLCGPMAAYVNPHGYVHETLTVYKACNLNLIGRPSTEHSWFPGYAWTVAQCKICA'
        'SHIGWKFTATKKDMSPQKFWGLTRSALLPTIPDTEDEISPDKVILCL'
    )
    ```

## 3. Predict

```python
sample = SampleInput(
    smiles="CC1(C)[C@H](NC(=O)c2ccc(N3CCN(CCCOc4ccc(C(=O)NC5CCC(=O)NC5=O)nc4)CC3)nc2)"
           "C(C)(C)[C@H]1Oc1ccc(C#N)c(Cl)c1",
    poi_name="AR",
    poi_sequence=AR_SEQ,
    ligase_name="CRBN",
    ligase_sequence=E3_SEQ,
    cell_line="Unknown",
    assay_type="Unknown",
    treatment_time=24.0,
)

# predict() returns {task_name: EnsemblePrediction}; this ensemble only has a 'bin' task
task_results = predictor.predict(sample, tasks=["bin"])
result = task_results["bin"]

print(result.summary())
```

```text
Task: BIN
Label: Binary Activity
Number of models: ...
Prediction: 0.8421 ± 0.0733
95% CI (percentile): [0.7103, 0.9532]
95% CI (SEM):        [0.8210, 0.8632]
```

`result` is an
[`EnsemblePrediction`][tackai.ensemble_predictor.EnsemblePrediction]: besides
`.summary()`, it exposes `weighted_mean`, `uncertainty_std`, both 95%
confidence-interval flavors, and every contributing model's individual
prediction — see [Making Predictions](../guide/making-predictions.md) for
the full field reference and batch/DataFrame variants.

## Next steps

- [Loading a Pretrained Ensemble](../guide/loading-models.md) — `subfolder`
  choices, local vs. Hub, auth for private repos.
- [Making Predictions](../guide/making-predictions.md) — dict inputs,
  batches, `predict_dataframe`.
- [Fast Screening](../guide/screening.md) — score thousands of SMILES
  against one fixed target/ligase/cell-line context efficiently.
- The [tutorial notebook](../tutorial/ensemble_predictor_tutorial.ipynb) for
  a longer, narrated walkthrough with plots.
