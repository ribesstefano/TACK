from __future__ import annotations

__version__ = "1.0.0"
__author__ = "Stefano Ribes, Nils Dunlop"

# PEP 562 lazy-import map: public name → defining module.
# Heavy dependencies (torch, transformers, sklearn, …) are not loaded until
# the first attribute access, keeping CLI startup fast.
_LAZY: dict[str, str] = {
    "ProteinEmbedding":               "tackai.data.embeddings.protein_embeddings",
    "CellEmbedding":                  "tackai.data.embeddings.cell_embeddings",
    "MolEmbedding":                   "tackai.data.embeddings.mol_embeddings",
    "MLPModel":                       "tackai.models.mlp_model",
    "BERTModel":                      "tackai.models.text_emb_model",
    "MultiEmbeddingsRegressionModel": "tackai.models.multi_emb_model",
    "TACKModel":                      "tackai.models.tack_model",
    "EnsemblePredictor":              "tackai.ensemble_predictor",
    "SampleInput":                    "tackai.ensemble_predictor",
    "PreprocessedContext":            "tackai.ensemble_predictor",
    "load_config_from_yaml":          "tackai.config",
    "save_config_to_yaml":            "tackai.config",
    "get_cache_dir":                  "tackai.data.utils",
    "load_protein2embedding":         "tackai.data.utils",
    "load_cell2embedding":            "tackai.data.utils",
    "load_curated_dataset":           "tackai.data.utils",
    "avail_cell_lines":               "tackai.data.utils",
    "avail_e3_ligases":               "tackai.data.utils",
    "avail_uniprots":                 "tackai.data.utils",
    "DegradationComplexDataModule":   "tackai.data.datamodule",
    "load_datamodule":                "tackai.data.datamodule",
}


def __getattr__(name: str):
    if name in _LAZY:
        import importlib
        mod = importlib.import_module(_LAZY[name])
        value = getattr(mod, name)
        globals()[name] = value  # cache so subsequent accesses bypass __getattr__
        return value
    raise AttributeError(f"module 'tackai' has no attribute {name!r}")


def __dir__():
    return list(globals()) + list(_LAZY)
