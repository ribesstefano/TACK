"""Run PROTAC-Splitter on one or more TACK-style CSVs and save the resulting
E3-ligand / linker / POI-warhead fragments (plus the warhead's generic Murcko
scaffold) as a separate sidecar CSV per input file, keyed on unique SMILES.

Splitting is done once per unique SMILES (not once per row) since the same
compound is assayed against many targets/cell lines across these datasets.

NOTE: the XGBoost stage currently crashes on this container's scikit-learn
1.7.2 (the pretrained graph-edge-classifier pickle references
sklearn.compose._column_transformer._RemainderColsList, which isn't present
in 1.7.2 -- an upstream PROTAC-Splitter/sklearn-pin incompatibility, not a
TACK issue). This script therefore uses model="adaptive" with the XGBoost
and Transformer stages explicitly disabled, so it only ever runs the
heuristic parameter grid (no pretrained model file touched at all). That
grid alone resolves noticeably more molecules than the single default
heuristic call (e.g. 363/822 additional TACKv2 rescues in testing), though
a residual fraction of PROTACs still can't be split by heuristics alone.

Usage (inside the training.sif container):
    python scripts/run_protac_splitter.py \
        data/yaochen/TACKv2.csv \
        data/yaochen/AutoTPDplus-PROTAC-training.csv \
        data/yaochen/PROTAC-test.csv \
        --num-proc 16
"""
import argparse
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol, MakeScaffoldGeneric

from protac_splitter import split_protac


def get_scaffold(smiles: str, generic: bool = True) -> str:
    """Murcko scaffold for a SMILES string; falls back to the input SMILES
    if parsing fails or the scaffold is empty (e.g. acyclic fragments)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles
    if generic:
        scaffold = Chem.MolToSmiles(MakeScaffoldGeneric(mol))
    else:
        scaffold = Chem.MolToSmiles(GetScaffoldForMol(mol))
    return scaffold if len(scaffold) > 0 else smiles


def split_one_csv(path: Path, smiles_col: str, model: str, num_proc: int) -> pd.DataFrame:
    df = pd.read_csv(path)
    smiles = df[smiles_col].dropna().unique().tolist()
    print(f"{path.name}: {len(df)} rows, {len(smiles)} unique SMILES to split")

    extra_kwargs = {}
    if model == "adaptive":
        extra_kwargs = dict(adaptive_use_xgboost=False, adaptive_use_transformer=False)
    results = split_protac(smiles, model=model, num_proc=num_proc, verbose=1, **extra_kwargs)

    rows = []
    n_failed = 0
    for smi, r in zip(smiles, results):
        pred = r.get("default_pred_n0")
        parts = pred.split(".") if isinstance(pred, str) else []
        if len(parts) != 3:
            n_failed += 1
            e3, linker, poi = None, None, None
        else:
            e3, linker, poi = parts
        rows.append({
            "SMILES": smi,
            "E3_Ligand_SMILES": e3,
            "Linker_SMILES": linker,
            "POI_Warhead_SMILES": poi,
            "POI_Warhead_Scaffold": get_scaffold(poi, generic=True) if poi else None,
            "split_model_name": r.get("model_name"),
        })
    print(f"{path.name}: {n_failed}/{len(smiles)} SMILES failed to split into 3 fragments")
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("csv_paths", nargs="+", type=Path)
    ap.add_argument("--smiles-col", default="SMILES")
    ap.add_argument("--model", default="adaptive")
    ap.add_argument("--num-proc", type=int, default=16)
    ap.add_argument("--output-suffix", default="_protac_split")
    args = ap.parse_args()

    for path in args.csv_paths:
        out_path = path.with_name(f"{path.stem}{args.output_suffix}.csv")
        result_df = split_one_csv(path, args.smiles_col, args.model, args.num_proc)
        result_df.to_csv(out_path, index=False)
        print(f"Saved {len(result_df)} rows -> {out_path}\n")


if __name__ == "__main__":
    main()
