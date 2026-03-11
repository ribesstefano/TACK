#!/usr/bin/env python3
"""
Author: Yossra Gharbi

Normalize PROTAC-DB and TPDDB POI/E3 fields using UniProt.

Outputs (CSV):
- protacdb_normalized.csv
- tpddb_normalized.csv

Optional audit:
- If --audit is enabled, the script writes simple raw/post-normalization audit tables into --audit-dir.

Usage examples:
  python normalize_poi_e3_pipeline.py \
    --protacdb /path/to/PROTAC-DB.csv \
    --tpddb /path/to/tpddb_protacs.csv \
    --out-dir data/normalized \
    --audit --audit-dir data/audit

Notes:
- This script queries UniProt REST API. You need internet access.
"""

import argparse
import csv
import re
import time
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

def _clean_string_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        out[c] = out[c].astype("string").str.strip().replace({"": pd.NA})
    return out


def _parse_protacdb_line(line: str):
    s = line.strip()
    if not s:
        return None
    s = s.rstrip(";")
    if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
        s = s[1:-1]
    s = s.replace('""', '"')
    return next(csv.reader([s], delimiter=",", quotechar='"', doublequote=True, escapechar="\\"))


def load_protacdb_custom(path: str | Path) -> pd.DataFrame:
    rows = []
    path = str(path)
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        header = _parse_protacdb_line(f.readline())
        ncols = len(header)
        for line in f:
            if not line.strip():
                continue
            row = _parse_protacdb_line(line)
            if row is None:
                continue
            if len(row) != ncols:
                row = (row + [""] * ncols)[:ncols]
            rows.append(row)
    return pd.DataFrame(rows, columns=header)


def _get_with_retry(url: str, params: dict, headers: dict | None = None, timeout: int = 60,
                    retries: int = 3, backoff: float = 1.5) -> requests.Response:
    last_err = None
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=timeout)
            if r.status_code in (429, 500, 502, 503, 504):
                # transient / rate-limit
                time.sleep(backoff * (attempt + 1))
                last_err = RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
                continue
            return r
        except Exception as e:
            last_err = e
            time.sleep(backoff * (attempt + 1))
    raise RuntimeError(f"Request failed after {retries} attempts: {last_err}")


def fetch_uniprot_info(uniprots, chunk_size: int = 25) -> pd.DataFrame:

    base_url = "https://rest.uniprot.org/uniprotkb/search"

    ids = (
        pd.Series(uniprots)
        .dropna()
        .astype("string").str.strip()
        .replace({"": pd.NA})
        .dropna()
        .str.replace(r"-\d+$", "", regex=True)
        .unique()
        .tolist()
    )

    out = []
    for i in range(0, len(ids), chunk_size):
        chunk = ids[i:i + chunk_size]
        query = " OR ".join([f"accession:{u}" for u in chunk])

        r = _get_with_retry(
            base_url,
            params={
                "query": f"({query})",
                "format": "tsv",
                "fields": "accession,gene_primary,protein_name",
                "size": len(chunk),
            },
            headers={"User-Agent": "protac-normalize/1.0"},
            timeout=60,
        )
        if r.status_code != 200:
            raise RuntimeError(f"UniProt request failed: {r.status_code}\n{r.text[:500]}")

        tmp = pd.read_csv(StringIO(r.text), sep="\t")
        if tmp.empty:
            continue

        tmp = tmp.iloc[:, :3].copy()
        tmp.columns = ["uniprot", "gene_primary", "protein_name"]
        out.append(tmp)

    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["uniprot", "gene_primary", "protein_name"])


def fetch_uniprot_for_gene(gene_symbol: str) -> dict | None:

    url = "https://rest.uniprot.org/uniprotkb/search"
    query = f"(gene_exact:{gene_symbol}) AND (organism_id:9606) AND (reviewed:true)"

    r = _get_with_retry(
        url,
        params={
            "query": query,
            "format": "tsv",
            "fields": "accession,gene_primary,protein_name",
            "size": 5,
        },
        headers={"User-Agent": "protac-e3-normalize/1.0"},
        timeout=60,
    )
    if r.status_code != 200:
        raise RuntimeError(f"UniProt request failed: {r.status_code}\n{r.text[:500]}")

    tab = pd.read_csv(StringIO(r.text), sep="\t")
    if tab.empty:
        return None

    return {
        "uniprot": tab.iloc[0, 0],
        "gene_primary": tab.iloc[0, 1] if tab.shape[1] > 1 else None,
        "protein_name": tab.iloc[0, 2] if tab.shape[1] > 2 else None,
    }



# Curated rules (only used when PROTAC-DB target UniProt is missing)
POI_ALIAS = {
    "c-Src": "SRC",
    "BRD4-L": "BRD4",
    "BRD4-S": "BRD4",
    "Alpha-tubulin": "ALPHA_TUBULIN",
    "Beta3-tubulin": "TUBB3",
    "p-p38": "P38",
}

FUSION_RE = re.compile(r"^([A-Za-z0-9]+-[A-Za-z0-9]+)(?:\s+(.*))?$")


def norm_poi_no_uniprot(poi_raw):
    if pd.isna(poi_raw):
        return pd.NA, pd.NA, pd.NA

    s = str(poi_raw).strip()
    if not s:
        return pd.NA, pd.NA, pd.NA

    s = re.sub(r"\s+", " ", s)
    s = s.replace("fusion protein", "").strip()

    if s in POI_ALIAS:
        base = POI_ALIAS[s]
        if poi_raw in ["BRD4-L", "BRD4-S"]:
            return base, poi_raw.split("-")[-1], "isoform"
        if poi_raw == "p-p38":
            return base, "phospho", "state"
        if poi_raw == "Alpha-tubulin":
            return base, pd.NA, "family"
        if poi_raw == "Beta3-tubulin":
            return base, pd.NA, "family"
        return base, pd.NA, "alias"

    m = FUSION_RE.match(s)
    if m and "-" in m.group(1):
        base = m.group(1)
        tail = (m.group(2) or "").strip()
        # notebook convention
        if base == "BCR-ABL":
            base = "BCR/ABL"
        variant = tail if tail else pd.NA
        return base, variant, "fusion"

    if s in {"RAR", "PDE4", "HSP90"}:
        return s, pd.NA, "family"

    if s == "NS3":
        return "NS3", pd.NA, "viral"

    parts = s.split(" ", 1)
    base = parts[0]
    variant = parts[1].strip() if len(parts) == 2 else pd.NA
    return base, variant, "other"


# Curated alias→gene symbol for PROTAC-DB E3 ligases
E3_ALIAS_TO_GENE = {
    "CRBN": "CRBN",
    "VHL": "VHL",
    "MDM2": "MDM2",
    "KEAP1": "KEAP1",
    "Keap1": "KEAP1",
    "AhR": "AHR",
    "DCAF16": "DCAF16",
    "DCAF1": "DCAF1",
    "DCAF15": "DCAF15",
    "DCAF11": "DCAF11",
    "KLHL20": "KLHL20",
    "KLHDC2": "KLHDC2",
    "FBXO22": "FBXO22",
    "RNF114": "RNF114",
    "RNF4": "RNF4",
    "FEM1B": "FEM1B",
    "cIAP1": "BIRC2",
    "XIAP": "BIRC4",
    "IAP": None,
    "UBR box": None,
    "BRD4": None,
}


# Normalization functions

def normalize_protacdb(raw_file: str | Path, output_csv: str | Path) -> pd.DataFrame:
    df_raw = load_protacdb_custom(raw_file)

    out = df_raw[["Compound ID", "Uniprot", "Target", "E3 ligase", "Smiles"]].copy()
    out.columns = ["compound_id", "target_uniprot", "poi_raw", "e3_raw", "smiles"]
    out = _clean_string_columns(out)

    out["target_uniprot_base"] = out["target_uniprot"].str.replace(r"-\d+$", "", regex=True)

    tinfo = fetch_uniprot_info(out["target_uniprot_base"].dropna().unique())
    t_u2gene = tinfo.set_index("uniprot")["gene_primary"].to_dict()
    t_u2prot = tinfo.set_index("uniprot")["protein_name"].to_dict()

    out["poi_gene"] = out["target_uniprot_base"].map(t_u2gene)
    out["poi_protein"] = out["target_uniprot_base"].map(t_u2prot)
    out["poi_norm"] = out["poi_gene"]

    # Manual fallback only when target UniProt is missing
    mask_missing_target_uniprot = out["target_uniprot"].isna() & out["poi_raw"].notna()
    tmp = out.loc[mask_missing_target_uniprot, "poi_raw"].apply(norm_poi_no_uniprot)
    out.loc[mask_missing_target_uniprot, "poi_norm"] = tmp.apply(lambda x: x[0])

    out["e3_gene"] = out["e3_raw"].apply(
        lambda x: E3_ALIAS_TO_GENE.get(x, str(x).upper()) if pd.notna(x) else pd.NA
    )

    genes = sorted(out["e3_gene"].dropna().unique())
    e3_rows = []
    for g in genes:
        hit = fetch_uniprot_for_gene(g)
        e3_rows.append({
            "e3_gene": g,
            "e3_uniprot": (hit or {}).get("uniprot", pd.NA),
            "e3_gene_primary": (hit or {}).get("gene_primary", pd.NA),
            "e3_protein": (hit or {}).get("protein_name", pd.NA),
        })
    e3_map = pd.DataFrame(e3_rows)

    out = out.merge(e3_map, on="e3_gene", how="left")
    out["e3_norm"] = out["e3_gene_primary"].fillna(out["e3_gene"])

    final = out[[
        "target_uniprot", "poi_raw", "poi_gene", "poi_protein", "poi_norm",
        "e3_uniprot", "e3_raw", "e3_gene", "e3_protein", "e3_norm",
        "smiles",
    ]].copy()

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(output_csv, index=False)
    return final


def normalize_tpddb(raw_file: str | Path, output_csv: str | Path) -> pd.DataFrame:
    df_raw = pd.read_csv(raw_file)

    out = df_raw[["POI_UniProt", "POI_Name", "Ligase_UniProt", "Ligase_Name", "SMILES"]].copy()
    out.columns = ["target_uniprot", "poi_raw", "e3_uniprot", "e3_raw", "smiles"]
    out = _clean_string_columns(out)

    out["target_uniprot_base"] = out["target_uniprot"].str.replace(r"-\d+$", "", regex=True)
    out["e3_uniprot_base"] = out["e3_uniprot"].str.replace(r"-\d+$", "", regex=True)

    tinfo = fetch_uniprot_info(out["target_uniprot_base"].dropna().unique())
    einfo = fetch_uniprot_info(out["e3_uniprot_base"].dropna().unique())

    t_u2gene = tinfo.set_index("uniprot")["gene_primary"].to_dict()
    t_u2prot = tinfo.set_index("uniprot")["protein_name"].to_dict()
    e_u2gene = einfo.set_index("uniprot")["gene_primary"].to_dict()
    e_u2prot = einfo.set_index("uniprot")["protein_name"].to_dict()

    out["poi_gene"] = out["target_uniprot_base"].map(t_u2gene)
    out["poi_protein"] = out["target_uniprot_base"].map(t_u2prot)
    out["poi_norm"] = out["poi_gene"]

    out["e3_gene"] = out["e3_uniprot_base"].map(e_u2gene)
    out["e3_protein"] = out["e3_uniprot_base"].map(e_u2prot)
    out["e3_norm"] = out["e3_gene"]

    final = out[[
        "target_uniprot", "poi_raw", "poi_gene", "poi_protein", "poi_norm",
        "e3_uniprot", "e3_raw", "e3_gene", "e3_protein", "e3_norm",
        "smiles",
    ]].copy()

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(output_csv, index=False)
    return final


# Audit (optional)
def _write_value_counts(series: pd.Series, out_csv: Path, name: str):
    vc = series.dropna().astype("string").str.strip().replace({"": pd.NA}).dropna().value_counts()
    df = vc.reset_index()
    df.columns = [name, "count"]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)


def run_audit(protac_raw_path: Path, tpddb_raw_path: Path,
              prot_norm: pd.DataFrame, tp_norm: pd.DataFrame, audit_dir: Path):
    audit_dir.mkdir(parents=True, exist_ok=True)

    # Raw audits
    prot_raw = load_protacdb_custom(protac_raw_path)
    _write_value_counts(prot_raw["Target"], audit_dir / "protacdb_raw_poi_counts.csv", "poi_raw")
    _write_value_counts(prot_raw["E3 ligase"], audit_dir / "protacdb_raw_e3_counts.csv", "e3_raw")

    tpd_raw = pd.read_csv(tpddb_raw_path)
    if "POI_Name" in tpd_raw.columns:
        _write_value_counts(tpd_raw["POI_Name"], audit_dir / "tpddb_raw_poi_counts.csv", "poi_raw")
    if "Ligase_Name" in tpd_raw.columns:
        _write_value_counts(tpd_raw["Ligase_Name"], audit_dir / "tpddb_raw_e3_counts.csv", "e3_raw")

    # Post-normalization missingness summaries
    def miss_table(name: str, df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame([{
            "dataset": name,
            "rows": int(len(df)),
            "missing_poi_norm": int(df["poi_norm"].isna().sum()),
            "missing_e3_norm": int(df["e3_norm"].isna().sum()),
            "missing_target_uniprot": int(df["target_uniprot"].isna().sum()),
            "missing_e3_uniprot": int(df["e3_uniprot"].isna().sum()),
        }])

    miss = pd.concat([miss_table("PROTAC-DB", prot_norm), miss_table("TPDDB", tp_norm)], ignore_index=True)
    miss.to_csv(audit_dir / "post_normalization_missingness.csv", index=False)


def main():
    p = argparse.ArgumentParser(description="Normalize PROTAC-DB and TPDDB POI/E3 fields using UniProt.")
    p.add_argument("--protacdb", required=True, help="Path to raw PROTAC-DB CSV (e.g., PROTAC-DB.csv)")
    p.add_argument("--tpddb", required=True, help="Path to raw TPDDB CSV (e.g., tpddb_protacs.csv)")
    p.add_argument("--out-dir", default=".", help="Output directory for normalized CSVs (default: current dir)")
    p.add_argument("--protacdb-out", default="protacdb_normalized.csv", help="Output filename for PROTAC-DB normalized CSV")
    p.add_argument("--tpddb-out", default="tpddb_normalized.csv", help="Output filename for TPDDB normalized CSV")
    p.add_argument("--audit", action="store_true", help="Write audit CSVs (raw + post-normalization)")
    p.add_argument("--audit-dir", default="audit", help="Directory for audit outputs (relative to out-dir by default)")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prot_out = out_dir / args.protacdb_out
    tpd_out = out_dir / args.tpddb_out

    prot_norm = normalize_protacdb(args.protacdb, prot_out)
    tpd_norm = normalize_tpddb(args.tpddb, tpd_out)

    print("Wrote:")
    print(" -", prot_out, prot_norm.shape)
    print(" -", tpd_out, tpd_norm.shape)

    if args.audit:
        audit_dir = Path(args.audit_dir)
        if not audit_dir.is_absolute():
            audit_dir = out_dir / audit_dir
        run_audit(Path(args.protacdb), Path(args.tpddb), prot_norm, tpd_norm, audit_dir)
        print("Audit written to:", audit_dir)


if __name__ == "__main__":
    main()
