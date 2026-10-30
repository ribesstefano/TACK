#!/usr/bin/env python3
"""Read-only audit of AutoTPD training exports and the PROTAC test set.

Uses only the Python standard library. By default writes a Markdown report to
docs/surrogate-data-audit-2026-10-30.md; it never alters the input CSVs.
"""

import argparse
import csv
import hashlib
import math
import re
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FILES = {
    "tpddb": ROOT / "data/AutoTPD-TPDdb.csv",
    "plus": ROOT / "data/auto_tpd_plus/AutoTPDplus-PROTAC.csv",
    "test": ROOT / "data/auto_tpd_plus/PROTAC-test.csv",
}


def clean(value):
    return (value or "").strip()


def first(row, *names):
    for name in names:
        value = clean(row.get(name))
        if value:
            return value
    return ""


def norm_doi(value):
    value = clean(value).lower()
    value = re.sub(r"^https?://(dx\.)?doi\.org/", "", value)
    return value.rstrip(" .")


def numeric(value):
    try:
        result = float(clean(value))
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    return rows


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def molecule_id(row):
    # Connectivity identity is preferred to avoid treating stereoisomers,
    # salts, or alternate canonical strings as independent split units.
    value = first(row, "Connectivity_Key", "StdInChIKey", "InChIKey", "Compound_Key")
    if value:
        return "key:" + value.upper()
    value = first(row, "SMILES")
    return "smiles:" + value if value else ""


def context(row):
    return tuple(
        first(row, *keys).casefold()
        for keys in (
            ("Degradation_Target_Uniprot", "Degradation_Target_Gene", "Degradation_Target"),
            ("Recruiter_Uniprot", "Recruiter_Gene", "Recruiter"),
            ("Cell_Line_STD_Accession", "Cell_Line_STD", "Cell_Line"),
            ("Assay_STD", "Assay"),
        )
    )


def endpoint_signature(row):
    return tuple(
        clean(row.get(name))
        for name in (
            "DC50", "DC50_units", "DC50_h", "DC50_operator", "DC50_range_min", "DC50_range_max",
            "Dmax", "Dmax_h", "Dmax_operator", "Dmax_range_min", "Dmax_range_max",
        )
    )


def describe_counter(counter, limit=12):
    if not counter:
        return "- none"
    return "\n".join("- `{}`: {}".format(k or "(blank)", v) for k, v in counter.most_common(limit))


def dataset_stats(rows):
    stats = {"rows": len(rows)}
    stats["dois"] = {norm_doi(r.get("DOI")) for r in rows if norm_doi(r.get("DOI"))}
    stats["doi_rows"] = sum(bool(norm_doi(r.get("DOI"))) for r in rows)
    for label, cols in {
        "target": ("Degradation_Target_Gene", "Degradation_Target"),
        "e3": ("Recruiter_Gene", "Recruiter"),
        "assay": ("Assay_STD", "Assay"),
        "cell": ("Cell_Line_STD_Accession", "Cell_Line_STD", "Cell_Line"),
    }.items():
        stats[label] = Counter(first(r, *cols) for r in rows)
    stats["missing"] = Counter()
    for col in ("DOI", "Reference", "TPD_ID", "SMILES", "Connectivity_Key", "Degradation_Target", "Recruiter", "Cell_Line", "Assay_STD", "DC50", "Dmax", "DC50_h", "Dmax_h"):
        stats["missing"][col] = sum(not clean(r.get(col)) for r in rows)
    stats["operator"] = {col: Counter(clean(r.get(col)) for r in rows if clean(r.get(col))) for col in ("DC50_operator", "Dmax_operator", "Dmax_conc_operator")}
    stats["source_ref"] = Counter(clean(r.get("Reference")) for r in rows if clean(r.get("Reference")))
    stats["tpd_id"] = len({clean(r.get("TPD_ID")) for r in rows if clean(r.get("TPD_ID"))})
    stats["dmax_outside"] = sum((numeric(r.get("Dmax")) is not None and not 0 <= numeric(r.get("Dmax")) <= 100) for r in rows)
    stats["dc50_nonpositive"] = sum((numeric(r.get("DC50")) is not None and numeric(r.get("DC50")) <= 0) for r in rows)
    both = [r for r in rows if numeric(r.get("DC50_h")) is not None and numeric(r.get("Dmax_h")) is not None]
    stats["both_time"] = len(both)
    stats["time_disagree"] = sum(numeric(r.get("DC50_h")) != numeric(r.get("Dmax_h")) for r in both)
    stats["time_pairs"] = Counter((clean(r.get("DC50_h")), clean(r.get("Dmax_h"))) for r in both)
    stats["identities"] = defaultdict(list)
    for i, row in enumerate(rows):
        ident = molecule_id(row)
        if ident:
            stats["identities"][ident].append((i, row))
    stats["unique_molecules"] = len(stats["identities"])
    return stats


def duplicate_counts(datasets):
    seen = defaultdict(list)
    for source, rows in datasets.items():
        for row in rows:
            mol = molecule_id(row)
            if not mol:
                continue
            sig = (mol, context(row), endpoint_signature(row))
            seen[sig].append(source)
    repeats = sum(max(0, len(sources) - 1) for sources in seen.values())
    duplicated_signatures = sum(len(sources) > 1 for sources in seen.values())
    cross = Counter()
    for sources in seen.values():
        if len(set(sources)) > 1:
            cross[" + ".join(sorted(set(sources)))] += 1
    return repeats, duplicated_signatures, cross


def context_conflicts(rows):
    by_molecule = defaultdict(set)
    for row in rows:
        mol = molecule_id(row)
        if mol:
            by_molecule[mol].add(context(row))
    return sum(len(contexts) > 1 for contexts in by_molecule.values()), by_molecule


def fmt_counts(counter, total, limit=12):
    lines = []
    for name, count in counter.most_common(limit):
        lines.append("- {}: {} ({:.1f}%)".format(name or "(missing)", count, 100 * count / total if total else 0))
    if len(counter) > limit:
        lines.append("- Other values: {} categories".format(len(counter) - limit))
    return "\n".join(lines) if lines else "- none"


def normalized_distribution(rows, kind):
    columns = {
        "target": ("Degradation_Target_Gene", "Degradation_Target"),
        "E3": ("Recruiter_Gene", "Recruiter"),
        "assay": ("Assay_STD", "Assay"),
    }[kind]
    return Counter(first(row, *columns).casefold() or "(missing)" for row in rows)


def audit(paths, output):
    data = {name: read_csv(path) for name, path in paths.items()}
    stats = {name: dataset_stats(rows) for name, rows in data.items()}
    train_dois = stats["plus"]["dois"] | stats["tpddb"]["dois"]
    test_dois = stats["test"]["dois"]
    test_train_overlap = test_dois & train_dois
    repeats, dup_sigs, cross = duplicate_counts(data)
    all_train = data["tpddb"] + data["plus"]
    conflict_count, by_molecule = context_conflicts(all_train)
    test_molecules = {molecule_id(r) for r in data["test"] if molecule_id(r)}
    train_molecules = {molecule_id(r) for r in all_train if molecule_id(r)}
    molecule_overlap = test_molecules & train_molecules
    ambiguous_groups = sum(len(rows) > 1 for rows in by_molecule.values())

    report = [
        "# Raw surrogate-data audit (2026-10-30)",
        "",
        "Generated by `scripts/audit_surrogate_data.py` using Python standard library only. The audit reads raw CSVs, makes no edits to them, and does not apply TACK's feature-encoding or endpoint eligibility filters.",
        "",
        "## Inputs and integrity",
        "",
        "| Dataset | Rows | SHA-256 | DOI rows / unique DOI | Unique molecule keys |",
        "|---|---:|---|---:|---:|",
    ]
    for name, path in paths.items():
        s = stats[name]
        report.append("| {} (`{}`) | {} | `{}` | {} / {} | {} |".format(name, path.relative_to(ROOT), s["rows"], sha256(path), s["doi_rows"], len(s["dois"]), s["unique_molecules"]))
    report += [
        "",
        "Counts below are raw-row counts unless explicitly called unique identities. Empty strings are treated as missing. Numeric checks count parseable finite values only.",
        "",
        "## Dataset profiles",
    ]
    for name in ("tpddb", "plus", "test"):
        s = stats[name]
        report += [
            "",
            "### {} ({} rows)".format(name, s["rows"]),
            "",
            "- Target distribution:\n{}".format(fmt_counts(s["target"], s["rows"])),
            "- E3 distribution:\n{}".format(fmt_counts(s["e3"], s["rows"])),
            "- Assay distribution:\n{}".format(fmt_counts(s["assay"], s["rows"])),
            "- Missing fields:\n{}".format("\n".join("- {}: {} ({:.1f}%)".format(k, v, 100 * v / s["rows"]) for k, v in s["missing"].items())),
            "- Nonempty endpoint operators:\n{}".format("\n".join("  - {}: {}".format(k, describe_counter(v, 20).replace("\n", "; ")) for k, v in s["operator"].items())),
            "- Parseable out-of-range checks: Dmax outside [0,100] = {}; DC50 <= 0 = {}.".format(s["dmax_outside"], s["dc50_nonpositive"]),
            "- Rows with both endpoint times: {}; different DC50_h and Dmax_h: {}. Most common (DC50_h, Dmax_h) pairs:\n{}".format(s["both_time"], s["time_disagree"], fmt_counts(Counter({"({}, {})".format(*pair): n for pair, n in s["time_pairs"].most_common(10)}), s["both_time"])),
        ]
        if name == "tpddb":
            report += [
                "- DOI values present: {} rows ({} unique); alternate `Reference` values: {} rows ({} unique); nonempty `TPD_ID`: {} unique.".format(s["doi_rows"], len(s["dois"]), sum(bool(clean(r.get("Reference"))) for r in data[name]), len(s["source_ref"]), s["tpd_id"]),
                "- Frequent alternate `Reference` identifiers (many are patent titles):\n{}".format(fmt_counts(s["source_ref"], s["rows"], 15)),
            ]
    report += [
        "",
        "## Distribution shift (raw marginal proportions)",
        "",
        "The table uses case-folded values and the first available field in each normalized field pair (target gene/name, E3 gene/name, assay standard/raw). It is descriptive: it is not matched on endpoint availability, molecule identity, or encoder eligibility.",
        "",
        "| Field / category | TPDdb (n=8664) | AutoTPDplus (n=6719) | Test (n=209) |",
        "|---|---:|---:|---:|",
    ]
    for field in ("target", "E3", "assay"):
        distributions = {name: normalized_distribution(data[name], field) for name in data}
        categories = set().union(*(set(counts) for counts in distributions.values()))
        common = sorted(categories, key=lambda category: sum(distributions[name][category] for name in data), reverse=True)[:12]
        for category in common:
            vals = []
            for name in ("tpddb", "plus", "test"):
                n = stats[name]["rows"]
                count = distributions[name][category]
                vals.append("{} ({:.1f}%)".format(count, 100 * count / n if n else 0))
            report.append("| {} / {} | {} | {} | {} |".format(field, category, *vals))
    report += [
        "",
        "## Repeated exports and context conflicts",
        "",
        "- Exact measurement-signature repeats across the supplied files: {} repeated rows across {} repeated `(molecule, context, endpoints/operators/times)` signatures.".format(repeats, dup_sigs),
        "- Signatures represented in multiple files: {}.".format("; ".join("{}: {} signatures".format(k, v) for k, v in cross.items()) or "none"),
        "- Training molecule identities with more than one recorded biological context: {} of {} identities with a usable molecule key.".format(conflict_count, len(by_molecule)),
        "- Test molecule identity overlap with the combined training exports: {} of {} test identities.".format(len(molecule_overlap), len(test_molecules)),
        "- Test DOI overlap with training DOI values: {} of {} test DOIs ({} test rows carry an overlapping DOI).".format(len(test_train_overlap), len(test_dois), sum(norm_doi(r.get("DOI")) in test_train_overlap for r in data["test"])),
        "",
        "The duplicate signature is deliberately conservative: only equal molecule key, normalized biological context, raw endpoint values, units, times, operators and range bounds count. It detects repeated exports of the same recorded measurement but does not infer that different endpoint values are independent replicates. Context conflict counts likewise flag context diversity, not an error; labels must retain target, E3, cell, assay and endpoint time.",
        "",
        "## Publication and source identifiers",
        "",
        "The TPDdb CSV has no populated DOI in this snapshot. Its `Reference` field is populated on all rows and contains {} distinct strings; `TPD_ID` is populated with {} distinct IDs. These are workable **source-family / patent-group proxies**, not verified publication identifiers. AutoTPDplus has DOI on {} / {} rows ({} distinct DOIs). The test has DOI on {} / {} rows ({} distinct DOIs).".format(len(stats["tpddb"]["source_ref"]), stats["tpddb"]["tpd_id"], stats["plus"]["doi_rows"], stats["plus"]["rows"], len(stats["plus"]["dois"]), stats["test"]["doi_rows"], stats["test"]["rows"], len(test_dois)),
        "",
        "A publication-grouped holdout is feasible for the DOI-bearing AutoTPDplus source, but cannot cover TPDdb records reliably by DOI. Use DOI as a normalized group key where available; group TPDdb rows by `Reference` at minimum and keep every row from a reference together. Do not treat missing DOI as one shared group or mint one pseudo-publication per row. Keep a separate `source_family` field (DOI, TPDdb reference, or unknown) so coverage and fallback choices are explicit.",
        "",
        "## Recommended split design",
        "",
        "1. Freeze the historical test as a benchmark; report that {} test DOIs overlap training DOI metadata and check molecule/connectivity overlap before interpreting it as publication-held-out. The raw identity overlap here is {} / {}; the loader's broader canonical/connectivity identity union should remain authoritative.".format(len(test_train_overlap), len(molecule_overlap), len(test_molecules)),
        "2. Create a development-only publication/source holdout by deterministic group assignment, not row sampling: reserve roughly 15–20% of DOI groups from AutoTPDplus and 15–20% of distinct TPDdb `Reference` groups, stratifying only at the source-group level when group counts permit. Publish the exact group assignment, seed, source and row counts.",
        "3. Before assigning rows, build a leakage graph over all three sources. Connect records sharing connectivity identity (and retain raw SMILES/canonical identity aliases where parseable); union connected components so salts, tautomers, stereoisomers, alternate exports, compounds, and all endpoint/context rows for the same chemical entity stay in one partition. If a component crosses source groups, hold the entire component with its publication/source group or exclude it from the strict source holdout and report the loss.",
        "4. Keep context distinct within molecule identity. Never deduplicate or average across target/E3/cell/assay/time. For repeated same-context measurements, preserve operator, ranges, endpoint-specific times and provenance; report exact replicate conflicts separately.",
        "5. Keep inner/outer CV groups chemical-series-disjoint and assign molecules once across all tasks before endpoint masks. Report row and independent-group counts, endpoint coverage, source/target/E3/assay distributions, and overlap exclusions for every partition.",
        "",
        "### Limitations",
        "",
        "This is a raw CSV audit, not an eligibility or label-quality adjudication. It does not parse SMILES, normalize cell/assay aliases, resolve citation text, validate patent-family membership, apply censored-label semantics, or reproduce `FusionData` dropped-row counts. The molecule key preference is based on supplied connectivity/InChIKey columns; test keys without those columns can only fall back to literal SMILES, so canonical/connectivity overlap may be higher. DOI overlap is not a complete provenance overlap check: papers may be represented by patent references, and distinct reference strings may belong to the same patent family. Verify record provenance before presenting the proposed split as publication-level generalization.",
        "",
        "## Reproduction",
        "",
        "```bash",
        "python3 scripts/audit_surrogate_data.py",
        "```",
        "",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(report), encoding="utf-8")
    print("Wrote {}".format(output))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tpddb", type=Path, default=DEFAULT_FILES["tpddb"])
    parser.add_argument("--plus", type=Path, default=DEFAULT_FILES["plus"])
    parser.add_argument("--test", type=Path, default=DEFAULT_FILES["test"])
    parser.add_argument("--output", type=Path, default=ROOT / "docs/surrogate-data-audit-2026-10-30.md")
    args = parser.parse_args()
    audit({"tpddb": args.tpddb, "plus": args.plus, "test": args.test}, args.output)


if __name__ == "__main__":
    main()
