"""Aggregate per-database SMILES CSVs produced by the --smiles_only flag.

Each input CSV must have two columns: SMILES and Database.  When the same
canonical SMILES appears in more than one source, the Database values are
sorted alphabetically and joined with a semicolon.

Typical usage (after running each curation script with --smiles_only):
    python -m tack_dataset.aggregate_smiles --output_dir data/curation
"""
import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description='Aggregate SMILES CSVs from all curation scripts into one deduplicated file.'
    )
    parser.add_argument(
        '--output_dir', type=str, default='data/curation',
        help='Directory that contains the per-database SMILES CSVs and where the '
             'aggregated file will be written (default: data/curation).',
    )
    parser.add_argument(
        '--protacdb_csv', type=str, default=None,
        help='Path to the PROTAC-DB SMILES CSV '
             '(default: <output_dir>/protacdb_smiles.csv).',
    )
    parser.add_argument(
        '--protacpedia_csv', type=str, default=None,
        help='Path to the PROTACpedia SMILES CSV '
             '(default: <output_dir>/protacpedia_smiles.csv).',
    )
    parser.add_argument(
        '--tpddb_csv', type=str, default=None,
        help='Path to the TPDdb SMILES CSV '
             '(default: <output_dir>/tpddb_smiles.csv).',
    )
    parser.add_argument(
        '--output_file', type=str, default='tack_smiles.csv',
        help='Filename for the aggregated output CSV (default: tack_smiles.csv).',
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    csv_paths = [
        Path(args.protacdb_csv)    if args.protacdb_csv    else output_dir / 'protacdb_smiles.csv',
        Path(args.protacpedia_csv) if args.protacpedia_csv else output_dir / 'protacpedia_smiles.csv',
        Path(args.tpddb_csv)       if args.tpddb_csv       else output_dir / 'tpddb_smiles.csv',
    ]

    dfs = []
    for path in csv_paths:
        if not path.exists():
            print(f"Warning: {path} not found, skipping.")
            continue
        df = pd.read_csv(path)[['SMILES', 'Database']].dropna(subset=['SMILES']).drop_duplicates()
        dfs.append(df)
        print(f"Loaded {len(df):,} SMILES from {path.name}")

    if not dfs:
        print("No input files found. Exiting.")
        return

    combined = pd.concat(dfs, ignore_index=True)
    print(f"Total entries before aggregation: {len(combined):,}")

    aggregated = (
        combined
        .groupby('SMILES', sort=False)['Database']
        .apply(lambda dbs: ('; '.join(sorted(set(dbs))).strip()))
        .reset_index()
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / args.output_file
    aggregated[['SMILES', 'Database']].to_csv(output_path, index=False)
    print(f"Saved {len(aggregated):,} unique SMILES to {output_path}")

    print("\nDatabase combination counts:")
    for combo, count in aggregated['Database'].value_counts().items():
        print(f"  {combo}: {count:,}")


if __name__ == '__main__':
    main()
