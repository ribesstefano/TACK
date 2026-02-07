# TACK Dataset Curation Pipeline

This directory contains scripts for curating PROTAC degradation data from multiple sources: TPD-DB, PROTAC-DB, and PROTACpedia.

## Overview

The curation pipeline processes raw data through several stages:
1. **Scraping**: Download and extract data from TPD-DB
2. **Parsing**: Clean and structure the scraped data
3. **Individual Curation**: Process each database separately
4. **Merging**: Combine all sources with deduplication and conflict resolution
5. **Splitting**: Create train/validation/test splits with clustering-based strategies

## Prerequisites

Please install the required Python packages in the `requirements.txt` file in the root directory:

```bash
pip install -r requirements.txt
```

Please make sure that the PYTHONPATH environment variable includes the root directory of the project to allow imports from `tack_dataset`:

```bash
export PYTHONPATH=$(pwd):$PYTHONPATH
```

**NOTE**: All commands should be run from the root directory of the project.

## Data Sources

- **TPD-DB**: Targeted Protein Degradation Database from https://tpddb.idrblab.net
- **PROTAC-DB**: Downloaded from http://cadd.zju.edu.cn/protacdb/
- **PROTACpedia**: Manually curated literature data from https://protacpedia.weizmann.ac.il/ptcb/main

Please ensure you have the raw data files downloaded and placed in the `data/original/` directory before running the scripts:

- for PROTACpedia, download the CSV file and place it in `data/original/` renamed as `PROTAC-Pedia.csv`
- for PROTAC-DB, download the CSV file and place it in `data/original/` renamed as `PROTAC-DB.csv`

## Pipeline Steps

### 1. 🌐 Scrape TPD-DB Data

Download PROTAC and molecular glue data from TPD-DB and save the text file into `data/original/`. Then run the scraping script to download the corresponding HTML pages:

```bash
python tack_dataset/tpddb_scraping.py \
    --protac-txt data/original/PROTAC_main_table.txt \
    -v
```

**Options:**
- `--protac-txt`: Path to TPD-DB PROTAC main table (required)
- `--glue-txt`: Path to TPD-DB molecular glue table (optional)
- `--limit`: Number of entries to process (useful for testing)
- `-v`: Verbose logging

**Output:** Raw HTML files in `data/original/html/`

### 2. 📄 Parse TPD-DB Data

Extract structured data from HTML files:

```bash
python tack_dataset/tpddb_parsing.py -v
```

**Output:** Parsed CSV files in `data/parsed/`

### 3. 🧹 Clean TPD-DB Data

Apply quality cleaning and standardization:

```bash
python tack_dataset/curate_tpddb.py
```

**Cleaning steps:**
- Aggregate parsed CSVs into a single dataset
- Remove invalid SMILES
- Standardize cell line names using Cellosaurus
- Map proteins to UniProt IDs
- Handle missing values and outliers

**Output:** `data/curation/tpddb_protac_glues_dc50_dmax.csv`

### 4. 📚 Curate PROTAC-Pedia

Process PROTACpedia data:

```bash
python tack_dataset/curate_protacpedia.py
```

**Features:**
- Parse free-text assay descriptions
- Extract DC50/Dmax values with operators (>, <, ~)
- Handle complex multi-cell/multi-target experiments
- Apply manual curation overrides for special cases

**Output:** `data/curation/protacpedia_protac_dc50_dmax.csv`

### 5. 🗃️ Curate PROTAC-DB

Process the PROTAC-DB dataset:

```bash
python tack_dataset/curate_protacdb.py
```

**Processing:**
- Parse multiple assay types (DC50/Dmax, percent degradation, IC50)
- Clean E3 ligase and target names
- Fetch protein sequences from UniProt
- Standardize cell lines
- Apply mutations to protein sequences

**Output:** Multiple CSV files in `data/curation/`:
- `protacdb_protac_dc50_dmax.csv`
- `protacdb_protac_percent_degradation.csv`

### 6. Merge All Sources

Combine and deduplicate data from all sources:

```bash
python tack_dataset/curate_protacdb_tpddb_protacpedia.py
```

**Merging strategy:**
- Priority order: TPD-DB > PROTAC-DB > PROTACpedia
- Remove duplicates based on key columns: SMILES, POI, ligase, cell line, assay type
- Handle conflicting values by preferring higher-priority source
- Convert units to standard (nM for DC50, % for Dmax)

**Output:** `data/curation/protacdb_tpddb_protacpedia_protac_dc50_dmax_activities.csv`

### 7. ✂️ Create Data Splits

Generate training+validation/hold-out splits:

```bash
python tack_dataset/data_splitting.py
```

**Splitting strategies:**
- Random split (baseline)
- Butina clustering (chemical similarity)
- Scaffold clustering (Bemis-Murcko)
- POI clustering (protein sequence similarity)

**Output:** Processed datasets with cluster assignments in `data/tack/`

## Complete Pipeline

Run all steps in sequence:

```bash
# 1. Scrape TPD-DB
python tack_dataset/tpddb_scraping.py \
    --protac-txt data/original/PROTAC_main_table.txt \
    -v

# 2. Parse TPD-DB
python tack_dataset/tpddb_parsing.py -v

# 3. Clean TPD-DB
python tack_dataset/tpddb_cleaning.py

# 4. Curate PROTACpedia
python tack_dataset/curate_protacpedia.py

# 5. Curate PROTAC-DB
python tack_dataset/curate_protacdb.py

# 6. Merge all sources
python tack_dataset/curate_protacdb_tpddb_protacpedia.py

# 7. Create splits
python tack_dataset/data_splitting.py
```

## Logging

By default, all scripts will log detailed information to both the console and log files in the `logs/` directory. Log files are named with timestamps for easy tracking of different runs.

## Output Structure

```
data
├── curation
│   ├── activities.csv
│   ├── protacdb_protac_dc50_dmax.csv
│   ├── protacdb_tpddb_protacpedia_protac_dc50_dmax_activities.csv
│   ├── protac_mode_of_action.csv
│   ├── protacpedia_protac_dc50_dmax.csv
│   ├── target2uniprots.json
│   ├── tpddb_protac_glues_dc50_dmax.csv
│   ├── uniprot2targets.json
│   └── uniprot_cache.json
├── original
│   ├── MG_activity.csv
│   ├── MG_main_table.csv
│   ├── PROTAC_activity.csv
│   ├── PROTAC_activity.txt
│   ├── PROTAC-DB.csv
│   ├── PROTAC_main_table.csv
│   ├── PROTAC_main_table.txt
│   └── PROTAC-Pedia.csv
└── tack
    ├── protacdb_tpddb_protacpedia_protac_dc50_activities_processed.csv
    ├── protacdb_tpddb_protacpedia_protac_dc50_dmax_activities_processed.csv
    ├── protacdb_tpddb_protacpedia_protac_dmax_activities_processed.csv
    └── protacdb_tpddb_protacpedia_protac_multitask_activities_processed.csv
```

Final curated datasets for the TACK dataset will be available in `data/tack/` with standardized formats and cluster assignments for model training and evaluation.

## Disclaimer

The Python files: `tack_dataset/curate_protacpedia.py`, `tack_dataset/curate_protacdb.py`, `tack_dataset/curate_protacdb_tpddb_protacpedia.py`, `tack_dataset/data_splitting.py`, are derived from Jupyter notebooks under the `tack_dataset/notebooks/` directory. The notebooks contain detailed explanations and visualizations of the curation process, while the Python scripts are optimized for reproducibility and automation. Please refer to the notebooks for a deeper understanding of the data processing steps and rationale behind key decisions.