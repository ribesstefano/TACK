<h1 align="center">TACK</h1>

<h4 align="center"><i>A statistical evaluation of degradation activity on a novel TArgeting Chimeras Knowledge dataset</i></h4>

<p align="center">
  <img src="misc/tack.drawio.png" alt="Overview of the TACK dataset and training pipeline" width="80%">
</p>

TACK combines data from multiple sources (TPDdb, PROTAC-DB, and PROTACpedia) to create the largest publicly available dataset for training and evaluating machine learning models that predict PROTAC-induced protein degradation activities.

[![Dataset on HF](https://huggingface.co/datasets/huggingface/badges/resolve/main/dataset-on-hf-md-dark.svg)](https://huggingface.co/datasets/ailab-bio/TACK)
[![Models](https://img.shields.io/badge/Models-Zenodo-green)](https://zenodo.org/uploads/15691822)
[![Paper](https://img.shields.io/badge/Paper-KDD%202026-blue)](https://arxiv.org/abs/2605.19579)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

## 📚 Overview

This repository provides:
- **Curated Dataset**: High-quality PROTAC degradation data with DC50/Dmax measurements
- **Data Curation Pipeline**: Scripts to reproduce the dataset from raw sources
- **Training Framework**: Model training with nested 5×5 cross-validation
- **Ensemble Selection**: Caruana's greedy forward selection with uncertainty quantification
- **Benchmark Suite**: Standardized evaluation protocols and baselines
- **Python API for Ensemble Models**: Pre-trained ensembles for predicting Dmax, DC50, and binary degradation activity

Please refer to the [`tack_dataset/README.md`](tack_dataset/README.md) for detailed instructions on dataset curation, to [`scripts/README.md`](scripts/README.md) for model training and ensemble selection, and to [`notebooks/ensemble_predictor_tutorial.ipynb`](notebooks/ensemble_predictor_tutorial.ipynb) for interactive tutorials on using the pre-trained ensemble predictor.

### Key Features

- ✅ **Multi-source integration** with deduplication and quality control
- ✅ **Scaffold-based data splitting** to prevent information leakage
- ✅ **Rigorous statistical evaluation** via repeated cross-validation
- ✅ **Uncertainty quantification** through ensemble disagreement
- ✅ **Multiple model architectures**: MLP, XGBoost
- ✅ **Hyperparameter optimization** using Optuna

## 🚀 Quick Start

### Installation

TACK uses [`uv`](https://docs.astral.sh/uv/) for environment and dependency management.

**1. Install uv** (skip if already available, e.g., via an HPC module):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# or on HPC: module load uv
```

**2. Clone the repository:**

```bash
git clone https://github.com/ribesstefano/TACK.git
cd TACK
```

**3. Create a virtual environment and install the package:**

```bash
# Core dependencies only
uv venv --python 3.13
source .venv/bin/activate
uv pip install -e .
```

To also install plotting, notebook, and development tools:

```bash
uv pip install -e ".[dev]"
```

**4. (GPU clusters) Install PyTorch with the correct CUDA version:**

GPU computing is generally discouraged, since the models are very small and the performance bottlneck is in data encoding. Nevertheless, if one wishes to train and/or run inference on GPU, replace `cu121` with the CUDA version available on your system (e.g. `cu118`, `cu124`) in the following command:

```bash
uv pip install torch --extra-index-url https://download.pytorch.org/whl/cu121
```

**5. Register the environment as a Jupyter kernel** (only needed for notebooks):

```bash
python -m ipykernel install --user --name tack --display-name "TACK"
```

**6. Set up cache and model files for inference with pre-trained ensembles:**

Please refer to the [README section on "Pre-trained Models & Cache Files"](README.md#pre-trained-models--cache-files) for detailed instructions on downloading and configuring the necessary files for inference.

For running inference with the pre-trained ensemble, please refer to the [ensemble predictor tutorial notebook](notebooks/ensemble_predictor_tutorial.ipynb) for step-by-step instructions on how to use the `EnsemblePredictor` class with the downloaded models and cache files.

### Download the Dataset

The TACK dataset is available on Hugging Face at [this link](https://huggingface.co/datasets/ailab-bio/TACK), it can be accessed via:

```python
from datasets import load_dataset

# Load specific configurations
dmax_ds = load_dataset("ailab-bio/TACK", "Dmax", split="train")
dc50_ds = load_dataset("ailab-bio/TACK", "DC50", split="train")
bin_ds = load_dataset("ailab-bio/TACK", "multitask", split="train")
```

> [!NOTE]
> For reproducibility, training can also be performed using local CSV files via `--custom_dataset_csv`.

## 🗄️ Pre-trained Models & Cache Files

Running inference with a pre-trained ensemble requires two sets of files that
are **not** included in this repository due to their size:

| Archive | Contents | Purpose |
|---|---|---|
| `cache.zip` | `cell2cell_id.json`, `cell2description.json`, `cell2data.json`, `cell_embeddings_model=sentence-transformer_pooling=sum.npz`, `morgan_fp_radius16_size512.npz`, `rdkit_descriptors.npz` | Pre-computed embeddings and molecular descriptors read at inference time |
| `ensembles.zip` | `ensembles/<task>_<type>/ensemble_weights_*.json`, `*_hparams.yaml`, `*_state.pt`, model checkpoints (`.ckpt` / XGBoost `.json`) | Trained ensemble weights and fitted data-processing state |

Both archives are available on Zenodo: **[https://doi.org/10.5281/zenodo.15691822](https://doi.org/10.5281/zenodo.15691822)**

### Setup

**1. Download and unpack the archives:**

```bash
# choose any writable location; this example uses ~/tack-artifacts
mkdir -p ~/tack-artifacts/cache ~/tack-artifacts/ensembles

unzip cache.zip  -d ~/tack-artifacts/cache
unzip ensembles.zip -d ~/tack-artifacts/ensembles
```

**2. Point `TACKAI_CACHE` at the cache directory:**

Copy the example file `.env.example` to `.env` and edit the `TACKAI_CACHE` variable to point to the location of the unpacked cache files:

```bash
cp .env.example .env
# Then edit .env and set:
TACKAI_CACHE=~/tack-artifacts/cache/tack/
```

`tackai` reads this variable at startup via `get_cache_dir()`.  If it is not
set the default falls back to `~/.cache/tackai/`.

**3. Run inference with the pre-trained ensemble:**

See [this tutorial (TODO)](notebooks/README.md).

### What belongs in each archive

The separation between **cache** and **models** is intentional:

- **Cache files** are dataset-wide and shared across all tasks (binary, Dmax,
  DC50).  They are expensive to recompute (ESM protein embeddings, sentence
  embeddings for cell lines) and must exactly match the versions used during
  training.
- **Model files** are task-specific.  Each ensemble folder contains the
  `_hparams.yaml` / `_state.pt` pair for the `DegradationComplexDataModule`
  (which stores fitted scalers and encoders) and the model checkpoints selected
  by Caruana's greedy forward search.

If you retrain models yourself the cache files can be reused as-is; only the
model archive needs to be regenerated.

## 📊 Data Curation

For re-running data curation, please refer to the instruction in this [README](tack_dataset/README.md) file.

## 📂 Repository Structure

```
TACK/
├── configs/               # YAML configuration files
├── data/                  # Processed dataset files
├── logs/                  # Log files from training
├── misc/                  # Images and miscellaneous files
├── notebooks/             # Jupyter notebooks for exploration
├── predictions/           # Model predictions on CV splits
├── protac_stan/           # PROTAC-STAN reproduction scripts
├── scripts/               # Training and ensemble scripts
├── ensemble_results/      # Ensemble selection results
├── pyproject.toml         # Package metadata and dependencies (uv)
└── README.md
```

## 📈 Reproducing the Results

### Train a Model

```bash
python scripts/train_models.py \
    --model_type xgboost \
    --task dmax \
    --group scaffold \
    --batch_size 64
```

### Construct and Evaluate Ensemble

```bash
python scripts/ensemble_comparison.py \
    --task dmax \
    --prediction_dir ./predictions \
    --output_dir ./ensemble_results
```

### 🧬 PROTAC-STAN Evaluation

See the [PROTAC-STAN evaluation instructions](protac_stan/README.md) for reproducing results on TACK with 5×5 cross-validation.

## 📄 License

The TACK dataset and code are released under the MIT License. See `LICENSE` for details.

<!-- Add citation -->

## 📑 Citation

If you use TACK in your research, please cite the following paper:

```bibtex
@misc{ribes2026tackstatisticalevaluationdegradation,
      title={{TACK: A statistical evaluation of degradation activity on a novel TArgeting Chimeras Knowledge dataset}}, 
      author={Stefano Ribes and Nils Dunlop and Rocío Mercado},
      year={2026},
      eprint={2605.19579},
      archivePrefix={arXiv},
      primaryClass={q-bio.QM},
      url={https://arxiv.org/abs/2605.19579}, 
}
```

## 🤝 Acknowledgements

The authors acknowledge funding provided by the Chalmers Gender Initiative for Excellence (Genie), and by the Wallenberg AI, Autonomous Systems, and Software Program (WASP), supported by the Knut and Alice Wallenberg Foundation.
The authors thank Yossra Gharbi, Alexander Persson, and Felix Erngård for helpful discussions.
The computations and data storage were enabled by resources provided by Chalmers e-Commons and by the National Academic Infrastructure for Supercomputing in Sweden (NAISS), partially funded by the Swedish Research Council through grant agreement no. 2022-06725.
