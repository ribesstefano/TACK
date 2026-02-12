<h1 align="center">TACK</h1>

<h4 align="center"><i>A statistical evaluation of degradation activity on a novel TArgeting Chimeras Knowledge dataset</i></h4>

<p align="center">
  <img src="misc/tack.drawio.png" alt="Overview of the TACK dataset and training pipeline" width="100%">
</p>

TACK combines data from multiple sources (TPD-DB, PROTAC-DB, and PROTAC-Pedia) to create the largest publicly available dataset for training and evaluating machine learning models that predict PROTAC-induced protein degradation activities.

---

## 📚 Overview

This repository provides:
- **Curated Dataset**: High-quality PROTAC degradation data with DC50/Dmax measurements
- **Data Curation Pipeline**: Scripts to reproduce the dataset from raw sources
- **Training Framework**: Model training with nested 5×5 cross-validation
- **Ensemble Selection**: Caruana's greedy forward selection with uncertainty quantification
- **Benchmark Suite**: Standardized evaluation protocols and baselines

Please refer to the [`tack_dataset/README.md`](tack_dataset/README.md) for detailed instructions on dataset curation and to [`scripts/README.md`](scripts/README.md) for model training and ensemble selection.

### Key Features

- ✅ **Multi-source integration** with deduplication and quality control
- ✅ **Scaffold-based data splitting** to prevent information leakage
- ✅ **Rigorous statistical evaluation** via repeated cross-validation
- ✅ **Uncertainty quantification** through ensemble disagreement
- ✅ **Multiple model architectures**: MLP, XGBoost
- ✅ **Hyperparameter optimization** using Optuna

---

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/ribesstefano/TACK.git
cd TACK
pip install -r requirements.txt
export PYTHONPATH=$(pwd):$PYTHONPATH
```

### Download the Dataset

The TACK dataset will soon be available on Hugging Face:

```python
from datasets import load_dataset

# Load specific configurations
dmax_ds = load_dataset("ailab-bio/TACK", "Dmax", split="train")
dc50_ds = load_dataset("ailab-bio/TACK", "DC50", split="train")
multitask_ds = load_dataset("ailab-bio/TACK", "multitask", split="train")
```

> [!NOTE]
> For reproducibility, training can also be performed using local CSV files via `--custom_dataset_csv`.

### Train a Model

```bash
python scripts/train_models.py \
    --model_type xgboost \
    --task dmax \
    --group scaffold \
    --batch_size 64
```

### Construct Ensemble

```bash
python scripts/ensemble_comparison.py \
    --task dmax \
    --prediction_dir ./predictions \
    --output_dir ./ensemble_results
```

---

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
├── requirements.txt       # Python dependencies
└── README.md
```

---

## 🧬 PROTAC-STAN Evaluation

See the [PROTAC-STAN evaluation instructions](protac_stan/README.md) for reproducing results on TACK with 5×5 cross-validation.

---

## 📄 License

The TACK dataset and code are released under the MIT License. See `LICENSE` for details.

---

## 🤝 Acknowledgements

We thank the contributors of TPD-DB, PROTAC-DB, and PROTAC-Pedia for making this resource possible.
