# PROTAC-STAN Evaluation

This directory contains updated training scripts for reproducing [PROTAC-STAN](https://github.com/PROTACs/PROTAC-STAN) results on the TACK dataset with 5×5 cross-validation.

---

## Setup

### 1. Clone Repositories

Clone both repositories into the same parent directory:

```bash
git clone https://github.com/ribesstefano/TACK.git
git clone https://github.com/PROTACs/PROTAC-STAN.git
```

Your directory structure should look like:
```
parent/
├── TACK/
│   └── protac_stan/
│       ├── containers/
│       │   ├── esm-plus.def
│       │   └── protac-stan.def
│       ├── esm_embed/
│       │   └── embed_proteins.py
│       ├── scripts/
│       │   ├── prepare_data.py
│       │   └── train_cv.py
│       └── data_loader.py
└── PROTAC-STAN/
    └── ...
```

### 2. Copy Updated Files

Copy the updated scripts into the PROTAC-STAN repository:

```bash
mkdir -p PROTAC-STAN/scripts
cp TACK/protac_stan/scripts/* PROTAC-STAN/scripts/
cp TACK/protac_stan/esm_embed/embed_proteins.py PROTAC-STAN/esm_embed/
cp TACK/protac_stan/data_loader.py PROTAC-STAN/
```

---

## Installation

### Option A: Conda Environment

```bash
conda create -n protac-stan python=3.11.5
conda activate protac-stan

conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia
conda install numpy==1.26.4

pip install torch_geometric==2.5.1 rdkit==2023.9.2 pandas==2.1.1 toml==0.10.2
pip install wandb datasets huggingface_hub scikit-learn python-dotenv

# Optional: accelerate PyG
pip install https://data.pyg.org/whl/torch-2.1.0%2Bcu118/torch_scatter-2.1.2%2Bpt21cu118-cp311-cp311-linux_x86_64.whl
```

### Option B: Apptainer Containers (HPC)

Build the containers from the definition files:

```bash
cp -r TACK/protac_stan/containers PROTAC-STAN/
cd PROTAC-STAN
apptainer build protac-stan.sif containers/protac-stan.def
apptainer build esm-plus.sif containers/esm-plus.def
```

---

## Download ESM Model Weights

Create the model directory and download the required weights:

```bash
mkdir -p PROTAC-STAN/esm_embed/model
cd PROTAC-STAN/esm_embed/model

wget https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t33_650M_UR50D.pt
wget https://huggingface.co/Oxer11/ESM-S/resolve/main/esm_650m_s.pth
```

See [ESM](https://github.com/facebookresearch/esm) and [ESM-S](https://github.com/DeepGraphLearning/esm-s) for more details.

---

## Configuration

Create a `.env` file in the PROTAC-STAN directory (all optional):

```bash
HF_TOKEN=your_huggingface_token
WANDB_API_KEY=your_wandb_api_key
```

---

## Training Pipeline

All commands below assume you are in the `PROTAC-STAN/` directory.

### Step 1: Generate ESM-S Protein Embeddings

This step requires GPU access. Run once per dataset.

The embedding script depends on `torchdrug`, which is only available inside the
`esm-plus.sif` container.

**With Apptainer (HPC) — recommended:**

Two flags are required:
- `--writable-tmpfs`: allows the container to write compiled library caches to
  a temporary in-memory overlay (needed by `lmdb`/`torchdrug`)
- `-B ~/.cache/huggingface:/root/.cache/huggingface`: makes the host HuggingFace
  cache available inside the container so the TACK dataset can be downloaded

```bash
cd PROTAC-STAN

# Load your HuggingFace token from the .env file
export HF_TOKEN=$(grep HF_TOKEN .env | cut -d'"' -f2)

apptainer exec --nv --writable-tmpfs \
    -B ~/.cache/huggingface:/root/.cache/huggingface \
    --env HF_TOKEN="$HF_TOKEN" \
    esm-plus.sif bash -c "
    pip install 'lmdb==1.3.0' --force-reinstall -q &&
    python esm_embed/embed_proteins.py \
        --model_dir esm_embed/model \
        --output_dir data/custom
"
```

This loads all TACK configs from HuggingFace by default. To use a local CSV
instead (avoids network access and the HF token requirement):

```bash
apptainer exec --nv --writable-tmpfs \
    -B ~/.cache/huggingface:/root/.cache/huggingface \
    --env HF_TOKEN="$HF_TOKEN" \
    esm-plus.sif bash -c "
    pip install 'lmdb==1.3.0' --force-reinstall -q &&
    python esm_embed/embed_proteins.py \
        --model_dir esm_embed/model \
        --output_dir data/custom \
        --custom_dataset_csv path/to/your/data.csv
```

The script is incremental: if output files already exist, only proteins with
missing embeddings are processed. Each run saves both `.pkl` and `.npz`
versions of the outputs.

**With Conda environment:**
```bash
cd PROTAC-STAN/esm_embed
python embed_proteins.py --output_dir ../data/custom
```

### Step 2: Prepare Data

Prepare the dataset for training. This creates CV splits and builds PyG graph data:

```bash
cd PROTAC-STAN

# Choose your task
python scripts/prepare_data.py --task dc50
python scripts/prepare_data.py --task dmax
python scripts/prepare_data.py --task bin
```

**Custom CSV dataset:**
```bash
python scripts/prepare_data.py --task dc50 --custom_dataset_csv path/to/your/data.csv
```

**With Apptainer (HPC):**
```bash
apptainer exec protac-stan.sif python scripts/prepare_data.py --task dc50
```

### Step 3: Train

Run 5×5 cross-validation training:

```bash
python scripts/train_cv.py --task dc50
python scripts/train_cv.py --task dmax
python scripts/train_cv.py --task bin
```

**Disable WandB logging:**
```bash
python scripts/train_cv.py --task bin --no_wandb
```

**With Apptainer (HPC):**
```bash
apptainer exec protac-stan.sif python scripts/train_cv.py --task bin
```

---

## Expected Output

Training produces:
- `results_{task}_{timestamp}/` - Results directory containing:
  - `predictions/` - Per-fold predictions and held-out evaluations
  - `checkpoints/` - Model weights for each fold
  - `summary.csv` - Aggregated metrics across all folds

If WandB is enabled, training metrics are logged to:
`https://wandb.ai/your-username/protac-stan-{task}`

---

## Reference

For more details on PROTAC-STAN architecture and methodology, see the [original repository](https://github.com/PROTACs/PROTAC-STAN) and paper:

> Chen et al. "Interpretable PROTAC Degradation Prediction With Structure-Informed Deep Ternary Attention Framework" *Advanced Science* (2025). [DOI](https://doi.org/10.1002/advs.202508138)	
