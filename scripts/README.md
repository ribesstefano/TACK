# 🤖 Model Training and Ensemble Selection Scripts

This directory contains scripts for training PROTAC degradation prediction models on TACK and constructing optimal ensembles with uncertainty quantification.

## 📋 Overview

The training pipeline implements:
- **Repeated 5×5 Cross-Validation**: Rigorous statistical evaluation with group-based splitting
- **Multiple Model Architectures**: MLP and XGBoost models
- **Hyperparameter Optimization**: Optuna-based tuning for XGBoost and MLP models
- **Ensemble Selection**: Caruana's greedy forward selection with uncertainty quantification

## 🔧 Prerequisites

Please install the required Python packages as described in the [README](README.md).

## 🎯 Training Models

### Basic Usage

Train a model with default settings:

```bash
python scripts/train_models.py \
    --model_type mlp \
    --task dmax \
    --group scaffold
```

### 🎨 Available Options

**Model Types:**
- `mlp`: Multi-layer perceptron with molecular descriptors
- `xgboost`: Gradient boosting (supports hyperparameter tuning)

**Tasks:**
- `dmax`: Dmax regression (% degradation)
- `dc50`: DC50 regression (pDC50 scale)
- `bin`: Binary activity classification
- `dmax_bin`: Binary classification from Dmax (thresholded at 80%)
- `dc50_bin`: Binary classification from DC50 (thresholded at 100 nM)
- `multitask`: Joint Dmax + DC50 prediction (either one of the regression targets)

**Data Splitting Strategies:**
- `random`: Random scaffold-free splitting
- `scaffold`: Bemis-Murcko scaffold clustering
- `butina`: Tanimoto-based chemical clustering

### 📝 Example Commands

```bash
# Train XGBoost with scaffold splitting for DC50 prediction
python scripts/train_models.py \
    --model_type xgboost \
    --task dc50 \
    --group scaffold \
    --batch_size 64 \
    --checkpoint_dir ./checkpoints \
    --predictions_dir ./predictions

# Train MLP model for binary activity classification
python scripts/train_models.py \
    --model_type mlp \
    --task bin \
    --group random \
    --batch_size 16
```

### ⚙️ Configuration Files

Custom configurations can be provided via YAML files:

```bash
python scripts/train_models.py \
    --model_type mlp \
    --task bin \
    --data_config configs/data/bin-fp.yaml \
    --model_config configs/model/mlp-bin.yaml
```

**Example configs in `configs/`:**
- `data/mlp_descriptors.yaml`: MLP with RDKit descriptors
- `model/xgboost_tuned.yaml`: Optimized XGBoost hyperparameters

## 🏆 Ensemble Selection

### Caruana's Greedy Forward Selection

Construct optimal ensembles from trained models:

```bash
python scripts/ensemble_selection.py \
    --task dmax \
    --prediction_dir ./predictions \
    --output_dir ./ensemble_results \
    --hillclimb_perc 0.2
```

Available tasks: `dmax`, `dc50`, `bin`.

### 🎲 How It Works

1. **Held-Out Split**: 20% for ensemble selection, 80% for evaluation
2. **Candidate Pool**: All 500 models (25 CV folds × 20 feature configs)
3. **Greedy Selection**: Iteratively add models that improve performance
4. **Bagging**: 10 random subsamples for robustness
5. **Evaluation**: Test on unseen 80% evaluation set

### 📊 Output

The script generates:
- **Ensemble weights** (JSON): Model weights for deployment
- **Comparison plots**: Performance vs baselines
- **Uncertainty metrics**: Calibration analysis
- **Results CSV**: Detailed metrics for all methods

**Example output structure:** by default the results will be saved under a newly created `ensemble_results` directory. Example after running the script for the `dmax`, `dc50`, and `bin` tasks:

```
ensemble_results/
├── ensemble_comparison_bin.png
├── ensemble_comparison_dc50.png
├── ensemble_comparison_dmax.png
├── ensemble_comparison_results_bin.csv
├── ensemble_comparison_results_bin.log
├── ensemble_comparison_results_dc50.csv
├── ensemble_comparison_results_dc50.log
├── ensemble_comparison_results_dmax.csv
├── ensemble_comparison_results_dmax.log
├── ensemble_comparison_results_summary.log
├── ensemble_uncertainty_analysis_bin.png
├── ensemble_uncertainty_analysis_dc50.png
├── ensemble_uncertainty_analysis_dmax.png
├── ensemble_uncertainty_metrics_bin.csv
├── ensemble_uncertainty_metrics_dc50.csv
├── ensemble_uncertainty_metrics_dmax.csv
├── ensemble_weights_bin_architecture_level_expanded.json
├── ensemble_weights_bin_architecture_level.json
├── ensemble_weights_bin_caruana_ensemble.json
├── ensemble_weights_dc50_architecture_level_expanded.json
├── ensemble_weights_dc50_architecture_level.json
├── ensemble_weights_dc50_caruana_ensemble.json
├── ensemble_weights_dmax_architecture_level_expanded.json
├── ensemble_weights_dmax_architecture_level.json
└── ensemble_weights_dmax_caruana_ensemble.json
```
