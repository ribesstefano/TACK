# Contributing Guidelines

## Commands

### Training

```bash
cd /proj/berzelius-2026-62/users/x_steri/TACK

sbatch scripts/slurm/train_array.sh dmax random
sbatch scripts/slurm/train_array.sh dmax scaffold
sbatch scripts/slurm/train_array.sh dc50 random
sbatch scripts/slurm/train_array.sh dc50 scaffold
sbatch scripts/slurm/train_array.sh bin  random
sbatch scripts/slurm/train_array.sh bin  scaffold

# Optional
sbatch scripts/slurm/train_array.sh dmax butina
sbatch scripts/slurm/train_array.sh dc50 butina
sbatch scripts/slurm/train_array.sh bin  butina
```

### Evaluation

To evaluate:

```bash
tack evaluate \
  --predictions-dir outputs/tack2_dmax_scaffold/predictions \
  --output-dir outputs/tack2_dmax_scaffold/evaluation \
  --checkpoints-dir outputs/tack2_dmax_scaffold/checkpoints \
  --tasks dmax

tack evaluate \
  --predictions-dir outputs/tack2_dc50_scaffold/predictions \
  --output-dir outputs/tack2_dc50_scaffold/evaluation \
  --checkpoints-dir outputs/tack2_dc50_scaffold/checkpoints \
  --tasks dc50

tack evaluate \
  --predictions-dir outputs/tack2_bin_scaffold/predictions \
  --output-dir outputs/tack2_bin_scaffold/evaluation \
  --checkpoints-dir outputs/tack2_bin_scaffold/checkpoints \
  --tasks bin


declare -a groups=("scaffold" "random")
declare -a tasks=("bin" "dmax" "dc50")
for group in "${groups[@]}"
do
  for task in "${tasks[@]}"
  do
  tack evaluate \
    --predictions-dir outputs/tack2_${task}_${group}/predictions \
    --output-dir outputs/tack2_${task}_${group}/evaluation \
    --checkpoints-dir outputs/tack2_${task}_${group}/checkpoints \
    --tasks ${task}
  done
done
```

Output:

```
INFO Discovered 10 run(s).
INFO Levene p=0.023, max variance ratio=4.54 → parametric
INFO Levene p=0.0792, max variance ratio=3.78 → parametric
INFO Levene p=0.991, max variance ratio=1.22 → parametric
INFO Wrote evaluation report to outputs/tack2_bin_scaffold/evaluation/report.md
INFO Wrote evaluation report to: outputs/tack2_bin_scaffold/evaluation/report.md
INFO Discovered 10 run(s).
INFO Levene p=0.831, max variance ratio=1.96 → parametric
INFO Levene p=0.335, max variance ratio=1.96 → parametric
INFO Levene p=0.956, max variance ratio=1.44 → parametric
INFO Wrote evaluation report to outputs/tack2_dmax_scaffold/evaluation/report.md
INFO Wrote evaluation report to: outputs/tack2_dmax_scaffold/evaluation/report.md
INFO Discovered 10 run(s).
INFO Levene p=5.56e-05, max variance ratio=10.81 → non-parametric
INFO Levene p=0.0173, max variance ratio=5.98 → parametric
INFO Levene p=0.367, max variance ratio=2.12 → parametric
INFO Wrote evaluation report to outputs/tack2_dc50_scaffold/evaluation/report.md
INFO Wrote evaluation report to: outputs/tack2_dc50_scaffold/evaluation/report.md
INFO Discovered 10 run(s).
INFO Levene p=9.27e-13, max variance ratio=14.36 → non-parametric
INFO Levene p=3.24e-05, max variance ratio=8.89 → parametric
INFO Levene p=0.998, max variance ratio=1.20 → parametric
INFO Wrote evaluation report to outputs/tack2_bin_random/evaluation/report.md
INFO Wrote evaluation report to: outputs/tack2_bin_random/evaluation/report.md
INFO Discovered 10 run(s).
INFO Levene p=0.194, max variance ratio=3.38 → parametric
INFO Levene p=0.48, max variance ratio=1.96 → parametric
INFO Levene p=0.892, max variance ratio=1.38 → parametric
INFO Wrote evaluation report to outputs/tack2_dmax_random/evaluation/report.md
INFO Wrote evaluation report to: outputs/tack2_dmax_random/evaluation/report.md
INFO Discovered 10 run(s).
INFO Levene p=7.84e-12, max variance ratio=14.96 → non-parametric
INFO Levene p=5.89e-05, max variance ratio=7.48 → parametric
INFO Levene p=0.749, max variance ratio=1.61 → parametric
INFO Wrote evaluation report to outputs/tack2_dc50_random/evaluation/report.md
INFO Wrote evaluation report to: outputs/tack2_dc50_random/evaluation/report.md
```

### Ensembles

To compare and collect ensembles:

```bash
python scripts/ensemble_comparison.py \
    --prediction_dir outputs/tack2_dc50_scaffold/predictions \
    --output_dir outputs/tack2_dc50_scaffold/ensemble_comparison \
    --task dc50

python scripts/collect_ensemble_checkpoints.py \
    --weights /home/x_steri/storage/TACK/outputs/tack2_dc50_scaffold/ensemble_comparison/ensemble_weights_dc50_caruana_all_models.json \
    --checkpoints /home/x_steri/storage/TACK/outputs/tack2_dc50_scaffold/checkpoints \
    --output /home/x_steri/storage/TACK/outputs/tack2_dc50_scaffold/ensemble_caruana

python scripts/collect_ensemble_checkpoints.py \
    --weights /home/x_steri/storage/TACK/outputs/tack2_dc50_scaffold/ensemble_comparison/ensemble_weights_dc50_best_architecture.json \
    --checkpoints /home/x_steri/storage/TACK/outputs/tack2_dc50_scaffold/checkpoints \
    --output /home/x_steri/storage/TACK/outputs/tack2_dc50_scaffold/ensemble_best_arch 

python scripts/ensemble_comparison.py \
    --prediction_dir outputs/tack2_dmax_scaffold/predictions \
    --output_dir outputs/tack2_dmax_scaffold/ensemble_comparison \
    --task dmax

python scripts/collect_ensemble_checkpoints.py \
    --weights /home/x_steri/storage/TACK/outputs/tack2_dmax_scaffold/ensemble_comparison/ensemble_weights_dmax_caruana_all_models.json \
    --checkpoints /home/x_steri/storage/TACK/outputs/tack2_dmax_scaffold/checkpoints \
    --output /home/x_steri/storage/TACK/outputs/tack2_dmax_scaffold/ensemble_caruana

python scripts/collect_ensemble_checkpoints.py \
    --weights /home/x_steri/storage/TACK/outputs/tack2_dmax_scaffold/ensemble_comparison/ensemble_weights_dmax_best_architecture.json \
    --checkpoints /home/x_steri/storage/TACK/outputs/tack2_dmax_scaffold/checkpoints \
    --output /home/x_steri/storage/TACK/outputs/tack2_dmax_scaffold/ensemble_best_arch 

python scripts/ensemble_comparison.py \
    --prediction_dir outputs/tack2_bin_scaffold/predictions \
    --output_dir outputs/tack2_bin_scaffold/ensemble_comparison \
    --task bin

python scripts/collect_ensemble_checkpoints.py \
    --weights /home/x_steri/storage/TACK/outputs/tack2_bin_scaffold/ensemble_comparison/ensemble_weights_bin_caruana_all_models.json \
    --checkpoints /home/x_steri/storage/TACK/outputs/tack2_bin_scaffold/checkpoints \
    --output /home/x_steri/storage/TACK/outputs/tack2_bin_scaffold/ensemble_caruana

python scripts/collect_ensemble_checkpoints.py \
    --weights /home/x_steri/storage/TACK/outputs/tack2_bin_scaffold/ensemble_comparison/ensemble_weights_bin_best_architecture.json \
    --checkpoints /home/x_steri/storage/TACK/outputs/tack2_bin_scaffold/checkpoints \
    --output /home/x_steri/storage/TACK/outputs/tack2_bin_scaffold/ensemble_best_arch

declare -a groups=("scaffold" "random")
declare -a tasks=("bin" "dmax" "dc50")
for group in "${groups[@]}"
do
  for task in "${tasks[@]}"
  do
    python scripts/ensemble_comparison.py \
        --prediction_dir outputs/tack2_${task}_${group}/predictions \
        --output_dir outputs/tack2_${task}_${group}/ensemble_comparison \
        --task ${task}

    python scripts/collect_ensemble_checkpoints.py \
        --weights /home/x_steri/storage/TACK/outputs/tack2_${task}_${group}/ensemble_comparison/ensemble_weights_${task}_caruana_all_models.json \
        --checkpoints /home/x_steri/storage/TACK/outputs/tack2_${task}_${group}/checkpoints \
        --output /home/x_steri/storage/TACK/outputs/tack2_${task}_${group}/ensemble_caruana \
        --overwrite

    python scripts/collect_ensemble_checkpoints.py \
        --weights /home/x_steri/storage/TACK/outputs/tack2_${task}_${group}/ensemble_comparison/ensemble_weights_${task}_best_architecture.json \
        --checkpoints /home/x_steri/storage/TACK/outputs/tack2_${task}_${group}/checkpoints \
        --output /home/x_steri/storage/TACK/outputs/tack2_${task}_${group}/ensemble_best_arch \
        --overwrite

  done
done
```

To run Boltz-2:

```bash
python /proj/berzelius-2026-62/users/x_steri/boltz/submit-boltz.py \
    -i /proj/berzelius-2026-62/users/x_steri/TACK/boltz_complex_yaml \
    -o /proj/berzelius-2026-62/users/x_steri/TACK/boltz_complex_output \
    --per-file \
    --use_msa_server \
    --write_embeddings \
    --recycling_steps 23 \
    --diffusion_samples 5
```
