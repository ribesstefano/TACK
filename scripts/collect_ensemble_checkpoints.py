"""
Script to collect model and datamodule checkpoints needed for an ensemble
into a self-contained output directory.

Given a checkpoints directory and an ensemble weights JSON file, it:
  1. Creates the output directory.
  2. Copies the weights JSON into it.
  3. Creates a `checkpoints/` sub-directory and copies into it, for every
     model listed in the weights file:
       - the model checkpoint (.ckpt for MLP/Lightning, .json for XGBoost)
       - the corresponding datamodule _hparams.yaml and _state.pt files

Usage:
    python scripts/collect_ensemble_checkpoints.py \\
        --weights ensemble_results/ensemble_weights_dmax_caruana_ensemble.json \\
        --checkpoints checkpoints/ \\
        --output my_ensemble/

Author: Stefano Ribes
"""
import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Union


MODEL_EXTENSIONS = ['.ckpt', '.json', '.ubj', '.pkl']


def parse_model_name(model_name: str) -> dict:
    """Extract data_config, group, and fold from a model name string.

    Model name convention:
        model=<arch>_<task>_protac-data=<data_config>-group=<group>-fold=<fold>
    """
    parts = model_name.split('-')
    data_config = None
    group = None
    fold = None
    for part in parts:
        if part.startswith('data='):
            data_config = part[len('data='):]
        elif part.startswith('group='):
            group = part[len('group='):]
        elif part.startswith('fold='):
            fold = part[len('fold='):]
    return {'data_config': data_config, 'group': group, 'fold': fold}


def find_model_file(model_name: str, checkpoints_dir: Path) -> Union[Path, None]:
    """Find the model checkpoint file for a given model name."""
    for ext in MODEL_EXTENSIONS:
        candidate = checkpoints_dir / f"{model_name}{ext}"
        if candidate.exists():
            return candidate
    return None


def find_datamodule_files(model_name: str, checkpoints_dir: Path) -> tuple[Union[Path, None], Union[Path, None]]:
    """Find the _hparams.yaml and _state.pt files for a given model name."""
    info = parse_model_name(model_name)
    data_config = info['data_config']
    group = info['group']
    fold = info['fold']

    if data_config is None or group is None or fold is None:
        return None, None

    dm_base = f"datamodule-data={data_config}-group={group}-fold={fold}"
    hparams = checkpoints_dir / f"{dm_base}_hparams.yaml"
    state = checkpoints_dir / f"{dm_base}_state.pt"

    hparams = hparams if hparams.exists() else None
    state = state if state.exists() else None
    return hparams, state


def main():
    parser = argparse.ArgumentParser(
        description="Collect ensemble checkpoints into a self-contained directory."
    )
    parser.add_argument(
        '--weights', required=True,
        help="Path to the ensemble weights JSON file."
    )
    parser.add_argument(
        '--checkpoints', required=True,
        help="Directory containing all model and datamodule checkpoints."
    )
    parser.add_argument(
        '--output', required=True,
        help="Name/path of the output directory to create."
    )
    parser.add_argument(
        '--overwrite', action='store_true',
        help="Delete the output directory first if it already exists."
    )
    args = parser.parse_args()

    weights_file = Path(args.weights)
    checkpoints_dir = Path(args.checkpoints)
    output_dir = Path(args.output)

    # Validate inputs
    if not weights_file.exists():
        print(f"Error: weights file not found: {weights_file}")
        sys.exit(1)
    if not checkpoints_dir.is_dir():
        print(f"Error: checkpoints directory not found: {checkpoints_dir}")
        sys.exit(1)
    if output_dir.exists():
        if args.overwrite:
            print(f"Overwriting existing output directory: {output_dir}")
            shutil.rmtree(output_dir)
        else:
            print(f"Error: output directory already exists: {output_dir}")
            sys.exit(1)

    # Load weights file
    with open(weights_file) as f:
        weights_doc = json.load(f)
    weights = weights_doc.get('weights', {})
    if not weights:
        print("Error: no 'weights' key found in the weights file.")
        sys.exit(1)

    print(f"Ensemble task : {weights_doc.get('task', 'unknown')}")
    print(f"Ensemble method: {weights_doc.get('method', 'unknown')}")
    print(f"Models in file : {len(weights)}")

    # Create output structure
    output_dir.mkdir(parents=True)
    ckpt_out = output_dir / 'checkpoints'
    ckpt_out.mkdir()

    # Copy weights file
    shutil.copy2(weights_file, output_dir / weights_file.name)
    print(f"\nCopied weights file -> {output_dir / weights_file.name}")

    # Collect checkpoints
    missing_models = []
    missing_datamodules = []
    copied = 0

    print(f"\nCollecting checkpoints into {ckpt_out} ...\n")
    for model_name in weights:
        model_file = find_model_file(model_name, checkpoints_dir)
        hparams_file, state_file = find_datamodule_files(model_name, checkpoints_dir)

        # Model checkpoint
        if model_file is None:
            print(f"  [MISSING model]      {model_name}")
            missing_models.append(model_name)
        else:
            shutil.copy2(model_file, ckpt_out / model_file.name)
            print(f"  [model]   {model_file.name}")

        # Datamodule files
        if hparams_file is None and state_file is None:
            print(f"  [MISSING datamodule] {model_name}")
            missing_datamodules.append(model_name)
        else:
            if hparams_file:
                shutil.copy2(hparams_file, ckpt_out / hparams_file.name)
                print(f"  [dm]      {hparams_file.name}")
            if state_file:
                shutil.copy2(state_file, ckpt_out / state_file.name)
                print(f"  [dm]      {state_file.name}")

        if model_file is not None and (hparams_file or state_file):
            copied += 1
        print()

    # Summary
    print("=" * 60)
    print(f"Done. Fully collected: {copied} / {len(weights)} models")
    if missing_models:
        print(f"\nMissing model checkpoints ({len(missing_models)}):")
        for name in missing_models:
            print(f"  {name}")
    if missing_datamodules:
        print(f"\nMissing datamodule files ({len(missing_datamodules)}):")
        for name in missing_datamodules:
            print(f"  {name}")
    print(f"\nOutput directory: {output_dir.resolve()}")


if __name__ == '__main__':
    main()
