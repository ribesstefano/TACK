from pathlib import Path
from typing import Dict, Union, Any

import numpy as np
import pandas as pd
from datasets import load_dataset
import matplotlib.pyplot as plt
import seaborn as sns

from tackai.training import create_cv_splits
from tackai import DegradationComplexDataModule


def get_bin_label(
        row: Union[pd.Series, Dict[str, Any]],
        dmax_threshold: float = 80.0,
        dc50_threshold: float = 100.0,
) -> Union[int, float]:
    """ Get binary activity label based on Dmax and DC50 thresholds.
    
    Args:
        row (pd.Series | Dict[str, Any]): A row from the dataframe containing 'Dmax' and 'DC50' columns.
        dmax_threshold (float): Threshold for Dmax to consider a compound active.
        dc50_threshold (float): Threshold for DC50 to consider a compound active.
        
    Returns:
        int | float: 1 for active, 0 for inactive, np.nan for undefined.
    """
    dc50 = row.get('DC50', np.nan)
    dmax = row.get('Dmax', np.nan)

    # Return inactive (0) if either DC50 is above threshold or Dmax is below threshold
    if pd.notna(dc50) and float(dc50) >= dc50_threshold:
        return 0
    if pd.notna(dmax) and float(dmax) < dmax_threshold:
        return 0

    # If either DC50 or Dmax is missing, we cannot determine activity, return np.nan
    if pd.isna(dc50) or pd.isna(dmax):
        return np.nan

    # Return active (1) if DC50 is below threshold and Dmax is above or equal to threshold
    if float(dc50) < dc50_threshold and float(dmax) >= dmax_threshold:
        return 1
    return 0

def map_bin_labels(example: Dict[str, Any]) -> Dict[str, Any]:
    """ Map binary activity labels to the example based on Dmax and DC50 values. """
    row = {
        'Dmax': example.get('Value_Dmax', np.nan),
        'DC50': example.get('Value_DC50', np.nan),
    }
    example['Activity'] = get_bin_label(row)
    return example

def convert_dc50_to_pdc50(example, type_col='Value_Type', value_col='Value'):
    """ Convert DC50 in nano Molar to pDC50 (-log10(M)). """
    if type_col in example and example[type_col] == 'Dmax':
        return example
    example['Value'] = DegradationComplexDataModule.convert_dc50_to_pdc50(example[value_col])
    return example

def split_held_out(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """ Split the dataframe into held-out and non-held-out sets based on the
    'SMILES_Held_Out' column.
    
    Args:
        df (pd.DataFrame): The input dataframe containing a 'SMILES_Held_Out' column.
        
    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: A tuple containing two dataframes:
            - The first dataframe contains non-held-out samples.
            - The second dataframe contains held-out samples.
    """
    held_out_df = df[df['SMILES_Held_Out']].copy().reset_index(drop=True)
    df = df[~df['SMILES_Held_Out']].copy().reset_index(drop=True)
    return df, held_out_df

def main():
    held_out_dfs = {}
    dfs = {}
    for task in ['bin', 'dc50', 'dmax']:
        # Download and prepare dataset
        ds_config = 'default'
        if 'dmax' in task:
            ds_config = 'Dmax'
        elif 'dc50' in task:
            ds_config = 'DC50'
        elif task == 'bin':
            ds_config = 'multitask'
        ds = load_dataset(
            "ailab-bio/TACK",
            ds_config,
            split="train",
        )
        df = ds.to_pandas()

        # Process labels based on the task
        if task == 'bin':
            df = df.apply(map_bin_labels, axis=1)
        elif task == 'dc50':
            df = df.apply(convert_dc50_to_pdc50, axis=1)
            # Rename 'Value' column to 'pDC50' for clarity
            df.rename(columns={'Value': 'pDC50'}, inplace=True)
        elif task == 'dmax':
            df.rename(columns={'Value': 'Dmax'}, inplace=True)
        
        # Split held-out samples
        df, held_out_df = split_held_out(df)
        
        dfs[task] = df
        held_out_dfs[task] = held_out_df

        print(f"Loaded {len(dfs[task])} samples for task {task}")
        print('SMILES_Scaffold_Cluster' in dfs[task].columns)

    group2column = {
        'random': None,
        'scaffold': 'SMILES_Scaffold_Cluster',
        'butina': 'SMILES_Butina_Cluster',
    }

    plot_data = []

    for task, train_val_df in dfs.items():
        print(f"Task: {task}")
        for fold in create_cv_splits(train_val_df, group_col=group2column['scaffold']):
            train_df = train_val_df.iloc[fold['train_idx']].copy()
            val_df = train_val_df.iloc[fold['test_idx']].copy()
            
            if task == 'bin':
                # print(f"Train positives: {train_df['Activity'].sum()}, Train negatives: {(train_df['Activity'] == 0).sum()}")
                # print(f"Val positives: {val_df['Activity'].sum()}, Val negatives: {(val_df['Activity'] == 0).sum()}")
                train_pos = train_df['Activity'].sum()
                train_neg = (train_df['Activity'] == 0).sum()
                val_pos = val_df['Activity'].sum()
                val_neg = (val_df['Activity'] == 0).sum()
            
                print(f"Train size: {len(train_df)}, Val size: {len(val_df)} - Train perc: {train_pos / len(train_df) * 100:.2f}% positives, Val perc: {val_pos / len(val_df) * 100:.2f}% positives")
            else:
                print(f"Train size: {len(train_df)}, Val size: {len(val_df)}")
            
            train_df['Task'] = task
            train_df['Set'] = 'Train'
            train_df['Fold'] = fold['fold'] + 1
            val_df['Task'] = task
            val_df['Set'] = 'Validation'
            val_df['Fold'] = fold['fold'] + 1
            
            plot_data.append(train_df)
            plot_data.append(val_df)

    plot_df = pd.concat(plot_data, ignore_index=True)
    plot_df.head()

    plot_dir = Path('plots')
    plot_dir.mkdir(exist_ok=True)

    colors = {
        'blue': '#4B9ECE',
        'orange': '#FFAA6E',
        'light_blue': '#50B1D8',
        'dark_orange': '#FF8428',
    }

    tasks = ['bin', 'dc50', 'dmax']
    target_cols = {
        'bin': 'Activity',
        'dc50': 'pDC50',
        'dmax': 'Dmax'
    }

    cv_repeats = 5

    fig, axes = plt.subplots(3, 1, figsize=(3 * cv_repeats, 7), sharex=True, constrained_layout=True)

    for idx, task in enumerate(tasks):
        target_col = target_cols[task]
        df_task = plot_df[plot_df['Task'] == task].dropna(subset=[target_col])
        df_task = df_task[['Fold', 'Set', target_col]].copy()
        hold_out_df = held_out_dfs[task]
        hold_out_mean = hold_out_df[target_col].mean()

        # Rename columns for visualization
        if task == 'dc50':
            df_task.rename(columns={'pDC50': '$pDC_{50}$'}, inplace=True)
            target_col = '$pDC_{50}$'
        elif task == 'dmax':
            df_task.rename(columns={'Dmax': '$D_{max}$'}, inplace=True)
            target_col = '$D_{max}$'

        ax = axes[idx]
        if task == 'bin':
            sns.barplot(
                data=df_task,
                x='Fold',
                y=target_col,
                hue='Set',
                palette=[colors['blue'], colors['dark_orange']],
                ax=ax
            )
            ax.grid(axis='y', alpha=0.3)
        else:
            sns.boxplot(
                data=df_task,
                x='Fold',
                y=target_col,
                hue='Set',
                palette=[colors['blue'], colors['dark_orange']],
                flierprops={'marker': '.', 'markersize': 3, 'alpha': 0.3},
                ax=ax
            )
        # Add held-out mean line
        if not hold_out_df.empty:
            ax.axhline(hold_out_mean, color='black', linestyle='--', label='Hold-out Mean')
        # ax.set_title(panel_titles[task])
        if idx == 2:
            ax.set_xlabel('Fold Index')
        else:
            ax.set_xlabel('')

        ax.set_ylabel(target_col)

        # Remove legend from individual axes
        ax.legend_.remove()

    # Create a single legend at the bottom
    plt.tight_layout(rect=[0, 0.07, 1, 1])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.03), ncol=3, title=None)
    plt.savefig(plot_dir / 'cv_distribution.pdf', bbox_inches='tight')
    plt.close()
    
    print(f"Plots saved to {plot_dir / 'cv_distribution.pdf'}")


if __name__ == "__main__":
    main()