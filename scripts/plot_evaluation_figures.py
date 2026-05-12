#!/usr/bin/env python3
"""
Generate Figure 2 (dataset distributions) and Figure 3 (model evaluation)
for the TACK paper.

Each sub-figure is saved individually (fig2a, fig2b, fig2c_*, fig3a, fig3b)
and both figures are also saved as combined layouts.
"""
import argparse
import logging
import warnings
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.transforms as mtransforms
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from datasets import load_dataset
from scipy.stats import f_oneway
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from statsmodels.sandbox.stats.multicomp import MultiComparison
from tqdm import tqdm

from tackai.models_comparison import calc_classification_metrics, calc_regression_metrics

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

COLORS = {
    'blue': '#4B9ECE',
    'orange': '#FFAA6E',
    'green': '#9DCE9C',
    'purple': '#C8ABDA',
}

DC50_THRESHOLD_NM = 100.0
DMAX_THRESHOLD_PCT = 80.0
PRECISION_THRESHOLD = 0.8

# Best methods selected from best_features_selection.ipynb
BEST_DC50_METHOD = 'MLP-DC50 Cell-OneHot E3-OneHot Mol-Desc POI-OneHot Time'
BEST_DMAX_METHOD = 'XGB-DMAX Cell-Text E3-OneHot Mol-Desc POI-Vec Time'
BEST_BIN_MLP = 'MLP-BIN Cell-OneHot E3-OneHot Mol-Desc POI-OneHot Time'
BEST_BIN_XGB = 'XGB-BIN Cell-Text E3-ESM-S Mol-Desc POI-ESM-S POI/E3-ESM-S-PCA Time'

BIN_METHOD_ORDER = ['PROTAC-STAN', 'MLP', 'XGB']

METRIC_NAMES = {
    'roc_auc': 'ROC-AUC',
    'pr_auc': 'PR-AUC',
    'mcc': 'MCC',
    'recall': f'Recall (Prec. ≥ {PRECISION_THRESHOLD})',
    'mae': 'MAE',
    'mse': 'MSE',
    'rmse': 'RMSE',
    'r2': 'R²',
    'rho': "Spearman’s ρ",
}

METRIC_DIRECTION = {
    'roc_auc': 'maximize', 'pr_auc': 'maximize', 'mcc': 'maximize',
    'recall': 'maximize', 'r2': 'maximize', 'rho': 'maximize',
    'mae': 'minimize', 'mse': 'minimize', 'rmse': 'minimize',
}


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def dc50_to_pdc50(value: float) -> float:
    return -np.log10(value * 1e-9 + 1e-12)


def set_size(
    width_pt: float,
    fraction: float = 1.0,
    subplots: tuple = (1, 1),
) -> tuple:
    fig_width_in = width_pt * fraction / 72.27
    golden_ratio = (5**0.5 - 1) / 2
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])
    return fig_width_in, fig_height_in


def clean_method_name(method: str, data: str = None) -> str:
    if 'protac-stan' in method.lower():
        return 'PROTAC-STAN'

    prefix = ''
    if 'xgb' in method.lower():
        prefix = 'XGB'
    elif 'mlp' in method.lower():
        prefix = 'MLP'

    for tag, suffix in [('_qr', '-QR'), ('_mve', '-MVE'), ('_bin', '-BIN'),
                        ('_dmax', '-DMAX'), ('_dc50', '-DC50')]:
        if tag in method.lower():
            prefix += suffix

    data_info = method if data is None else data
    feature_map = [
        ('fp512r16', 'FP'), ('_desc', 'Mol-Desc'), ('_poi_ord', 'POI-Ord'),
        ('_lig_ord', 'E3-Ord'), ('_assay_time', 'Time'), ('_cell_ord', 'Cell-Ord'),
        ('_cell_text', 'Cell-Text'), ('_cell_pt', 'Cell-Emb'),
        ('_cell_onehot', 'Cell-OneHot'), ('_poi_pt', 'POI-Emb'),
        ('_poi_onehot', 'POI-OneHot'), ('_poi_vec', 'POI-Vec'),
        ('_lig_pt', 'E3-Emb'), ('_lig_onehot', 'E3-OneHot'),
        ('_lig_vec', 'E3-Vec'), ('_poi_emb', 'POI-ESM-S'),
        ('_lig_emb', 'E3-ESM-S'), ('_poi_pca44_lig_pca', 'POI/E3-ESM-S-PCA'),
    ]
    features = sorted(label for key, label in feature_map if key in data_info.lower())
    return prefix + ' ' + ' '.join(features)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_tack_dataset() -> pd.DataFrame:
    logger.info('Downloading TACK dataset from HuggingFace...')
    ds = load_dataset('ailab-bio/TACK', 'default', split='train')
    return ds.to_pandas()


def load_predictions(
    predictions_dir: Path,
    protac_stan_dir: Path,
    methods_filter: list = None,
) -> pd.DataFrame:
    prediction_files = list(predictions_dir.glob('*.csv'))
    protac_stan_files = (
        list(protac_stan_dir.glob('*.csv')) if protac_stan_dir.exists() else []
    )

    results = []

    for file in tqdm(prediction_files, desc='Loading predictions'):
        parts = file.stem.split('=')
        model_name = parts[1].split('-')[0]
        data_name = parts[2].split('-')[0]
        clean = clean_method_name(model_name, data_name)
        if methods_filter is not None and clean not in methods_filter:
            continue

        df = pd.read_csv(file)
        df = _normalise_prediction_df(df, clean, is_protac_stan=False, stem=file.stem)
        results.append(df)

    for file in protac_stan_files:
        if methods_filter is not None and 'PROTAC-STAN' not in methods_filter:
            continue
        df = pd.read_csv(file)
        df = _normalise_prediction_df(df, 'PROTAC-STAN', is_protac_stan=True, stem=file.stem)
        results.append(df)

    if not results:
        raise ValueError('No prediction files found.')

    combined = pd.concat(results, ignore_index=True)
    combined = _drop_incomplete_folds(combined)
    combined = _convert_dc50_column(combined)
    return combined


def _normalise_prediction_df(
    df: pd.DataFrame,
    method: str,
    is_protac_stan: bool,
    stem: str,
) -> pd.DataFrame:
    df = df.copy()
    df['method'] = method
    if is_protac_stan:
        df['group'] = 'scaffold'
    df = df.rename(columns={'group': 'split', 'value_type': 'task', 'confidence': 'prob'})
    df['task'] = (df['task']
                  .str.replace('dmax', 'Dmax', regex=False)
                  .str.replace('dc50', 'DC50', regex=False)
                  .str.replace('binary_class', 'bin', regex=False)
                  .str.replace('multitask', 'bin', regex=False)
                  .str.replace('heldout', 'bin', regex=False))

    if df['task'].iloc[0] == 'bin' and 'prob' not in df.columns:
        df['prob'] = df['pred'].copy()
        df['pred'] = (df['prob'] >= 0.5).astype(int)

    df['set'] = 'test' if ('test' in stem or 'heldout' in stem) else 'val'
    return df


def _drop_incomplete_folds(df: pd.DataFrame) -> pd.DataFrame:
    max_folds = df.groupby('method')['fold'].nunique().max()
    bad_idx = []
    for (method, task), group in df.groupby(['method', 'task']):
        n = group['fold'].nunique()
        if n < max_folds:
            logger.warning('%s / %s: %d folds (expected %d), dropping.', method, task, n, max_folds)
            bad_idx.extend(group.index.tolist())
    return df.drop(bad_idx).reset_index(drop=True)


def _convert_dc50_column(df: pd.DataFrame) -> pd.DataFrame:
    mask = df['task'] == 'DC50'
    df.loc[mask, 'target'] = df.loc[mask, 'target'].apply(dc50_to_pdc50)
    df.loc[mask, 'pred'] = df.loc[mask, 'pred'].apply(dc50_to_pdc50)
    return df


# ---------------------------------------------------------------------------
# Figure 3 – model evaluation
# ---------------------------------------------------------------------------

def _draw_parity_scatter(
    ax: plt.Axes,
    df: pd.DataFrame,
    thresh: float,
    ax_name: str,
    color: str,
    font_size: int = 9,
    text_pos: tuple = (0.05, 0.95),
) -> None:
    df_metrics = calc_regression_metrics(
        df, cycle_col='fold', val_col='target', pred_col='pred', thresh=thresh
    )
    y_true = df['target'] > thresh
    y_pred = df['pred'] > thresh
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)

    ax.scatter(df['pred'], df['target'], alpha=0.5, s=7, color=color,
               rasterized=True, linewidths=0.1)
    val_min = min(df['target'].min(), df['pred'].min())
    val_max = max(df['target'].max(), df['pred'].max())
    ax.plot([val_min, val_max], [val_min, val_max], 'k--', lw=1)
    ax.axhline(y=thresh, color='red', linestyle='--', lw=1, alpha=0.7)
    ax.axvline(x=thresh, color='red', linestyle='--', lw=1, alpha=0.7)

    rmse = np.sqrt(df_metrics['mse'].mean())
    text = (
        f"MAE: {df_metrics['mae'].mean():.2f}\n"
        f"MSE: {df_metrics['mse'].mean():.2f}\n"
        f"RMSE: {rmse:.2f}\n"
        f"$R^2$: {df_metrics['r2'].mean():.2f}\n"
        f"$\\rho$: {df_metrics['rho'].mean():.2f}\n"
        f"Precision: {prec:.2f}\n"
        f"Recall: {rec:.2f}\n"
        f"AUC: {auc:.2f}"
    )
    ax.text(*text_pos, text, transform=ax.transAxes, verticalalignment='top',
            fontsize=font_size - 3,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8,
                      edgecolor='black', linewidth=0.5))
    ax.set_xlabel(f'Predicted {ax_name}', fontsize=font_size, fontweight='bold')
    ax.set_ylabel(f'Measured {ax_name}', fontsize=font_size, fontweight='bold')
    ax.tick_params(axis='both', labelsize=font_size - 2)
    ax.grid(alpha=0.3)


def _run_anova(df: pd.DataFrame, col: str, group_var: str = 'method') -> float:
    groups = [g[col].values for _, g in df.groupby(group_var)]
    return f_oneway(*groups)[1]


def _color_matches(color, target: str) -> bool:
    if isinstance(color, str):
        return color == target
    if isinstance(color, np.ndarray):
        return np.allclose(color.flatten(), mcolors.to_rgba(target))
    return False


def _recolor_tukey_ax(ax: plt.Axes) -> None:
    """Replace statsmodels default b/r/0.5 colors with the TACK palette."""
    color_map = {
        'b': COLORS['green'],
        'r': COLORS['purple'],
        '0.5': '#CCCCCC',
    }
    for container in ax.containers:
        for child in container.get_children():
            current = child.get_color() if hasattr(child, 'get_color') else None
            for src, dst in color_map.items():
                if current is not None and _color_matches(current, src):
                    child.set_color(dst)
    for line in ax.get_lines():
        current = line.get_color()
        ls = line.get_linestyle()
        if ls == '--' and _color_matches(current, '0.7'):
            line.set_color('#808080')
            line.set_linewidth(1.0)
            line.set_alpha(0.5)
        else:
            for src, dst in color_map.items():
                if _color_matches(current, src):
                    line.set_color(dst)


def _draw_ci_subplot(
    ax: plt.Axes,
    df_metrics: pd.DataFrame,
    metric: str,
    method_order: list,
    font_size: int = 9,
) -> None:
    if df_metrics[metric].isna().any():
        logger.warning('Metric %s has NaN values; skipping.', metric)
        return

    mc = MultiComparison(
        df_metrics[metric].values,
        df_metrics['method'].values,
        group_order=method_order,
    )
    tukey_result = mc.tukeyhsd(alpha=0.05)

    direction = METRIC_DIRECTION.get(metric, 'maximize')
    asc = direction == 'minimize'
    best = df_metrics.groupby('method')[metric].mean().sort_values(ascending=asc).index[0]
    
    print(df_metrics.groupby('method')[metric].mean().sort_values(ascending=asc).to_markdown())
    print()
    print(tukey_result.summary())
    print()

    p_anova = _run_anova(df_metrics, metric)
    tukey_result.plot_simultaneous(comparison_name=best, ax=ax)
    _recolor_tukey_ax(ax)

    ax.set_xlabel(METRIC_NAMES.get(metric, metric), fontsize=font_size, fontweight='bold')
    ax.set_title(f'p = {p_anova:.2e}', fontsize=font_size - 3)
    ax.tick_params(axis='both', labelsize=font_size - 2)
    ax.grid(alpha=0.3)

    best_val = df_metrics.groupby('method')[metric].mean()[best]
    logger.info('Best %s: %s = %.4f', metric, best, best_val)


def _compute_bin_metrics(results_df: pd.DataFrame) -> tuple:
    """Return (df_metrics, metric_list) for the binary classification task."""
    bin_method_map = {
        BEST_BIN_MLP: 'MLP',
        BEST_BIN_XGB: 'XGB',
        'PROTAC-STAN': 'PROTAC-STAN',
    }
    val_bin = results_df[
        (results_df['set'] == 'val') &
        (results_df['task'] == 'bin') &
        (results_df['method'].isin(bin_method_map))
    ].copy()
    val_bin['method'] = val_bin['method'].map(bin_method_map)

    df_metrics = calc_classification_metrics(
        val_bin,
        cycle_col='fold',
        val_col='target',
        prob_col='prob',
        pred_col='pred',
        precision_threshold=PRECISION_THRESHOLD,
    )
    metric_list = [c for c in df_metrics.columns[3:] if c != 'tnr']
    return df_metrics, metric_list

    
def make_fig3(results_df: pd.DataFrame, plot_dir: Path) -> None:
    TEXT_WIDTH = 506.295 * 1.5 # LaTeX \textwidth in points
    
    pdc50_thresh = dc50_to_pdc50(DC50_THRESHOLD_NM)
    test_df = results_df[results_df['set'] == 'test']

    df_dc50 = test_df[(test_df['method'] == BEST_DC50_METHOD) & (test_df['task'] == 'DC50')].copy()
    df_dmax = test_df[(test_df['method'] == BEST_DMAX_METHOD) & (test_df['task'] == 'Dmax')].copy()

    df_metrics, metric_list = _compute_bin_metrics(results_df)

    fig_w, _ = set_size(TEXT_WIDTH, fraction=1.0)
    fig_h = fig_w * 0.32
    
    fig3 = plt.figure(figsize=(fig_w, fig_h), layout='constrained')
    
    # 1. Outer GridSpec: 1 row, 3 columns. 
    # The 3rd column is given twice the width to accommodate the 2x2 CI grid.
    gs_outer = fig3.add_gridspec(1, 3, width_ratios=[1.2, 1.2, 2])

    # --- Part A: Scatter Plots ---
    ax_dc50 = fig3.add_subplot(gs_outer[0])
    ax_dmax = fig3.add_subplot(gs_outer[1])
    
    _draw_parity_scatter(
        ax_dc50, df_dc50, pdc50_thresh, r'$pDC_{50}$',
        COLORS['blue'], font_size=11,
    )
    _draw_parity_scatter(
        ax_dmax, df_dmax, DMAX_THRESHOLD_PCT,
        r'$D_{\mathrm{max}}$ (%)', COLORS['orange'],
        font_size=11, text_pos=(0.05, 0.95),
    )
    
    # --- Part B: CI Plots (Nested 2x2 Grid) ---
    # 2. Subdivide the 3rd outer column into a 2x2 grid.
    gs_ci = gs_outer[2].subgridspec(2, 2)
    
    # Top-right shares Y with Top-left. Bottom-right shares Y with Bottom-left.
    ax_tl = fig3.add_subplot(gs_ci[0, 0])
    ax_tr = fig3.add_subplot(gs_ci[0, 1], sharey=ax_tl)
    ax_bl = fig3.add_subplot(gs_ci[1, 0])
    ax_br = fig3.add_subplot(gs_ci[1, 1], sharey=ax_bl)
    
    axes_ci = [ax_tl, ax_tr, ax_bl, ax_br]

    for i, metric in enumerate(metric_list[:4]):
        ax = axes_ci[i]
        _draw_ci_subplot(ax, df_metrics, metric, BIN_METHOD_ORDER, font_size=10)
        
        # Strip y-axis labels and ticks ONLY from the right-hand plots
        if i in [1, 3]:
            ax.tick_params(labelleft=False)
            ax.set_ylabel('')

    # Pad remaining axes if metric_list is somehow shorter than 4
    for j in range(len(metric_list), 4):
        axes_ci[j].axis('off')
    
    # Set the parity plots to have a square aspect ratio
    ax_dc50.set_box_aspect(1)
    ax_dmax.set_box_aspect(1)
    
    # Print the subfigure labels (a) and (b)
    # NOTE: We need to use blended transforms to position the labels at the same
    # height relative to the overall figure, rather than relative to each
    # individual subplot, because the CI subplots are shorter-than-wide and
    # would end

    # Create blended transforms
    # X is relative to the specific axis, Y is relative to the figure (0.0 to 1.0)
    trans_a = mtransforms.blended_transform_factory(ax_dc50.transAxes, fig3.transFigure)
    trans_b = mtransforms.blended_transform_factory(axes_ci[0].transAxes, fig3.transFigure)

    # Plot the labels using a unified Y coordinate (e.g., 0.95 = 95% up the figure)
    # You might need to tweak 0.95 slightly depending on how much whitespace is at the top
    ax_dc50.text(-0.2, 0.96, 'a)', transform=trans_a, fontsize=14, fontweight='bold', va='top')
    axes_ci[0].text(-0.40, 0.96, 'b)', transform=trans_b, fontsize=14, fontweight='bold', va='top')

    fig3.set_size_inches(fig_w, fig_h)

    _save(fig3, plot_dir, 'fig3')


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save(fig: plt.Figure, plot_dir: Path, stem: str) -> None:
    for ext in ('pdf', 'svg'):
        fig.savefig(plot_dir / f'{stem}.{ext}')
    logger.info('Saved %s.', stem)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Generate TACK paper figures (Fig 2 and Fig 3).'
    )
    parser.add_argument(
        '--plot_dir', type=Path, default=Path('./plots'),
        help='Directory where figures are saved.',
    )
    parser.add_argument(
        '--predictions_dir', type=Path, default=Path('./predictions'),
        help='Directory containing model prediction CSV files.',
    )
    parser.add_argument(
        '--protac_stan_dir', type=Path, default=Path('./protac_stan_predictions'),
        help='Directory containing PROTAC-STAN prediction CSV files.',
    )
    parser.add_argument(
        '--skip_fig2', action='store_true',
        help='Skip Figure 2 (requires HuggingFace dataset access).',
    )
    parser.add_argument(
        '--skip_fig3', action='store_true',
        help='Skip Figure 3 (requires prediction files).',
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = parse_args()
    args.plot_dir.mkdir(parents=True, exist_ok=True)

    # if not args.skip_fig2:
    #     df = load_tack_dataset()

    if not args.skip_fig3:
        methods_needed = [BEST_DC50_METHOD, BEST_DMAX_METHOD,
                          BEST_BIN_MLP, BEST_BIN_XGB, 'PROTAC-STAN']
        results_df = load_predictions(
            args.predictions_dir,
            args.protac_stan_dir,
            methods_filter=methods_needed,
        )
        make_fig3(results_df, args.plot_dir)

    logger.info('Done. Figures saved to %s', args.plot_dir)


if __name__ == '__main__':
    main()
