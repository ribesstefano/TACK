"""
Statistical analysis and plotting routines for model comparison.
Taken from: https://github.com/polaris-hub/polaris-method-comparison/blob/main/ADME_example/model_comparison.py
"""
import math
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pingouin as pg
from scipy import stats
from scipy.stats import spearmanr, f_oneway
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.libqsturng import psturng, qsturng
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import scikit_posthocs as sp
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
    precision_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    recall_score,
)
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import seaborn as sns

def calc_regression_metrics(
    df: pd.DataFrame,
    cycle_col: str,
    val_col: str,
    pred_col: str,
    thresh: float,
) -> pd.DataFrame:
    """ Calculate regression metrics (MAE, MSE, R2, prec, recall) for each method and split.

    Args:
        df (pd.DataFrame): Input dataframe; must contain columns [method, split] as well as the columns specified in the other arguments.
        cycle_col (str): Column indicating the cross-validation fold.
        val_col (str): Column with the ground truth value.
        pred_col (str): Column with predictions.
        thresh (float): Threshold for binary classification.

    Returns:
        pd.DataFrame: A dataframe with columns [cv_cycle, method, split, mae, mse, r2, rho, prec, recall, roc_auc].
    """
    df_in = df.copy()
    metric_ls = ["mae", "mse", "r2", "rho", "prec", "recall", "roc_auc"]
    metric_list = []
    df_in['true_class'] = df_in[val_col] > thresh
    # Make sure the thresh variable creates 2 classes
    assert len(df_in.true_class.unique()) == 2, "Binary classification requires two classes"
    df_in['pred_class'] = df_in[pred_col] > thresh

    for k, v in df_in.groupby([cycle_col, "method", "split"]):
        cycle, method, split = k
        mae = mean_absolute_error(v[val_col], v[pred_col])
        mse = mean_squared_error(v[val_col], v[pred_col])
        r2 = r2_score(v[val_col], v[pred_col])
        recall = recall_score(v.true_class, v.pred_class)
        prec = precision_score(v.true_class, v.pred_class)
        roc_auc = roc_auc_score(v.true_class, v.pred_class)
        rho, _ = spearmanr(v[val_col], v[pred_col])
        metric_list.append([cycle, method, split, mae, mse, r2, rho, prec, recall, roc_auc])
    metric_df = pd.DataFrame(metric_list, columns=["cv_cycle", "method", "split"] + metric_ls)
    return metric_df


def rm_tukey_hsd(
    df: pd.DataFrame,
    metric: str,
    group_col: str,
    alpha: float = 0.05,
    sort: bool = False,
    direction_dict: Optional[Dict[str, str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """ Perform repeated measures Tukey HSD test on the given dataframe.

    Args:
        df (pd.DataFrame): Input dataframe containing the data.
        metric (str): The metric column name to perform the test on.
        group_col (str): The column name indicating the groups.
        alpha (float): Significance level for the test. Default is 0.05.
        sort (bool): Whether to sort the output tables. Default is False.
        direction_dict (Optional[Dict[str, str]]): Maps metric name to 'maximize' or 'minimize', used when `sort` is True. Default is None.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]: A tuple containing:
            - result_tab: DataFrame with pairwise comparisons and adjusted p-values.
            - df_means: DataFrame with mean values for each group.
            - df_means_diff: DataFrame with mean differences between groups.
            - pc: DataFrame with adjusted p-values for pairwise comparisons.
    """
    if sort and direction_dict and metric in direction_dict:
        if direction_dict[metric] == 'maximize':
            df_means = df.groupby(group_col).mean(numeric_only=True).sort_values(metric, ascending=False)
        elif direction_dict[metric] == 'minimize':
            df_means = df.groupby(group_col).mean(numeric_only=True).sort_values(metric, ascending=True)
        else:
            raise ValueError("Invalid direction. Expected 'maximize' or 'minimize'.")
    else:
        df_means = df.groupby(group_col).mean(numeric_only=True)

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning,
                                message='divide by zero encountered in scalar divide')
        aov = pg.rm_anova(dv=metric, within=group_col, subject='cv_cycle', data=df, detailed=True)
    mse = aov.loc[1, 'MS']
    df_resid = aov.loc[1, 'DF']

    methods = df_means.index
    n_groups = len(methods)
    n_per_group = df[group_col].value_counts().mean()

    tukey_se = np.sqrt(2 * mse / (n_per_group))
    q = qsturng(1 - alpha, n_groups, df_resid)

    num_comparisons = len(methods) * (len(methods) - 1) // 2
    result_tab = pd.DataFrame(index=range(num_comparisons),
                              columns=["group1", "group2", "meandiff", "lower", "upper", "p-adj"])

    df_means_diff = pd.DataFrame(index=methods, columns=methods, data=0.0)
    pc = pd.DataFrame(index=methods, columns=methods, data=1.0)

    # Calculate pairwise mean differences and adjusted p-values
    row_idx = 0
    for i, method1 in enumerate(methods):
        for j, method2 in enumerate(methods):
            if i < j:
                group1 = df[df[group_col] == method1][metric]
                group2 = df[df[group_col] == method2][metric]
                mean_diff = group1.mean() - group2.mean()
                studentized_range = np.abs(mean_diff) / tukey_se
                adjusted_p = psturng(studentized_range * np.sqrt(2), n_groups, df_resid)
                if isinstance(adjusted_p, np.ndarray):
                    adjusted_p = adjusted_p[0]
                lower = mean_diff - (q / np.sqrt(2) * tukey_se)
                upper = mean_diff + (q / np.sqrt(2) * tukey_se)
                result_tab.loc[row_idx] = [method1, method2, mean_diff, lower, upper, adjusted_p]
                pc.loc[method1, method2] = adjusted_p
                pc.loc[method2, method1] = adjusted_p
                df_means_diff.loc[method1, method2] = mean_diff
                df_means_diff.loc[method2, method1] = -mean_diff
                row_idx += 1

    df_means_diff = df_means_diff.astype(float)

    result_tab["group1_mean"] = result_tab["group1"].map(df_means[metric])
    result_tab["group2_mean"] = result_tab["group2"].map(df_means[metric])

    result_tab.index = result_tab['group1'] + ' - ' + result_tab['group2']

    return result_tab, df_means, df_means_diff, pc


def recall_at_precision(
    y_true: Union[np.ndarray, pd.Series],
    y_score: Union[np.ndarray, pd.Series],
    precision_threshold: float = 0.5,
    direction: str = 'greater',
) -> Tuple[float, Optional[float]]:
    """ Find the best recall achieved at or beyond a target precision.

    Args:
        y_true (Union[np.ndarray, pd.Series]): Ground truth binary labels.
        y_score (Union[np.ndarray, pd.Series]): Predicted scores or probabilities.
        precision_threshold (float): Minimum (if `direction='greater'`) or maximum (if `direction='lesser'`) precision to satisfy. Default is 0.5.
        direction (str): One of 'greater' or 'lesser'; selects whether `precision_threshold` is a lower or upper bound. Default is 'greater'.

    Returns:
        tuple[float, Optional[float]]: A tuple of (recall, threshold). If no threshold satisfies `precision_threshold`, the recall at the best achieved precision is returned with a `None` threshold.
    """
    if direction not in ['greater', 'lesser']:
        raise ValueError("Invalid direction. Expected one of: ['greater', 'lesser']")

    # This handles the "greater" logic automatically
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_score)

    # Filter for where precision meets your threshold
    if direction == 'greater':
        valid_indices = np.where(precisions >= precision_threshold)[0]
    else:  # direction == 'lesser'
        valid_indices = np.where(precisions <= precision_threshold)[0]

    if len(valid_indices) == 0:
        # Option A: Return 0 instead of NaN if that makes sense for your report
        # Option B: Return the recall at the maximum precision achieved
        if direction == 'greater':
            idx_prec = np.argmax(precisions)
        else:
            idx_prec = np.argmin(precisions)
        return recalls[idx_prec], None

    # Take the first index where precision is met
    # (Usually you want the maximum recall at that precision)
    best_idx = valid_indices[0]
    return recalls[best_idx], thresholds[min(best_idx, len(thresholds)-1)]


def calc_classification_metrics(
    df_in: pd.DataFrame,
    cycle_col: str,
    val_col: str,
    prob_col: str,
    pred_col: str,
    precision_threshold: float = 0.8,
) -> pd.DataFrame:
    """ Calculate classification metrics (ROC-AUC, PR-AUC, MCC, recall/TNR at precision) for each method and split.

    Args:
        df_in (pd.DataFrame): Input dataframe; must contain columns [method, split] as well as the columns specified in the other arguments.
        cycle_col (str): Column indicating the cross-validation fold.
        val_col (str): Column with the ground truth binary label.
        prob_col (str): Column with predicted probabilities/scores.
        pred_col (str): Column with predicted binary class.
        precision_threshold (float): Precision threshold used for the recall/TNR-at-precision metrics. Default is 0.8.

    Returns:
        pd.DataFrame: A dataframe with columns [cv_cycle, method, split, roc_auc, pr_auc, mcc, recall, tnr].
    """
    metric_list = []
    for k, v in df_in.groupby([cycle_col, "method", "split"]):
        cycle, method, split = k
        roc_auc = roc_auc_score(v[val_col], v[prob_col])
        pr_auc = average_precision_score(v[val_col], v[prob_col])
        mcc = matthews_corrcoef(v[val_col], v[pred_col])

        recall, _ = recall_at_precision(v[val_col].astype(bool), v[prob_col], precision_threshold, direction='greater')
        tnr, _ = recall_at_precision(~v[val_col].astype(bool), v[prob_col], precision_threshold, direction='lesser')

        metric_list.append([cycle, method, split, roc_auc, pr_auc, mcc, recall, tnr])

    metric_df = pd.DataFrame(metric_list, columns=["cv_cycle", "method", "split",
                                                    "roc_auc", "pr_auc", "mcc", "recall", "tnr"])
    return metric_df

# -------------- Plotting routines -------------------#


def make_boxplots_parametric(
    df: pd.DataFrame,
    metric_ls: List[str],
    precision_threshold: float = 0.5,
) -> None:
    """ Create boxplots for each metric using repeated measures ANOVA.

    Args:
        df (pd.DataFrame): Input dataframe containing the data.
        metric_ls (List[str]): List of metric column names to create boxplots for.
        precision_threshold (float): Precision threshold used in axis labels for precision-conditioned metrics. Default is 0.5.

    Returns:
        None
    """
    sns.set_context('notebook')
    sns.set(rc={'figure.figsize': (4, 3)}, font_scale=1.5)
    sns.set_style('whitegrid')
    _, axes = plt.subplots(1, len(metric_ls), sharex=False, sharey=False, figsize=(45, 8))
    # figure, axes = plt.subplots(1, 3, sharex=False, sharey=False, figsize=(16, 8))

    metric2name = {
        'mae': 'MAE',
        'mse': 'MSE',
        'r2': 'R2',
        'rho': "Spearman's Rho",
        'roc_auc': 'ROC-AUC',
        'pr_auc': 'PR-AUC',
        'mcc': 'Matthews Correlation Coefficient (MCC)',
        'recall': f'Recall at Precision ≥ {precision_threshold}',
        'prec': 'Precision',
        'tnr': f'True Negative Rate (at Precision ≥ {precision_threshold})',
    }

    for i, metric in enumerate(metric_ls):
        model = AnovaRM(data=df, depvar=metric, subject='cv_cycle', within=['method']).fit()
        p_value = model.anova_table['Pr > F'].iloc[0]
        ax = sns.boxplot(y=metric, x="method", hue="method", ax=axes[i], data=df, palette="Set2", legend=False)
        ax.set_title(f"ANOVA p={p_value:.1e}")
        ax.set_xlabel("")
        ax.set_ylabel(metric2name.get(metric, metric))
        x_tick_labels = ax.get_xticklabels()
        label_text_list = [x.get_text() for x in x_tick_labels]
        new_xtick_labels = ["\n".join(x.split("_")) for x in label_text_list]
        ax.set_xticks(list(range(0, len(x_tick_labels))))
        ax.set_xticklabels(new_xtick_labels)
        ax.tick_params(axis='x', rotation=0)
    # plt.tight_layout()


def make_boxplots_nonparametric(
    df: pd.DataFrame,
    metric_ls: List[str],
    precision_threshold: float = 0.5,
) -> None:
    """ Create boxplots for each metric using the Friedman test.

    Args:
        df (pd.DataFrame): Input dataframe containing the data.
        metric_ls (List[str]): List of metric column names to create boxplots for.
        precision_threshold (float): Precision threshold used in axis labels for precision-conditioned metrics. Default is 0.5.

    Returns:
        None
    """
    sns.set_context('notebook')
    sns.set(rc={'figure.figsize': (4, 3)}, font_scale=1.5)
    sns.set_style('whitegrid')
    _, axes = plt.subplots(1, len(metric_ls), sharex=False, sharey=False, figsize=(45, 8))

    metric2name = {
        'mae': 'MAE',
        'mse': 'MSE',
        'r2': 'R2',
        'rho': "Spearman's Rho",
        'roc_auc': 'ROC-AUC',
        'pr_auc': 'PR-AUC',
        'mcc': 'Matthews Correlation Coefficient (MCC)',
        'recall': f'Recall at Precision ≥ {precision_threshold}',
        'prec': 'Precision',
        'tnr': f'True Negative Rate (at Precision ≥ {precision_threshold})',
    }

    for i, metric in enumerate(metric_ls):
        friedman = pg.friedman(df, dv=metric, within="method", subject="cv_cycle")['p_unc'].values[0]
        ax = sns.boxplot(y=metric, x="method", hue="method", ax=axes[i], data=df, palette="Set2", legend=False)
        ax.set_title(f"Friedman p={friedman:.1e}")
        ax.set_xlabel("")
        ax.set_ylabel(metric2name.get(metric, metric))
        x_tick_labels = ax.get_xticklabels()
        label_text_list = [x.get_text() for x in x_tick_labels]
        new_xtick_labels = ["\n".join(x.split("_")) for x in label_text_list]
        ax.set_xticks(list(range(0, len(x_tick_labels))))
        ax.set_xticklabels(new_xtick_labels)
        ax.tick_params(axis='x', rotation=90)
    # plt.tight_layout()

def make_sign_plots_nonparametric(df: pd.DataFrame, metric_ls: List[str]) -> None:
    """ Create significance heatmaps for each metric using the Conover-Friedman post-hoc test.

    Args:
        df (pd.DataFrame): Input dataframe containing the data.
        metric_ls (List[str]): List of metric column names to create sign plots for.

    Returns:
        None
    """
    heatmap_args = {'linewidths': 0.25, 'linecolor': '0.1', 'clip_on': True, 'square': True}
    cmap = {
        'diag': 'white',
        'non-significant': 'lightgrey',
        'p < 0.001': 'steelblue',
        'p < 0.01': 'darkturquoise',
        'p < 0.05': 'paleturquoise',
    }
    # NOTE: The order is important for the colormap and must match the order in
    # which the categories are defined (see documentation for more details)
    cmap = list(cmap.values())
    sns.set_theme(rc={'figure.figsize': (4, 3)}, font_scale=1.5)
    _, axes = plt.subplots(1, len(metric_ls), sharex=False, sharey=True, figsize=(26, 8))

    for i, stat in enumerate(metric_ls):
        pivot_df = df.pivot(index='cv_cycle', columns='method', values=stat)
        pc = sp.posthoc_conover_friedman(pivot_df, p_adjust="holm")
        sub_ax, _ = sp.sign_plot(pc, **heatmap_args, ax=axes[i], xticklabels=True, cmap=cmap)  # Update xticklabels parameter
        sub_ax.set_title(stat.upper())

def make_critical_difference_diagrams(df: pd.DataFrame, metric_ls: List[str]) -> None:
    """ Create critical difference diagrams for each metric using the Conover-Friedman post-hoc test.

    Args:
        df (pd.DataFrame): Input dataframe containing the data.
        metric_ls (List[str]): List of metric column names to create diagrams for.

    Returns:
        None
    """
    _, axes = plt.subplots(6, 1, sharex=True, sharey=False, figsize=(16, 10))
    for i, stat in enumerate(metric_ls):
        pivot_df = df.pivot(index='cv_cycle', columns='method', values=stat)
        pc = sp.posthoc_conover_friedman(pivot_df, p_adjust="holm")
        avg_rank = df.groupby("cv_cycle")[stat].rank(pct=True).groupby(df.method).mean()
        sp.critical_difference_diagram(avg_rank, pc, ax=axes[i])
        axes[i].set_title(stat.upper())
    plt.tight_layout()

def make_normality_diagnostic(
    df: pd.DataFrame,
    metric_ls: List[str],
    precision_threshold: float = 0.5,
) -> None:
    """ Create a normality diagnostic plot grid with histograms and QQ plots for the given metrics.

    Args:
        df (pd.DataFrame): Input dataframe containing the data.
        metric_ls (List[str]): List of metrics to create plots for.
        precision_threshold (float): Precision threshold used in axis labels for precision-conditioned metrics. Default is 0.5.

    Returns:
        None
    """
    df_norm = df.copy()

    for metric in metric_ls:
        df_norm[metric] = df_norm[metric] - df_norm.groupby("method")[metric].transform("mean")

    df_norm = df_norm.melt(id_vars=["cv_cycle", "method", "split"],
                                   value_vars=metric_ls,
                                   var_name="metric",
                                   value_name="value")

    sns.set_context('notebook', font_scale=1.5)
    sns.set_style('whitegrid')

    metrics = df_norm['metric'].unique()
    n_metrics = len(metrics)

    _, axes = plt.subplots(2, n_metrics, figsize=(20, 10))

    # metric2name = {
    #     'mae': 'MAE',
    #     'mse': 'MSE',
    #     'r2': 'R2',
    #     'rho': "Spearman's Rho",
    #     'roc_auc': 'ROC-AUC',
    #     'pr_auc': 'PR-AUC',
    #     'mcc': 'Matthews Correlation\nCoefficient (MCC)',
    #     'recall': f'Recall at Precision ≥ {precision_threshold}',
    #     'prec': 'Precision',
    #     'tnr': f'True Negative Rate\n(at Precision ≥ {precision_threshold})',
    # }
    metric2name = {
        'mae': 'MAE',
        'mse': 'MSE',
        'rmse':'RMSE',
        'r2': 'R²',
        'rho': "Spearman's ρ",
        'roc_auc': 'ROC-AUC',
        'pr_auc': 'PR-AUC',
        'mcc': 'MCC',
        'prec': 'Precision',
        'recall': f'Recall (Prec. ≥ {precision_threshold})',
        'tnr': f'TNR (Prec. ≥ {precision_threshold})',
    }

    colors = {
        'blue': '#4B9ECE',
        'orange': '#FFAA6E',
        'light_blue': '#50B1D8',
        'dark_orange': '#FF8428',
        'green': '#9DCE9C',
        'purple': '#C8ABDA',
    }

    for i, metric in enumerate(metrics):
        ax = axes[0, i]
        sns.histplot(df_norm[df_norm['metric'] == metric]['value'], kde=True, ax=ax, color=colors['blue'])
        ax.set_title(f'{metric2name[metric]}', fontsize=16, fontweight='bold')
        # Change x-axis label
        ax.set_xlabel('Value')
        # If it's not the first plot, remove y-axis label
        if i != 0:
            ax.set_ylabel('')
        ax.grid(alpha=0.5)

    for i, metric in enumerate(metrics):
        ax = axes[1, i]
        metric_data = df_norm[df_norm['metric'] == metric]['value']
        stats.probplot(metric_data, dist="norm", plot=ax)
        ax.set_title("")
        # If it's not the first plot, remove y-axis label
        if i != 0:
            ax.set_ylabel('')
        ax.grid(alpha=0.5)

    plt.tight_layout()


def mcs_plot(
    pc: pd.DataFrame,
    effect_size: pd.DataFrame,
    means: pd.Series,
    labels: bool = True,
    cmap: Optional[str] = None,
    cbar_ax_bbox: Optional[Tuple[float, float, float, float]] = None,
    ax: Optional[Axes] = None,
    show_diff: bool = True,
    cell_text_size: int = 16,
    axis_text_size: int = 12,
    show_cbar: bool = True,
    reverse_cmap: bool = False,
    vlim: Optional[float] = None,
    **kwargs: Any,
) -> Axes:
    """ Create a multiple comparison of means plot using a heatmap.

    Args:
        pc (pd.DataFrame): DataFrame containing p-values for pairwise comparisons.
        effect_size (pd.DataFrame): DataFrame containing effect sizes for pairwise comparisons.
        means (pd.Series): Series containing mean values for each group.
        labels (bool): Whether to show labels on the axes. Default is True.
        cmap (Optional[str]): Colormap to use for the heatmap. Default is None.
        cbar_ax_bbox (Optional[Tuple[float, float, float, float]]): Bounding box for the colorbar axis. Default is None.
        ax (Optional[matplotlib.axes.Axes]): The axes on which to plot the heatmap. Default is None.
        show_diff (bool): Whether to show the mean differences in the plot. Default is True.
        cell_text_size (int): Font size for the cell text. Default is 16.
        axis_text_size (int): Font size for the axis text. Default is 12.
        show_cbar (bool): Whether to show the colorbar. Default is True.
        reverse_cmap (bool): Whether to reverse the colormap. Default is False.
        vlim (Optional[float]): Limit for the colormap. Default is None.
        **kwargs (Any): Additional keyword arguments for the heatmap.

    Returns:
        matplotlib.axes.Axes: The axes with the heatmap.
    """
    for key in ['cbar', 'vmin', 'vmax', 'center']:
        if key in kwargs:
            del kwargs[key]

    if not cmap:
        cmap = "coolwarm"
    if reverse_cmap:
        cmap = cmap + "_r"

    significance = pc.copy().astype(object)
    significance[(pc < 0.001) & (pc >= 0)] = '***'
    significance[(pc < 0.01) & (pc >= 0.001)] = '**'
    significance[(pc < 0.05) & (pc >= 0.01)] = '*'
    significance[(pc >= 0.05)] = ''

    # np.fill_diagonal(significance.values, '')

    # Create a DataFrame for the annotations
    if show_diff:
        annotations = effect_size.round(3).astype(str) + significance
    else:
        annotations = significance

    hax = sns.heatmap(effect_size, cmap=cmap, annot=annotations, fmt='', cbar=show_cbar, ax=ax,
                      annot_kws={"size": cell_text_size},
                      vmin=-2*vlim if vlim else None, vmax=2*vlim if vlim else None, **kwargs)

    if labels:
        label_list = list(means.index)
        x_label_list = [x + f'\n{means.loc[x].round(2)}' for x in label_list]
        y_label_list = [x + f'\n{means.loc[x].round(2)}\n' for x in label_list]
        hax.set_xticklabels(x_label_list, size=axis_text_size, ha='center', va='top', rotation=0,
                            rotation_mode='anchor')
        hax.set_yticklabels(y_label_list, size=axis_text_size, ha='center', va='center', rotation=90,
                            rotation_mode='anchor')

    hax.set_xlabel('')
    hax.set_ylabel('')

    return hax


def make_mcs_plot_grid(
    df: pd.DataFrame,
    metric_ls: List[str],
    group_col: str,
    alpha: float = .05,
    figsize: Tuple[float, float] = (20, 10),
    direction_dict: Optional[Dict[str, str]] = None,
    effect_dict: Optional[Dict[str, float]] = None,
    show_diff: bool = True,
    cell_text_size: int = 16,
    axis_text_size: int = 12,
    title_text_size: int = 16,
    sort_axes: bool = False,
    precision_threshold: float = 0.5,
    metrics_per_row: int = 3,
) -> None:
    """ Create a grid of multiple comparison of means plots using Tukey HSD test results.

    Args:
        df (pd.DataFrame): Input dataframe containing the data.
        metric_ls (List[str]): List of statistical metrics to create plots for.
        group_col (str): The column name indicating the groups.
        alpha (float): Significance level for the Tukey HSD test. Default is 0.05.
        figsize (Tuple[float, float]): Size of the figure. Default is (20, 10).
        direction_dict (Optional[Dict[str, str]]): Dictionary indicating whether to minimize or maximize each metric. Default is None.
        effect_dict (Optional[Dict[str, float]]): Dictionary with effect size limits for each metric. Default is None.
        show_diff (bool): Whether to show the mean differences in the plot. Default is True.
        cell_text_size (int): Font size for the cell text. Default is 16.
        axis_text_size (int): Font size for the axis text. Default is 12.
        title_text_size (int): Font size for the title text. Default is 16.
        sort_axes (bool): Whether to sort the axes. Default is False.
        precision_threshold (float): Precision threshold used in metric display names. Default is 0.5.
        metrics_per_row (int): Number of subplot columns per row. Default is 3.

    Returns:
        None
    """
    metric2name = {
        'mae': 'MAE',
        'mse': 'MSE',
        'rmse':'RMSE',
        'r2': 'R²',
        'rho': "Spearman's ρ",
        'roc_auc': 'ROC-AUC',
        'pr_auc': 'PR-AUC',
        'mcc': 'MCC',
        'prec': 'Precision',
        'recall': f'Recall (Prec. ≥ {precision_threshold})',
        'tnr': f'TNR (Prec. ≥ {precision_threshold})',
    }

    # Avoid mutable default arguments (dicts are shared across calls otherwise).
    direction_dict = dict(direction_dict) if direction_dict else {}
    effect_dict = dict(effect_dict) if effect_dict else {}

    nrow = math.ceil(len(metric_ls) / metrics_per_row)
    _, ax = plt.subplots(nrow, metrics_per_row, figsize=figsize)

    # Set defaults
    for key in ['r2', 'rho', 'prec', 'recall', 'mae', 'mse', 'roc_auc']:
        direction_dict.setdefault(key, 'maximize' if key in ['r2', 'rho', 'prec', 'recall', 'roc_auc'] else 'minimize')

    for key in ['r2', 'rho', 'prec', 'recall', 'roc_auc']:
        effect_dict.setdefault(key, 0.1)

    direction_dict = {k.lower(): v for k, v in direction_dict.items()}
    effect_dict = {k.lower(): v for k, v in effect_dict.items()}

    for i, stat in enumerate(metric_ls):
        stat = stat.lower()

        row = i // int(metrics_per_row)
        col = i % int(metrics_per_row)

        if stat not in direction_dict:
            raise ValueError(f"Stat '{stat}' is missing in direction_dict. Please set its value.")
        if stat not in effect_dict:
            raise ValueError(f"Stat '{stat}' is missing in effect_dict. Please set its value.")

        reverse_cmap = False
        if direction_dict[stat] == 'minimize':
            reverse_cmap = True

        _, df_means, df_means_diff, pc = rm_tukey_hsd(df, stat, group_col, alpha,
                                                       sort_axes, direction_dict)

        hax = mcs_plot(pc, effect_size=df_means_diff, means=df_means[stat],
                       show_diff=show_diff, ax=ax[row, col], cbar=True,
                       cell_text_size=cell_text_size, axis_text_size=axis_text_size,
                       reverse_cmap=reverse_cmap, vlim=effect_dict[stat])
        hax.set_title(metric2name[stat], fontsize=title_text_size, fontweight='bold')

    # If there are less plots than cells in the grid, hide the remaining cells
    if (len(metric_ls) % metrics_per_row) != 0:
        for i in range(len(metric_ls), nrow * metrics_per_row):
            row = i // metrics_per_row
            col = i % metrics_per_row
            ax[row, col].set_visible(False)

    plt.tight_layout()


def make_scatterplot(
    df: pd.DataFrame,
    val_col: str,
    pred_col: str,
    thresh: float,
    cycle_col: str = "cv_cycle",
    group_col: str = "method",
) -> None:
    """ Create scatter plots for each method showing the relationship between predicted and measured values.

    Args:
        df (pd.DataFrame): Input dataframe containing the data.
        val_col (str): The column name for the ground truth values.
        pred_col (str): The column name for the predicted values.
        thresh (float): Threshold for binary classification.
        cycle_col (str): The column name indicating the cross-validation fold. Default is "cv_cycle".
        group_col (str): The column name indicating the groups/methods. Default is "method".

    Returns:
        None
    """
    df_split_metrics = calc_regression_metrics(df, cycle_col=cycle_col, val_col=val_col, pred_col=pred_col,
                                               thresh=thresh)
    methods = sorted(df[group_col].unique())

    _, axs = plt.subplots(nrows=1, ncols=len(methods), figsize=(6 * len(methods), 6))

    for i, (ax, method) in enumerate(zip(axs, methods)):
        df_method = df.query(f"{group_col} == @method")
        df_metrics = df_split_metrics.query(f"{group_col} == @method")
        ax.scatter(df_method[pred_col], df_method[val_col], alpha=0.3)
        ax.plot([df_method[val_col].min(), df_method[val_col].max()],
                [df_method[val_col].min(), df_method[val_col].max()], 'k--', lw=1)

        ax.axhline(y=thresh, color='r', linestyle='--')
        ax.axvline(x=thresh, color='r', linestyle='--')
        ax.set_title(method)

        y_true = df_method[val_col] > thresh
        y_pred = df_method[pred_col] > thresh
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        roc_auc = roc_auc_score(y_true, y_pred)

        metrics_text = (f"MAE: {df_metrics['mae'].mean():.2f}\n" +
                        f"MSE: {df_metrics['mse'].mean():.2f}\n" +
                        f"$R^2$: {df_metrics['r2'].mean():.2f}\n" +
                        f"Spearman's $\\rho$: {df_metrics['rho'].mean():.2f}\n" +
                        f"Precision: {precision:.2f}\n" +
                        f"Recall: {recall:.2f}\n" +
                        f"ROC-AUC: {roc_auc:.2f}")
        ax.text(0.05, .5, metrics_text, transform=ax.transAxes,
                verticalalignment='top', fontsize=12,
                bbox={"boxstyle": 'round', "facecolor": 'white', "alpha": 0.8},
        )

        ax.set_xlabel('Predicted')
        if i == 0:
            ax.set_ylabel('Measured')
        else:
            # Remove y-axis ticks for other plots
            # ax.set_yticks([])
            ax.set_ylabel('')

        ax.grid(axis='both', alpha=0.5)

    plt.tight_layout()


def ci_plot(result_tab: pd.DataFrame, ax_in: Axes, name: str) -> None:
    """ Create a confidence interval plot for the given result table.

    Args:
        result_tab (pd.DataFrame): DataFrame containing the results with columns 'meandiff', 'lower', and 'upper'.
        ax_in (matplotlib.axes.Axes): The axes on which to plot the confidence intervals.
        name (str): The title of the plot.

    Returns:
        None
    """
    result_err = np.array([result_tab['meandiff'] - result_tab['lower'],
                           result_tab['upper'] - result_tab['meandiff']])
    sns.set(rc={'figure.figsize': (6, 2)})
    sns.set_context('notebook')
    sns.set_style('whitegrid')
    ax = sns.pointplot(x=result_tab.meandiff, y=result_tab.index, marker='o', linestyle='', ax=ax_in)
    ax.errorbar(y=result_tab.index, x=result_tab['meandiff'], xerr=result_err, fmt='o', capsize=5)
    ax.axvline(0, ls="--", lw=3)
    ax.set_xlabel("Mean Difference")
    ax.set_ylabel("")
    ax.set_title(name)
    ax.set_xlim(-0.2, 0.2)


def make_ci_plot_grid(df_in: pd.DataFrame, metric_list: List[str], group_col: str = "method") -> None:
    """ Create a grid of confidence interval plots for multiple metrics using Tukey HSD test results.

    Args:
        df_in (pd.DataFrame): Input dataframe containing the data.
        metric_list (List[str]): List of metric column names to create confidence interval plots for.
        group_col (str): The column name indicating the groups. Default is "method".

    Returns:
        None
    """
    figure, axes = plt.subplots(len(metric_list), 1, figsize=(8, 3 * len(metric_list)), sharex=False)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    for i, metric in enumerate(metric_list):
        df_tukey, _, _, _ = rm_tukey_hsd(df_in, metric, group_col=group_col)
        ci_plot(df_tukey, ax_in=axes[i], name=metric)
    figure.suptitle("Multiple Comparison of Means\nTukey HSD, FWER=0.05")
    plt.tight_layout()


def make_curve_plots(
    df_plot: pd.DataFrame,
    val_col: str,
    prob_col: str,
    precision_threshold: float = 0.8,
) -> None:
    """ Create ROC and precision-recall curve plots for each method.

    Args:
        df_plot (pd.DataFrame): Input dataframe containing a 'method' column plus `val_col` and `prob_col`.
        val_col (str): The column name for the ground truth binary label.
        prob_col (str): The column name for the predicted probability/score.
        precision_threshold (float): Precision threshold at which to mark the recall/TNR operating point. Default is 0.8.

    Returns:
        None
    """
    color_map = plt.get_cmap('tab10')
    # Sort df_plot by method names
    df_plot = df_plot.sort_values(by='method')
    le = LabelEncoder()
    df_plot['color'] = le.fit_transform(df_plot['method'])
    colors = color_map(df_plot['color'].unique())

    _, axes = plt.subplots(1, 2, figsize=(12, 8))
    for (k, v), color in zip(df_plot.groupby("method"), colors):
        roc_auc = roc_auc_score(v[val_col], v[prob_col])
        pr_auc = average_precision_score(v[val_col], v[prob_col])
        fpr, recall_pos, thresholds_roc = roc_curve(v[val_col], v[prob_col])
        precision, recall, thresholds_pr = precision_recall_curve(v[val_col], v[prob_col])

        _, threshold_recall_pos = recall_at_precision(v[val_col].astype(bool), v[prob_col], precision_threshold, direction='greater')
        _, threshold_recall_neg = recall_at_precision(~v[val_col].astype(bool), v[prob_col], precision_threshold, direction='lesser')

        fpr_recall_pos = fpr[np.abs(thresholds_roc - threshold_recall_pos).argmin()]
        fpr_recall_neg = fpr[np.abs(thresholds_roc - threshold_recall_neg).argmin()]
        recall_recall_pos = recall[np.abs(thresholds_pr - threshold_recall_pos).argmin()]
        recall_recall_neg = recall[np.abs(thresholds_pr - threshold_recall_neg).argmin()]

        axes[0].plot(fpr, recall_pos, label=f"{k} (ROC AUC={roc_auc:.03f})", color=color, alpha=0.75)
        axes[1].plot(recall, precision, label=f"{k} (PR AUC={pr_auc:.03f})", color=color, alpha=0.75)

        axes[0].axvline(fpr_recall_pos, color=color, linestyle=':', alpha=0.75)
        axes[0].axvline(fpr_recall_neg, color=color, linestyle='--', alpha=0.75)
        axes[1].axvline(recall_recall_pos, color=color, linestyle=':', alpha=0.75)
        axes[1].axvline(recall_recall_neg, color=color, linestyle='--', alpha=0.75)

    axes[0].plot([0, 1], [0, 1], "--", color="black", lw=0.5)
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve")
    # Make legend outside of plot, on the bottom, with smaller font
    axes[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), fontsize='small')


    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curve")
    axes[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), fontsize='small')

    plt.tight_layout()


def run_anova(df_in: pd.DataFrame, col: str, group_var: str = "method") -> float:
    """ Run a one-way ANOVA on `col`, grouped by `group_var`.

    Args:
        df_in (pd.DataFrame): Input dataframe containing the data.
        col (str): The column name to test.
        group_var (str): The column name indicating the groups. Default is "method".

    Returns:
        float: The p-value of the one-way ANOVA.
    """
    res_list = []
    for _, v in df_in.groupby(group_var):
        res_list.append(v[col].values)
    return f_oneway(*res_list)[1]


def make_simultaneous_ci_plot(
    df_in: pd.DataFrame,
    metric_list: List[str],
    group_col: str = "method",
    alpha: float = 0.05,
    direction_dict: Optional[Dict[str, str]] = None,
) -> None:
    """ Create simultaneous confidence interval plots for multiple metrics using Tukey HSD test results.

    Args:
        df_in (pd.DataFrame): Input dataframe containing the data.
        metric_list (List[str]): List of metric column names to create confidence interval plots for.
        group_col (str): The column name indicating the groups. Default is "method".
        alpha (float): Significance level for the Tukey HSD test. Default is 0.05.
        direction_dict (Optional[Dict[str, str]]): Dictionary indicating whether to minimize or maximize each metric, used to pick the reference method per plot. Default is None.

    Returns:
        None
    """
    tuckey_metrics = {}
    for i, metric in enumerate(metric_list):
        # If any NaN values are present, skip plotting
        if df_in[metric].isna().any():
            print(f"WARNING: Metric {metric} contains NaN values, skipping...")
            continue

        tuckey_metric = pairwise_tukeyhsd(endog=df_in[metric],
                                          groups=df_in[group_col],
                                          alpha=alpha)
        # print(tuckey_metric)
        tuckey_metrics[metric] = tuckey_metric

    metric2name = {
        'mae': 'Mean Absolute Error (MAE)',
        'mse': 'Mean Squared Error (MSE)',
        'r2': 'R-squared (R2)',
        'rho': "Spearman's Rho",
        'roc_auc': 'ROC-AUC',
        'pr_auc': 'PR-AUC',
        'mcc': 'Matthews Correlation Coefficient (MCC)',
        'recall': 'Recall (Sensitivity)',
        'tnr': 'True Negative Rate (Specificity)',
    }

    # Change the axes dimensions to be a bigger square
    _, axes = plt.subplots(1, len(tuckey_metrics), figsize=(10 * len(tuckey_metrics), 5), sharey=True)

    for i, (metric, tuckey_metric) in enumerate(tuckey_metrics.items()):
        if direction_dict and metric in direction_dict:
            best_method = df_in.groupby(group_col)[metric].mean().reset_index().sort_values(by=metric, ascending=direction_dict[metric]=='minimize').iloc[0][group_col]
            tuckey_metric.plot_simultaneous(comparison_name=best_method, ax=axes[i], figsize=(20, 5))
            metric_anova = run_anova(df_in, metric, group_var=group_col)
            axes[i].set_xlabel(metric2name.get(metric, metric), fontsize=14)
            axes[i].set_title(f"p = {metric_anova:.2e}", fontsize=14)

            best_method = best_method.replace('\n', ' ')
            print(f"- Best method for {metric}: {best_method}")

    # Change the font size of the y-axis labels
    for ax in axes:
        ax.tick_params(axis='y', labelsize=14)
    plt.suptitle(f"Simultaneous Confidence Intervals\nTukey HSD, FWER={alpha}", fontsize=16)
