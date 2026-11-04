"""
TACK Dataset - Publication-Ready Figures

Generates two publication-ready figures:
  Figure 1 — pDC50 and Dmax distributions (training / held-out, 2x2 grid)
  Figure 2 — Active/inactive stacked bars for POI, E3 ligase, and cell line

Activity definition: DC50 < 100 nM AND Dmax > 80 % (both required).
pDC50 = -log10(DC50 in M) = 9 - log10(DC50_nM)
"""
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import seaborn as sns
from scipy import stats
from datasets import load_dataset
from dotenv import load_dotenv

warnings.filterwarnings('ignore')
load_dotenv()

# Set HuggingFace cache directory from environment variable (if set)
HF_HOME = os.getenv('HF_HOME')
if HF_HOME:
    os.environ['HF_HOME'] = HF_HOME
    print(f"Using HuggingFace cache directory: {HF_HOME}")
else:
    print("No HuggingFace cache directory set (HF_HOME). Using default cache location.")

# ============================================================================
# CONFIGURATION
# ============================================================================

OUTDIR = Path("plots")
OUTDIR.mkdir(exist_ok=True)
OUTDIR = Path("plots") / "data"
OUTDIR.mkdir(exist_ok=True)

DPI = 300
FONT_FAMILY = 'sans-serif'
TITLE_FONTSIZE = 14
LABEL_FONTSIZE = 12
TICK_FONTSIZE = 10
LEGEND_FONTSIZE = 10

FIG_DOUBLE_COL = 7.0

COLORS = {
    'blue':        '#4B9ECE',
    'orange':      '#FFAA6E',
    'light_blue':  '#50B1D8',
    'dark_orange': '#FF8428',
    'green':       '#9DCE9C',
    'purple':      '#C8ABDA',
}

# Activity thresholds
DC50_THRESH = 100.0   # nM
DMAX_THRESH = 80.0    # %

plt.rcParams.update({
    'font.family':        FONT_FAMILY,
    'font.size':          TICK_FONTSIZE,
    'axes.titlesize':     TITLE_FONTSIZE,
    'axes.labelsize':     LABEL_FONTSIZE,
    'xtick.labelsize':    TICK_FONTSIZE,
    'ytick.labelsize':    TICK_FONTSIZE,
    'legend.fontsize':    LEGEND_FONTSIZE,
    'figure.dpi':         100,
    'savefig.dpi':        DPI,
    'savefig.bbox':       'tight',
    'savefig.pad_inches': 0.1,
})

# ============================================================================
# DATA LOADING
# ============================================================================

def load_tack_data() -> dict:
    """Load TACK dataset from HuggingFace (multitask, DC50, Dmax configs)."""
    print("Loading TACK dataset from HuggingFace...")
    data = {}
    for config in ('default', 'multitask', 'DC50', 'Dmax'):
        print(f"  Loading config: {config}")
        ds = load_dataset("ailab-bio/TACK", config)
        df = ds['train'].to_pandas()
        df['source_config'] = config
        data[config] = df
        print(f"    Loaded {len(df):,} rows")
    return data


# ============================================================================
# FIGURE 1: pDC50 and Dmax distributions (2x2 grid)
# ============================================================================

def to_pdc50(dc50_nM: pd.Series) -> pd.Series:
    """Convert DC50 (nM) to pDC50 = -log10(DC50 in M) = 9 - log10(DC50_nM)."""
    positive = dc50_nM[dc50_nM > 0]
    return 9.0 - np.log10(positive)


def darken_color(color: str, factor: float = 0.80) -> tuple:
    """Return a darker version of *color* by scaling its RGB components down."""
    r, g, b = mcolors.to_rgb(color)
    return (r * factor, g * factor, b * factor)


def create_figure1_distributions(data: dict) -> plt.Figure:
    """
    2x2 grid of histograms: pDC50 and Dmax for training and held-out sets.

    Row a) = Training, row b) = Held-out.
    For Dmax, values outside [0, 100] are clipped to the boundary bins and
    drawn fully opaque to flag out-of-range observations.
    """
    print("\n" + "=" * 60)
    print("FIGURE 1: pDC50 and Dmax Value Distributions (2x2 grid)")
    print("=" * 60)

    dc50_df = data['DC50'].copy()
    dmax_df = data['Dmax'].copy()

    is_heldout_dc50 = dc50_df['SMILES_Held_Out'] == True
    is_heldout_dmax = dmax_df['SMILES_Held_Out'] == True

    dc50_train_vals = pd.to_numeric(dc50_df.loc[~is_heldout_dc50, 'Value'], errors='coerce').dropna()
    dc50_held_vals = pd.to_numeric(dc50_df.loc[ is_heldout_dc50, 'Value'], errors='coerce').dropna()
    dmax_train_vals = pd.to_numeric(dmax_df.loc[~is_heldout_dmax, 'Value'], errors='coerce').dropna()
    dmax_held_vals = pd.to_numeric(dmax_df.loc[ is_heldout_dmax, 'Value'], errors='coerce').dropna()

    pdc50_train = to_pdc50(dc50_train_vals)
    pdc50_held = to_pdc50(dc50_held_vals)

    print(f"Training:  pDC50={len(pdc50_train):,}  Dmax={len(dmax_train_vals):,}")
    print(f"Held-out:  pDC50={len(pdc50_held):,}   Dmax={len(dmax_held_vals):,}")

    fig, axes = plt.subplots(2, 2, figsize=(FIG_DOUBLE_COL, 7.0))

    pdc50_bins = np.linspace(4, 12, 20)
    dmax_bins = np.linspace(0, 100, 15)

    def plot_histogram(
        ax: plt.Axes,
        values: pd.Series,
        color: str,
        xlabel: str,
        bins: np.ndarray,
        xlim: tuple,
        ylim: tuple = None,
        yticks: np.ndarray = None,
        show_overflow: bool = False,
    ) -> None:
        """
        Histogram with a darker KDE line on top.

        When show_overflow=True, values outside xlim are clipped to the boundary
        bins and drawn with alpha=1.0 to visually flag out-of-range counts.
        """
        kde_color = darken_color(color)

        if show_overflow:
            in_range = values[(values >= xlim[0]) & (values <= xlim[1])]
            out_of_range = values[(values < xlim[0]) | (values > xlim[1])]
            ax.hist(in_range, bins=bins,
                    color=color, edgecolor='white', zorder=3)
            if len(out_of_range) > 0:
                ax.hist(out_of_range,
                        color=color, edgecolor='white', alpha=0.7, zorder=2)
            kde_vals = in_range
        else:
            ax.hist(values, bins=bins, range=xlim,
                    color=color, edgecolor='white', zorder=2)
            kde_vals = values

        if len(kde_vals) > 10:
            kde = stats.gaussian_kde(kde_vals)
            x = np.linspace(xlim[0], xlim[1], 200)
            ax.plot(x, kde(x) * len(kde_vals) * (bins[1] - bins[0]),
                    color=kde_color, linewidth=3, zorder=5)

        ax.set_xlabel(xlabel, fontweight='bold')
        ax.set_ylabel('Count', fontweight='bold')
        ax.set_xlim(min(values), max(values))
        if ylim is not None:
            ax.set_ylim(ylim)
        if yticks is not None:
            ax.set_yticks(yticks)
        ax.grid(axis='y', alpha=0.3, linestyle='-', linewidth=0.5)

    pdc50_label = r'$pDC_{50}$ $(-\log_{10}(M))$'
    dmax_label = r'$D_{max}$ $(\%)$'

    train_ylim = (0, 700)
    heldout_ylim = (0, 60)
    train_yticks = np.arange(-10, 701, 120)
    heldout_yticks = np.arange(-1.0, 61, 10)

    plot_histogram(
        ax=axes[0, 0],
        values=pdc50_train,
        color=COLORS['blue'],
        xlabel=pdc50_label,
        bins=pdc50_bins,
        xlim=(min(pdc50_train), max(pdc50_train)), # (4, 12),
        ylim=train_ylim,
        yticks=train_yticks,
    )
    plot_histogram(
        ax=axes[0, 1],
        values=dmax_train_vals,
        color=COLORS['dark_orange'],
        xlabel=dmax_label,
        bins=dmax_bins,
        xlim=(0, 100),
        ylim=train_ylim,
        yticks=train_yticks,
        show_overflow=True,
    )
    plot_histogram(
        ax=axes[1, 0],
        values=pdc50_held,
        color=COLORS['blue'],
        xlabel=pdc50_label,
        bins=pdc50_bins,
        xlim=(min(pdc50_held), max(pdc50_held)), # (4, 12),
        ylim=heldout_ylim,
        yticks=heldout_yticks,
    )
    plot_histogram(
        ax=axes[1, 1],
        values=dmax_held_vals,
        color=COLORS['dark_orange'],
        xlabel=dmax_label,
        bins=dmax_bins,
        xlim=(0, 100),
        ylim=heldout_ylim,
        yticks=heldout_yticks,
        show_overflow=True,
    )

    # Row section titles
    fig.text(0.5, 0.98, 'Training', ha='center', va='top',    fontweight='bold', fontsize=LABEL_FONTSIZE)
    fig.text(0.5, 0.48, 'Held-out', ha='center', va='bottom', fontweight='bold', fontsize=LABEL_FONTSIZE)

    # Panel labels a) and b) in bold at top-left of each row
    for ax, label, pos in zip([axes[0, 0], axes[1, 0]], ['a)', 'b)'], [(-0.2, 1.12), (-0.2, 1.22)]):
        ax.text(*pos, label, transform=ax.transAxes,
                fontweight='bold', fontsize=LABEL_FONTSIZE * 1.5, va='top', ha='left',
                clip_on=False)

    plt.tight_layout()
    plt.subplots_adjust(top=0.93, hspace=0.50)

    for fmt in ('pdf', 'png', 'svg'):
        fig.savefig(OUTDIR / f"figure1_distributions.{fmt}", format=fmt)
    print(f"Saved: {OUTDIR}/figure1_distributions.[pdf|png|svg]")

    plt.close(fig)
    return fig


# ============================================================================
# FIGURE 2: Active/inactive distributions for POI, E3, Cell Line
# ============================================================================

def get_active_inactive_label(data: dict) -> pd.DataFrame:
    """
    Classify each unique PROTAC (by SMILES + context) as active or inactive.

    Active requires BOTH: DC50 < 100 nM AND Dmax > 80 %.
    PROTACs with only one measurement are classified as inactive.
    """
    key_cols = ['SMILES', 'POI_Name', 'Ligase_Name', 'Cell_Line']
    
    df = data['multitask']
    is_active = (df['Value_DC50'] < DC50_THRESH) & (df['Value_Dmax'] > DMAX_THRESH)
    df['is_active'] = is_active.astype(float)

    return df[key_cols + ['is_active']]


def create_figure2_distributions(data: dict) -> plt.Figure:
    """
    Three vertically stacked bar charts showing active/inactive breakdown
    for the top-8 POIs, E3 ligases, and cell lines.

    Activity: DC50 < 100 nM AND Dmax > 80 % (both required).
    Panel label c) appears at the top-left of the first (POI) chart.
    """
    print("\n" + "=" * 60)
    print("FIGURE 2: Active/Inactive Distribution by POI, E3, and Cell Line")
    print("=" * 60)

    combined_df = get_active_inactive_label(data)

    total_entries = len(combined_df)
    active_count = (combined_df['is_active'] == 1).sum()
    inactive_count = (combined_df['is_active'] == 0).sum()
    print(f"Total: {total_entries:,}  Active: {active_count:,}  Inactive: {inactive_count:,}")

    def calc_activity_stats(df: pd.DataFrame, group_col: str, top_n: int) -> pd.DataFrame:
        top_groups = df[group_col].value_counts().head(top_n).index.tolist()
        rows = []
        for group in top_groups:
            subset = df[df[group_col] == group]
            total = len(subset)
            active = int((subset['is_active'] == 1).sum())
            inactive = int((subset['is_active'] == 0).sum())
            rows.append({
                'name':         group,
                'total':        total,
                'active':       active,
                'inactive':     inactive,
                'active_pct':   active / total * 100 if total > 0 else 0.0,
                'inactive_pct': inactive / total * 100 if total > 0 else 0.0,
                'total_pct':    total / len(df) * 100,
            })
        return pd.DataFrame(rows)

    TOP_N = 8
    poi_stats = calc_activity_stats(combined_df, 'POI_Name', TOP_N)
    e3_stats = calc_activity_stats(combined_df, 'Ligase_Name', TOP_N)
    cell_stats = calc_activity_stats(combined_df, 'Cell_Line', TOP_N)

    fig, axes = plt.subplots(3, 1, figsize=(FIG_DOUBLE_COL, 7.0))

    COLOR_INACTIVE = COLORS['blue']
    COLOR_ACTIVE = COLORS['dark_orange']

    def plot_stacked_bar(
        ax: plt.Axes,
        stats_df: pd.DataFrame,
        xlabel: str,
        total_n: int,
    ) -> None:
        names = stats_df['name'].tolist()
        totals = stats_df['total'].tolist()
        actives = stats_df['active'].tolist()
        inacts = stats_df['inactive'].tolist()

        x = np.arange(len(names))
        width = 0.6

        inact_h = [v / total_n * 100 for v in inacts]
        act_h = [v / total_n * 100 for v in actives]
        total_h = [v / total_n * 100 for v in totals]
        inact_w = [iv / t * 100 if t > 0 else 0 for iv, t in zip(inacts,  totals)]
        act_w = [av / t * 100 if t > 0 else 0 for av, t in zip(actives, totals)]

        ax.bar(x, inact_h, width, label='Inactive (%)',
               color=COLOR_INACTIVE, edgecolor='white', linewidth=0.5)
        ax.bar(x, act_h, width, bottom=inact_h,
               label='Active (%)', color=COLOR_ACTIVE, edgecolor='white', linewidth=0.5)

        INNER_FS = 8.5 # Font size inside the bars
        MIN_H = 3  # minimum segment height (% of total) to fit a label inside

        for i in range(len(names)):
            ih, ah, iw, aw, th = inact_h[i], act_h[i], inact_w[i], act_w[i], total_h[i]

            if ih > MIN_H:
                ax.text(i, ih / 2, f'{iw:.1f}%', ha='center', va='center',
                        fontsize=INNER_FS, color='white')
            if ah > MIN_H:
                ax.text(i, ih + ah / 2, f'{aw:.1f}%', ha='center', va='center',
                        fontsize=INNER_FS, color='white')
            elif ih < MIN_H:
                pass
            elif th > 1 and ih <= MIN_H:
                # Bar too small for either segment label — show the dominant percentage
                pct = aw if aw >= iw else iw
                ax.text(i, ih + ah / 2, f'{pct:.1f}%', ha='center', va='center',
                        fontsize=INNER_FS, color='white')

            ax.text(i, th + 0.3, f'{th:.1f}%', ha='center', va='bottom', fontsize=INNER_FS)

        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=0, ha='center', fontsize=TICK_FONTSIZE)
        ax.set_ylabel('Percentage', fontweight='bold')
        ax.set_xlabel(xlabel, fontweight='bold', fontsize=LABEL_FONTSIZE)
        ax.set_ylim(0, max(total_h) * 1.12)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', alpha=0.3, linestyle='-', linewidth=0.5)
        ax.set_axisbelow(True)

    plot_stacked_bar(axes[0], poi_stats, 'Protein of Interest', total_entries)
    plot_stacked_bar(axes[1], e3_stats, 'E3 Ligase', total_entries)
    plot_stacked_bar(axes[2], cell_stats, 'Cell Line', total_entries)

    # Panel label c) at top-left of the first (POI) subplot
    axes[0].text(-0.10, 1.10, 'c)', transform=axes[0].transAxes,
                 fontweight='bold', fontsize=LABEL_FONTSIZE * 1.5, va='top', ha='left')

    handles = [
        mpatches.Patch(facecolor=COLOR_INACTIVE, edgecolor='none', label='Inactive (%)'),
        mpatches.Patch(facecolor=COLOR_ACTIVE,   edgecolor='none', label='Active (%)'),
    ]
    for ax in axes:
        legend = ax.legend(handles=handles, loc='upper right', frameon=True,
                           fancybox=False, fontsize=LEGEND_FONTSIZE, framealpha=0.5)
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_edgecolor('#cccccc')
        legend.get_frame().set_linewidth(0.5)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.45)

    for fmt in ('pdf', 'png', 'svg'):
        fig.savefig(OUTDIR / f"figure2_distributions.{fmt}", format=fmt)
    print(f"Saved: {OUTDIR}/figure2_distributions.[pdf|png|svg]")

    plt.close(fig)
    return fig

def print_statistics(data: dict) -> None:
    """Print basic statistics about the dataset."""    
    df = data['default']
    print("=" * 20)
    print("Dataset Statistics:")
    print("=" * 20)
    print(f"Total records: {len(df):,}")
    print(f"Unique PROTACs: {df['SMILES'].nunique():,}")
    print(f"Degradation endpoints: {len(df[df['Value'].notna()]):,}")
    print(f"POIs: {df['POI_Name'].nunique():,}")
    print(f"E3 Ligases: {df['Ligase_Name'].nunique():,}")
    print(f"Cell Lines: {df['Cell_Line'].nunique():,}")
    print(f"Held-out: {len(df[df['SMILES_Held_Out']]):,}")
    print()
    for db in ['TPDdb', 'PROTAC-DB', 'PROTACpedia']:
        db_df = df[df['SMILES_Held_Out'] & (df['Database'].str.contains(db))].copy()
        print(f"({db}) POIs: {db_df['POI_Name'].nunique():,}")
        print(f"({db}) E3 Ligases: {db_df['Ligase_Name'].nunique():,}")
        print(f"({db}) Cell Lines: {db_df['Cell_Line'].nunique():,}")
        print(f"({db}) Held-out: {len(db_df):,}")
    print()
    num_dc50 = len(df[df['Value_Type'] == 'DC50'])
    num_dmax = len(df[df['Value_Type'] == 'Dmax'])
    print(f"DC50 records: {num_dc50:,}")
    print(f"Dmax records: {num_dmax:,}")
    
    df = data['multitask']
    active = df[(df['Value_DC50'] < DC50_THRESH) & (df['Value_Dmax'] > DMAX_THRESH)]
    print(f"Number of data points with both DC50 and Dmax: {len(df[(df['Value_DC50'].notna()) & (df['Value_Dmax'].notna())]):,} (perc: {(len(df[(df['Value_DC50'].notna()) & (df['Value_Dmax'].notna())]) / len(data['default']) * 100):.1f}%)")
    print(f"Active PROTACs (DC50 < {DC50_THRESH} nM AND Dmax > {DMAX_THRESH} %): {len(active):,} (perc: {(len(active) / len(df) * 100):.1f}%)")
    print(f"Number of unique active PROTACs: {active['SMILES'].nunique():,}")

# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    print("=" * 60)
    print("TACK Dataset — Publication Figure Generation")
    print("=" * 60)
    print(f"\nOutput directory: {OUTDIR.absolute()}")

    data = load_tack_data()
    
    print_statistics(data)
    create_figure1_distributions(data)
    create_figure2_distributions(data)

    print("\n" + "=" * 60)
    print("All figures generated successfully!")
    print("=" * 60)
    print(f"\nOutput files in: {OUTDIR.absolute()}")
    for f in sorted(OUTDIR.glob("*")):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()
