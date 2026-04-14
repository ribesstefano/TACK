
import re
import os
import sys
import logging
import warnings
import random
from pathlib import Path
from typing import List, Literal, Tuple, Dict, Optional

import optuna
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from typing import Dict, Union
from scipy.stats import skew
from sklearn.cluster import HDBSCAN
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold, KFold
from sklearn.preprocessing import (
    MultiLabelBinarizer, LabelEncoder, OneHotEncoder, OrdinalEncoder
)
from sklearn.model_selection import RepeatedKFold
import umap.umap_ as umap
from rdkit import RDLogger
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit.DataStructs import ExplicitBitVect
from Bio.Align import PairwiseAligner, substitution_matrices

import matplotlib.pyplot as plt
import seaborn as sns
from datasets import Dataset, DatasetDict, load_dataset
from useful_rdkit_utils.split_utils import (
    taylor_butina_clustering,
    get_bemis_murcko_clusters,
)

from tack_dataset.logging_utils import setup_logging
from tack_dataset.curation_utils import set_global_logging_level
from tack_dataset.protein_utils import generate_normalized_alignment_matrix
from tack_dataset.clustering_utils import (
    cluster_prot_sequences,
    evaluate_clusters,
)


# Filter out annoying Pytorch Lightning printouts
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', '.*Covariance of the parameters could not be estimated.*')
warnings.filterwarnings('ignore', '.*You seem to be using the pipelines sequentially on GPU.*')
# Disable RDKit warnings
RDLogger.DisableLog('rdApp.*')

log_file = setup_logging(
    log_dir=Path('logs'),
    log_base_name='data_splitting',
    # log_file='logs/data_splitting.log',
    verbose=1, # Enable INFO level logging
)
logger = logging.getLogger(__name__)

# Setup working directories:

data_dir = Path(os.path.join(os.getcwd(), '.', 'data'))
data_curation_dir = data_dir / 'curation'
data_tack_dir = data_dir / 'tack'

for d in [data_dir, data_curation_dir, data_tack_dir]:
    if not os.path.exists(d):
        os.makedirs(d)

# ## Load Data

df = pd.read_csv(os.path.join(data_curation_dir, 'protacdb_tpddb_protacpedia_protac_dc50_dmax_activities.csv'))
df

# ## Cluster POI sequences based on NAS matrix

poi_sequences = df['POI_Sequence'].unique().tolist()
seq2name = {seq: df[df['POI_Sequence'] == seq]['POI_Name'].iloc[0] for seq in poi_sequences}

# Print some statistics
logger.info(f"Total unique POI sequences: {len(poi_sequences)}")
for seq, name in list(seq2name.items())[:5]:
    logger.info(f"POI Name: {name}, Sequence Length: {len(seq)}")

plt.figure(figsize=(8, 8))
# Plot the distribution of POI_Name (with a number on the side of each bar) of
# the ones that have more than 50 occurrences
tmp = df['POI_Name'].value_counts()
tmp = tmp[tmp > 50]
sns.barplot(x=tmp.values, y=tmp.index)
plt.title('Distribution of POI Names')
plt.xlabel('Count')
plt.ylabel('POI Name')
# Add the value on the side of each bar
for i, v in enumerate(tmp.values):
    plt.text(v + 1, i, str(v), color='black', va='center', fontsize=6)
plt.grid(axis='x', alpha=0.5)
# Set the y-axis font size
plt.yticks(fontsize=6)
# plt.show()

# Plot the distribution of values that have between 2 and 50 occurrences
plt.figure(figsize=(8, 12))
tmp = df['POI_Name'].value_counts()
tmp = tmp[(tmp >= 10) & (tmp <= 50)]
sns.barplot(x=tmp.values, y=tmp.index)
plt.title('Distribution of POI Names')
plt.xlabel('Count')
plt.ylabel('POI Name')
# Add the value on the side of each bar
for i, v in enumerate(tmp.values):
    plt.text(v + 1, i, str(v), color='black', va='center', fontsize=6)
plt.grid(axis='x', alpha=0.5)
# Set the y-axis font size
plt.yticks(fontsize=6)
# plt.show()

# Print all POI_Names that have more than 100 occurrences, and the ones that have only 1 occurrence
logger.info("POI Names with more than 100 occurrences:")
for name, count in df['POI_Name'].value_counts().items():
    if count > 100:
        logger.info(f"• {name}: {count}")
logger.info("\nPOI Names with only 1 occurrence:")
for name, count in df['POI_Name'].value_counts().items():
    if count == 1:
        logger.info(f"• {name}")

poi_nas_dist_matrix = generate_normalized_alignment_matrix(
    sequences=poi_sequences,
    mode="global",
    gap_open=-10.0,
    gap_extend=-0.5,
    matrix="BLOSUM62",
    clip=(0.0, 1.0),
    return_distance=True,
    eps=1e-12,
    sanitize=True,
)

seq_names = [seq2name[seq] for seq in poi_sequences]

plt.figure(figsize=(16, 14))
sns.heatmap(
    1 - poi_nas_dist_matrix,
    xticklabels=seq_names,
    yticklabels=seq_names,
    cmap="viridis",
    square=True,
    cbar_kws={"label": "Normalized Alignment Score (NAS)"},
    annot=False if len(seq_names) > 20 else True,  # Annotate only if there are few sequences
    fmt=".3f",  # Format for the annotations
)
plt.title("Global Alignment Similarity Matrix")
plt.xlabel("Sequences")
plt.ylabel("Sequences")
plt.yticks(rotation=0)
plt.tight_layout()
# plt.savefig(os.path.join(data_curation_dir, f"{name.lower().replace(' ', '_')}_matrix.png"))

if len(seq_names) > 30:
    # Disable x and y ticks to avoid clutter
    plt.xticks([])
    plt.yticks([])
# plt.show()

# Sort sequences by average (1 - poi_nas_dist_matrix) to print the top-5 most
# different sequences
avg_dists = poi_nas_dist_matrix.mean(axis=1)
sorted_indices = np.argsort(avg_dists)[::-1]  # Descending order
logger.info("Top 10 most different POI sequences based on average distance:")
for idx in sorted_indices[:10]:
    logger.info(f"POI Name: {seq2name[poi_sequences[idx]]}, Average Distance: {avg_dists[idx]:.4f}, Number of samples in dataset: {df[df['POI_Sequence'] == poi_sequences[idx]].shape[0]}")

# Cluster the POI sequences using HDBSCAN with optimized parameters
seq2cluster = cluster_prot_sequences(poi_sequences)

# Add cluster labels to the original dataframe
df['POI_Cluster'] = df['POI_Sequence'].map(seq2cluster)

# # Evaluate clustering results
# report = evaluate_clusters(poi_nas_dist_matrix, labels, metric="precomputed")
# for metric, value in report.items():
#     logger.info(f"{metric}: {value:.4f}" if isinstance(value, float) else f"{metric}: {value}")

# ## Cluster SMILES based on Tanimoto distance matrix
# 
# Use Butina clustering to cluster the SMILES based on the Tanimoto distance matrix.

# ### Get Fingerprints


radius = 16
fp_size = 512

fp_generator = AllChem.GetMorganGenerator(
    radius=radius,
    fpSize=fp_size,
    useBondTypes=True,
    includeChirality=True,
    # countSimulation=True,
    # onlyNonzeroInvariants=True,
    # includeRedundantEnvironments=True,
)

smiles = df['SMILES'].unique().tolist()
smiles2mol = {smi: Chem.MolFromSmiles(smi) for smi in smiles}
smiles2fp = {smi: fp_generator.GetFingerprintAsNumPy(mol) for smi, mol in smiles2mol.items()}

# Isolate all the SMILES that have overlapping fingerprints
overlapping_smiles = []
for i in range(len(smiles)):
    fp_i = smiles2fp[smiles[i]]
    for j in range(i + 1, len(smiles)):
        fp_j = smiles2fp[smiles[j]]
        if np.array_equal(fp_i, fp_j):
            overlapping_smiles.append(smiles[i])
            overlapping_smiles.append(smiles[j])
overlapping_smiles = list(set(overlapping_smiles))

logger.info(f"Number of unique SMILES: {len(smiles):,}")
logger.info(f"Number of SMILES with overlapping fingerprints: {len(overlapping_smiles)} ({len(overlapping_smiles) / len(smiles) * 100:.2f}%)")

def np2bitvect(fp_array: np.ndarray) -> ExplicitBitVect:
    """Convert a numpy array fingerprint to RDKit ExplicitBitVect."""
    bitvect = ExplicitBitVect(len(fp_array))
    for bit_idx in range(len(fp_array)):
        if fp_array[bit_idx] > 0:
            bitvect.SetBit(bit_idx)
    return bitvect

fps = [fp_generator.GetFingerprint(Chem.MolFromSmiles(smi)) for smi in smiles]
# bitvects = [np2bitvect(fp) for fp in fps]
smiles2bitvect = {smi: bv for smi, bv in zip(smiles, fps)}

# Check that the lengths match
# assert len(bitvects) == len(smiles), "Mismatch between number of fingerprints and SMILES"
assert len(fps) == len(smiles), "Mismatch between number of fingerprints and SMILES"

# ### Isolate Held-Out SMILES

def get_avg_dist(fp, fp_list):
    dists = DataStructs.BulkTanimotoSimilarity(fp, fp_list, returnDistance=True)
    return np.mean(dists)

smiles_wo_operator = df[df['Value_Operator'].isna()]['SMILES'].dropna().unique().tolist()

held_out_smiles = []

for task in ['Dmax', 'DC50']:
    subset = df[df['Value_Type'] == task].copy()
    n_held_out = int(0.1 * subset.shape[0])
    
    fps = [smiles2bitvect[smi] for smi in subset['SMILES'].unique().tolist()]
    
    # Assign average distance to each SMILES in the subset
    subset['Avg_Tanimoto_Dist'] = subset['SMILES'].apply(lambda smi: get_avg_dist(smiles2bitvect[smi], fps))
    
    # Identify held-out SMILES so that they cover ~10% of the dataset:
    # - avoid picking SMILES with operators
    # - TODO: bin the labels and pick from each bin to ensure diversity -> for now they just look fine, see the plots below
    subset = subset[subset['SMILES'].isin(smiles_wo_operator)]
    subset = subset.drop_duplicates(subset=['SMILES'])
    subset = subset.sort_values(by='Avg_Tanimoto_Dist', ascending=False)
    held_out_smiles_task = subset['SMILES'].unique().tolist()[:n_held_out]
    held_out_smiles.extend(held_out_smiles_task)
    
    # Remove held-out SMILES that have overlapping fingerprints
    held_out_smiles_task = [smi for smi in held_out_smiles_task if smi not in overlapping_smiles]
    held_out_df = df[df['SMILES'].isin(held_out_smiles_task)]
    
    logger.info(f"Config: {task}")
    logger.info(f"Number of held-out SMILES: {len(held_out_smiles_task)} ({len(held_out_smiles_task) / subset.shape[0] * 100:.2f}%)")
    logger.info(f"Size of held-out set for {task}: {len(held_out_df)} ({len(held_out_df) / len(df[df['Value_Type'] == task]) * 100:.2f}%)")
    logger.info('')

held_out_smiles = list(set(held_out_smiles))
held_out_df = df[df['SMILES'].isin(held_out_smiles)]
logger.info(f"Total number of held-out SMILES across both tasks: {len(held_out_smiles)} ({len(held_out_smiles) / len(df['SMILES'].unique()) * 100:.2f}%)")
logger.info(f"Size of the held-out set across both tasks: {len(held_out_df)} ({len(held_out_df) / len(df) * 100:.2f}%)")

df['SMILES_Held_Out'] = df['SMILES'].apply(lambda smi: smi in held_out_smiles)

# Plot the distribution of 'Value' and 'Value_Type' == Dmax and DC50, for held-out SMILES
for task in ['Dmax', 'DC50']:
    # Isolate the data for the current task and label held-out vs training+validation
    tmp = df[df['Value_Type'] == task].copy()
    tmp['Dataset Split'] = tmp['SMILES'].apply(lambda smi: 'Held-Out' if smi in held_out_smiles else 'Training+Validation')
    
    # Convert 'Value' to p-log10 scale if task is DC50 (for better visualization)
    if task == 'DC50':
        tmp['Value'] = -np.log10(tmp['Value'] + 1e-12)
    
    plt.figure(figsize=(8, 6))
    sns.histplot(
        data=tmp,
        x='Value',
        hue='Dataset Split',
        bins=30,
        kde=True,
        element='step',
        stat='density',
    )
    plt.title(f'Distribution of {task} Values for Held-Out SMILES')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.grid(axis='y', alpha=0.5)
    # plt.show()

# Print the Database count of the held-out SMILES
logger.info('')
logger.info(f"Total number of entries in the dataset for held-out SMILES: {held_out_df.shape[0]}")
logger.info(held_out_df.groupby('Database').size())

# ### Butina and Scaffold Cluster the Data

smiles_to_cluster = [s for s in smiles if s not in held_out_smiles]

scaffold_clusters = get_bemis_murcko_clusters(smiles_to_cluster)
clusters_metrics = evaluate_clusters(
    X=np.array([smiles2fp[smi] for smi in smiles_to_cluster]),
    clusters=scaffold_clusters,
    metric='jaccard',
)
for metric, value in clusters_metrics.items():
    logger.info(f"{metric}: {value:.4f}" if isinstance(value, float) else f"{metric}: {value}")

# Store results for different cutoffs
cutoff_range = np.arange(0.3, 0.71, 0.01)
results = []

X = np.array([smiles2fp[smi] for smi in smiles_to_cluster])

for cutoff in tqdm(cutoff_range, desc="Getting and evaluating clusters at different cutoffs"):
    clusters = taylor_butina_clustering(
        [smiles2bitvect[smi] for smi in smiles_to_cluster],
        cutoff=cutoff,
    )
    
    # Skip if only one cluster (not meaningful)
    if len(set(clusters)) < 2:
        continue
    
    stats = evaluate_clusters(X, np.array(clusters), metric="jaccard")
    
    results.append({
        'cutoff': cutoff,
        'silhouette': stats['silhouette'],
        'avg_cluster_size': stats['avg_cluster_size'],
        'avg_cluster_data_ratio': stats['avg_cluster_data_ratio'],
        'std_cluster_size': stats['std_cluster_size'],
        'min_cluster_size': stats['min_cluster_size'],
        'median_cluster_size': stats['median_cluster_size'],
        'max_cluster_size': stats['max_cluster_size'],
        'cluster_size_skewness': stats['cluster_size_skewness'],
        'num_clusters': stats['num_clusters']
    })

# Convert to DataFrame for easier plotting
results_df = pd.DataFrame(results)

logger.info(results_df.round(2))

# Print metrics at a specific cutoff, e.g., 0.48
best_cutoff = 0.5
logger.info(f"Metrics at cutoff {best_cutoff}:")

butina_clusters = taylor_butina_clustering(
    [smiles2bitvect[smi] for smi in smiles_to_cluster],
    cutoff=best_cutoff,
)
stats = evaluate_clusters(
    X=np.array([smiles2fp[smi] for smi in smiles_to_cluster]),
    clusters=butina_clusters,
    metric='jaccard',
)
for metric, value in stats.items():
    logger.info(f"{metric}: {value:.4f}" if isinstance(value, float) else f"{metric}: {value}")

# Visualize held-out SMILES in UMAP embedding
held_out_indices = [i for i, smi in enumerate(smiles) if smi in held_out_smiles]

smiles2scaffold = {s: c for s, c in zip(smiles_to_cluster, scaffold_clusters)}
smiles2butina = {s: c for s, c in zip(smiles_to_cluster, butina_clusters)}

# Canonicalize SMILES in the original dataframe before mapping
df['SMILES'] = df['SMILES'].apply(lambda smi: Chem.MolToSmiles(Chem.MolFromSmiles(smi)))

# Map cluster labels back to the original dataframe
df['SMILES_Scaffold_Cluster'] = df['SMILES'].map(smiles2scaffold).fillna(-1).astype(int)
df['SMILES_Butina_Cluster'] = df['SMILES'].map(smiles2butina).fillna(-1).astype(int)

# Assign held-out SMILES as a separate column
df['SMILES_Held_Out'] = df['SMILES'].apply(lambda smi: smi in held_out_smiles)

# Check that all SMILES with cluster -1 are marked as held-out
unclustered_smiles = df[df['SMILES_Scaffold_Cluster'] == -1]['SMILES'].unique().tolist()
unclustered_smiles += df[df['SMILES_Butina_Cluster'] == -1]['SMILES'].unique().tolist()
unclustered_smiles = list(set(unclustered_smiles))
for smi in unclustered_smiles:
    assert smi in held_out_smiles, f"SMILES {smi} has cluster -1 but is not marked as held-out."

# Print unique clusters found
logger.info(f"Unique Bemis-Murcko clusters found: {df['SMILES_Scaffold_Cluster'].unique()}")
logger.info(f"Unique Butina clusters found: {df['SMILES_Butina_Cluster'].unique()}")

# The following is another method to obtain held-out data based on clustering the SMILES.
# 
# We decided to instead isolate the held-out data first, based on their Tanimoto distances to the rest of the data, and then cluster the remaining data. Because of this, the following code is no longer used in the final analysis, but is kept here for reference.

# from collections import defaultdict
# from rdkit import DataStructs

# held_out_clusters = defaultdict(list)

# # For each cluster type, get the clusters with most dissimilar SMILES based on
# # Tanimoto distance
# for cluster_type in ['SMILES_Scaffold_Cluster', 'SMILES_Butina_Cluster']:
#     logger.info(f"\nAnalyzing clusters for: {cluster_type}")
#     unique_clusters = df[cluster_type].unique()
#     most_dissimilar_cluster = None
#     max_distance = -1.0
#     cluster_distances = []

#     for cluster_id in unique_clusters:
#         # Get SMILES in the current cluster and outside the cluster
#         cluster_smiles = df[df[cluster_type] == cluster_id]['SMILES'].unique().tolist()
#         non_cluster_smiles = df[df[cluster_type] != cluster_id]['SMILES'].unique().tolist()
        
#         # Ignore clusters that have SMILES with overlapping fingerprints
#         if any(smi in overlapping_smiles for smi in cluster_smiles):
#             continue
        
#         # Turn SMILES into fingerprints (bit vectors)
#         cluster_fps = [smiles2bitvect[smi] for smi in cluster_smiles]
#         non_cluster_fps = [smiles2bitvect[smi] for smi in non_cluster_smiles]
#         # Compute pairwise Tanimoto distances using BulkTanimotoSimilarity
#         mean_dist = 0.0
#         for fp in cluster_fps:
#             dists = DataStructs.BulkTanimotoSimilarity(fp, non_cluster_fps, returnDistance=True)
#             # Get the maximum distance found
#             mean_dist_in_cluster = np.mean(dists)
#             mean_dist += mean_dist_in_cluster
#             if mean_dist_in_cluster > max_distance:
#                 max_distance = mean_dist_in_cluster
#                 most_dissimilar_cluster = cluster_id
#         mean_dist /= len(cluster_fps)
#         cluster_distances.append((cluster_id, mean_dist))

#     logger.info(f"Cluster ID with most dissimilar SMILES: {most_dissimilar_cluster} (Mean Tanimoto Distance: {max_distance:.4f})")

#     # Sort and print the top-N most dissimilar clusters
#     cluster_distances.sort(key=lambda x: x[1], reverse=True)
#     logger.info("Top 3 most dissimilar clusters based on mean Tanimoto distance:")
#     num_included_clusters = 0
#     total_size = 0
#     total_Dmax_size = 0
#     total_DC50_size = 0
#     for cid, dist in cluster_distances:
#         logger.info(f"• Cluster ID: {cid}, Mean Tanimoto Distance: {dist:.4f} (len={len(df[df[cluster_type] == cid])})")
#         total_size += len(df[df[cluster_type] == cid])
#         total_Dmax_size += len(df[(df[cluster_type] == cid) & (df['Value_Type'] == 'Dmax')])
#         total_DC50_size += len(df[(df[cluster_type] == cid) & (df['Value_Type'] == 'DC50')])
#         # if total_Dmax_size > len(df) * 0.05 and total_DC50_size > len(df) * 0.05:
#         if total_size > len(df) * 0.1:
#             break
#         num_included_clusters += 1
#         held_out_clusters[cluster_type].append(cid)

#     logger.info(f"Total number of samples in top-{num_included_clusters} dissimilar clusters: {total_size}")
# # UMAP plot all the clusters, highlighting the held-out clusters
# for cluster_type in ['SMILES_Scaffold_Cluster', 'SMILES_Butina_Cluster']:
#     held_out = held_out_clusters[cluster_type]
#     colors = ['red' if cid in held_out else 'blue' for cid in df[cluster_type]]
#     fig = go.Figure()
#     fig.add_trace(
#         go.Scatter(
#             x=embedding[:, 0],
#             y=embedding[:, 1],
#             mode='markers',
#             marker=dict(color=colors, showscale=False),
#             text=[f"SMILES: {smi}, Cluster: {cid}" for smi, cid in zip(df['SMILES'], df[cluster_type])],
#             name=f'{cluster_type} with Held-Out Highlighted'
#         )
#     )
#     # Update layout
#     fig.update_layout(
#         title=f"UMAP Projection of Molecule Clusters ({cluster_type})",
#         xaxis_title="UMAP 1",
#         yaxis_title="UMAP 2",
#         template="plotly_white",
#         hovermode="closest",
#     )
#     fig.show()
# # Add a held-out flag to the dataframe
# for cluster_type in ['SMILES_Scaffold_Cluster', 'SMILES_Butina_Cluster']:
#     held_out = held_out_clusters[cluster_type]
#     df[f'{cluster_type}_Held_Out'] = df[cluster_type].apply(lambda cid: cid in held_out)

# for config in ['Dmax', 'DC50']:
#     tmp = df[df['Value_Type'].str.contains(config)]
#     # Print the held-out size and total size
#     for cluster_type in ['SMILES_Scaffold_Cluster', 'SMILES_Butina_Cluster']:
#         held_out_size = tmp[tmp[f'{cluster_type}_Held_Out']].shape[0]
#         total_size = tmp.shape[0]
#         logger.info(f"{config} - {cluster_type}: Held-out size: {held_out_size}, Total size: {total_size} ({held_out_size / total_size * 100:.2f}%)")

# ### Plot 5x5 CV Folds


def create_cv_splits(
    ds: Union[Dataset, pd.DataFrame],
    n_splits: int = 5,
    n_repeats: int = 5,
    group_col: Optional[str] = None,
    base_seed: int = 42,
):
    """Create repeated k-fold cross-validation splits.
    
    Args:
        ds: Dataset or DataFrame to split.
        n_splits: Number of folds.
        n_repeats: Number of repeats.
        group_col: Column name for grouping (if None, uses standard K-Fold).
        base_seed: Base random seed.
        
    Yields:
        Dictionary with repeat, cv_fold, fold, train_idx, test_idx.
    """
    if group_col is None:
        # Get a dummy feature matrix X
        X = np.zeros((len(ds), 1))
        kf = RepeatedKFold(
            n_splits=n_splits,
            n_repeats=n_repeats,
            random_state=base_seed,
        )
        for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
            yield {
                "repeat": fold // n_splits,
                "cv_fold": fold % n_splits,
                "fold": fold,
                "train_idx": train_idx,
                "test_idx": test_idx,
            }
    else:
        # Get the feature matrix X from the group_col
        # TODO: Most likely, having a dummy X is sufficient, but to be safe,
        # we will encode all other columns as ordinal features
        groups = np.array(ds[group_col])
        X = ds.drop(columns=[group_col]).values if isinstance(ds, pd.DataFrame) else ds.remove_columns([group_col]).to_pandas().values
        X = OrdinalEncoder().fit_transform(X)
        for repeat in range(n_repeats):
            kf = GroupKFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=base_seed + repeat,
            )
            for fold, (train_idx, test_idx) in enumerate(kf.split(X, groups=groups)):
                yield {
                    "repeat": repeat,
                    "cv_fold": fold,
                    "fold": repeat * n_splits + fold,
                    "train_idx": train_idx,
                    "test_idx": test_idx,
                }

def print_split_stats(ds, splits):
    """ Print statistics about the provided splits, including sizes, and
    verify no group leakage and no repeating folds with same data.
    
    Args:
        ds (Dataset): The dataset being split.
        splits (List[dict]): The list of split dictionaries containing    
    """
    # Verify split sizes and stratification
    logger.info("\nSplit statistics:")
    curr_repeat = -1
    for split in splits:  # Show first repeat only
        train_ds = ds.select(split["train_idx"])
        test_ds = ds.select(split["test_idx"])
        
        train_size = len(train_ds)
        test_size = len(test_ds)
        train_pct = train_size / len(ds) * 100
        test_pct = test_size / len(ds) * 100

        if curr_repeat == -1:
            curr_repeat = split['repeat']
        if split['repeat'] != curr_repeat:
            curr_repeat = split['repeat']
            logger.info('')
        
        logger.info(f"  Repeat {split['repeat']} Fold {split['fold']}: "
                f"Train={train_size:4d} ({train_pct:.1f}%), "
                f"Test={test_size:4d} ({test_pct:.1f}%)")
        

    # Verify no group leakage
    logger.info("\n✅ Verifying no group leakage...")
    for split in splits:
        train_groups = set(ds.select(split["train_idx"])["POI_Cluster"])
        test_groups = set(ds.select(split["test_idx"])["POI_Cluster"])
        overlap = train_groups & test_groups
        if overlap:
            logger.info(f"  ⚠️  WARNING: Repeat {split['repeat']} Fold {split['fold']} has group leakage!")
        else:
            logger.info(f"  ✓ Repeat {split['repeat']} Fold {split['fold']}: No group leakage")

    # Check that there are no repeating folds with the same data in them
    logger.info("\n✅ Verifying no repeating folds with same data...")
    seen_folds = {}
    num_overlaps = 0
    for split in splits:
        train_idx_tuple = tuple(sorted(split["train_idx"]))
        test_idx_tuple = tuple(sorted(split["test_idx"]))
        fold_key = (train_idx_tuple, test_idx_tuple)
        
        if fold_key in seen_folds:
            prev_repeat, prev_fold = seen_folds[fold_key]
            logger.info(f"  ⚠️  WARNING: Repeat {split['repeat']} Fold {split['fold']} is identical to "
                f"Repeat {prev_repeat} Fold {prev_fold}!")
            num_overlaps += 1
        else:
            seen_folds[fold_key] = (split['repeat'], split['fold'])
            
    if num_overlaps == 0:
        logger.info(f"🚀 All folds are unique!\n")

    return num_overlaps

def plot_folds_distribution(df, cv_splits, title="Fold Distribution", held_out_df=None, plot_held_out=True):
    """ Plot the distribution of the 'Value' column in each fold as boxplots. 
    
    Args:
        df (pd.DataFrame): The dataframe containing the data.
        cv_splits (List[dict]): The list of cross-validation splits.
        title (str): The title of the plot.
        held_out_df (Optional[pd.DataFrame]): Dataframe of held-out set to overlay.
    """
    # Build a long dataframe: one row per test sample per split
    records = []
    for s in cv_splits:
        rep = s["repeat"]
        fold = s["fold"]
        for idx in s["test_idx"]:
            records.append({
                "(Replicate, Fold)": f"({rep}, {fold})",
                "Value": df.loc[idx, "Value"]
            })

    plot_df = pd.DataFrame(records)

    plt.figure(figsize=(16, 6))
    sns.boxplot(
        data=plot_df,
        x="(Replicate, Fold)",
        y="Value",
        color="steelblue",
        fliersize=2,
        linewidth=1
    )
    
    if held_out_df is not None:
        # # Overlay held-out set distribution
        # sns.scatterplot(
        #     x=[-0.4 + i for i in range(len(cv_splits))],
        #     y=[held_out_df["Value"].median()] * len(cv_splits),
        #     color="red",
        #     s=100,
        #     label="Held-out Set Median"
        # )
        # Plot the held-out set median as a horizontal line instead
        plt.axhline(y=held_out_df["Value"].median(), color='red', linestyle='--', label='Held-out Set Median')
        plt.legend(loc='upper right')
        plt.grid(axis='y', alpha=0.5)

    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.title(title)
    # plt.show()
    
    # Plot the 'Value' distribution in the held-out set as an histogram
    if held_out_df is not None and plot_held_out:
        plt.figure(figsize=(6, 4))
        sns.histplot(
            data=held_out_df,
            x="Value",
            # bins=20,
            color="orange",
            kde=True,
        )
        plt.grid(axis='y', alpha=0.5)
        plt.title("Held-out Set Value Distribution")
        plt.tight_layout()
        # plt.show()

for task in ['Dmax', 'DC50']:
    # Isolate task dataframe and remove held-out samples
    subset_df = df[df['Value_Type'] == task].reset_index(drop=True).copy()
    held_out_df = subset_df[subset_df['SMILES_Held_Out']].reset_index(drop=True)
    held_out_df = held_out_df[held_out_df['Value_Type'] == task]
    subset_df = subset_df[~subset_df['SMILES_Held_Out']].reset_index(drop=True)
    
    if task == 'DC50':
        # Convert 'Value' to p-log10 scale
        subset_df['Value'] = -np.log10(subset_df['Value'] + 1e-12)
        held_out_df['Value'] = -np.log10(held_out_df['Value'] + 1e-12)
    
    logger.info(f"\nCreating CV splits for task: {task}, Total samples: {len(subset_df)}")
    
    for clustering in [None, 'SMILES_Scaffold_Cluster', 'SMILES_Butina_Cluster', 'POI_Cluster']:
        n_clusters = subset_df[clustering].nunique() if clustering is not None else 'N/A'
        logger.info('-' * 80)
        logger.info(f"Using clustering: {clustering}, Number of unique clusters: {n_clusters}")
        logger.info('-' * 80)
        cv_splits = list(create_cv_splits(
            ds=subset_df,
            n_splits=5,
            n_repeats=5,
            group_col=clustering,
            base_seed=1234,
        ))
        
        ds = Dataset.from_pandas(subset_df, preserve_index=False)
        num_overlaps = print_split_stats(ds, cv_splits)
        
        clustering_name = clustering if clustering is not None else "Random Splitting"
        clustering_name = clustering_name.replace('SMILES_', '').replace('_', ' ').replace('Cluster', 'Clustering')
        
        plot_folds_distribution(
            df=subset_df,
            cv_splits=cv_splits,
            title=f"Fold Distribution for {task} and {clustering_name}",
            held_out_df=held_out_df[held_out_df['Value_Type'] == task],
            plot_held_out=True if clustering is None else False,
        )

# ## Push to Hugging Face

dmax_df = df[df['Value_Type'] == 'Dmax']
dc50_df = df[df['Value_Type'] == 'DC50']

df.to_csv(data_tack_dir / 'protacdb_tpddb_protacpedia_protac_dc50_dmax_activities_processed.csv', index=False)
dmax_df.to_csv(data_tack_dir / 'protacdb_tpddb_protacpedia_protac_dmax_activities_processed.csv', index=False)
dc50_df.to_csv(data_tack_dir / 'protacdb_tpddb_protacpedia_protac_dc50_activities_processed.csv', index=False)

ds = Dataset.from_pandas(df, preserve_index=False)
dmax_ds = Dataset.from_pandas(dmax_df, preserve_index=False)
dc50_ds = Dataset.from_pandas(dc50_df, preserve_index=False)

# Push to HuggingFace Hub (privately)
try:
    for config, dataset in [
        ('default', ds),
        ('Dmax', dmax_ds),
        ('DC50', dc50_ds),
    ]:
        dataset.push_to_hub(
            "ailab-bio/TACK",
            config_name=config,
            private=True,
        )
        logger.info('Dataset pushed to Hugging Face Hub')
        logger.info(dataset)
        # Log the number of held-out samples in the dataset card
        num_held_out = dataset.filter(lambda x: x['SMILES_Held_Out'])['SMILES_Held_Out'].count()
        total_samples = len(dataset)
        logger.info(f"Number of held-out samples in {config} dataset: {num_held_out} ({num_held_out / total_samples * 100:.2f}%)")
except Exception as e:
    logger.info(f"Error pushing to Hugging Face Hub: {e}")

# ## Multitask Assembly and Clustering

# Prepare Keys for Merging
# We create a temporary column to handle NaNs in 'Assay_Time' so that 
# 'Unspecified' (NaN) matches 'Unspecified' (NaN).
# Pandas default merge does not match NaN == NaN.
dc50_df['Assay_Time_Filled'] = dc50_df['Assay_Time'].fillna(-1)
dmax_df['Assay_Time_Filled'] = dmax_df['Assay_Time'].fillna(-1)

# Fill NaN assay names with "Unknown" (placeholder)
dc50_df['Assay_Filled'] = dc50_df['Assay'].fillna("Unknown")
dmax_df['Assay_Filled'] = dmax_df['Assay'].fillna("Unknown")

# Define the list of keys to merge on
keys = ['SMILES', 'POI_Name', 'Cell_Line', 'Ligase_Name', 'Reference', 'Assay_Time_Filled', 'Assay_Filled']

# Rename Value Columns
# This prevents collision and clearly labels which value is DC50 vs Dmax
value_cols = ['Value', 'Value_Type', 'Value_Unit', 'Value_Operator', 'Value_Category',
              'Value_Range_Min', 'Value_Range_Max', 'Value_Error', 'Value_Concentration',
              'Value_Concentration_Unit', 'TPD_ID']

dc50_renamed = dc50_df.rename(columns={c: f"{c}_DC50" for c in value_cols})
dmax_renamed = dmax_df.rename(columns={c: f"{c}_Dmax" for c in value_cols})

# Perform the Merge
# Inner join ensures we only keep rows that exist in BOTH files with matching keys
multitask_df = pd.merge(dc50_renamed, dmax_renamed, on=keys, how='inner', suffixes=('', '_y'))

# Clean Up Metadata
# Consolidate duplicate metadata columns (like POI_Sequence, Description, etc.)
y_cols = [c for c in multitask_df.columns if c.endswith('_y')]

for y_col in y_cols:
    base_col = y_col[:-2]  # Remove '_y' suffix
    if base_col in multitask_df.columns:
        # Fill missing info in the main column with data from the secondary (_y) column
        multitask_df[base_col] = multitask_df[base_col].fillna(multitask_df[y_col])

# Drop the temporary key and the redundant _y columns
multitask_df_clean = multitask_df.drop(columns=y_cols + ['Assay_Time_Filled', 'Assay_Filled'])
multitask_df = multitask_df_clean.dropna(subset=['Value_DC50', 'Value_Dmax'], how='any')

# Log the number of held-out samples in the multitask dataset
num_held_out_multitask = multitask_df[multitask_df['SMILES_Held_Out']].shape[0]
total_multitask_samples = multitask_df.shape[0]
logger.info(f"Number of held-out samples in multitask dataset: {num_held_out_multitask} ({num_held_out_multitask / total_multitask_samples * 100:.2f}%)")

# Save Output
multitask_df.to_csv(data_tack_dir / "protacdb_tpddb_protacpedia_protac_multitask_activities_processed.csv", index=False)

def convert_to_plog10(value):
    """ Convert a value to p-log10 scale. """
    return -np.log10(value + 1e-12)

def plot_folds_distribution_multitask(df, cv_splits, title="Fold Distribution", held_out_df=None, plot_held_out=True):
    """ Plot the distribution of the 'Value' column in each fold as boxplots. 
    
    Args:
        df (pd.DataFrame): The dataframe containing the data.
        cv_splits (List[dict]): The list of cross-validation splits.
        title (str): The title of the plot.
        held_out_df (Optional[pd.DataFrame]): Dataframe of held-out set to overlay.
    """
    # Build a long dataframe: one row per test sample per split
    records = []
    for s in cv_splits:
        rep = s["repeat"]
        fold = s["fold"]
        for idx in s["test_idx"]:
            records.append({
                "(Replicate, Fold)": f"({rep}, {fold})",
                "Value_Dmax": df.loc[idx, "Value_Dmax"] / 10,
                "Value_DC50": convert_to_plog10(df.loc[idx, "Value_DC50"]),
            })

    plot_df = pd.DataFrame(records)
    
    # Melt the dataframe to have a single 'Value' column and a 'Task' column
    plot_df = plot_df.melt(
        id_vars="(Replicate, Fold)",
        value_vars=["Value_Dmax", "Value_DC50"],
        var_name="Task",
        value_name="Value"
    )

    plt.figure(figsize=(16, 6))
    sns.boxplot(
        data=plot_df,
        x="(Replicate, Fold)",
        y="Value",
        color="steelblue",
        hue="Task",
        fliersize=2,
        linewidth=1,
        palette={"Value_Dmax": "C1", "Value_DC50": "C0"}
    )
    
    # Get the median of the Dmax and DC50 values, then plot as horizontal lines
    if held_out_df is not None:
        dmax_median = held_out_df[held_out_df["Value_Type"] == "Dmax"]["Value"].median() / 10
        dc50_median = convert_to_plog10(held_out_df[held_out_df["Value_Type"] == "DC50"]["Value"].median())
        # Plot horizontal lines for medians
        plt.axhline(y=dmax_median, color='red', linestyle='--', label='Held-out Set Dmax Median')
        plt.axhline(y=dc50_median, color='green', linestyle='--', label='Held-out Set DC50 Median')
        plt.legend(loc='upper right')
        plt.grid(axis='y', alpha=0.5)


for clustering in [None, 'SMILES_Scaffold_Cluster', 'SMILES_Butina_Cluster', 'POI_Cluster']:
    n_clusters = multitask_df[clustering].nunique() if clustering is not None else 'N/A'
    logger.info('-' * 80)
    logger.info(f"Using clustering: {clustering}, Number of unique clusters: {n_clusters}")
    logger.info('-' * 80)
    cv_splits = list(create_cv_splits(
        ds=multitask_df,
        n_splits=5,
        n_repeats=5,
        group_col=clustering,
        base_seed=1234,
    ))
    
    ds = Dataset.from_pandas(multitask_df, preserve_index=False)
    num_overlaps = print_split_stats(ds, cv_splits)
    
    clustering_name = clustering if clustering is not None else "Random Splitting"
    clustering_name = clustering_name.replace('SMILES_', '').replace('_', ' ').replace('Cluster', 'Clustering')
    
    plot_folds_distribution_multitask(
        df=multitask_df,
        cv_splits=cv_splits,
        title=f"Fold Distribution for {task} and {clustering_name}",
        held_out_df=held_out_df,
        plot_held_out=True if clustering is None else False,
    )
    # plt.show()

ds = Dataset.from_pandas(multitask_df, preserve_index=False)

try:
    ds.push_to_hub(
        "ailab-bio/TACK",
        # "ailab-bio/PROTAC-Degradation-Predictor-Dataset",
        config_name="multitask",
        private=True,
    )
except Exception as e:
    logger.info(f"Error pushing multitask dataset to Hugging Face Hub: {e}")
logger.info(ds)

filenames = {
    'all data': data_tack_dir / 'protacdb_tpddb_protacpedia_protac_dc50_dmax_activities_processed.csv',
    'Dmax only': data_tack_dir / 'protacdb_tpddb_protacpedia_protac_dmax_activities_processed.csv',
    'DC50 only': data_tack_dir / 'protacdb_tpddb_protacpedia_protac_dc50_activities_processed.csv',
    'multitak data': data_tack_dir / 'protacdb_tpddb_protacpedia_protac_multitask_activities_processed.csv',
}
logger.info('=' * 80)
logger.info("Final TACK Dataset Files and Sizes:")
logger.info('=' * 80)
for name, path in filenames.items():
    logger.info(f"{name}: {path}, Size: {path.stat().st_size / 1e6:.2f} MB")

