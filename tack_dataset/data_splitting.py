# %% [markdown]
# # Data Splitting and Clustering

# %% [markdown]
# ## Setup

# %%
import re
import os
import sys
import logging
import warnings
import random
from pathlib import Path
from typing import List, Literal, Tuple, Dict, Optional
from copy import deepcopy

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
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score, calinski_harabasz_score,
)
import umap.umap_ as umap
from rdkit import RDLogger
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit.DataStructs import ExplicitBitVect
from Bio.Align import PairwiseAligner, substitution_matrices
from Bio.Phylo.TreeConstruction import DistanceMatrix, DistanceTreeConstructor
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import Dataset, DatasetDict, load_dataset
from useful_rdkit_utils.split_utils import (
    taylor_butina_clustering,
    get_bemis_murcko_clusters,
)

from tack_dataset.logging_utils import setup_logging

# %% [markdown]
# Filter out some warnings...

# %%
def set_global_logging_level(level=logging.ERROR, prefices=[""]):
    """
    Override logging levels of different modules based on their name as a prefix.
    It needs to be invoked after the modules have been loaded so that their loggers have been initialized.

    Args:
        - level: desired level. e.g. logging.INFO. Optional. Default is logging.ERROR
        - prefices: list of one or more str prefices to match (e.g. ["transformers", "torch"]). Optional.
          Default is `[""]` to match all active loggers.
          The match is a case-sensitive `module_name.startswith(prefix)`
    """
    prefix_re = re.compile(fr'^(?:{ "|".join(prefices) })')
    for name in logging.root.manager.loggerDict:
        if re.match(prefix_re, name):
            logging.getLogger(name).setLevel(level)


# Filter out annoying Pytorch Lightning printouts
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', '.*Covariance of the parameters could not be estimated.*')
warnings.filterwarnings('ignore', '.*You seem to be using the pipelines sequentially on GPU.*')
# Disable RDKit warnings
RDLogger.DisableLog('rdApp.*')

log_file = setup_logging(
    log_dir=Path('logs'),
    log_base_name='data_splitting',
    verbose=1, # Enable INFO level logging
)
logger = logging.getLogger(__name__)

# %% [markdown]
# Setup working directories:

# %%
data_dir = Path(os.path.join(os.getcwd(), '.', 'data'))
data_curation_dir = data_dir / 'curation'
data_tack_dir = data_dir / 'tack'

for d in [data_dir, data_curation_dir, data_tack_dir]:
    if not os.path.exists(d):
        os.makedirs(d)

# %% [markdown]
# ## Clustering Utilities
# 
# The followings are metrics to evaluate the quality of the clustering (used in Optuna optimization, for example):

# %%
def evaluate_clusters(
        X: np.ndarray,
        clusters,
        metric: str = 'euclidean',
) -> Dict[str, Union[float, int]]:
    """ Compute clustering metrics and assess cluster size distribution.
    
    Args:
        X (np.ndarray): The input data as a 2D array of shape (n_samples, n_features).
        clusters: An array-like structure containing cluster labels for each sample in X.
        metric (str): The distance metric to use for clustering. Default is 'euclidean'. If 'precomputed', X should be a distance matrix of shape (n_samples, n_samples). If 'precomputed', the davis_bouldin and calinski_harabasz scores will not be computed.
    
    Returns:
        Dict[str, Union[float, int]]: A dictionary containing various clustering metrics:
            - silhouette: Silhouette score of the clustering.
            - davies_bouldin: Davies-Bouldin index of the clustering.
            - calinski_harabasz: Calinski-Harabasz index of the clustering.
            - avg_cluster_size: Average size of the clusters.
            - avg_cluster_data_ratio: Average size of the clusters relative to the total number of samples.
            - std_cluster_size: Standard deviation of the cluster sizes.
            - min_cluster_size: Minimum size of the clusters.
            - median_cluster_size: Median size of the clusters.
            - max_cluster_size: Maximum size of the clusters.
            - cluster_size_skewness: Skewness of the cluster sizes, indicating imbalance.
            - num_clusters: Number of unique clusters.
    """
    
    unique_clusters = list(set(clusters))
    
    if len(unique_clusters) < 2:  # Avoid single-cluster issues
        return {
            "silhouette": -1,
            "davies_bouldin": float("inf"),
            "calinski_harabasz": -1,
            "avg_cluster_size": len(X),
            "avg_cluster_data_ratio": 1,
            "std_cluster_size": 0,
            "min_cluster_size": len(X),
            "median_cluster_size": len(X),
            "max_cluster_size": len(X),
            "cluster_size_skewness": 0,
            "num_clusters": 1,
        }

    # Compute standard clustering metrics
    silhouette = silhouette_score(X, clusters, metric=metric)
    if metric == 'precomputed':
        # If the metric is precomputed, we cannot compute Davies-Bouldin and
        # Calinski-Harabasz scores
        davies_bouldin = float("inf")
        calinski_harabasz = -1
    else:
        davies_bouldin = davies_bouldin_score(X, clusters)
        calinski_harabasz = calinski_harabasz_score(X, clusters)

    # Compute cluster size statistics
    cluster_sizes = [len(np.where(clusters == i)[0]) for i in np.unique(clusters)]
    avg_cluster_size = np.mean(cluster_sizes)
    avg_cluster_data_ratio = avg_cluster_size / len(X)
    std_cluster_size = np.std(cluster_sizes)
    median_cluster_size = np.median(cluster_sizes)
    min_cluster_size = np.min(cluster_sizes)
    max_cluster_size = np.max(cluster_sizes)
    cluster_size_skewness = skew(cluster_sizes, nan_policy="omit")  # Indicates imbalance in cluster sizes

    return {
        "silhouette": silhouette,
        "davies_bouldin": davies_bouldin,
        "calinski_harabasz": calinski_harabasz,
        "avg_cluster_size": avg_cluster_size,
        "avg_cluster_data_ratio": avg_cluster_data_ratio,
        "std_cluster_size": std_cluster_size,
        "min_cluster_size": min_cluster_size,
        "median_cluster_size": median_cluster_size,
        "max_cluster_size": max_cluster_size,
        "cluster_size_skewness": cluster_size_skewness,
        "num_clusters": len(unique_clusters),
    }

# %%
def sequence_clustering_objective(
        trial: optuna.Trial,
        nas_matrix: np.ndarray,
        is_distance: bool = True,
) -> float:
    """ Objective function for Optuna to optimize HDBSCAN clustering parameters based on silhouette score.
    
    Args:
        trial (optuna.Trial): An Optuna trial object for suggesting hyperparameters.
        nas_matrix (np.ndarray): A 2D numpy array representing the NAS similarity distance matrix (Normalized Alignment Matrix).
        is_distance (bool): Whether the provided nas_matrix is a distance matrix. If False, it is treated as a similarity matrix. Default is True.
        
    Returns:
        float: The silhouette score of the clustering. Returns -1.0 if clustering is invalid.
    """
    # NOTE: The NAS matrix is symmetric and square with shape (n_sequences, n_sequences)
    n_sequences = nas_matrix.shape[0]
    
    dist_matrix = 1.0 - nas_matrix if not is_distance else nas_matrix

    # Setup HDBSCAN parameters to optimize
    min_cluster_size = trial.suggest_int('min_cluster_size', 2, int(n_sequences * 0.2))
    min_samples = trial.suggest_int('min_samples', 1, int(n_sequences * 0.2))
    cluster_selection_epsilon = trial.suggest_float('cluster_selection_epsilon', 0.0, 0.9)
    alpha = trial.suggest_float('alpha', 1.0, 2.0)
    leaf_size = trial.suggest_int('leaf_size', 10, 50)
    cluster_selection_method = trial.suggest_categorical('cluster_selection_method', ['eom', 'leaf'])

    # Perform HDBSCAN clustering
    labels = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=cluster_selection_epsilon,
        metric='precomputed',
        alpha=alpha,
        leaf_size=leaf_size,
        cluster_selection_method=cluster_selection_method,
        allow_single_cluster=False,
    ).fit_predict(dist_matrix)

    # Only compute silhouette if more than 1 cluster and at least 2 samples per
    # cluster, otherwise return -1.0
    if len(set(labels)) < 2 or (labels >= 0).sum() < 2:
        return -1.0

    return silhouette_score(dist_matrix, labels, metric='precomputed')


# # Example of usage
# sampler = optuna.samplers.TPESampler(seed=42)
# study = optuna.create_study(direction='maximize', sampler=sampler)
# study.optimize(
#     lambda trial: sequence_clustering_objective(trial, nas_matrix=np.random.rand(100, 100)),
#     n_trials=300,
# )

# logger.info("Best silhouette score:", study.best_value)
# logger.info("Best parameters:", study.best_params)

# %% [markdown]
# ## Protein Clustering Utilities

# %%
def _sanitize_protein(
    seq: str,
    allowed: str,
    *,
    strip_gaps: bool = True,
    strip_whitespace: bool = True,
    drop_stops: bool = True,
    map_rare_to_x: bool = True,
    rare_map: dict = None,
) -> str:
    """Sanitize a protein sequence to be compatible with substitution matrix alphabet.
    
    This function cleans and standardizes protein sequences by removing unwanted
    characters, mapping rare amino acids, and ensuring compatibility with
    substitution matrices used in sequence alignment.
    
    Args:
        seq: Input protein sequence string to sanitize
        allowed: String containing allowed amino acid characters (e.g., from substitution matrix alphabet)
        strip_gaps: If True, remove gap characters ('-') from sequence
        strip_whitespace: If True, remove all whitespace characters
        drop_stops: If True, remove stop codon symbols ('*')
        map_rare_to_x: If True, map any character not in allowed alphabet to 'X'
        rare_map: Optional dictionary for explicit character mapping before fallback-to-X
                 (e.g., {'U': 'X', 'O': 'X'} for selenocysteine and pyrrolysine)
    
    Returns:
        Sanitized protein sequence string compatible with the substitution matrix
        
    Raises:
        ValueError: If map_rare_to_x=False and sequence contains characters not in allowed alphabet
        
    Example:
        >>> allowed = "ARNDCQEGHILKMFPSTWYVBZX*"
        >>> _sanitize_protein("MET-LYS U", allowed, rare_map={'U': 'X'})
        'METKX'
    """
    # Handle None input gracefully
    if seq is None:
        return ""
    
    # Convert to uppercase string for standardization
    s = str(seq).upper()
    
    # Remove whitespace characters (spaces, tabs, newlines)
    if strip_whitespace:
        s = re.sub(r"\s+", "", s)
    
    # Remove alignment gap characters
    if strip_gaps:
        s = s.replace("-", "")
    
    # Remove stop codon symbols (conservative approach for scoring)
    if drop_stops:
        s = s.replace("*", "")
    
    # Apply explicit character mappings for rare amino acids
    # This allows controlled mapping before the general fallback-to-X
    if rare_map:
        for bad, good in rare_map.items():
            s = s.replace(bad, good)
    
    # Handle characters not in the allowed alphabet
    if map_rare_to_x:
        # Replace any remaining non-standard characters with 'X' (unknown amino acid)
        s = "".join(ch if ch in allowed else "X" for ch in s)
    else:
        # Strict mode: raise error if any disallowed characters remain
        bad = {ch for ch in set(s) if ch not in allowed}
        if bad:
            raise ValueError(f"Sequence contains unsupported letters: {sorted(bad)}")
    
    return s

def generate_normalized_alignment_matrix(
    sequences: List[str],
    *,
    mode: Literal["global", "local"] = "local",
    gap_open: float = -10.0,
    gap_extend: float = -0.5,
    matrix: str = "BLOSUM62",
    clip: Tuple[float, float] = (0.0, 1.0),
    return_distance: bool = False,
    eps: float = 1e-12,
    sanitize: bool = True,
) -> np.ndarray:
    """Generate a Normalized Alignment Score (NAS) matrix for protein sequences.

    This function computes pairwise sequence similarity using normalized alignment scores,
    where NAS(i,j) = S(i,j) / sqrt(S(i,i) * S(j,j)). This normalization makes the scores
    comparable across sequences of different lengths and compositions.

    Args:
        sequences: List of protein sequences to compare
        mode: Alignment mode - "local" (Smith-Waterman) for finding best matching regions,
              or "global" (Needleman-Wunsch) for end-to-end alignment
        gap_open: Penalty for opening a gap in the alignment (negative value)
        gap_extend: Penalty for extending an existing gap (negative value, less severe than gap_open)
        matrix: Name of substitution matrix to use (e.g., "BLOSUM62", "PAM250")
        clip: Tuple of (min, max) values to clip the normalized scores to prevent extreme values
        return_distance: If True, return distance matrix (1 - NAS) instead of similarity matrix
        eps: Small epsilon value to prevent division by zero in normalization
        sanitize: If True, clean sequences using _sanitize_protein function

    Returns:
        Symmetric matrix of normalized alignment scores (or distances if return_distance=True).
        Shape: (n_sequences, n_sequences), dtype: float32
        - Diagonal elements are 1.0 (perfect self-similarity)
        - Off-diagonal elements range from clip[0] to clip[1]

    Notes:
        - Uses BioPython's PairwiseAligner for sequence alignment scoring
        - Self-scores S(i,i) are computed first to enable normalization
        - Empty sequences receive zero self-score to avoid numerical issues
        - Normalization formula: NAS(i,j) = S(i,j) / sqrt(S(i,i) * S(j,j) + eps)

    Example:
        >>> seqs = ["MKVLWAALLVTFLAGCQAKVEQAVETEPEPELRQQTEWQSGQRWELALGRFWDYLRWVQTLSEQVQEELLSSQVTQELRALMDETAQ"]
        >>> matrix = generate_normalized_alignment_matrix(seqs, mode="global")
        >>> logger.info(matrix.shape)  # (1, 1)
        >>> logger.info(matrix[0, 0])  # 1.0 (perfect self-similarity)
    """
    n = len(sequences)
    
    # Handle edge case: empty input
    if n == 0:
        return np.zeros((0, 0), dtype=np.float32)

    # Load substitution matrix and get allowed amino acid alphabet
    subs = substitution_matrices.load(matrix)
    allowed = subs.alphabet  # e.g., 'ARNDCQEGHILKMFPSTWYVBZX*' for BLOSUM62
    
    # Define mapping for rare/non-standard amino acids before fallback to 'X'
    # U=Selenocysteine, O=Pyrrolysine, J=Leucine/Isoleucine ambiguity
    rare_map = {"U": "X", "O": "X", "J": "X"}

    # Sanitize all sequences for consistent processing
    seqs = []
    for s in sequences:
        if sanitize:
            # Clean sequence: remove gaps, whitespace, stops; map rare AAs to X
            s = _sanitize_protein(
                s, allowed,
                strip_gaps=True, 
                strip_whitespace=True, 
                drop_stops=True,
                map_rare_to_x=True, 
                rare_map=rare_map
            )
        else:
            # Use sequence as-is, but handle None values
            s = s or ""
        seqs.append(s)

    # Configure pairwise sequence aligner
    aligner = PairwiseAligner()
    aligner.substitution_matrix = subs
    aligner.mode = mode  # "local" (Smith–Waterman) or "global" (Needleman–Wunsch)
    aligner.open_gap_score = gap_open    # Penalty for starting a gap
    aligner.extend_gap_score = gap_extend # Penalty for extending a gap

    # Step 1: Compute self-alignment scores for normalization
    # S(i,i) represents the maximum possible score for sequence i
    self_scores = np.empty(n, dtype=np.float64)
    for i, s in tqdm(enumerate(seqs), total=n, desc="Computing self-alignment scores"):
        if s:  # Non-empty sequence
            # Self-alignment score, clipped to non-negative to handle gap penalties
            self_scores[i] = max(aligner.score(s, s), 0.0)
        else:  # Empty sequence
            self_scores[i] = 0.0

    # Step 2: Compute pairwise Normalized Alignment Scores
    # Initialize with identity matrix (diagonal = 1.0 for perfect self-similarity)
    nas = np.eye(n, dtype=np.float64)
    
    # Fill upper triangle, then mirror to lower triangle for symmetry
    for i in tqdm(range(n), total=n, desc="Computing pairwise NAS"):
        si = seqs[i]
        for j in range(i + 1, n):
            sj = seqs[j]
            
            # Compute raw alignment score S(i,j)
            if si and sj:  # Both sequences non-empty
                sij = aligner.score(si, sj)
            else:  # At least one sequence is empty
                sij = 0.0
            
            # Normalize: NAS(i,j) = S(i,j) / sqrt(S(i,i) * S(j,j))
            # Add epsilon to denominator to prevent division by zero
            denom = (self_scores[i] * self_scores[j]) ** 0.5 + eps
            val = sij / denom
            
            # Apply clipping to prevent extreme values
            if clip is not None:
                lo, hi = clip
                if val < lo: 
                    val = lo
                elif val > hi: 
                    val = hi
            
            # Fill both symmetric positions
            nas[i, j] = nas[j, i] = val

    # Return distance matrix (1 - similarity) or similarity matrix
    if return_distance:
        return (1.0 - nas).astype(np.float32)
    else:
        return nas.astype(np.float32)

# %% [markdown]
# Another approach for clustering would be to build a NJ (Neighbor-Joining) tree from the NAS matrix and then use a clustering algorithm on the tree. This would allow for hierarchical clustering based on the evolutionary relationships between sequences.
# 
# Since the proteins are assumed to be quite diverse, we will use HDBSCAN on the NAS matrix instead.

# %%
# --------------------------
# 2) Build NJ tree from NAS
# --------------------------
def nas_to_biopython_distance_matrix(nas: np.ndarray, names: List[str]) -> DistanceMatrix:
    """Biopython's NJ expects a DistanceMatrix (lower triangular incl. 0 diagonal)."""
    assert nas.shape[0] == nas.shape[1] == len(names)
    dist = 1.0 - nas  # convert similarity -> distance in [0,1]
    # Biopython DistanceMatrix takes only lower triangle (row i has i+1 elements)
    matrix = [[0.0]]
    for i in range(1, len(names)):
        row = [float(dist[i, j]) for j in range(i + 1)]
        matrix.append(row)
    return DistanceMatrix(names, matrix)

def neighbor_joining_tree_from_nas(nas: np.ndarray, names: List[str]):
    """Construct an NJ tree whose patristic distances approximate the input distances."""
    dm = nas_to_biopython_distance_matrix(nas, names)
    constructor = DistanceTreeConstructor()
    tree = constructor.nj(dm)  # Neighbor-Joining per Saitou & Nei (1987)
    return tree

# --------------------------
# 3) Convert NJ tree -> flat clusters by cutting long branches
#    (simple, transparent rule: cut any branch with length > threshold)
# --------------------------
def clusters_from_tree_cut(tree, max_branch_len: float, names: Optional[List[str]] = None) -> Dict[int, List[str]]:
    """
    Cut every edge longer than `max_branch_len`. Connected components of the remaining
    graph define clusters; leaves within each component are a cluster.
    """
    # Work on a copy
    t = deepcopy(tree)

    # Break long edges by detaching their child clade
    for clade in list(t.find_clades(order="preorder")):
        if clade.branch_length is not None and clade.branch_length > max_branch_len:
            parent = t.get_path(clade)[-2] if len(t.get_path(clade)) >= 2 else None
            if parent is not None:
                # detach: replace parent's clades with all except this child
                parent.clades = [c for c in parent.clades if c is not clade]

    # After cuts, collect connected leaf sets by traversing from each remaining top-level clade
    clusters = {}
    cid = 0
    for clade in t.root.clades:
        leaves = [leaf.name for leaf in clade.get_terminals()]
        if leaves:
            clusters[cid] = leaves
            cid += 1

    # Handle the degenerate case where all leaves remain directly attached to root
    if not clusters:
        leaves = [leaf.name for leaf in t.get_terminals()]
        clusters[0] = leaves
    return clusters

# --------------------------
# 4) Optional: turn cluster dict -> label vector aligned to `names`
# --------------------------
def labels_from_clusters_dict(clusters: Dict[int, List[str]], names: List[str]) -> np.ndarray:
    name_to_cluster = {name: -1 for name in names}
    for cid, leaf_names in clusters.items():
        for nm in leaf_names:
            if nm in name_to_cluster:
                name_to_cluster[nm] = cid
    return np.array([name_to_cluster[nm] for nm in names], dtype=int)

# ------------------------------------------------------------------------------
# Custom NJ clustering
# ------------------------------------------------------------------------------
# # 2) Build Neighbor-Joining tree from NAS
# tree = neighbor_joining_tree_from_nas(nas_matrix, seq_names)
# logger.info(f"NJ tree has {len(tree.get_terminals())} leaves and {len(tree.get_nonterminals())} internal nodes.")

# # Visualize (ASCII and/or matplotlib)
# Phylo.draw_ascii(tree)        # quick console view

# # ax = plt.gca()  # Get current axes for matplotlib
# # # Change the size of the figure if needed
# # ax.figure.set_size_inches(8, 12)  # Optional: adjust figure size
# # # Set xlimits if needed
# # ax.set_xlim(0, 1.0)  # Optional: adjust x-axis
# # Phylo.draw(tree, axes=ax)            # matplotlib figure (optional)

# # 3) Cut the NJ tree into flat clusters, then evaluate
# #    Choose a branch-length threshold; start coarse (e.g., 0.2) and tune.
# clusters_dict = clusters_from_tree_cut(tree, max_branch_len=0.20)
# labels = labels_from_clusters_dict(clusters_dict, seq_names)
# logger.info(f"Found {len(clusters_dict)} clusters with max branch length 0.20")

# # 4) Evaluate with your function: pass a **distance** matrix (1-NAS) and metric='precomputed'
# dist_matrix = 1.0 - nas_matrix
# report = evaluate_clusters(dist_matrix, labels, metric='precomputed')
# for metric, value in report.items():
#     logger.info(f"{metric}: {value:.4f}" if isinstance(value, float) else f"{metric}: {value}")

# # Inspect clusters
# for cid, members in clusters_dict.items():
#     logger.info(f"[Cluster {cid}]  n={len(members)}  -> {members}")

# # ------------------------------------------------------------------------------
# # SciKit-Bio based clustering
# # ------------------------------------------------------------------------------
# from skbio import DistanceMatrix
# from skbio.tree import nj

# dist = 1.0 - e3_nas_local_matrix  # Convert similarity to distance
# dm = DistanceMatrix(dist, list(e3_seq2name.values()))
# tree = nj(dm)
# logger.info(tree.ascii_art())

# # ------------------------------------------------------------------------------
# # SciPy based clustering
# # ------------------------------------------------------------------------------
# from scipy.spatial.distance import squareform
# from scipy.cluster.hierarchy import linkage, fcluster

# # 1) Similarity -> distance in [0,1]
# dist = 1.0 - nas_matrix
# condensed = squareform(dist, checks=False)  # SciPy needs condensed form

# # 2) Hierarchical clustering (average linkage is a good default for sequence distances)
# Z = linkage(condensed, method='average')

# # 3) Flat clusters by distance threshold (tune 't' to your scale; e.g., t=0.3)
# labels = fcluster(Z, t=0.30, criterion='distance')

# # 4) Evaluate with your function, which supports 'precomputed' distances
# report = evaluate_clusters(dist, labels, metric='precomputed')
# logger.info(report)

# # 5) Inspect cluster membership by protein names (seq_names in your code)
# clusters = {}
# for name, lab in zip(seq_names, labels):
#     clusters.setdefault(lab, []).append(name)

# for cid, members in sorted(clusters.items()):
#     logger.info(f"[Cluster {cid}] n={len(members)} -> {members}")


# %% [markdown]
# ## Load Data

# %%
df = pd.read_csv(os.path.join(data_curation_dir, 'protacdb_tpddb_protacpedia_protac_dc50_dmax_activities.csv'))
df

# %% [markdown]
# ## Cluster POI sequences based on NAS matrix

# %%
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

# %%
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

# %%
# Cluster the POI sequences using HDBSCAN with optimized parameters
sampler = optuna.samplers.TPESampler(seed=42)
study = optuna.create_study(direction='maximize', sampler=sampler)
study.optimize(
    lambda trial: sequence_clustering_objective(
        trial,
        nas_matrix=poi_nas_dist_matrix,
        is_distance=True,
    ),
    n_trials=100,
)

logger.info(f"Best silhouette score: {study.best_value}")
logger.info(f"Best parameters: {study.best_params}")

# %%
# Apply HDBSCAN with the best parameters found
best_params = study.best_params
labels = HDBSCAN(
    min_cluster_size=best_params["min_cluster_size"],
    min_samples=best_params["min_samples"],
    cluster_selection_epsilon=best_params["cluster_selection_epsilon"],
    metric="precomputed",
    alpha=best_params["alpha"],
    leaf_size=best_params["leaf_size"],
    cluster_selection_method=best_params["cluster_selection_method"],
).fit_predict(poi_nas_dist_matrix)

# Assign cluster labels to sequences
seq2cluster = {}
for seq, label in zip(poi_sequences, labels):
    seq2cluster[seq] = label

# Add cluster labels to the original dataframe
df['POI_Cluster'] = df['POI_Sequence'].map(seq2cluster)

# Evaluate clustering results
report = evaluate_clusters(poi_nas_dist_matrix, labels, metric="precomputed")
for metric, value in report.items():
    logger.info(f"{metric}: {value:.4f}" if isinstance(value, float) else f"{metric}: {value}")

# %% [markdown]
# ## Cluster SMILES based on Tanimoto distance matrix
# 
# Use Butina clustering to cluster the SMILES based on the Tanimoto distance matrix.

# %% [markdown]
# ### Get Fingerprints

# %%
# Canonicalize SMILES strings
df['SMILES'] = df['SMILES'].apply(lambda smi: Chem.MolToSmiles(Chem.MolFromSmiles(smi)))

# %%
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
# Canonicalize SMILES
smiles = [Chem.MolToSmiles(Chem.MolFromSmiles(smi)) for smi in smiles]
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

# %%
def np2bitvect(fp_array: np.ndarray) -> ExplicitBitVect:
    """Convert a numpy array fingerprint to RDKit ExplicitBitVect."""
    bitvect = ExplicitBitVect(len(fp_array))
    for bit_idx in range(len(fp_array)):
        if fp_array[bit_idx] > 0:
            bitvect.SetBit(bit_idx)
    return bitvect

fps = [fp_generator.GetFingerprintAsNumPy(Chem.MolFromSmiles(smi)) for smi in smiles]
bitvects = [np2bitvect(fp) for fp in fps]
smiles2bitvect = {smi: bv for smi, bv in zip(smiles, bitvects)}

# Check that the lengths match
assert len(bitvects) == len(smiles), "Mismatch between number of fingerprints and SMILES"
assert len(fps) == len(smiles), "Mismatch between number of fingerprints and SMILES"

# %% [markdown]
# ### Isolate Held-Out SMILES

def get_avg_dist(fp, fp_list):
    dists = DataStructs.BulkTanimotoSimilarity(fp, fp_list, returnDistance=True)
    return np.mean(dists)

smiles_wo_operator = df[df['Value_Operator'].isna()]['SMILES'].dropna().unique().tolist()

held_out_smiles = []

for task in ['Dmax', 'DC50']:
    subset = df[df['Value_Type'] == task].copy()
    n_held_out = int(0.05 * subset.shape[0])
    
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

# %% [markdown]
# ### Butina and Scaffold Cluster the Data

# %%
smiles_to_cluster = [s for s in smiles if s not in held_out_smiles]

# %%
scaffold_clusters = get_bemis_murcko_clusters(smiles_to_cluster)
clusters_metrics = evaluate_clusters(
    X=np.array([smiles2fp[smi] for smi in smiles_to_cluster]),
    clusters=scaffold_clusters,
    metric='jaccard',
)
for metric, value in clusters_metrics.items():
    logger.info(f"{metric}: {value:.4f}" if isinstance(value, float) else f"{metric}: {value}")

# %%
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

logger.info(results_df)

# %%
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

# %%
# Visualize held-out SMILES in UMAP embedding
held_out_indices = [i for i, smi in enumerate(smiles) if smi in held_out_smiles]

# %%
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

# %% [markdown]
# The following is another method to obtain held-out data based on clustering the SMILES.
# 
# We decided to instead isolate the held-out data first, based on their Tanimoto distances to the rest of the data, and then cluster the remaining data. Because of this, the following code is no longer used in the final analysis, but is kept here for reference.

# %%
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

# %% [markdown]
# ### Plot 5x5 CV Folds

# %%

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

# %%
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

# %% [markdown]
# ## Push to Hugging Face

# %%
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
        # dataset.push_to_hub(
        #     "ailab-bio/PROTAC-Degradation-Predictor-Dataset",
        #     config_name=config,
        #     private=True,
        # )
        logger.info(dataset)
except Exception as e:
    logger.info(f"Error pushing to Hugging Face Hub: {e}")

# %% [markdown]
# ## Multitask Assembly and Clustering

# %%
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
              'Value_Concentration_Unit', 'Value_Mean', 'TPD_ID']

dc50_renamed = dc50_df.rename(columns={c: f"{c}_DC50" for c in value_cols})
dmax_renamed = dmax_df.rename(columns={c: f"{c}_Dmax" for c in value_cols})

# Perform the Merge
# Inner join ensures we only keep rows that exist in BOTH files with matching keys
multitask_df = pd.merge(dc50_renamed, dmax_renamed, on=keys, how='inner', suffixes=('', '_y'))

# 5. Clean Up Metadata
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

# Save Output
multitask_df.to_csv(data_tack_dir / "protacdb_tpddb_protacpedia_protac_multitask_activities_processed.csv", index=False)

# %%
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

# %%
ds = Dataset.from_pandas(multitask_df, preserve_index=False)

try:
    pass
    # ds.push_to_hub(
    #     "ailab-bio/PROTAC-Degradation-Predictor-Dataset",
    #     config_name="multitask",
    #     private=True,
    # )
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

