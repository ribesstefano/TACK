""" 
"""
from typing import List, Literal, Tuple, Dict, Optional, Union
from copy import deepcopy

import numpy as np
import optuna
from scipy.stats import skew
from sklearn.cluster import HDBSCAN
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)
from Bio import Phylo
from Bio.Phylo.TreeConstruction import DistanceMatrix, DistanceTreeConstructor

from tack_dataset.protein_utils import generate_normalized_alignment_matrix


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

    Example:
        >>> sampler = optuna.samplers.TPESampler(seed=42)
        >>> study = optuna.create_study(direction='maximize', sampler=sampler)
        >>> study.optimize(
        >>>     lambda trial: sequence_clustering_objective(trial, nas_matrix=np.random.rand(100, 100)),
        >>>     n_trials=300,
        >>> )
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


def nas_to_biopython_distance_matrix(
        nas: np.ndarray,
        names: List[str],
) -> DistanceMatrix:
    """ Biopython's NJ expects a DistanceMatrix (lower triangular incl. 0
        diagonal).
        
    Args:
        nas (np.ndarray): A 2D numpy array representing the NAS similarity matrix (Normalized Alignment Matrix) of shape (n_sequences, n_sequences).
        names (List[str]): A list of sequence names corresponding to the rows/columns of the NAS matrix. The length of this list should match the dimensions of the NAS matrix.
        
    Returns:
        DistanceMatrix: A Biopython DistanceMatrix object constructed from the NAS matrix, where distances
    """
    assert nas.shape[0] == nas.shape[1] == len(names)
    dist = 1.0 - nas  # convert similarity -> distance in [0,1]
    # Biopython DistanceMatrix takes only lower triangle (row i has i+1 elements)
    matrix = [[0.0]]
    for i in range(1, len(names)):
        row = [float(dist[i, j]) for j in range(i + 1)]
        matrix.append(row)
    return DistanceMatrix(names, matrix)


def neighbor_joining_tree_from_nas(nas: np.ndarray, names: List[str]) -> Phylo:
    """ Construct an NJ tree whose patristic distances approximate the input
        distances. See: Neighbor-Joining per Saitou & Nei (1987).
        
    Args:
        nas (np.ndarray): A 2D numpy array representing the NAS similarity matrix (Normalized Alignment Matrix) of shape (n_sequences, n_sequences).
        names (List[str]): A list of sequence names corresponding to the rows/columns of the NAS matrix. The length of this list should match the dimensions of the NAS matrix.
        
    Returns:
        Phylo: A Biopython Phylo tree object constructed using the Neighbor-Joining algorithm based on the provided NAS matrix. The tree's branch lengths are derived from the distances in
    """
    dm = nas_to_biopython_distance_matrix(nas, names)
    constructor = DistanceTreeConstructor()
    tree = constructor.nj(dm)
    return tree


def labels_from_clusters_dict(clusters: Dict[int, List[str]], names: List[str]) -> np.ndarray:
    name_to_cluster = {name: -1 for name in names}
    for cid, leaf_names in clusters.items():
        for nm in leaf_names:
            if nm in name_to_cluster:
                name_to_cluster[nm] = cid
    return np.array([name_to_cluster[nm] for nm in names], dtype=int)


def clusters_from_tree_cut(
        tree: Phylo,
        max_branch_len: float,
        names: Optional[List[str]] = None,
) -> Dict[int, List[str]]:
    """ Cut every edge longer than `max_branch_len`. Connected components of the
        remaining graph define clusters; leaves within each component are a
        cluster.
        
        Another approach for clustering would be to build a NJ (Neighbor-Joining)
        tree from the NAS matrix and then use a clustering algorithm on the tree. This would allow for hierarchical clustering based on the evolutionary relationships between sequences.
        
    Args:
        tree (Phylo): A Biopython Phylo tree object.
        max_branch_len (float): The threshold for cutting branches. Edges with branch length greater than this value will be cut.
        names (Optional[List[str]]): An optional list of leaf names to consider. If provided, only these names will be included in the clusters. If None, all leaf names in the tree will be considered.
        
    Returns:
        Dict[int, List[str]]: A dictionary mapping cluster IDs (integers) to lists of leaf names that belong to each cluster. Cluster IDs are assigned sequentially starting from 0.

    Example:
        >>> tree = neighbor_joining_tree_from_nas(nas_matrix, seq_names)
        >>> clusters_dict = clusters_from_tree_cut(tree, max_branch_len=0.20)
        >>> labels = labels_from_clusters_dict(clusters_dict, seq_names)
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


def cluster_prot_sequences(
    sequences: List[str],
    n_trials: int = 100,
    nas_kwargs: Optional[Dict[str, Union[str, float, bool]]] = None,
    hdbscan_kwargs: Optional[Dict[str, Union[int, float, str]]] = None,
) -> Dict[str, int]:
    """ Cluster a list of protein sequences using HDBSCAN with an optimized
        distance matrix derived from NAS.
        
    Args:
        sequences (List[str]): A list of protein sequences to cluster.
        n_trials (int): The number of Optuna trials to perform for optimizing HDBSCAN parameters. Default is 100. If set to 0, no optimization will be performed and default HDBSCAN parameters will be used.
        nas_kwargs (Optional[Dict[str, Union[str, float, bool]]]): Optional keyword arguments for generating the NAS distance matrix. If None, default parameters will be used.
        hdbscan_kwargs (Optional[Dict[str, Union[int, float, str]]]): Optional keyword arguments for HDBSCAN clustering. If None and n_trials > 0, parameters will be optimized using Optuna. If None and n_trials = 0, default HDBSCAN parameters will be used.
        
    Returns:
        Dict[str, int]: A dictionary mapping each input sequence to its assigned cluster label. Cluster labels are integers, where -1 typically indicates noise points that do not belong to any cluster.
    """
    if nas_kwargs is None:
        nas_kwargs = {
            "mode": "global",
            "gap_open": -10.0,
            "gap_extend": -0.5,
            "matrix": "BLOSUM62",
            "clip": (0.0, 1.0),
            "return_distance": True,
            "eps": 1e-12,
            "sanitize": True,
        }

    nas_dist_matrix = generate_normalized_alignment_matrix(
        sequences,
        **nas_kwargs,
    )
    
    if n_trials > 0:
        # Cluster the POI sequences using HDBSCAN with optimized parameters
        sampler = optuna.samplers.TPESampler(seed=42)
        study = optuna.create_study(direction='maximize', sampler=sampler)
        study.optimize(
            lambda trial: sequence_clustering_objective(
                trial,
                nas_matrix=nas_dist_matrix,
                is_distance=True,
            ),
            n_trials=100,
        )

        # Apply HDBSCAN with the best parameters found
        hdbscan_params = study.best_params
    elif hdbscan_kwargs is not None:
        hdbscan_params = hdbscan_kwargs
    else:
        # Use default parameters if no optimization is performed
        hdbscan_params = {
            "min_cluster_size": 5,
            "min_samples": 1,
            "cluster_selection_epsilon": 0.0,
            "alpha": 1.0,
            "leaf_size": 15,
            "cluster_selection_method": 'eom',
        }

    # Perform HDBSCAN clustering
    labels = HDBSCAN(
        min_cluster_size=hdbscan_params["min_cluster_size"],
        min_samples=hdbscan_params["min_samples"],
        cluster_selection_epsilon=hdbscan_params["cluster_selection_epsilon"],
        metric="precomputed",
        alpha=hdbscan_params["alpha"],
        leaf_size=hdbscan_params["leaf_size"],
        cluster_selection_method=hdbscan_params["cluster_selection_method"],
    ).fit_predict(nas_dist_matrix)

    # Assign cluster labels to sequences
    seq2cluster = {}
    for seq, label in zip(sequences, labels):
        seq2cluster[seq] = label
    
    return seq2cluster
    