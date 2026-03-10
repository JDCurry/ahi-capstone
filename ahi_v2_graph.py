#!/usr/bin/env python3
"""
AHI v2: County Adjacency Graph Construction
=============================================

Builds a k-NN spatial graph from WA county centroids for the spatial mesh.
Uses Haversine distance for geographic accuracy.

Author: Joshua D. Curry
"""

import math
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, Optional
import logging

logger = logging.getLogger(__name__)


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Haversine distance in kilometers between two lat/lon points."""
    R = 6371.0  # Earth radius in km
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def build_distance_matrix(centroids: pd.DataFrame) -> np.ndarray:
    """Compute pairwise Haversine distance matrix from centroids DataFrame."""
    n = len(centroids)
    lats = centroids['lat'].values
    lons = centroids['lon'].values
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = haversine_distance(lats[i], lons[i], lats[j], lons[j])
            dist[i, j] = d
            dist[j, i] = d
    return dist


def build_adjacency_graph(
    centroids_path: Optional[str] = None,
    k: int = 5,
    county_map: Optional[Dict[str, int]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, int], list]:
    """
    Build county adjacency graph from centroids CSV.

    Args:
        centroids_path: Path to centroids CSV. Auto-detected if None.
        k: Number of nearest neighbors per county.
        county_map: Optional mapping from county name to integer ID.
                    If provided, adjacency matrix uses these IDs.

    Returns:
        adjacency: (N, N) boolean tensor - True where counties are neighbors
        distances: (N, N) float tensor - Haversine distances in km
        county_to_idx: mapping from county name to graph index
        county_names: ordered list of county names
    """
    # Find centroids file
    if centroids_path is None:
        search_paths = [
            Path(__file__).parent / "data" / "WA_County_Master_Table__Final_with_Schools_.centroids.csv",
            Path(__file__).parent / "data" / "WA_County_Master_Table__Final_with_Schools_.cleaned.csv",
        ]
        for p in search_paths:
            if p.exists():
                centroids_path = str(p)
                break
        if centroids_path is None:
            raise FileNotFoundError(
                "County centroids CSV not found. Searched:\n" +
                "\n".join(f"  {p}" for p in search_paths)
            )

    df = pd.read_csv(centroids_path)
    logger.info(f"Loaded {len(df)} counties from {centroids_path}")

    # Normalize county names
    if 'county_name' in df.columns:
        names = df['county_name'].str.strip().tolist()
    elif 'COUNTY' in df.columns:
        names = df['COUNTY'].astype(str).tolist()
    else:
        raise ValueError(f"No county name column found. Columns: {df.columns.tolist()}")

    n = len(names)
    county_to_idx = {name: i for i, name in enumerate(names)}

    # Build distance matrix
    dist_matrix = build_distance_matrix(df)

    # Build k-NN adjacency (symmetric)
    adj = np.eye(n, dtype=bool)  # self-loops
    for i in range(n):
        # Find k nearest neighbors (excluding self)
        dists = dist_matrix[i].copy()
        dists[i] = float('inf')  # exclude self
        neighbors = np.argsort(dists)[:k]
        for j in neighbors:
            adj[i, j] = True
            adj[j, i] = True  # symmetric

    adjacency = torch.tensor(adj, dtype=torch.bool)
    distances = torch.tensor(dist_matrix, dtype=torch.float32)

    # Log stats
    neighbor_counts = adj.sum(axis=1) - 1  # exclude self-loop
    logger.info(f"Adjacency graph: {n} counties, k={k}")
    logger.info(f"  Neighbors per county: min={neighbor_counts.min()}, "
                f"max={neighbor_counts.max()}, mean={neighbor_counts.mean():.1f}")

    return adjacency, distances, county_to_idx, names


def get_batch_adjacency(
    full_adjacency: torch.Tensor,
    county_ids: torch.Tensor,
    num_graph_nodes: int = 39,
) -> torch.Tensor:
    """
    Extract adjacency submatrix for a batch of county IDs.

    During training, batches contain random county-day samples.
    This extracts the relevant subgraph for spatial attention.

    Args:
        full_adjacency: (N, N) boolean adjacency matrix
        county_ids: (batch,) integer county indices into the graph
        num_graph_nodes: total nodes in full graph

    Returns:
        batch_adj: (batch, batch) boolean adjacency for this batch
    """
    # Clamp IDs to valid range
    ids = county_ids.clamp(0, num_graph_nodes - 1).long()
    # Index into full adjacency: rows then columns
    batch_adj = full_adjacency[ids][:, ids]
    return batch_adj


def verify_adjacency(adjacency: torch.Tensor, county_names: list):
    """Sanity check the adjacency graph."""
    n = adjacency.shape[0]
    name_to_idx = {name: i for i, name in enumerate(county_names)}

    # Check symmetry
    assert (adjacency == adjacency.T).all(), "Adjacency matrix is not symmetric"

    # Check self-loops
    assert adjacency.diagonal().all(), "Missing self-loops"

    # Spot-check known neighbors
    checks = [
        ("King County", ["Snohomish County", "Pierce County", "Kitsap County"]),
        ("Pierce County", ["King County", "Thurston County"]),
        ("Spokane County", ["Lincoln County", "Stevens County"]),
    ]

    all_pass = True
    for county, expected_neighbors in checks:
        if county not in name_to_idx:
            continue
        idx = name_to_idx[county]
        for neighbor in expected_neighbors:
            if neighbor not in name_to_idx:
                continue
            nidx = name_to_idx[neighbor]
            if not adjacency[idx, nidx]:
                logger.warning(f"Expected {county} adjacent to {neighbor}, but not found")
                all_pass = False

    if all_pass:
        logger.info("Adjacency graph verification passed")
    return all_pass


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    adjacency, distances, county_to_idx, names = build_adjacency_graph(k=5)

    print(f"\nAdjacency matrix shape: {adjacency.shape}")
    print(f"Counties: {len(names)}")
    print(f"Edges (excluding self): {(adjacency.sum() - len(names)).item() // 2}")

    # Print neighbor list for a few counties
    for county in ["King County", "Pierce County", "Spokane County"]:
        if county in county_to_idx:
            idx = county_to_idx[county]
            neighbor_idxs = adjacency[idx].nonzero(as_tuple=True)[0]
            neighbors = [names[i] for i in neighbor_idxs if i != idx]
            dists = [f"{distances[idx, i]:.0f}km" for i in neighbor_idxs if i != idx]
            print(f"\n{county} neighbors:")
            for n, d in zip(neighbors, dists):
                print(f"  {n}: {d}")

    verify_adjacency(adjacency, names)
