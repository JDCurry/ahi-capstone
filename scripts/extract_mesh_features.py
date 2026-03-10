#!/usr/bin/env python3
"""
Extract AHI v2 mesh representations for hybrid XGBoost model.

Outputs per-sample feature vectors:
  temporal_out (128d) + spatial_out (128d) + coupled (128d) = 384d
  + raw features (21d) = 405d total

These enriched features are fed to XGBoost for the hybrid configuration.
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
import sys
import logging
import argparse
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from ahi_v2_model import AHIv2Model, AHIv2Config
from ahi_v2_graph import build_adjacency_graph, get_batch_adjacency
from train_diffusion import prepare_dataloaders, TrainingConfig

logger = logging.getLogger(__name__)


def extract_features(
    model_path: str,
    data_path: str = "data/hazard_lm_clean_labeled.parquet",
    output_dir: str = "outputs/ahi_v2_hybrid",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    batch_size: int = 64,
):
    """Extract v2 intermediate representations for all splits."""

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Load model
    config = AHIv2Config()
    model = AHIv2Model(config)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # Build adjacency graph
    adjacency, _, _, county_names = build_adjacency_graph()
    adjacency = adjacency.to(device)
    num_nodes = len(county_names)

    # Load data splits
    v1_config = TrainingConfig(
        data_path=data_path,
        batch_size=batch_size,
        num_workers=4,
    )
    train_loader, val_loader, test_loader, _ = prepare_dataloaders(v1_config)

    for split_name, loader in [('train', train_loader), ('val', val_loader), ('test', test_loader)]:
        logger.info(f"Extracting features for {split_name} split...")

        all_temporal = []
        all_spatial = []
        all_coupled = []
        all_labels = {h: [] for h in config.hazards}

        with torch.no_grad():
            for batch in tqdm(loader, desc=split_name):
                static_cont = batch['static_cont'].to(device)
                temporal = batch['temporal'].to(device)
                state_ids = batch['state_id'].to(device)
                county_ids = batch['county_id'].to(device)
                nlcd_ids = batch['nlcd_id'].to(device)

                spatial_mask = get_batch_adjacency(adjacency, county_ids, num_nodes)

                outputs = model(
                    static_cont, temporal, county_ids, state_ids, nlcd_ids,
                    spatial_mask=spatial_mask,
                    return_intermediates=True,
                )

                all_temporal.append(outputs['_temporal_out'].cpu().numpy())
                all_spatial.append(outputs['_spatial_out'].cpu().numpy())
                all_coupled.append(outputs['_coupled'].cpu().numpy())

                for h in config.hazards:
                    all_labels[h].extend(batch[f'{h}_label'].numpy().flatten())

        # Concatenate
        temporal_feats = np.concatenate(all_temporal, axis=0)
        spatial_feats = np.concatenate(all_spatial, axis=0)
        coupled_feats = np.concatenate(all_coupled, axis=0)

        # Combined: 384d mesh representations
        mesh_feats = np.concatenate([temporal_feats, spatial_feats, coupled_feats], axis=1)
        labels = {h: np.array(v) for h, v in all_labels.items()}

        # Save
        np.save(Path(output_dir) / f"{split_name}_mesh_features.npy", mesh_feats)
        np.save(Path(output_dir) / f"{split_name}_labels.npy",
                np.stack([labels[h] for h in config.hazards], axis=1))

        logger.info(f"  {split_name}: {mesh_feats.shape[0]} samples, "
                     f"{mesh_feats.shape[1]} features")

    logger.info(f"Features saved to {output_dir}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="outputs/ahi_v2/best_model.pt")
    parser.add_argument('--data', type=str, default="data/hazard_lm_clean_labeled.parquet")
    parser.add_argument('--output', type=str, default="outputs/ahi_v2_hybrid")
    args = parser.parse_args()

    extract_features(args.model, args.data, args.output)
