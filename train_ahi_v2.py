#!/usr/bin/env python3
"""
AHI v2 Training Pipeline: Stacked Mesh Diffusion
==================================================

Training script for the dual-mesh architecture with:
- Gate warmup (coupling frozen for first N epochs)
- Warm-start from v1 checkpoint
- Spatial mask construction per batch
- Diagnostic logging (gate value, diffusion times, routing weights)

Reuses HazardDataset, focal loss, seasonal penalty from train_diffusion.py.

Author: Joshua D. Curry
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import json
import logging
import argparse
from tqdm import tqdm
import warnings
import sys

warnings.filterwarnings('ignore')

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
# Also check for train_diffusion in Desktop hazard-lm
_desktop_hl = Path(r"C:\Users\JDC\Desktop\hazard-lm")
if _desktop_hl.exists():
    sys.path.insert(0, str(_desktop_hl))

from ahi_v2_model import AHIv2Model, AHIv2Config, create_ahi_v2
from ahi_v2_graph import build_adjacency_graph, get_batch_adjacency, verify_adjacency

# Import reusable training utilities from v1
from train_diffusion import (
    HazardDataset,
    prepare_dataloaders,
    TrainingConfig,
    focal_loss_with_logits,
    seasonal_penalty,
    compute_ece,
    compute_metrics,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# Date-Grouped Sampler: Coherent Spatial Snapshots
# =============================================================================

class DateGroupedSampler(Sampler):
    """
    Yields indices grouped by date so each batch is a complete spatial snapshot.

    Every date in the dataset has exactly 39 counties (verified).
    Each batch = all 39 counties for one date, enabling meaningful spatial attention.

    From Simplicial Computation: spatial attention requires *coherent* input —
    counties from the same time point. Random county-day mixing produces
    incoherent spatial signal that the coupling gate correctly suppresses to ~0.

    Args:
        date_indices: list of lists — date_indices[i] = row indices for date i
        shuffle: whether to shuffle the date order (not the within-date order)
    """

    def __init__(self, date_indices: list, shuffle: bool = True):
        self.date_indices = date_indices
        self.shuffle = shuffle

    def __iter__(self):
        order = list(range(len(self.date_indices)))
        if self.shuffle:
            import random
            random.shuffle(order)
        for date_idx in order:
            yield from self.date_indices[date_idx]

    def __len__(self):
        return sum(len(group) for group in self.date_indices)


def prepare_date_grouped_loaders(
    data_path: str,
    val_split: float = 0.1,
    test_split: float = 0.1,
    num_workers: int = 4,
    hazards: list = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, pd.DataFrame, int]:
    """
    Prepare dataloaders with date-grouped batching.

    Each batch contains exactly one date's worth of counties (39 rows).
    This ensures spatial attention sees a coherent spatial snapshot.

    Returns:
        train_loader, val_loader, test_loader, train_df, counties_per_date
    """
    hazards = hazards or ['fire', 'flood', 'wind', 'winter', 'seismic']

    logger.info(f"Loading data from {data_path}")
    df = pd.read_parquet(data_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['date', 'county']).reset_index(drop=True)

    # Verify structure
    counties_per_date = df.groupby('date')['county'].nunique()
    n_counties = int(counties_per_date.mode().iloc[0])
    logger.info(f"Counties per date: {n_counties} (consistent: {(counties_per_date == n_counties).all()})")

    # Split by date (chronological)
    dates = sorted(df['date'].unique())
    n_dates = len(dates)
    test_start = int(n_dates * (1 - test_split))
    val_start = int(n_dates * (1 - test_split - val_split))

    train_dates = set(dates[:val_start])
    val_dates = set(dates[val_start:test_start])
    test_dates = set(dates[test_start:])

    train_mask = df['date'].isin(train_dates)
    val_mask = df['date'].isin(val_dates)
    test_mask = df['date'].isin(test_dates)

    train_df = df[train_mask].reset_index(drop=True)
    val_df = df[val_mask].reset_index(drop=True)
    test_df = df[test_mask].reset_index(drop=True)

    logger.info(f"Train: {len(train_df):,} ({len(train_dates)} dates) | "
                f"Val: {len(val_df):,} ({len(val_dates)} dates) | "
                f"Test: {len(test_df):,} ({len(test_dates)} dates)")

    # Identify features
    exclude_cols = {'date', 'county', 'state', 'county_id', 'state_id',
                    'fire_label', 'flood_label', 'wind_label', 'winter_label',
                    'seismic_label', 'tornado_label', 'any_hazard'}
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    temporal_cols = [c for c in feature_cols if c.startswith(('lag_', 'roll_', 'delta_'))]
    label_cols = [f'{h}_label' for h in hazards]

    # Build v1-compatible config for HazardDataset
    v1_cfg = TrainingConfig(data_path=data_path, batch_size=n_counties)

    train_ds = HazardDataset(train_df, feature_cols, temporal_cols, label_cols, v1_cfg)
    val_ds = HazardDataset(val_df, feature_cols, temporal_cols, label_cols, v1_cfg)
    test_ds = HazardDataset(test_df, feature_cols, temporal_cols, label_cols, v1_cfg)

    # Build date-grouped indices for each split
    def build_date_groups(split_df):
        """Group row indices by date for the sampler."""
        split_df = split_df.copy()
        split_df['_idx'] = range(len(split_df))
        groups = split_df.groupby('date')['_idx'].apply(list).tolist()
        return groups

    train_groups = build_date_groups(train_df)
    val_groups = build_date_groups(val_df)
    test_groups = build_date_groups(test_df)

    train_loader = DataLoader(
        train_ds,
        batch_size=n_counties,
        sampler=DateGroupedSampler(train_groups, shuffle=True),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=n_counties,
        sampler=DateGroupedSampler(val_groups, shuffle=False),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=n_counties,
        sampler=DateGroupedSampler(test_groups, shuffle=False),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader, train_df, n_counties


# =============================================================================
# v2 Training Configuration
# =============================================================================

@dataclass
class V2TrainingConfig:
    """Training configuration for AHI v2."""

    # Data (same as v1)
    data_path: str = "data/hazard_lm_clean_labeled.parquet"
    val_split: float = 0.1
    test_split: float = 0.1

    # Training
    batch_size: int = 32
    epochs: int = 30
    learning_rate: float = 1e-4
    weight_decay: float = 0.05
    warmup_steps: int = 500
    max_grad_norm: float = 1.0
    early_stop_patience: int = 7  # More patient than v1 (5)

    # Loss
    use_focal_loss: bool = True
    focal_gamma: float = 2.0
    focal_alpha: float = 0.75

    # Class weights
    use_class_weights: bool = True
    class_weight_cap: float = 10.0

    # Gate warmup
    coupling_warmup_epochs: int = 3

    # Warm-start
    v1_checkpoint: Optional[str] = None

    # Hardware
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    mixed_precision: bool = True
    num_workers: int = 4

    # Output
    output_dir: str = "outputs/ahi_v2"
    save_every: int = 5
    log_every: int = 100

    # Hazards
    hazards: List[str] = field(default_factory=lambda: [
        "fire", "flood", "wind", "winter", "seismic"
    ])

    # Graph
    centroids_path: Optional[str] = None
    k_neighbors: int = 5

    def __post_init__(self):
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

    def to_v1_config(self) -> TrainingConfig:
        """Create a v1 TrainingConfig for reusable functions."""
        return TrainingConfig(
            data_path=self.data_path,
            val_split=self.val_split,
            test_split=self.test_split,
            batch_size=self.batch_size,
            epochs=self.epochs,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            num_workers=self.num_workers,
            device=self.device,
            mixed_precision=self.mixed_precision,
            use_focal_loss=self.use_focal_loss,
            focal_gamma=self.focal_gamma,
            focal_alpha=self.focal_alpha,
            use_class_weights=self.use_class_weights,
            class_weight_cap=self.class_weight_cap,
            output_dir=self.output_dir,
            hazards=self.hazards,
        )


# =============================================================================
# v2 Trainer
# =============================================================================

class AHIv2Trainer:
    """Training loop for AHI v2 stacked mesh architecture."""

    def __init__(
        self,
        model: AHIv2Model,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: V2TrainingConfig,
        train_df: pd.DataFrame,
        full_adjacency: torch.Tensor,
        county_names: list,
    ):
        self.model = model.to(config.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.full_adjacency = full_adjacency.to(config.device)
        self.county_names = county_names
        self.num_graph_nodes = len(county_names)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Scheduler
        total_steps = len(train_loader) * config.epochs
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=config.learning_rate,
            total_steps=total_steps,
            pct_start=0.1,
        )

        # Mixed precision
        self.scaler = GradScaler() if config.mixed_precision else None

        # Class weights (computed from training data)
        self.class_weights = {}
        if config.use_class_weights and train_df is not None:
            for hazard in config.hazards:
                label_col = f'{hazard}_label'
                if label_col in train_df.columns:
                    pos = (train_df[label_col] == 1).sum()
                    neg = (train_df[label_col] == 0).sum()
                    weight = min(neg / max(pos, 1), config.class_weight_cap)
                    self.class_weights[hazard] = weight
                    logger.info(f"  {hazard}: pos={pos:,}, neg={neg:,}, weight={weight:.2f}")

        # Tracking
        self.global_step = 0
        self.best_val_auc = 0.0
        self.epochs_since_improve = 0
        self.train_losses = []
        self.val_metrics_history = []
        self.gate_history = []

    def _compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        labels: Dict[str, torch.Tensor],
        months: torch.Tensor,
        epoch_losses: Dict[str, list],
    ) -> torch.Tensor:
        """Multi-hazard focal loss + seasonal penalty (same as v1)."""
        total_loss = torch.tensor(0.0, device=self.config.device)

        for hazard in self.config.hazards:
            logits = outputs[f'{hazard}_logits']
            target = labels[hazard]

            if self.config.use_focal_loss:
                cw = self.class_weights.get(hazard, 1.0)
                loss = focal_loss_with_logits(
                    logits, target,
                    gamma=self.config.focal_gamma,
                    alpha=self.config.focal_alpha,
                    class_weight=cw,
                )
            else:
                loss = F.binary_cross_entropy_with_logits(logits, target, reduction='none')

            # Seasonal penalty
            loss = seasonal_penalty(loss, months, target, logits, hazard)

            loss = loss.mean()
            total_loss = total_loss + loss
            epoch_losses[hazard].append(loss.item())

        return total_loss

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()

        # Gate warmup: freeze coupling for first N epochs
        gate_frozen = epoch < self.config.coupling_warmup_epochs
        self.model.set_coupling_frozen(gate_frozen)

        epoch_losses = {h: [] for h in self.config.hazards}
        epoch_losses['total'] = []

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config.epochs}")

        for batch in pbar:
            static_cont = batch['static_cont'].to(self.config.device)
            temporal = batch['temporal'].to(self.config.device)
            state_ids = batch['state_id'].to(self.config.device)
            county_ids = batch['county_id'].to(self.config.device)
            nlcd_ids = batch['nlcd_id'].to(self.config.device)
            months = batch['month'].to(self.config.device)

            labels = {
                h: batch[f'{h}_label'].to(self.config.device).unsqueeze(1)
                for h in self.config.hazards
            }

            # Build per-batch spatial adjacency mask
            spatial_mask = get_batch_adjacency(
                self.full_adjacency, county_ids, self.num_graph_nodes
            )

            # Forward + loss
            self.optimizer.zero_grad()

            if self.config.mixed_precision:
                with autocast():
                    outputs = self.model(
                        static_cont, temporal, county_ids, state_ids, nlcd_ids,
                        spatial_mask=spatial_mask,
                    )
                    total_loss = self._compute_loss(outputs, labels, months, epoch_losses)

                self.scaler.scale(total_loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(
                    static_cont, temporal, county_ids, state_ids, nlcd_ids,
                    spatial_mask=spatial_mask,
                )
                total_loss = self._compute_loss(outputs, labels, months, epoch_losses)

                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()

            self.scheduler.step()
            epoch_losses['total'].append(total_loss.item())
            self.global_step += 1

            pbar.set_postfix({
                'loss': f"{total_loss.item():.4f}",
                'gate': f"{self.model.coupling.gate.item():.4f}",
                'lr': f"{self.scheduler.get_last_lr()[0]:.2e}",
            })

        summary = {
            f'{h}_loss': np.mean(epoch_losses[h]) for h in self.config.hazards
        }
        summary['total_loss'] = np.mean(epoch_losses['total'])
        return summary

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        """Evaluate on validation/test set."""
        self.model.eval()

        all_probs = {h: [] for h in self.config.hazards}
        all_labels = {h: [] for h in self.config.hazards}

        for batch in tqdm(loader, desc="Evaluating", leave=False):
            static_cont = batch['static_cont'].to(self.config.device)
            temporal = batch['temporal'].to(self.config.device)
            state_ids = batch['state_id'].to(self.config.device)
            county_ids = batch['county_id'].to(self.config.device)
            nlcd_ids = batch['nlcd_id'].to(self.config.device)

            spatial_mask = get_batch_adjacency(
                self.full_adjacency, county_ids, self.num_graph_nodes
            )

            outputs = self.model(
                static_cont, temporal, county_ids, state_ids, nlcd_ids,
                spatial_mask=spatial_mask,
            )

            for hazard in self.config.hazards:
                probs = outputs[f'{hazard}_prob'].cpu().numpy().flatten()
                labels = batch[f'{hazard}_label'].numpy().flatten()
                all_probs[hazard].extend(probs)
                all_labels[hazard].extend(labels)

        all_probs = {h: np.array(v) for h, v in all_probs.items()}
        all_labels = {h: np.array(v) for h, v in all_labels.items()}

        metrics = compute_metrics(all_probs, all_labels)

        # Add v2-specific diagnostics
        diag = self.model.get_diagnostics()
        metrics['coupling_gate'] = diag['coupling_gate']
        metrics['mean_diffusion_time'] = diag['mean_diffusion_time']
        metrics['mma_scale'] = diag['mma_scale']

        return metrics

    def save_checkpoint(self, epoch: int, metrics: Dict[str, float]) -> bool:
        """Save checkpoint. Returns True if this is a new best."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'config': {
                k: v for k, v in self.config.__dict__.items()
                if not callable(v)
            },
        }

        path = Path(self.config.output_dir) / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, path)

        mean_auc = np.mean([v for k, v in metrics.items() if '_auc' in k and v > 0])
        improved = mean_auc > self.best_val_auc

        if improved:
            self.best_val_auc = mean_auc
            best_path = Path(self.config.output_dir) / "best_model.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"New best model! Mean AUC: {mean_auc:.4f}")

        return improved

    def train(self):
        """Full training loop with gate warmup."""
        logger.info("=" * 60)
        logger.info("AHI v2 STACKED MESH TRAINING (date-grouped batching)")
        logger.info(f"Device: {self.config.device}")
        logger.info(f"Epochs: {self.config.epochs}")
        logger.info(f"Batch size: {self.config.batch_size} (1 date = 1 spatial snapshot)")
        logger.info(f"Coupling warmup: {self.config.coupling_warmup_epochs} epochs")
        logger.info(f"Graph: {self.num_graph_nodes} counties, k={self.config.k_neighbors}")
        logger.info("=" * 60)

        val_metrics = {}

        for epoch in range(self.config.epochs):
            # Train
            train_metrics = self.train_epoch(epoch)

            # Validate
            val_metrics = self.evaluate(self.val_loader)

            # Log diagnostics
            diag = self.model.get_diagnostics()
            gate_val = diag['coupling_gate']
            self.gate_history.append({
                'epoch': epoch + 1,
                'gate': gate_val,
                'mma_routing': diag['mma_routing'],
                'mma_scale': diag['mma_scale'],
            })

            logger.info(f"\nEpoch {epoch+1}/{self.config.epochs}")
            logger.info(f"  Train Loss: {train_metrics['total_loss']:.4f}")
            logger.info(f"  Coupling Gate: {gate_val:.4f}"
                        f" {'(FROZEN)' if epoch < self.config.coupling_warmup_epochs else ''}")

            for hazard in self.config.hazards:
                auc = val_metrics.get(f'{hazard}_auc', 0)
                ece = val_metrics.get(f'{hazard}_ece', 0)
                logger.info(f"  {hazard.upper()}: AUC={auc:.4f}, ECE={ece:.4f}")

            mean_auc = np.mean([v for k, v in val_metrics.items() if '_auc' in k and v > 0])
            logger.info(f"  Mean AUC: {mean_auc:.4f} (best: {self.best_val_auc:.4f})")

            # Save checkpoint
            improved = False
            if (epoch + 1) % self.config.save_every == 0 or mean_auc > self.best_val_auc:
                improved = self.save_checkpoint(epoch + 1, val_metrics)

            # Early stopping
            if improved:
                self.epochs_since_improve = 0
            else:
                self.epochs_since_improve += 1

            self.val_metrics_history.append({
                'epoch': epoch + 1,
                **val_metrics,
            })

            if self.epochs_since_improve >= self.config.early_stop_patience:
                logger.info(f"Early stopping at epoch {epoch+1} "
                            f"(no improvement for {self.epochs_since_improve} epochs)")
                break

        # Save final checkpoint
        try:
            self.save_checkpoint(self.config.epochs, val_metrics)
        except Exception:
            pass

        # Save histories
        output_dir = Path(self.config.output_dir)
        with open(output_dir / "training_history.json", 'w') as f:
            json.dump({
                'train_losses': self.train_losses,
                'val_metrics': self.val_metrics_history,
            }, f, indent=2)

        with open(output_dir / "gate_history.json", 'w') as f:
            json.dump(self.gate_history, f, indent=2)

        logger.info("\n" + "=" * 60)
        logger.info("TRAINING COMPLETE")
        logger.info(f"Best validation AUC: {self.best_val_auc:.4f}")
        logger.info(f"Final coupling gate: {self.model.coupling.gate.item():.4f}")
        logger.info("=" * 60)

        return self.best_val_auc


# =============================================================================
# Main
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Train AHI v2 Stacked Mesh")
    parser.add_argument('--data_path', type=str, default="data/hazard_lm_clean_labeled.parquet")
    parser.add_argument('--v1_checkpoint', type=str, default=None,
                        help="Path to v1 best_model.pt for warm-start")
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--coupling_warmup', type=int, default=3)
    parser.add_argument('--k_neighbors', type=int, default=5)
    parser.add_argument('--output_dir', type=str, default="outputs/ahi_v2")
    parser.add_argument('--no_mma', action='store_true', help="Disable MMA bias field")
    parser.add_argument('--no_spatial', action='store_true', help="Disable spatial mesh (ablation)")
    parser.add_argument('--device', type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    config = V2TrainingConfig(
        data_path=args.data_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        coupling_warmup_epochs=args.coupling_warmup,
        k_neighbors=args.k_neighbors,
        output_dir=args.output_dir,
        v1_checkpoint=args.v1_checkpoint,
    )

    if args.device:
        config.device = args.device

    # ---- 1. Build adjacency graph ----
    logger.info("Building county adjacency graph...")
    adjacency, distances, county_to_idx, county_names = build_adjacency_graph(
        centroids_path=config.centroids_path,
        k=config.k_neighbors,
    )
    verify_adjacency(adjacency, county_names)

    # ---- 2. Load data with date-grouped batching ----
    # Each batch = all 39 counties for one date (coherent spatial snapshot)
    # This is critical: random county-day mixing produces incoherent spatial signal
    train_loader, val_loader, test_loader, train_df, n_counties = \
        prepare_date_grouped_loaders(
            data_path=config.data_path,
            val_split=config.val_split,
            test_split=config.test_split,
            num_workers=config.num_workers,
            hazards=config.hazards,
        )
    config.batch_size = n_counties  # Override to match date grouping
    logger.info(f"Date-grouped batching: {n_counties} counties per batch")

    # ---- 3. Create model ----
    model_config = AHIv2Config(
        use_mma=not args.no_mma,
        k_neighbors=args.k_neighbors,
        num_counties=len(county_names),
    )
    model = AHIv2Model(model_config)

    # Parameter count
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"AHI v2 model: {total_params:,} parameters")

    # Warm-start from v1
    if config.v1_checkpoint:
        logger.info(f"Warm-starting from v1: {config.v1_checkpoint}")
        result = AHIv2Model.warm_start_from_v1(model, config.v1_checkpoint)
        logger.info(f"  Loaded {len(result['loaded'])} params, skipped {len(result['skipped'])}")
    else:
        # Auto-detect v1 checkpoint
        candidates = [
            Path(__file__).parent / "best_model.pt",
            Path(__file__).parent / "outputs" / "diffusion_clean_v1" / "best_model.pt",
            Path(r"C:\Users\JDC\Desktop\hazard-lm\outputs\diffusion_clean_v1\best_model.pt"),
        ]
        for p in candidates:
            if p.exists():
                logger.info(f"Auto-detected v1 checkpoint: {p}")
                AHIv2Model.warm_start_from_v1(model, str(p))
                break

    # ---- 4. If spatial ablation, zero out spatial mesh ----
    if args.no_spatial:
        logger.info("ABLATION: Spatial mesh disabled (gate fixed at 0)")
        with torch.no_grad():
            model.coupling.gate.fill_(0.0)
        model.coupling.gate.requires_grad_(False)

    # ---- 5. Train ----
    trainer = AHIv2Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        train_df=train_df,
        full_adjacency=adjacency,
        county_names=county_names,
    )

    best_auc = trainer.train()

    # ---- 6. Evaluate on test set ----
    logger.info("\nFinal evaluation on test set...")
    # Load best model
    best_path = Path(config.output_dir) / "best_model.pt"
    if best_path.exists():
        checkpoint = torch.load(best_path, map_location=config.device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(config.device)

    test_metrics = trainer.evaluate(test_loader)
    logger.info("\nTest Results:")
    for hazard in config.hazards:
        auc = test_metrics.get(f'{hazard}_auc', 0)
        ece = test_metrics.get(f'{hazard}_ece', 0)
        ap = test_metrics.get(f'{hazard}_ap', 0)
        logger.info(f"  {hazard.upper()}: AUC={auc:.4f}, AP={ap:.4f}, ECE={ece:.4f}")

    mean_auc = np.mean([v for k, v in test_metrics.items() if '_auc' in k and v > 0])
    logger.info(f"\n  Mean Test AUC: {mean_auc:.4f}")

    # Save test results
    with open(Path(config.output_dir) / "test_results.json", 'w') as f:
        json.dump(test_metrics, f, indent=2)

    return mean_auc


if __name__ == "__main__":
    main()
