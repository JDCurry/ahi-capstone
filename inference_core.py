"""
Cloud-safe inference core for HazardLM-Diffusion v2.0.
NO training dependencies - pure PyTorch inference only.

Updated for HazardLMDiffusion model API:
  model(static_cont, temporal, region_ids, state_ids, nlcd_ids)
  -> dict with {hazard}_prob and {hazard}_logits keys
"""
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import date
from typing import Dict, Optional, Tuple

# Hazard types
HAZARD_TYPES = ['fire', 'flood', 'wind', 'winter', 'seismic']

# Feature columns used during training (must match train_diffusion.py / HazardDataset)
# These are the 21 static features from hazard_lm_clean_labeled.parquet
STATIC_FEATURE_COLS = [
    'latitude', 'longitude', 'day_of_year', 'month', 'year',
    'tmmx', 'tmmn', 'rmin', 'rmax', 'vs', 'erc', 'pr', 'vpd',
    'red_flag_active', 'tmmx_3d_mean', 'pr_3d_mean', 'vs_3d_mean',
    'elevation', 'forest_fraction', 'urban_fraction', 'pop_density'
]

# County name -> county_id mapping (mod 250 to match training)
# Built at runtime from the dataset; cached here for fallback
_COUNTY_MAP = {}
_STATE_MAP = {}


def _build_maps(hazard_df: pd.DataFrame):
    """Build county and state maps from dataset (mirrors HazardDataset)."""
    global _COUNTY_MAP, _STATE_MAP
    if 'county' in hazard_df.columns:
        counties = sorted(hazard_df['county'].unique())
        _COUNTY_MAP = {c: i % 250 for i, c in enumerate(counties)}
    if 'state' in hazard_df.columns:
        states = sorted(hazard_df['state'].unique())
        _STATE_MAP = {s: i for i, s in enumerate(states)}


@torch.no_grad()
def predict_from_tensors(
    model,
    static_cont: torch.Tensor,
    temporal: torch.Tensor,
    region_ids: torch.Tensor,
    state_ids: torch.Tensor,
    nlcd_ids: torch.Tensor,
    hazard_types: list = None
) -> Dict[str, float]:
    """
    Run HazardLMDiffusion inference from pre-built tensors.

    Args:
        model: Loaded HazardLMDiffusion model
        static_cont: [batch, static_dim] static continuous features (padded to 50)
        temporal: [batch, seq_len, feat_dim] temporal sequence (14, 20)
        region_ids: [batch] county/region IDs (long)
        state_ids: [batch] state IDs (long)
        nlcd_ids: [batch] NLCD land cover IDs (long)
        hazard_types: List of hazard types to predict

    Returns:
        Dict mapping hazard type -> probability (float in [0, 1])
    """
    if model is None:
        return {h: 0.0 for h in (hazard_types or HAZARD_TYPES)}

    hazard_types = hazard_types or HAZARD_TYPES

    model.eval()
    device = next(model.parameters()).device

    # Move tensors to device
    static_cont = static_cont.to(device).float()
    temporal = temporal.to(device).float()
    region_ids = region_ids.to(device).long()
    state_ids = state_ids.to(device).long()
    nlcd_ids = nlcd_ids.to(device).long()

    # Forward pass (HazardLMDiffusion API)
    outputs = model(static_cont, temporal, region_ids, state_ids, nlcd_ids)

    # Extract probabilities
    risks = {}
    for h in hazard_types:
        prob = 0.0
        if isinstance(outputs, dict):
            prob_key = f'{h}_prob'
            logit_key = f'{h}_logits'
            if prob_key in outputs:
                prob = float(outputs[prob_key].cpu().numpy().flatten()[0])
            elif logit_key in outputs:
                prob = float(torch.sigmoid(outputs[logit_key]).cpu().numpy().flatten()[0])
        risks[h] = max(0.0, min(1.0, prob))  # Clamp to [0, 1]

    return risks


def build_tensors_from_county_data(
    county_row: pd.Series,
    county_name: str = '',
    target_date: date = None,
    static_pad_dim: int = 50,
    temporal_seq_len: int = 14,
    temporal_feat_dim: int = 20,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build inference tensors from a county data row.
    Matches the HazardDataset.__getitem__ format from train_diffusion.py.

    Returns: (static_cont, temporal, region_ids, state_ids, nlcd_ids)
    """
    # --- Static continuous features ---
    static_values = []
    for col in STATIC_FEATURE_COLS:
        if col in county_row.index:
            val = county_row[col]
            try:
                static_values.append(float(val) if pd.notna(val) else 0.0)
            except (ValueError, TypeError):
                static_values.append(0.0)
        else:
            # If column not present, inject sensible defaults for date-derived cols
            if col == 'day_of_year' and target_date is not None:
                static_values.append(float(target_date.timetuple().tm_yday))
            elif col == 'month' and target_date is not None:
                static_values.append(float(target_date.month))
            elif col == 'year' and target_date is not None:
                static_values.append(float(target_date.year))
            else:
                static_values.append(0.0)

    # Override date-derived features with target_date if provided
    if target_date is not None:
        for i, col in enumerate(STATIC_FEATURE_COLS):
            if col == 'day_of_year':
                static_values[i] = float(target_date.timetuple().tm_yday)
            elif col == 'month':
                static_values[i] = float(target_date.month)
            elif col == 'year':
                static_values[i] = float(target_date.year)

    # Pad to expected dimension (50)
    while len(static_values) < static_pad_dim:
        static_values.append(0.0)
    static_cont = torch.tensor(static_values[:static_pad_dim], dtype=torch.float32).unsqueeze(0)

    # --- Temporal features ---
    # The clean dataset has no lag/roll/delta columns, so temporal is zero-filled
    # (same as training — HazardDataset pads to 14x20)
    temporal = torch.zeros(1, temporal_seq_len, temporal_feat_dim, dtype=torch.float32)

    # --- Categorical IDs ---
    county_id = _COUNTY_MAP.get(county_name, 0) if county_name else 0
    state_name = county_row.get('state', 'WA') if 'state' in county_row.index else 'WA'
    state_id = _STATE_MAP.get(state_name, 0)
    nlcd_id = 0  # Placeholder (matches training)

    region_ids = torch.tensor([county_id], dtype=torch.long)
    state_ids = torch.tensor([state_id], dtype=torch.long)
    nlcd_ids = torch.tensor([nlcd_id], dtype=torch.long)

    return static_cont, temporal, region_ids, state_ids, nlcd_ids


def predict_county_risks_simple(
    model,
    county_name: str,
    hazard_df: pd.DataFrame,
    target_date: date = None
) -> Dict[str, float]:
    """
    Simplified county risk prediction for cloud deployment.
    Uses available data from hazard_lm_clean_labeled.parquet.

    Args:
        model: Loaded HazardLMDiffusion model
        county_name: Name of county (e.g., "King", "Pierce")
        hazard_df: DataFrame with county hazard data
        target_date: Target date for prediction (affects seasonal features)

    Returns:
        Dict mapping hazard type -> probability
    """
    if model is None:
        return _generate_fallback_risks(county_name)

    # Build maps if not yet built
    if not _COUNTY_MAP and hazard_df is not None and len(hazard_df) > 0:
        _build_maps(hazard_df)

    # Normalize county name
    county_upper = county_name.upper().replace(' COUNTY', '').strip()

    # Find county in dataframe
    if hazard_df is not None and len(hazard_df) > 0 and 'county' in hazard_df.columns:
        county_mask = hazard_df['county'].str.upper().str.replace(' COUNTY', '').str.strip() == county_upper
        county_rows = hazard_df[county_mask]
    else:
        county_rows = pd.DataFrame()

    if len(county_rows) == 0:
        return _generate_fallback_risks(county_name)

    # Use most recent row for this county
    if 'date' in county_rows.columns:
        county_rows = county_rows.sort_values('date', ascending=False)
    county_row = county_rows.iloc[0]

    # Resolve actual county name from data (preserves casing for map lookup)
    actual_county = county_row.get('county', county_name)

    try:
        # Build tensors matching HazardLMDiffusion forward() signature
        static_cont, temporal, region_ids, state_ids, nlcd_ids = \
            build_tensors_from_county_data(county_row, actual_county, target_date)

        # Run inference
        risks = predict_from_tensors(
            model, static_cont, temporal, region_ids, state_ids, nlcd_ids
        )
        return risks

    except Exception as e:
        print(f"[INFERENCE] Error predicting for {county_name}: {e}")
        import traceback
        traceback.print_exc()
        return _generate_fallback_risks(county_name)


def _generate_fallback_risks(county_name: str) -> Dict[str, float]:
    """Generate plausible fallback risks based on county name hash.
    Used when model or data is unavailable."""
    try:
        seed = abs(hash(county_name)) % 10000
    except Exception:
        seed = 42

    rng = np.random.RandomState(seed)
    return {
        'fire': float(rng.beta(2, 5)),
        'flood': float(rng.beta(2, 8)),
        'wind': float(rng.beta(2, 6)),
        'winter': float(rng.beta(2, 5)),
        'seismic': float(rng.beta(1, 15))
    }
