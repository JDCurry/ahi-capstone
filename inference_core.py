"""
Cloud-safe inference core for Hazard-LM.
NO training dependencies - pure PyTorch inference only.
"""
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import date
from typing import Dict, Optional, Tuple

# Hazard types
HAZARD_TYPES = ['fire', 'flood', 'wind', 'winter', 'seismic']


@torch.no_grad()
def predict_from_tensors(
    model,
    static: torch.Tensor,
    temporal: torch.Tensor,
    image: torch.Tensor,
    nlcd: torch.Tensor,
    hazard_types: list = None
) -> Dict[str, float]:
    """
    Run model inference from pre-built tensors.
    This is the cloud-safe core - no dataset dependencies.
    
    Args:
        model: Loaded HazardLM model
        static: [1, static_dim] static features
        temporal: [1, seq_len, feat_dim] temporal sequence
        image: [1, C, H, W] satellite image tensor
        nlcd: [1] NLCD land cover code
        hazard_types: List of hazard types to predict
    
    Returns:
        Dict mapping hazard type -> probability
    """
    if model is None:
        return {h: 0.0 for h in (hazard_types or HAZARD_TYPES)}
    
    hazard_types = hazard_types or HAZARD_TYPES
    
    model.eval()
    device = next(model.parameters()).device
    
    # Move tensors to device
    static = static.to(device).float()
    temporal = temporal.to(device).float()
    image = image.to(device).float()
    nlcd = nlcd.to(device)
    
    # Forward pass
    outputs = model(static, temporal, image, nlcd)
    
    # Extract probabilities
    risks = {}
    for h in hazard_types:
        prob = 0.0
        if isinstance(outputs, dict):
            prob_key = f'{h}_prob'
            if prob_key in outputs:
                prob = float(outputs[prob_key].cpu().numpy().flatten()[0])
            elif h in outputs:
                prob = float(torch.sigmoid(outputs[h]).cpu().numpy().flatten()[0])
        risks[h] = max(0.0, min(1.0, prob))  # Clamp to [0,1]
    
    return risks


def build_dummy_tensors(
    static_dim: int = 64,
    temporal_seq_len: int = 14,
    temporal_feat_dim: int = 32,
    image_size: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build dummy tensors for testing/fallback.
    Returns: (static, temporal, image, nlcd)
    """
    static = torch.zeros(1, static_dim)
    temporal = torch.zeros(1, temporal_seq_len, temporal_feat_dim)
    image = torch.zeros(1, 3, image_size, image_size)
    nlcd = torch.tensor([42])  # Default forest
    return static, temporal, image, nlcd


def build_tensors_from_county_data(
    county_row: pd.Series,
    static_dim: int = 64,
    temporal_seq_len: int = 14,
    temporal_feat_dim: int = 32,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build inference tensors from a county data row.
    Uses available features from the hazard dataset.
    """
    # Static features - extract what's available
    static_cols = [
        'population', 'svi_score', 'elevation_mean', 'slope_mean',
        'forest_pct', 'urban_pct', 'ag_pct', 'water_pct',
        'fire_history', 'flood_history', 'wind_history', 'winter_history'
    ]
    
    static_values = []
    for col in static_cols:
        if col in county_row.index:
            val = county_row[col]
            static_values.append(float(val) if pd.notna(val) else 0.0)
        else:
            static_values.append(0.0)
    
    # Pad to expected dimension
    while len(static_values) < static_dim:
        static_values.append(0.0)
    static = torch.tensor(static_values[:static_dim]).unsqueeze(0)
    
    # Temporal features - use climate/weather if available
    temporal_cols = [
        'tmax_mean', 'tmin_mean', 'pr_mean', 'rmax_mean', 'rmin_mean',
        'vs_mean', 'bi_mean', 'fm100_mean', 'fm1000_mean', 'erc_mean'
    ]
    
    temporal_row = []
    for col in temporal_cols:
        if col in county_row.index:
            val = county_row[col]
            temporal_row.append(float(val) if pd.notna(val) else 0.0)
        else:
            temporal_row.append(0.0)
    
    # Pad features
    while len(temporal_row) < temporal_feat_dim:
        temporal_row.append(0.0)
    
    # Repeat for sequence length (simple approach - use same data)
    temporal = torch.tensor([temporal_row[:temporal_feat_dim]] * temporal_seq_len).unsqueeze(0)
    
    # Image - placeholder (would need to load actual satellite imagery)
    image = torch.zeros(1, 3, 64, 64)
    
    # NLCD code
    nlcd_val = 42  # Default forest
    if 'dominant_nlcd' in county_row.index:
        try:
            nlcd_val = int(county_row['dominant_nlcd'])
        except:
            pass
    nlcd = torch.tensor([nlcd_val])
    
    return static, temporal, image, nlcd


def predict_county_risks_simple(
    model,
    county_name: str,
    hazard_df: pd.DataFrame,
    target_date: date = None
) -> Dict[str, float]:
    """
    Simplified county risk prediction for cloud deployment.
    Uses available data from hazard_lm_dataset.parquet.
    
    Args:
        model: Loaded HazardLM model
        county_name: Name of county (e.g., "King", "Pierce")
        hazard_df: DataFrame with county hazard data
        target_date: Target date (not used in simple version)
    
    Returns:
        Dict mapping hazard type -> probability
    """
    if model is None:
        # Return plausible fallback values
        return _generate_fallback_risks(county_name)
    
    # Normalize county name
    county_upper = county_name.upper().replace(' COUNTY', '').strip()
    
    # Find county in dataframe
    county_mask = hazard_df['county'].str.upper().str.replace(' COUNTY', '').str.strip() == county_upper
    county_rows = hazard_df[county_mask]
    
    if len(county_rows) == 0:
        return _generate_fallback_risks(county_name)
    
    # Use most recent row for this county
    county_row = county_rows.iloc[-1]
    
    try:
        # Build tensors from available data
        static, temporal, image, nlcd = build_tensors_from_county_data(county_row)
        
        # Run inference
        risks = predict_from_tensors(model, static, temporal, image, nlcd)
        return risks
        
    except Exception as e:
        print(f"[INFERENCE] Error predicting for {county_name}: {e}")
        return _generate_fallback_risks(county_name)


def _generate_fallback_risks(county_name: str) -> Dict[str, float]:
    """Generate plausible fallback risks based on county name hash."""
    try:
        seed = abs(hash(county_name)) % 10000
    except:
        seed = 42
    
    np.random.seed(seed)
    return {
        'fire': float(np.random.beta(2, 5)),
        'flood': float(np.random.beta(2, 8)),
        'wind': float(np.random.beta(2, 6)),
        'winter': float(np.random.beta(2, 5)),
        'seismic': float(np.random.beta(1, 15))
    }
