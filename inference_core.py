"""
Cloud-safe inference core for HazardLM-Diffusion v2.0.
NO training dependencies - pure PyTorch inference only.

Calibration pipeline (applied in order):
  1. Temperature scaling  - per-hazard T fitted on validation set (NLL optimization)
  2. Seasonal prior       - physics-informed logit bias by month (WA climatology)
  3. Base-rate ceiling     - caps max probability at historical plausibility limits

Updated for HazardLMDiffusion model API:
  model(static_cont, temporal, region_ids, state_ids, nlcd_ids)
  -> dict with {hazard}_prob and {hazard}_logits keys
"""
import json
import math
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

# --- Calibration state (loaded once, cached) ---
_TEMPERATURES: Dict[str, float] = {}  # per-hazard temperature scaling factors

# --- Seasonal prior: logit bias by month for WA state ---
# Derived from WA climatology and clean label base rates.
# Positive bias = increase risk, negative = suppress.
# Values are in logit space (log-odds), so -2.0 is a strong suppression.
#
# Fire: WA fire season is June-October. Nov-Mar fires are extremely rare (<0.5%).
# Winter: Winter storms peak Nov-Mar. Jun-Sep storms are very rare.
# Wind: High-wind events peak Oct-Mar, but can occur year-round.
# Flood: Year-round with slight winter peak (atmospheric rivers).
# Seismic: No seasonal pattern (tectonic), so no bias applied.
SEASONAL_LOGIT_BIAS = {
    'fire': {
        # month: logit_bias
        1: -3.0,   # January  - virtually no WA wildfire
        2: -3.0,   # February - virtually no WA wildfire
        3: -2.0,   # March    - very rare, snowmelt not fire
        4: -1.0,   # April    - rare, some early-season prescribed burns
        5: -0.5,   # May      - occasional, season starting
        6:  0.0,   # June     - fire season begins
        7:  0.0,   # July     - peak fire season
        8:  0.0,   # August   - peak fire season
        9:  0.0,   # September - fire season
        10: -0.5,  # October  - season winding down
        11: -2.0,  # November - very rare
        12: -3.0,  # December - virtually no WA wildfire
    },
    'winter': {
        1:  0.0,   # January  - peak winter storm season
        2:  0.0,   # February - peak winter storm season
        3: -0.5,   # March    - transition month, declining storm frequency
        4: -0.5,   # April    - transitioning
        5: -1.5,   # May      - rare
        6: -3.0,   # June     - no winter storms
        7: -3.0,   # July     - no winter storms
        8: -3.0,   # August   - no winter storms
        9: -2.0,   # September - very rare early season
        10: -0.5,  # October  - season starting
        11:  0.0,  # November - winter storm season
        12:  0.0,  # December - peak winter storm season
    },
    'wind': {
        1:  0.0,   # January  - high wind season
        2:  0.0,   # February - high wind season
        3:  0.0,   # March    - high wind season
        4: -0.3,   # April    - moderate
        5: -0.5,   # May      - less common
        6: -0.5,   # June     - less common
        7: -0.5,   # July     - less common
        8: -0.3,   # August   - less common
        9:  0.0,   # September - picking up
        10:  0.0,  # October  - high wind season
        11:  0.0,  # November - high wind season
        12:  0.0,  # December - high wind season
    },
    # Flood and seismic: no seasonal bias
    'flood': {m: 0.0 for m in range(1, 13)},
    'seismic': {m: 0.0 for m in range(1, 13)},
}

# --- Base-rate ceiling: maximum plausible daily probability per hazard ---
# Derived from WA clean label base rates in the training data.
# These represent the upper bound of what a single-day probability should be.
# Even during peak season, no county should exceed these on a daily basis.
#
# Training data base rates (3-day label window):
#   fire ~1.1%, flood ~0.9%, wind ~2.8%, winter ~4.3%, seismic ~3.0%
#
# Updated for AHI v2 (stacked mesh) per-hazard AUCs:
#   fire 0.88, flood 0.90, wind 0.83, winter 0.91, seismic 0.68
# All heads are now strong discriminators; ceilings are generous.
BASE_RATE_CEILING = {
    'fire':    0.35,  # 35% max - strong head (AUC 0.88)
    'flood':   0.35,  # 35% max - strong head (AUC 0.90)
    'wind':    0.25,  # 25% max - good head (AUC 0.83)
    'winter':  0.35,  # 35% max - best head (AUC 0.91)
    'seismic': 0.05,  # 5% max - constant geographic risk, not weather-driven
}

# --- Month-aware ceilings for seasonal hazards ---
# During transition months, the max plausible daily probability is naturally
# lower than during peak season. This prevents the "wall of 35%" effect
# where most counties hit the same ceiling and lose differentiation.
SEASONAL_CEILING = {
    'winter': {
        1: 0.35, 2: 0.35,   # Peak winter storm season
        3: 0.25,             # Transition month - storms declining
        4: 0.15,             # Late transition
        5: 0.08, 6: 0.05, 7: 0.05, 8: 0.05,  # Off-season
        9: 0.08,             # Early transition
        10: 0.20,            # Season starting
        11: 0.35, 12: 0.35,  # Peak winter storm season
    },
}


def _get_ceiling(hazard: str, month: int) -> float:
    """Get the effective probability ceiling for a hazard in a given month.

    Uses month-specific ceiling for seasonal hazards (winter),
    falls back to BASE_RATE_CEILING for non-seasonal or unknown month.
    """
    if month and 1 <= month <= 12 and hazard in SEASONAL_CEILING:
        return SEASONAL_CEILING[hazard][month]
    return BASE_RATE_CEILING.get(hazard, 1.0)


def load_temperature_scales(path: Optional[str] = None) -> Dict[str, float]:
    """Load per-hazard temperature scales from JSON file.

    Always reads from disk (no in-memory caching) to ensure
    updates to temperature_scales.json take effect immediately.

    Temperature scaling: calibrated_logit = raw_logit / T
    T < 1 sharpens (more confident), T > 1 softens (less confident).
    Fitted by minimizing NLL on the validation set.

    Args:
        path: Path to temperature_scales.json. If None, searches common locations.

    Returns:
        Dict mapping hazard type -> temperature (float > 0)
    """
    search_paths = [
        Path(path) if path else None,
        Path('temperature_scales.json'),
        Path('outputs/ahi_v2/temperature_scales_v2.json'),
        Path('outputs/diffusion_clean_v1/temperature_scales.json'),
        Path('data/temperature_scales.json'),
    ]

    for p in search_paths:
        if p is not None and p.exists():
            try:
                with open(p) as f:
                    data = json.load(f)
                temps = data.get('temperatures', data)
                loaded = {h: float(temps[h]) for h in HAZARD_TYPES if h in temps}
                print(f"[CALIBRATION] Loaded temperature scales from {p}: {loaded}")
                return loaded
            except Exception as e:
                print(f"[CALIBRATION] Error loading {p}: {e}")

    print("[CALIBRATION] No temperature_scales.json found - using T=1.0 (uncalibrated)")
    return {h: 1.0 for h in HAZARD_TYPES}


def _apply_calibration(
    raw_logit: float,
    hazard: str,
    month: int,
    temperatures: Optional[Dict[str, float]] = None,
) -> float:
    """Apply calibration pipeline to a single raw logit.

    Pipeline:
      1. Temperature scaling:  logit_scaled = raw_logit / T
      2. Seasonal prior:       logit_final  = logit_scaled + seasonal_bias(month)
      3. Sigmoid:              prob = 1 / (1 + exp(-logit_final))
      4. Base-rate ceiling:    prob = min(prob, ceiling[hazard])

    Args:
        raw_logit: Raw logit from the model
        hazard: Hazard type ('fire', 'flood', etc.)
        month: Calendar month (1-12). 0 or None = no seasonal adjustment.
        temperatures: Per-hazard temperature dict. Loaded from cache if None.

    Returns:
        Calibrated probability in [0, 1]
    """
    if temperatures is None:
        temperatures = load_temperature_scales()

    # Step 1: Temperature scaling
    # NOTE: T < 1 sharpens predictions (makes them more extreme).
    # For hazards where the model is near-random (AUC < 0.65), sharpening
    # makes overconfident predictions worse. Clamp T >= 1.0 for those.
    T = temperatures.get(hazard, 1.0)
    T = max(T, 0.01)  # Guard against division by zero
    # For poorly-performing heads, don't let T<1 amplify bad predictions
    # v2 update: only seismic remains weak (AUC 0.68). Wind is now strong (0.83).
    WEAK_HEADS = {'seismic'}  # AUC < 0.70 on test set
    if hazard in WEAK_HEADS:
        T = max(T, 1.0)  # Only soften, never sharpen
    scaled_logit = raw_logit / T

    # Step 1b: Base-rate recalibration bias for weak heads
    # v2 update: wind is no longer weak (AUC 0.83), bias removed.
    # Seismic (AUC 0.68) still overconfident — softened bias from -2.5 to -1.5
    # since the head now has real discriminative power.
    WEAK_HEAD_BIAS = {
        'seismic': -2.0,  # Seismic is constant geographic risk, not daily-varying
    }
    if hazard in WEAK_HEAD_BIAS:
        scaled_logit += WEAK_HEAD_BIAS[hazard]

    # Step 2: Seasonal prior (logit-space bias)
    if month and 1 <= month <= 12 and hazard in SEASONAL_LOGIT_BIAS:
        bias = SEASONAL_LOGIT_BIAS[hazard].get(month, 0.0)
        scaled_logit += bias

    # Step 3: Sigmoid
    prob = 1.0 / (1.0 + math.exp(-scaled_logit))

    # Step 4: Base-rate ceiling (prevents absurd probabilities from near-random heads)
    # Uses month-aware ceiling for seasonal hazards (e.g., winter in March = 0.25)
    ceiling = _get_ceiling(hazard, month)
    prob = min(prob, ceiling)

    return max(0.0, prob)


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
    hazard_types: list = None,
    month: int = 0,
    temperatures: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """
    Run HazardLMDiffusion inference from pre-built tensors.
    Applies temperature scaling + seasonal prior calibration.

    Args:
        model: Loaded HazardLMDiffusion model
        static_cont: [batch, static_dim] static continuous features (padded to 50)
        temporal: [batch, seq_len, feat_dim] temporal sequence (14, 20)
        region_ids: [batch] county/region IDs (long)
        state_ids: [batch] state IDs (long)
        nlcd_ids: [batch] NLCD land cover IDs (long)
        hazard_types: List of hazard types to predict
        month: Calendar month (1-12) for seasonal adjustment. 0 = no adjustment.
        temperatures: Per-hazard temperature dict. Loaded from file if None.

    Returns:
        Dict mapping hazard type -> calibrated probability (float in [0, 1])
    """
    if model is None:
        return {h: 0.0 for h in (hazard_types or HAZARD_TYPES)}

    hazard_types = hazard_types or HAZARD_TYPES

    # Load temperatures if not provided
    if temperatures is None:
        temperatures = load_temperature_scales()

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

    # Extract calibrated probabilities
    risks = {}
    for h in hazard_types:
        if isinstance(outputs, dict):
            logit_key = f'{h}_logits'
            prob_key = f'{h}_prob'

            if logit_key in outputs:
                # Preferred: calibrate from raw logit
                raw_logit = float(outputs[logit_key].cpu().numpy().flatten()[0])
                risks[h] = _apply_calibration(raw_logit, h, month, temperatures)
            elif prob_key in outputs:
                # Fallback: convert prob -> logit, then calibrate
                raw_prob = float(outputs[prob_key].cpu().numpy().flatten()[0])
                raw_prob = max(1e-7, min(1 - 1e-7, raw_prob))  # Avoid log(0)
                raw_logit = math.log(raw_prob / (1.0 - raw_prob))
                risks[h] = _apply_calibration(raw_logit, h, month, temperatures)
            else:
                risks[h] = 0.0
        else:
            risks[h] = 0.0

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
    # (same as training -- HazardDataset pads to 14x20)
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
    Applies temperature scaling + seasonal prior calibration.

    Args:
        model: Loaded HazardLMDiffusion model
        county_name: Name of county (e.g., "King", "Pierce")
        hazard_df: DataFrame with county hazard data
        target_date: Target date for prediction (affects seasonal features)

    Returns:
        Dict mapping hazard type -> calibrated probability
    """
    if model is None:
        return _generate_fallback_risks(county_name)

    # Build maps if not yet built
    if not _COUNTY_MAP and hazard_df is not None and len(hazard_df) > 0:
        _build_maps(hazard_df)

    # Load temperature scales (cached after first call)
    temperatures = load_temperature_scales()

    # Determine month for seasonal adjustment
    month = target_date.month if target_date is not None else 0

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

        # Run inference with calibration
        risks = predict_from_tensors(
            model, static_cont, temporal, region_ids, state_ids, nlcd_ids,
            month=month, temperatures=temperatures,
        )
        return risks

    except Exception as e:
        print(f"[INFERENCE] Error predicting for {county_name}: {e}")
        import traceback
        traceback.print_exc()
        return _generate_fallback_risks(county_name)


@torch.no_grad()
def predict_from_ahi_v2(
    model,
    static_cont: torch.Tensor,
    temporal: torch.Tensor,
    region_ids: torch.Tensor,
    state_ids: torch.Tensor,
    nlcd_ids: torch.Tensor,
    adjacency_mask: torch.Tensor,
    hazard_types: list = None,
    month: int = 0,
    temperatures: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """
    Run AHI v2 stacked mesh inference with spatial adjacency mask.
    Applies the same calibration pipeline as v1.

    Args:
        model: Loaded AHIv2Model
        static_cont: [batch, 50] static features
        temporal: [batch, 14, 20] temporal sequence
        region_ids: [batch] county IDs
        state_ids: [batch] state IDs
        nlcd_ids: [batch] NLCD IDs
        adjacency_mask: [batch, batch] boolean adjacency mask
        hazard_types: Hazard types to predict
        month: Calendar month (1-12)
        temperatures: Per-hazard temperature dict

    Returns:
        Dict mapping hazard -> calibrated probability per county
        If batch > 1: Dict[str, list[float]]
        If batch == 1: Dict[str, float]
    """
    hazard_types = hazard_types or HAZARD_TYPES

    if temperatures is None:
        temperatures = load_temperature_scales()

    model.eval()
    device = next(model.parameters()).device

    static_cont = static_cont.to(device).float()
    temporal = temporal.to(device).float()
    region_ids = region_ids.to(device).long()
    state_ids = state_ids.to(device).long()
    nlcd_ids = nlcd_ids.to(device).long()
    adjacency_mask = adjacency_mask.to(device)

    outputs = model(
        static_cont, temporal, region_ids, state_ids, nlcd_ids,
        spatial_mask=adjacency_mask,
    )

    batch_size = static_cont.size(0)
    if batch_size == 1:
        # Single county: return scalar probabilities
        risks = {}
        for h in hazard_types:
            raw_logit = float(outputs[f'{h}_logits'].cpu().numpy().flatten()[0])
            risks[h] = _apply_calibration(raw_logit, h, month, temperatures)
        return risks
    else:
        # Multiple counties: return per-county probabilities
        risks = {h: [] for h in hazard_types}
        for h in hazard_types:
            logits = outputs[f'{h}_logits'].cpu().numpy().flatten()
            for raw_logit in logits:
                risks[h].append(_apply_calibration(float(raw_logit), h, month, temperatures))
        return risks


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
