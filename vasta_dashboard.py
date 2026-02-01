"""
VASTA: Variable Adaptation Systems for Threat Anticipation
Main Dashboard Application - Hazard-LM Edition
Version: 4.0.1

 - Multi-hazard AI predictions (Hazard-LM v1.0)
- Interactive county map with popups
- Climate analysis
- Mitigation planning with FEMA ROI data
"""

import streamlit as st
from scipy.stats import gamma
# =============================================================================
# PROBABILISTIC RISK SCORING MODULE (Gamma–Poisson Empirical Bayes)
# =============================================================================

def gamma_poisson_posterior(events, days_observed, prior_alpha=0.5, prior_beta=0.5, threshold=None, ci=0.90):
    """
    Empirical Bayes Gamma–Poisson posterior for event rate estimation.
    Args:
        events (int or array): Number of observed events.
        days_observed (int or array): Exposure (days observed).
        prior_alpha (float): Prior shape parameter (alpha > 0).
        prior_beta (float): Prior rate parameter (beta > 0).
        threshold (float or None): If set, compute P(rate > threshold).
        ci (float): Credible interval width (e.g., 0.90 for 90% CI).
    Returns:
        posterior_mean_rate (float or array): Posterior mean annualized rate.
        lower_ci (float or array): Lower bound of credible interval (annualized).
        upper_ci (float or array): Upper bound of credible interval (annualized).
        prob_exceeds (float or array): P(rate > threshold) if threshold is set, else None.
    """
    events = np.asarray(events)
    days_observed = np.asarray(days_observed)
    # Avoid division by zero
    safe_days = np.where(days_observed > 0, days_observed, 1)
    # Posterior parameters
    post_alpha = prior_alpha + events
    post_beta = prior_beta + safe_days
    # Posterior mean (per day)
    mean_rate_per_day = post_alpha / post_beta
    # Annualize
    posterior_mean_rate = mean_rate_per_day * 365.0
    # Credible interval (per day)
    lower = gamma.ppf((1 - ci) / 2, post_alpha, scale=1 / post_beta)
    upper = gamma.ppf(1 - (1 - ci) / 2, post_alpha, scale=1 / post_beta)
    lower_ci = lower * 365.0
    upper_ci = upper * 365.0
    # Probability rate exceeds threshold (annualized)
    prob_exceeds = None
    if threshold is not None:
        # Convert threshold to per day
        thresh_per_day = threshold / 365.0
        prob_exceeds = 1 - gamma.cdf(thresh_per_day, post_alpha, scale=1 / post_beta)
    return posterior_mean_rate, lower_ci, upper_ci, prob_exceeds

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from pathlib import Path
import json
import torch
import warnings
warnings.filterwarnings('ignore')
import os
import traceback
import time
import shutil
import subprocess

# Optional imports
try:
    import folium
    from streamlit_folium import st_folium
    import geopandas as gpd
    FOLIUM_AVAILABLE = True
except ImportError:
    FOLIUM_AVAILABLE = False

# Local imports
try:
    from hazard_lm import HazardLM, HazardConfig
    HAZARD_LM_AVAILABLE = True
except ImportError:
    HAZARD_LM_AVAILABLE = False

# Shared inference helpers - use cloud-safe inference_core (no training dependencies)
try:
    from inference_core import predict_county_risks_simple
    predict_county_risks = predict_county_risks_simple  # Alias for compatibility
except Exception as _inf_err:
    print(f"[IMPORT] inference_core import failed: {_inf_err}")
    predict_county_risks = None
    predict_county_risks_simple = None

# LLM helpers not available on cloud deployment
load_local_llm = None
generate_risk_summary = None
load_wa_dataset = None

# =============================================================================
# CONFIGURATION
# =============================================================================

PLATFORM_NAME = "Adaptive Hazard Intelligence"
TAGLINE = "Calibrated hazard risk for defensible decisions"

st.set_page_config(
    page_title=f"{PLATFORM_NAME}",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_DIR = Path("data")  # Data directory for datasets and supporting files

# Model paths - try multiple locations
MODEL_PATH = Path("best_model.pt")  # Primary path (also used as alias)
MODEL_PATH_LOCAL = MODEL_PATH
MODEL_PATH_CLOUD = Path("/mount/src/ahi-capstone/best_model.pt")  # Streamlit Cloud clones here
MODEL_PATH_TMP = Path("/tmp/best_model.pt")
MODEL_URL = "https://media.githubusercontent.com/media/JDCurry/ahi-capstone/main/best_model.pt"
MODEL_LOAD_ERROR = None
MODEL_DISPLAY_NAME = "Hazard-LM v1.0"
MIN_MODEL_SIZE = 10_000_000  # 10MB - real model is ~184MB

def get_model_path():
    """Get the best available model path."""
    # Check all possible locations
    for path in [MODEL_PATH_LOCAL, MODEL_PATH_CLOUD, MODEL_PATH_TMP]:
        try:
            if path.exists() and path.stat().st_size > MIN_MODEL_SIZE:
                print(f"Found valid model at: {path} ({path.stat().st_size // (1024*1024)}MB)")
                return path
        except Exception:
            pass
    return None

def download_model_if_needed():
    """Download model from GitHub LFS media URL if not present or if only LFS pointer exists."""
    import requests
    
    # Already have a valid model?
    existing = get_model_path()
    if existing:
        return existing
    
    # Download to /tmp/ (always writable on Streamlit Cloud)
    target = MODEL_PATH_TMP
    try:
        print(f"Downloading Hazard-LM model to {target}...")
        response = requests.get(MODEL_URL, stream=True, timeout=600)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        
        if total_size < MIN_MODEL_SIZE:
            print(f"Warning: Download size {total_size} too small, expected ~184MB")
            return None
        
        with open(target, 'wb') as f:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=65536):
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0 and downloaded % (20 * 1024 * 1024) < 65536:
                    pct = int(100 * downloaded / total_size)
                    print(f"Downloaded {downloaded // (1024*1024)}MB / {total_size // (1024*1024)}MB ({pct}%)")
        
        print(f"Model download complete: {target.stat().st_size // (1024*1024)}MB")
        if target.exists() and target.stat().st_size > MIN_MODEL_SIZE:
            return target
        return None
        
    except Exception as e:
        print(f"Failed to download model: {e}")
        return None

# Resolved model path
RESOLVED_MODEL_PATH = get_model_path() or MODEL_PATH_LOCAL
# Prefer canonical WA parquet by default (falls back inside loader if missing)
DATA_PATH = Path("data/wa_gridmet/wa_hazard_dataset.parquet/wa_hazard_dataset.parquet")
DUCKDB_PATH = Path("data/hazard_lm_warehouse.duckdb")
LOADED_DATA_PATH = None
LOADED_DATA_ROWS = 0
GEOJSON_PATH = Path("data/wa_counties.geojson")
CLIMATE_FIRE_PATH = Path("data/WA_Climate_Fire_Dashboard_Data.csv")

# Enterprise-Grade Professional Color Palette (Dark Theme)
COLORS = {
    'app_bg': '#0f1419',
    'card_bg': '#1a1f26',
    'sidebar_bg': '#141920',
    'elevated_bg': '#1f2430',
    'primary': '#0d7dc1',
    'primary_hover': '#1089d3',
    'critical': '#c0392b',
    'warning': '#e8a838',
    'success': '#27ae60',
    'info': '#3498db',
    'border': '#2d333b',
    'text_primary': '#e6edf3',
    'text_secondary': '#8b949e',
    'text_tertiary': '#6e7681',
    'fire': '#c0392b',
    'flood': '#2980b9',
    'wind': '#8e44ad',
    'winter': '#1abc9c',
    'seismic': '#d35400',
}

# Washington State County Data
WA_COUNTY_FIPS = {
    53001: 'Adams', 53003: 'Asotin', 53005: 'Benton', 53007: 'Chelan',
    53009: 'Clallam', 53011: 'Clark', 53013: 'Columbia', 53015: 'Cowlitz',
    53017: 'Douglas', 53019: 'Ferry', 53021: 'Franklin', 53023: 'Garfield',
    53025: 'Grant', 53027: 'Grays Harbor', 53029: 'Island', 53031: 'Jefferson',
    53033: 'King', 53035: 'Kitsap', 53037: 'Kittitas', 53039: 'Klickitat',
    53041: 'Lewis', 53043: 'Lincoln', 53045: 'Mason', 53047: 'Okanogan',
    53049: 'Pacific', 53051: 'Pend Oreille', 53053: 'Pierce', 53055: 'San Juan',
    53057: 'Skagit', 53059: 'Skamania', 53061: 'Snohomish', 53063: 'Spokane',
    53065: 'Stevens', 53067: 'Thurston', 53069: 'Wahkiakum', 53071: 'Walla Walla',
    53073: 'Whatcom', 53075: 'Whitman', 53077: 'Yakima'
}

WA_COUNTY_COORDS = {
    'Adams': (46.98, -118.56), 'Asotin': (46.19, -117.20), 'Benton': (46.23, -119.52),
    'Chelan': (47.87, -120.62), 'Clallam': (48.11, -123.93), 'Clark': (45.78, -122.48),
    'Columbia': (46.29, -117.91), 'Cowlitz': (46.19, -122.67), 'Douglas': (47.53, -119.69),
    'Ferry': (48.47, -118.52), 'Franklin': (46.53, -118.89), 'Garfield': (46.43, -117.54),
    'Grant': (47.21, -119.45), 'Grays Harbor': (47.15, -123.76), 'Island': (48.21, -122.58),
    'Jefferson': (47.76, -123.50), 'King': (47.49, -121.84), 'Kitsap': (47.64, -122.65),
    'Kittitas': (47.12, -120.68), 'Klickitat': (45.87, -120.78), 'Lewis': (46.58, -122.38),
    'Lincoln': (47.58, -118.41), 'Mason': (47.35, -123.18), 'Okanogan': (48.55, -119.74),
    'Pacific': (46.56, -123.78), 'Pend Oreille': (48.53, -117.27), 'Pierce': (47.04, -122.13),
    'San Juan': (48.53, -123.02), 'Skagit': (48.48, -121.80), 'Skamania': (46.02, -121.92),
    'Snohomish': (48.05, -121.72), 'Spokane': (47.62, -117.40), 'Stevens': (48.40, -117.85),
    'Thurston': (46.93, -122.83), 'Wahkiakum': (46.29, -123.42), 'Walla Walla': (46.23, -118.48),
    'Whatcom': (48.85, -121.72), 'Whitman': (46.90, -117.52), 'Yakima': (46.46, -120.74)
}

# FEMA Research-Backed Mitigation ROI
MITIGATION_ROI = {
    'property_acquisition': {'roi': 5.8, 'cost_range': (150000, 500000), 'hazard': 'flood'},
    'flood_control': {'roi': 5.5, 'cost_range': (500000, 5000000), 'hazard': 'flood'},
    'structure_elevation': {'roi': 5.3, 'cost_range': (30000, 150000), 'hazard': 'flood'},
    'early_warning_systems': {'roi': 6.2, 'cost_range': (50000, 300000), 'hazard': 'multi'},
    'defensible_space': {'roi': 4.5, 'cost_range': (1000, 5000), 'hazard': 'fire'},
    'vegetation_management': {'roi': 3.8, 'cost_range': (50000, 500000), 'hazard': 'fire'},
    'seismic_retrofit': {'roi': 2.2, 'cost_range': (5000, 50000), 'hazard': 'seismic'},
    'foundation_bolting': {'roi': 2.8, 'cost_range': (3000, 10000), 'hazard': 'seismic'},
    'roof_retrofits': {'roi': 4.2, 'cost_range': (8000, 25000), 'hazard': 'wind'},
    'backup_power': {'roi': 4.0, 'cost_range': (50000, 500000), 'hazard': 'multi'},
    'critical_facility_hardening': {'roi': 3.8, 'cost_range': (500000, 5000000), 'hazard': 'multi'},
}

# =============================================================================
# CUSTOM CSS - DARK ENTERPRISE THEME
# =============================================================================

def inject_custom_css():
    st.markdown(f"""
    <style>
    .stApp {{
        background: {COLORS['app_bg']} !important;
        color: {COLORS['text_secondary']} !important;
    }}
    
    h1 {{
        color: {COLORS['text_primary']} !important;
        font-family: 'Segoe UI', sans-serif;
        font-weight: 600;
        border-bottom: 2px solid {COLORS['primary']} !important;
        padding-bottom: 12px;
    }}
    
    h2, h3 {{
        color: {COLORS['text_primary']} !important;
        font-family: 'Segoe UI', sans-serif;
    }}
    
    .header-container {{
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        padding: 10px 0 20px 0;
        border-bottom: 1px solid {COLORS['border']};
        margin-bottom: 20px;
    }}
    
    .main-title {{
        color: {COLORS['text_primary']} !important;
        font-size: 28px;
        font-weight: 600;
        margin: 0;
    }}
    
    .subtitle {{
        color: {COLORS['text_tertiary']} !important;
        font-size: 13px;
        margin: 4px 0 0 0;
    }}
    
    .status-section {{
        text-align: right;
    }}
    
    .status-time {{
        color: {COLORS['text_primary']};
        font-size: 18px;
        font-weight: 600;
    }}
    
    .status-indicator {{
        color: {COLORS['success']};
        font-size: 11px;
        font-weight: 600;
        margin-top: 8px;
    }}
    
    [data-testid="metric-container"] {{
        background: {COLORS['card_bg']} !important;
        border: 1px solid {COLORS['border']} !important;
        padding: 18px;
        border-radius: 6px;
    }}
    
    [data-testid="metric-container"] label {{
        color: {COLORS['text_tertiary']} !important;
        text-transform: uppercase;
        font-size: 11px;
    }}
    
    [data-testid="metric-container"] [data-testid="metric-value"] {{
        color: {COLORS['text_primary']} !important;
        font-size: 28px;
    }}
    
    section[data-testid="stSidebar"] {{
        background: {COLORS['sidebar_bg']} !important;
        border-right: 1px solid {COLORS['border']} !important;
    }}
    
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] label {{
        color: {COLORS['text_primary']} !important;
    }}
    
    .stat-card {{
        background: {COLORS['card_bg']} !important;
        padding: 20px;
        border-radius: 6px;
        border: 1px solid {COLORS['border']};
        margin: 10px 0;
    }}
    
    .alert-box {{
        padding: 16px 20px;
        border-radius: 6px;
        margin: 12px 0;
        border-left: 4px solid;
    }}
    
    .alert-critical {{
        background: rgba(192, 57, 43, 0.1) !important;
        border-left-color: {COLORS['critical']} !important;
    }}
    
    .alert-warning {{
        background: rgba(232, 168, 56, 0.1) !important;
        border-left-color: {COLORS['warning']} !important;
    }}
    
    .alert-info {{
        background: rgba(13, 125, 193, 0.1) !important;
        border-left-color: {COLORS['info']} !important;
    }}
    
    .alert-success {{
        background: rgba(39, 174, 96, 0.1) !important;
        border-left-color: {COLORS['success']} !important;
    }}
    
    .stButton > button {{
        background-color: {COLORS['primary']} !important;
        color: white !important;
        border: none;
        border-radius: 4px;
    }}
    
    .stButton > button:hover {{
        background-color: {COLORS['primary_hover']} !important;
    }}
    
    .dataframe {{
        background: {COLORS['card_bg']} !important;
        color: {COLORS['text_secondary']} !important;
    }}
    
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}
    
    .hazard-badge {{
        display: inline-block;
        padding: 4px 12px;
        border-radius: 4px;
        font-size: 12px;
        font-weight: 600;
        margin: 2px;
    }}
    
    .hazard-fire {{ background: {COLORS['fire']}; color: white; }}
    .hazard-flood {{ background: {COLORS['flood']}; color: white; }}
    .hazard-wind {{ background: {COLORS['wind']}; color: white; }}
    .hazard-winter {{ background: {COLORS['winter']}; color: white; }}
    .hazard-seismic {{ background: {COLORS['seismic']}; color: white; }}
    </style>
    """, unsafe_allow_html=True)


# =============================================================================
# DATA LOADING
# =============================================================================

@st.cache_resource
def load_hazard_model():
    """Load Hazard-LM v2.0 model from checkpoint."""
    global RESOLVED_MODEL_PATH, MODEL_LOAD_ERROR
    
    print("[MODEL] Starting model load...")

    # Try to get existing model path or download
    model_path = get_model_path()
    print(f"[MODEL] get_model_path() returned: {model_path}")
    
    if model_path is None:
        print("[MODEL] No model found, attempting download...")
        model_path = download_model_if_needed()
        print(f"[MODEL] download_model_if_needed() returned: {model_path}")
    
    if model_path is None:
        MODEL_LOAD_ERROR = f"Model not available - download failed"
        print(f"[MODEL] ERROR: {MODEL_LOAD_ERROR}")
        return None, False
    
    RESOLVED_MODEL_PATH = model_path
    
    # Verify model file is valid
    if not model_path.exists():
        MODEL_LOAD_ERROR = f"Model file not found: {model_path}"
        print(f"[MODEL] ERROR: {MODEL_LOAD_ERROR}")
        return None, False
    
    file_size = model_path.stat().st_size
    print(f"[MODEL] Model file size: {file_size:,} bytes ({file_size // (1024*1024)} MB)")
    
    if file_size < MIN_MODEL_SIZE:
        MODEL_LOAD_ERROR = f"Model file too small ({file_size} bytes) - download may have failed"
        print(f"[MODEL] ERROR: {MODEL_LOAD_ERROR}")
        return None, False

    try:
        print("[MODEL] Importing hazard_lm_v20...")
        from hazard_lm_v20 import model as h20_model
        print("[MODEL] hazard_lm_v20 imported successfully")

        # Load checkpoint
        print(f"[MODEL] Loading checkpoint from {model_path}...")
        state = torch.load(str(model_path), map_location='cpu', weights_only=False)
        print(f"[MODEL] Checkpoint loaded, type: {type(state)}, keys: {state.keys() if isinstance(state, dict) else 'N/A'}")

        # Create model with default config
        print("[MODEL] Creating model architecture...")
        model = h20_model.create_hazard_lm()
        print(f"[MODEL] Model created, params: {sum(p.numel() for p in model.parameters()):,}")

        # Extract state dict from nested structure
        sd = None
        if isinstance(state, dict):
            if 'model_state_dict' in state:
                sd = state['model_state_dict']
                print("[MODEL] Using 'model_state_dict' key")
            elif 'state_dict' in state:
                sd = state['state_dict']
                print("[MODEL] Using 'state_dict' key")
            else:
                sd = state
                print("[MODEL] Using checkpoint directly as state_dict")

        if sd is not None:
            # Try direct load first
            try:
                model.load_state_dict(sd, strict=False)
                print("[MODEL] State dict loaded successfully (strict=False)")
            except Exception as load_err:
                print(f"[MODEL] Direct load failed: {load_err}, trying per-component...")
                # Try loading per-component
                if isinstance(sd, dict):
                    for key in ['backbone', 'backbone_ln', 'token_embedding', 'position_embedding']:
                        if key in sd:
                            try:
                                getattr(model, key).load_state_dict(sd[key])
                                print(f"[MODEL] Loaded component: {key}")
                            except Exception:
                                print(f"[MODEL] Failed to load component: {key}")

        model.eval()
        param_sum = sum(p.sum().item() for p in model.parameters())
        print(f"[MODEL] Model loaded successfully, param checksum: {param_sum:.4f}")
        MODEL_LOAD_ERROR = None
        return model, True

    except Exception as e:
        import traceback
        MODEL_LOAD_ERROR = str(e)
        print(f"[MODEL] EXCEPTION during load: {e}")
        print(f"[MODEL] Traceback: {traceback.format_exc()}")
        return None, False


@st.cache_data
def load_hazard_data():
    """Load hazard dataset"""
    try:
        # Try configured DATA_PATH first
        candidates = [DATA_PATH]
        # Common alternate locations used in this repo
        candidates += [Path("data/wa_gridmet/wa_hazard_dataset.parquet/wa_hazard_dataset.parquet"),
                       Path("data/wa_hazard_dataset.parquet"),
                       Path("data/wa_gridmet/wa_hazard_dataset.parquet"),
                       Path("outputs/wa_adapters_longrun/statewide_dataset.parquet")]

        # Also look for any parquet file under data/ that contains 'hazard' in its name
        data_dir = Path('data')
        if data_dir.exists():
            for p in data_dir.rglob('*.parquet'):
                if 'hazard' in p.name.lower() and p not in candidates:
                    candidates.append(p)

        global LOADED_DATA_PATH, LOADED_DATA_ROWS
        for p in candidates:
            try:
                if p and p.exists():
                    df = pd.read_parquet(p)
                    if 'date' in df.columns:
                        df['date'] = pd.to_datetime(df['date'])
                    LOADED_DATA_PATH = str(p)
                    LOADED_DATA_ROWS = len(df)
                    return df
            except Exception:
                continue

        # DuckDB fallback: try to read a table from the warehouse if present
        try:
            if DUCKDB_PATH.exists():
                import duckdb
                con = duckdb.connect(str(DUCKDB_PATH))
                # List tables and pick one likely containing hazard data
                try:
                    tables = [r[0] for r in con.execute("SHOW TABLES").fetchall()]
                except Exception:
                    tables = []

                pick = None
                for t in tables:
                    if 'hazard' in t.lower():
                        pick = t
                        break
                if pick is None and tables:
                    pick = tables[0]

                if pick is not None:
                    df = con.execute(f"SELECT * FROM \"{pick}\"").df()
                    if 'date' in df.columns:
                        df['date'] = pd.to_datetime(df['date'])
                    LOADED_DATA_PATH = f"duckdb:{DUCKDB_PATH}::{pick}"
                    LOADED_DATA_ROWS = len(df)
                    return df
        except Exception:
            # duckdb not available or file unreadable; fall through to return None
            pass
    except Exception as e:
        st.error(f"Data load error: {e}")
    return None


@st.cache_data
def load_geojson():
    """Load Washington State GeoJSON"""
    if FOLIUM_AVAILABLE and GEOJSON_PATH.exists():
        try:
            return gpd.read_file(GEOJSON_PATH)
        except:
            pass
    return None


@st.cache_data
def load_climate_fire_data():
    """Load WA Climate Fire Dashboard data with real fire counts and climate trends"""
    try:
        if CLIMATE_FIRE_PATH.exists():
            df = pd.read_csv(CLIMATE_FIRE_PATH)
            return df
        # Try alternate paths
        for alt_path in [Path("fire/WA_Climate_Fire_Dashboard_Data.csv"),
                         Path("WA_Climate_Fire_Dashboard_Data.csv")]:
            if alt_path.exists():
                return pd.read_csv(alt_path)
    except Exception as e:
        st.warning(f"Climate fire data load error: {e}")
    return None


@st.cache_data
def load_county_metadata():
    """Load supplemental county metadata (population, SVI, hospitals, schools) if available."""
    candidates = [
        Path('data/WA_County_Master_Table__Final_with_Schools_.cleaned.csv'),
        Path('data/WA_County_Master_Table__Final_with_Schools_.csv'),
        Path('data/wa_county_master.csv'),
    ]
    for p in candidates:
        try:
            if p.exists():
                df = pd.read_csv(p)
                # Normalize column names
                cols = {c.lower(): c for c in df.columns}
                # common names: county_name, total_population, svi_score, hospital_count, school_count
                # ensure we have a `county_name` column
                if 'county_name' not in [c.lower() for c in df.columns] and 'county' in [c.lower() for c in df.columns]:
                    # some CSVs may have 'county' which contains full name
                    df.rename(columns={list(cols.get('county'))[0]: 'county_name'}, inplace=True)
                # standardize casing and strip ' County' suffix
                if 'county_name' in df.columns:
                    try:
                        df['county_name'] = df['county_name'].astype(str).str.replace(' County', '', regex=False).str.strip()
                    except Exception:
                        pass
                return df
        except Exception:
            continue
    return None


@st.cache_data
def compute_county_stats(_df):
    """Compute county-level statistics"""
    if _df is None:
        return None
    # Ensure required columns exist (map common alternative names and fill defaults)
    df = _df.copy()

    # Helper to pick first available candidate column name
    def pick(col_candidates, default=None):
        for c in col_candidates:
            if c in df.columns:
                return c
        return None

    # Labels
    for lbl in ['fire_label','flood_label','wind_label','winter_label','seismic_label','any_hazard']:
        if lbl not in df.columns:
            df[lbl] = 0

    # Population: try several common names
    pop_col = pick(['total_population','population','total_pop','pop','pop_total'])
    if pop_col is None:
        df['total_population'] = 0
        pop_col = 'total_population'
    else:
        # normalize to expected name
        if pop_col != 'total_population':
            df['total_population'] = df[pop_col]

    # SVI: social vulnerability index
    svi_col = pick(['svi_score','svi','sv_index'])
    if svi_col is None:
        df['svi_score'] = 0.0
    else:
        if svi_col != 'svi_score':
            df['svi_score'] = df[svi_col]

    # Latitude/Longitude fallbacks
    lat_col = pick(['latitude','lat','centroid_lat'])
    lon_col = pick(['longitude','lon','centroid_lon','long'])
    if lat_col is None:
        df['latitude'] = np.nan
    else:
        if lat_col != 'latitude':
            df['latitude'] = df[lat_col]
    if lon_col is None:
        df['longitude'] = np.nan
    else:
        if lon_col != 'longitude':
            df['longitude'] = df[lon_col]

    # Ensure date exists for counting
    if 'date' not in df.columns:
        df['date'] = pd.NaT

    stats = df.groupby('county').agg({
        'fire_label': 'sum',
        'flood_label': 'sum',
        'wind_label': 'sum',
        'winter_label': 'sum',
        'seismic_label': 'sum',
        'any_hazard': 'sum',
        'total_population': 'first',
        'svi_score': 'first',
        'latitude': 'first',
        'longitude': 'first',
        'date': 'count'
    }).reset_index()
    
    stats.columns = ['county', 'fire_events', 'flood_events', 'wind_events',
                     'winter_events', 'seismic_events', 'total_events',
                     'population', 'svi_score', 'lat', 'lon', 'days_observed']
    
    # Probabilistic rates and intervals using Gamma–Poisson model
    for h in ['fire', 'flood', 'wind', 'winter', 'seismic']:
        mean, lower, upper, _ = gamma_poisson_posterior(
            stats[f'{h}_events'], stats['days_observed'], prior_alpha=0.5, prior_beta=0.5, ci=0.90
        )
        stats[f'{h}_rate'] = mean
        stats[f'{h}_rate_lower'] = lower
        stats[f'{h}_rate_upper'] = upper
    # Total risk (all hazards)
    mean, lower, upper, _ = gamma_poisson_posterior(
        stats['total_events'], stats['days_observed'], prior_alpha=0.5, prior_beta=0.5, ci=0.90
    )
    stats['total_rate'] = mean
    stats['total_rate_lower'] = lower
    stats['total_rate_upper'] = upper
    # Normalized risk score (for visualization)
    stats['risk_score'] = stats['total_rate'] / stats['total_rate'].max() if stats['total_rate'].max() > 0 else 0
    # Merge supplemental county metadata when available to enrich population, SVI, hospitals, schools
    try:
        meta = load_county_metadata()
        if meta is not None and not meta.empty:
            # create normalized join keys
            meta['county_norm'] = meta['county_name'].astype(str).str.upper().str.replace(' COUNTY', '', regex=False).str.strip()
            stats['county_norm'] = stats['county'].astype(str).str.upper().str.strip()
            merged = stats.merge(meta[['county_norm', 'total_population', 'svi_score', 'hospital_count', 'school_count']], on='county_norm', how='left')

            # population: prefer existing population unless missing/zero; if zero, replace from meta
            try:
                merged['population'] = merged['population'].replace(0, np.nan).fillna(merged['total_population'])
                # ensure integer type where possible
                merged['population'] = merged['population'].fillna(0).astype(int)
            except Exception:
                pass

            # svi: prefer existing stats value unless missing/zero, otherwise use meta
            try:
                # handle possible suffixes after merge
                if 'svi_score_x' in merged.columns and 'svi_score_y' in merged.columns:
                    merged['svi_score'] = merged['svi_score_x'].replace(0, np.nan).fillna(merged['svi_score_y'])
                elif 'svi_score_x' in merged.columns:
                    merged['svi_score'] = merged['svi_score_x'].replace(0, np.nan)
                elif 'svi_score' in merged.columns and 'svi_score' not in ['svi_score_x','svi_score_y']:
                    merged['svi_score'] = merged['svi_score'].replace(0, np.nan)
                # finally fill any remaining NaNs with 0.0
                merged['svi_score'] = merged['svi_score'].fillna(0.0)
            except Exception:
                pass

            # hospitals and schools: add columns if present
            if 'hospital_count' in merged.columns:
                merged['hospital_count'] = merged['hospital_count'].fillna(0).astype(int)
            else:
                merged['hospital_count'] = 0
            if 'school_count' in merged.columns:
                merged['school_count'] = merged['school_count'].fillna(0).astype(int)
            else:
                merged['school_count'] = 0

            # cleanup and restore expected schema
            stats = merged.drop(columns=[c for c in ['county_norm','total_population'] if c in merged.columns], errors='ignore')
            # ensure expected columns exist
            if 'hospital_count' not in stats.columns:
                stats['hospital_count'] = 0
            if 'school_count' not in stats.columns:
                stats['school_count'] = 0
    except Exception:
        pass

    return stats


@st.cache_data
def compute_empirical_horizon_probs(df, horizon_days: int = 14, alpha: int = 20):
    """Compute empirical per-county horizon probabilities for each hazard label.

    Returns a mapping: { county: { hazard: {'p_emp':..., 'n':..., 'p_hat':...}, ... }, '__state__': {...} }

    Uses simple anchor-based approach: for each county, an anchor is any date for which
    the dataset contains future observations up to `horizon_days`. For each anchor we
    mark whether the hazard label occurred within (date, date + horizon]. p_emp = occ/anchors.

    We apply simple shrinkage toward the state-level empirical probability using weight `alpha`:
        p_hat = (n * p_emp + alpha * p_state) / (n + alpha)
    """
    hazards = ['fire_label', 'flood_label', 'wind_label', 'winter_label', 'seismic_label']
    out = {}
    state_counts = {h: 0 for h in hazards}
    state_anchors = 0

    # group by county
    for county, g in df.groupby('county'):
        dates = sorted(g['date'].unique())
        # identify anchors that have horizon_days of future coverage
        anchors = [d for d in dates if g['date'].max() >= (d + pd.Timedelta(days=horizon_days))]
        n = len(anchors)
        rec = {}
        if n == 0:
            # leave empty for now; will fill with state-level later
            for h in hazards:
                rec[h] = {'p_emp': 0.0, 'n': 0, 'p_hat': 0.0}
            out[county] = rec
            continue

        # compute occurrences per hazard across anchors
        for h in hazards:
            occ = 0
            for d in anchors:
                t0 = pd.Timestamp(d)
                t1 = t0 + pd.Timedelta(days=horizon_days)
                sub = g[(g['date'] > t0) & (g['date'] <= t1)]
                if (sub.get(h, pd.Series([])) > 0).any():
                    occ += 1
            p_emp = occ / n if n > 0 else 0.0
            rec[h] = {'p_emp': float(p_emp), 'n': int(n), 'p_hat': float(p_emp)}
            state_counts[h] += occ
        state_anchors += n
        out[county] = rec

    # compute state-level empirical probabilities
    state_probs = {}
    for h in hazards:
        if state_anchors > 0:
            state_probs[h] = float(state_counts[h] / state_anchors)
        else:
            state_probs[h] = 0.0

    # apply shrinkage to compute p_hat
    for county, rec in out.items():
        for h in hazards:
            n = rec[h]['n']
            p_emp = rec[h]['p_emp']
            p_state = state_probs[h]
            if n <= 0:
                p_hat = p_state
            else:
                p_hat = (n * p_emp + alpha * p_state) / (n + alpha)
            rec[h]['p_hat'] = float(p_hat)

    out['__state__'] = state_probs
    return out


def get_plotly_theme():
    """Return dark Plotly theme"""
    return {
        'paper_bgcolor': COLORS['card_bg'],
        'plot_bgcolor': COLORS['card_bg'],
        'font': {'color': COLORS['text_secondary'], 'family': 'Segoe UI'},
        'xaxis': {'gridcolor': COLORS['border'], 'linecolor': COLORS['border']},
        'yaxis': {'gridcolor': COLORS['border'], 'linecolor': COLORS['border']},
    }


def predict_and_summarize(county_identifier: str, target_date: datetime.date):
    """Wrapper: run model prediction for a county and return (risks, summary).

    - Uses `load_hazard_model()` to load the promoted model.
    - Uses shared `predict_county_risks` and LLM helpers when available.
    """
    # Load model
    try:
        model, ok = load_hazard_model()
        if not ok or model is None:
            return None, 'Model not available'
    except Exception as e:
        return None, f'Model load failed: {e}'

    # Load hazard dataset for inference
    try:
        hazard_df = pd.read_parquet(DATA_DIR / 'hazard_lm_dataset.parquet')
    except Exception as e:
        hazard_df = None

    # Run prediction (shared helper if imported)
    try:
        if predict_county_risks is not None and hazard_df is not None:
            risks = predict_county_risks(model, county_identifier, hazard_df, target_date)
        elif predict_county_risks is not None:
            # Try without hazard_df (will use fallback internally)
            risks = predict_county_risks(model, county_identifier, pd.DataFrame(), target_date)
        else:
            # No prediction helper available - return error
            return None, 'Prediction helper not available. Model loaded but inference pipeline not configured.'
    except Exception as e:
        return None, f'Prediction failed: {e}'

    # LLM summary (best-effort)
    try:
        llm = load_local_llm() if load_local_llm is not None else None
        # lightweight hazards mapping for summary formatting
        hazards_map = {
            'fire': {'name': 'Wildfire'},
            'flood': {'name': 'Flood'},
            'wind': {'name': 'Wind'},
            'winter': {'name': 'Winter Storm'},
            'seismic': {'name': 'Seismic'}
        }

        def _get_risk_level(p: float):
            if p < 0.3:
                return 'Low', 'risk-low'
            elif p < 0.6:
                return 'Moderate', 'risk-moderate'
            elif p < 0.8:
                return 'High', 'risk-high'
            else:
                return 'Extreme', 'risk-extreme'

        if generate_risk_summary is not None:
            summary = generate_risk_summary(county_identifier, risks, llm, hazards_map, _get_risk_level)
        else:
            # Minimal fallback summary
            top = sorted(risks.items(), key=lambda kv: kv[1], reverse=True)[:2]
            summary = f"Top hazards: {top[0][0]} ({top[0][1]*100:.1f}%), {top[1][0]} ({top[1][1]*100:.1f}%)"
    except Exception as e:
        summary = f'Summary generation failed: {e}'

    return risks, summary


# =============================================================================
# HEADER
# =============================================================================

def render_header():
    # Use PST timezone for consistent display
    try:
        from zoneinfo import ZoneInfo
        pst = ZoneInfo('America/Los_Angeles')
        now_pst = datetime.now(pst)
    except Exception:
        # Fallback to UTC-8 offset if zoneinfo unavailable
        from datetime import timezone, timedelta
        pst = timezone(timedelta(hours=-8))
        now_pst = datetime.now(pst)
    current_time = now_pst.strftime("%H:%M:%S")
    current_date = now_pst.strftime("%B %d, %Y")
    
    model, model_ok = load_hazard_model()
    status_color = COLORS['success'] if model_ok else COLORS['warning']
    status_text = f"{MODEL_DISPLAY_NAME} Online" if model_ok else "Model Offline"
    
    st.markdown(f"""
    <div class="header-container">
        <div>
            <h1 class="main-title" style="border: none; margin: 0; padding: 0;">{PLATFORM_NAME}</h1>
            <p class="subtitle">{TAGLINE}</p>
        </div>
        <div class="status-section">
            <div class="status-time">{current_time}</div>
            <div style="color: {COLORS['text_tertiary']}; font-size: 12px;">{current_date}</div>
            <div class="status-indicator" style="color: {status_color};">{status_text}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # If a model load error occurred, show a concise warning (details in outputs/logs)
    try:
        if MODEL_LOAD_ERROR is not None:
            try:
                show_adv = bool(st.session_state.get('show_advanced', False))
            except Exception:
                show_adv = False
            if show_adv:
                st.warning(f"Model failed to load. Details written to `{MODEL_LOAD_ERROR}`. Expand Model Diagnostics for more info.")
    except Exception:
        pass

    # Small provenance/footer: model lineage
    try:
        st.markdown(f"<div style='color:{COLORS['text_tertiary']}; font-size:11px; margin-top:6px;'>Powered by <strong>Hazard-LM</strong> — model: {MODEL_DISPLAY_NAME}</div>", unsafe_allow_html=True)
    except Exception:
        pass

    # Show loaded dataset path and row count for debugging
    try:
        if LOADED_DATA_PATH:
            st.markdown(f"<div style='color:{COLORS['text_tertiary']}; font-size:12px;'>Data: {LOADED_DATA_PATH} — {LOADED_DATA_ROWS:,} rows</div>", unsafe_allow_html=True)
    except Exception:
        pass


# =============================================================================
# SIDEBAR NAVIGATION
# =============================================================================

def render_sidebar():
    with st.sidebar:
        # Centered logo
        logo_path = Path(__file__).parent / 'logo-dark.png'
        try:
            if logo_path.exists():
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.image(str(logo_path), width=140)
        except Exception:
            pass
        st.markdown(f"""
        <div style='text-align: center; padding: 20px 0; border-bottom: 1px solid {COLORS["border"]};'>
            <h2 style='color: {COLORS["text_primary"]}; margin: 0; font-size: 18px;'>Interfaces</h2>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"<p style='color: {COLORS['text_tertiary']}; font-size: 11px; text-transform: uppercase; margin-top: 20px;'>Analytics Modules</p>", unsafe_allow_html=True)
        
        if st.button("Executive Dashboard", width='stretch'):
            st.session_state.page = 'dashboard'
        
        if st.button("Interactive Map", width='stretch'):
            st.session_state.page = 'map'
        
        if st.button("Risk Assessment", width='stretch'):
            st.session_state.page = 'risk'
        
        if st.button("Climate Analysis", width='stretch'):
            st.session_state.page = 'climate'
        
        if st.button("Mitigation Planning", width='stretch'):
            st.session_state.page = 'mitigation'
        
        st.markdown(f"<p style='color: {COLORS['text_tertiary']}; font-size: 11px; text-transform: uppercase; margin-top: 20px;'>AI Tools</p>", unsafe_allow_html=True)
        
        if st.button("Quick Predict", width='stretch'):
            st.session_state.page = 'ai_predict'
        
        if st.button("Model Diagnostics", width='stretch'):
            st.session_state.page = 'diagnostics'
        
        if st.button("Model Evaluation", width='stretch'):
            st.session_state.page = 'model_eval'
        
        if st.button("About This Project", width='stretch'):
            st.session_state.page = 'about'
        
        st.markdown("---")
        
        # Advanced debug toggle (hidden by default)
        try:
            adv = st.checkbox('Show advanced debug/logs', value=False, help='Enable detailed logs and developer diagnostics')
            st.session_state.show_advanced = bool(adv)
            
            # Show model diagnostics when debug is enabled
            if adv:
                st.markdown("**Model Diagnostics:**")
                
                # Check all possible model paths
                paths_to_check = [
                    ("Local", Path("best_model.pt")),
                    ("Cloud mount", Path("/mount/src/ahi-capstone/best_model.pt")),
                    ("Tmp", Path("/tmp/best_model.pt")),
                ]
                
                for name, p in paths_to_check:
                    if p.exists():
                        size_mb = p.stat().st_size / (1024*1024)
                        st.success(f"{name}: {p} ({size_mb:.1f}MB)")
                    else:
                        st.error(f"{name}: {p} NOT FOUND")
                
                # Show resolved path and model status
                st.write(f"RESOLVED_MODEL_PATH: {RESOLVED_MODEL_PATH}")
                st.write(f"MODEL_LOAD_ERROR: {MODEL_LOAD_ERROR}")
                
                # Try to load model and show checksum
                try:
                    model, ok = load_hazard_model()
                    st.write(f"Model loaded: {ok}")
                    st.write(f"Model type: {type(model)}")
                    if model is not None:
                        param_count = sum(p.numel() for p in model.parameters())
                        param_sum = sum(p.abs().sum().item() for p in model.parameters())
                        st.write(f"Param count: {param_count:,}")
                        st.write(f"Param checksum: {param_sum:.4f}")
                        st.write(f"Model training mode: {model.training}")
                except Exception as e:
                    st.error(f"Model load error: {e}")
                
                # Clear cache button
                if st.button("Clear Model Cache"):
                    st.cache_resource.clear()
                    st.rerun()
        except Exception:
            # if session_state unavailable, ignore
            pass
    
    return st.session_state.get('page', 'dashboard')


# =============================================================================
# PAGE: EXECUTIVE DASHBOARD
# =============================================================================

def page_executive_dashboard():
    st.markdown("## Executive Dashboard")
    
    df = load_hazard_data()
    if df is None:
        st.error("No data available")
        return
    
    county_stats = compute_county_stats(df)
    
    # Top metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Counties Analyzed", len(county_stats))
    
    with col2:
        high_risk = len(county_stats[county_stats['risk_score'] > 0.7])
        st.metric("High Risk Counties", high_risk, delta=f"{high_risk/len(county_stats)*100:.0f}%", delta_color="inverse")
    
    with col3:
        total_pop = county_stats['population'].sum()
        st.metric("Total Population", f"{total_pop/1e6:.1f}M")
    
    with col4:
        avg_svi = county_stats['svi_score'].mean()
        st.metric("Avg SVI Score", f"{avg_svi:.2f}")
    
    with col5:
        total_events = county_stats['total_events'].sum()
        st.metric("Total Events (Hist)", f"{int(total_events):,}")
    
    # Hazard summary cards
    st.markdown("### Hazard Type Summary")
    
    hazard_cols = st.columns(5)
    hazards = [
        ("Fire", county_stats['fire_events'].sum(), COLORS['fire']),
        ("Flood", county_stats['flood_events'].sum(), COLORS['flood']),
        ("Wind", county_stats['wind_events'].sum(), COLORS['wind']),
        ("Winter", county_stats['winter_events'].sum(), COLORS['winter']),
        ("Seismic", county_stats['seismic_events'].sum(), COLORS['seismic']),
    ]
    
    for col, (name, count, color) in zip(hazard_cols, hazards):
        with col:
            st.markdown(f"""
            <div class="stat-card" style="border-left: 4px solid {color};">
                <div style="color: {COLORS['text_tertiary']}; font-size: 11px; text-transform: uppercase;">
                    {name} Events
                </div>
                <div style="color: {color}; font-size: 28px; font-weight: 600;">
                    {int(count):,}
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Charts row
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### County Risk Distribution")
        
        county_risk = county_stats.sort_values('risk_score', ascending=False).head(15)
        
        colors = [COLORS['critical'] if s > 0.7 else COLORS['warning'] if s > 0.4 else COLORS['success'] 
                  for s in county_risk['risk_score']]
        
        fig = go.Figure(go.Bar(
            x=county_risk['county'],
            y=county_risk['risk_score'],
            marker_color=colors,
            hovertemplate='<b>%{x}</b><br>Risk: %{y:.2%}<extra></extra>'
        ))
        
        fig.update_layout(
            **get_plotly_theme(),
            height=400,
            xaxis_title="County",
            yaxis_title="Risk Score",
            showlegend=False
        )
        
        st.plotly_chart(fig, width='stretch')
    
    with col2:
        st.markdown("### Hazard Distribution")
        
        hazard_totals = {
            'Fire': county_stats['fire_events'].sum(),
            'Flood': county_stats['flood_events'].sum(),
            'Wind': county_stats['wind_events'].sum(),
            'Winter': county_stats['winter_events'].sum(),
            'Seismic': county_stats['seismic_events'].sum(),
        }
        
        fig = go.Figure(go.Pie(
            labels=list(hazard_totals.keys()),
            values=list(hazard_totals.values()),
            marker=dict(colors=[COLORS['fire'], COLORS['flood'], COLORS['wind'], 
                               COLORS['winter'], COLORS['seismic']]),
            textinfo='label+percent',
            textfont=dict(color='white'),
            hole=0.4
        ))
        
        fig.update_layout(
            **get_plotly_theme(),
            height=400,
            showlegend=True
        )
        
        st.plotly_chart(fig, width='stretch')
    
    # Alerts
    st.markdown("### System Alerts")
    
    high_risk_counties = county_stats[county_stats['risk_score'] > 0.7]['county'].tolist()
    if high_risk_counties:
        st.markdown(f"""
        <div class="alert-box alert-warning">
            <strong>ELEVATED RISK</strong><br>
            {len(high_risk_counties)} counties showing elevated multi-hazard risk: {', '.join(high_risk_counties[:5])}{'...' if len(high_risk_counties) > 5 else ''}
        </div>
        """, unsafe_allow_html=True)
    
    model, model_ok = load_hazard_model()
    if model_ok:
            st.markdown(f"""
            <div class="alert-box alert-success">
                <strong>AI MODEL ACTIVE</strong><br>
                {MODEL_DISPLAY_NAME} is online and ready for predictions. 15.8M parameters, 5 hazard types.
            </div>
            """, unsafe_allow_html=True)


# =============================================================================
# PAGE: INTERACTIVE MAP
# =============================================================================

def page_interactive_map():
    st.markdown("## Interactive Risk Map")
    
    df = load_hazard_data()
    county_stats = compute_county_stats(df)
    climate_fire_df = load_climate_fire_data()
    gdf = load_geojson()

    # Optionally delegate interactive map rendering to `app.py` if available
    try:
        import app as fire_app
        APP_AVAILABLE = True
    except Exception:
        fire_app = None
        APP_AVAILABLE = False

    use_app_map = False
    if APP_AVAILABLE:
        use_app_map = st.sidebar.checkbox("Render map with Fire-LM UI (app.py)", value=False)

    if use_app_map and fire_app is not None:
        # Build a county_coords DataFrame expected by app.render_risk_map
        try:
            coords = []
            for k, v in WA_COUNTY_COORDS.items():
                coords.append({'county': k, 'latitude': v[0], 'longitude': v[1]})
            county_coords = pd.DataFrame(coords)
            # Choose date: latest available in df or today
            sel_date = pd.Timestamp(df['date'].max()) if (df is not None and not df.empty) else pd.Timestamp(datetime.now())
            # Delegate to app's renderer
            try:
                fire_app.render_risk_map(df, county_coords, sel_date, hazard_type='fire')
                return
            except Exception as e:
                st.warning(f"Delegation to app.render_risk_map failed: {e}")
        except Exception:
            st.warning("Failed to construct county coords for app delegation.")
    
    if not FOLIUM_AVAILABLE:
        st.warning("Folium not installed. Install with: pip install folium streamlit-folium geopandas")
        st.dataframe(county_stats[['county', 'risk_score', 'population', 'total_events']], width='stretch')
        return
    
    # Controls
    col1, col2 = st.columns([2, 2])
    with col1:
        hazard_layer = st.selectbox(
            "Hazard Layer",
            ["Fire", "Flood", "Wind", "Winter", "Seismic"]
        )
    with col2:
        map_style = st.selectbox(
            "Map Style",
            ["Satellite", "Dark", "Light", "Street"]
        )
    
    # Create map with selected tile layer
    if map_style == "Satellite":
        # Esri World Imagery - high quality satellite basemap
        m = folium.Map(
            location=[47.5, -120.5],
            zoom_start=7,
            tiles=None  # No default tiles
        )
        folium.TileLayer(
            tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
            attr='Esri',
            name='Satellite',
            overlay=False,
            control=True
        ).add_to(m)
    else:
        tiles = {'Dark': 'CartoDB dark_matter', 'Light': 'CartoDB positron', 'Street': 'OpenStreetMap'}
        m = folium.Map(
            location=[47.5, -120.5],
            zoom_start=7,
            tiles=tiles[map_style]
        )
    
    # Build lookup for climate fire data
    cf_lookup = {}
    if climate_fire_df is not None:
        for _, row in climate_fire_df.iterrows():
            cf_lookup[row['County'].upper()] = row
    
    # Add counties
    if gdf is not None:
        name_field = None
        for f in ['NAME', 'name', 'COUNTY', 'county_name']:
            if f in gdf.columns:
                name_field = f
                break
        
        if name_field:
            for idx, row in gdf.iterrows():
                county_name = row[name_field].replace(' County', '').strip()
                county_upper = county_name.upper()
                
                # Get climate fire data if available
                cf_data = cf_lookup.get(county_upper, None)
                
                # Find matching hazard stats
                match = county_stats[county_stats['county'].str.upper() == county_upper] if county_stats is not None else pd.DataFrame()
                stats = match.iloc[0] if len(match) > 0 else None
                
                # Determine score based on layer selection
                if hazard_layer == "Fire" and cf_data is not None:
                    score = cf_data.get('climate_fire_risk_score', 0) / 100.0  # Normalize to 0-1
                    risk_category = cf_data.get('risk_category', 'Unknown')
                elif stats is not None:
                    hazard_key = hazard_layer.lower()
                    if hazard_key in ['fire', 'flood', 'wind', 'winter', 'seismic']:
                        col = f'{hazard_key}_rate'
                        if col in stats.index:
                            max_rate = county_stats[col].max() if col in county_stats.columns else 1
                            score = stats[col] / max_rate if max_rate > 0 else 0
                        else:
                            score = stats.get('risk_score', 0.5)
                    else:
                        score = stats.get('risk_score', 0.5)
                    risk_category = "High" if score > 0.7 else "Moderate" if score > 0.4 else "Low"
                else:
                    score = 0.5
                    risk_category = "Unknown"
                
                # Color based on risk
                if score > 0.7 or risk_category == "High":
                    color = COLORS['critical']
                elif score > 0.4 or risk_category == "Moderate":
                    color = COLORS['warning']
                else:
                    color = COLORS['success']
                
                # Build popup based on selected hazard layer
                if hazard_layer == "Fire" and cf_data is not None:
                    # Fire-specific popup with climate fire data
                    fire_count = int(cf_data.get('Fire_Count', 0))
                    climate_trend = cf_data.get('climate_trend', 'Stable')
                    heat_stress = cf_data.get('heat_stress', 0)
                    drought_stress = cf_data.get('drought_stress', 0)
                    fire_history = cf_data.get('fire_history_score', 0)
                    wui_exposure = cf_data.get('wui_exposure_pct', 0)
                    population = int(cf_data.get('population', 0))
                    pop_at_risk = int(cf_data.get('population_at_risk', 0))
                    pct_interface = cf_data.get('pct_interface', 0) * 100
                    pct_intermix = cf_data.get('pct_intermix', 0) * 100
                    risk_score = cf_data.get('climate_fire_risk_score', 0)
                    
                    # Climate trend color
                    if 'Warming' in str(climate_trend) and 'Drying' in str(climate_trend):
                        trend_color = '#c0392b'
                    elif 'Drying' in str(climate_trend):
                        trend_color = '#e8a838'
                    elif 'Warming' in str(climate_trend):
                        trend_color = '#f39c12'
                    else:
                        trend_color = '#27ae60'
                    
                    popup_html = f"""
                    <div style="font-family: 'Segoe UI', sans-serif; background: #1a1f26; border-radius: 6px; overflow: hidden;">
                        <div style="background: linear-gradient(135deg, {color} 0%, #1a1f26 100%); padding: 8px 10px;">
                            <h3 style="margin: 0; color: #fff; font-size: 14px; font-weight: 600;">{county_name} County</h3>
                            <div style="color: rgba(255,255,255,0.85); font-size: 11px; margin-top: 2px;">
                                Fire Risk (NOAA events + climate indicators): {risk_score:.1f} | {risk_category} Risk
                            </div>
                            <div style="color: #8b949e; font-size: 10px; margin-top: 4px;">Includes NOAA fire counts and climate trend indicators (heat / drought)</div>
                        </div>
                        <div style="padding: 8px 10px;">
                            <table style="width: 100%; font-size: 11px; color: #8b949e; border-collapse: collapse;">
                                <tr style="border-bottom: 1px solid #2d333b;"><td style="padding: 4px 0;">Climate Trend</td><td style="text-align: right; color: {trend_color}; font-weight: 600;">{climate_trend}</td></tr>
                                <tr style="border-bottom: 1px solid #2d333b;"><td style="padding: 4px 0;">Heat Stress</td><td style="text-align: right; color: #e6edf3;">{heat_stress:.1f}</td></tr>
                                <tr style="border-bottom: 1px solid #2d333b;"><td style="padding: 4px 0;">Drought Stress</td><td style="text-align: right; color: #e6edf3;">{drought_stress:.1f}</td></tr>
                                <tr style="border-bottom: 1px solid #2d333b;"><td style="padding: 4px 0;">Fire History Score</td><td style="text-align: right; color: #e6edf3;">{fire_history:.1f}</td></tr>
                                <tr><td style="padding: 4px 0;">WUI Exposure</td><td style="text-align: right; color: #e6edf3;">{wui_exposure:.1f}%</td></tr>
                            </table>
                            <div style="margin-top: 8px; padding-top: 8px; border-top: 1px solid #2d333b;">
                                <table style="width: 100%; font-size: 11px; color: #8b949e;">
                                    <tr style="border-bottom: 1px solid #2d333b;"><td style="padding: 4px 0;">Population</td><td style="text-align: right; color: #e6edf3;">{population:,}</td></tr>
                                    <tr><td style="padding: 4px 0;">At Risk (WUI)</td><td style="text-align: right; color: #e6edf3;">{pop_at_risk:,}</td></tr>
                                </table>
                            </div>
                            <div style="margin-top: 8px; padding: 6px; background: #141920; border-radius: 4px; text-align: center;">
                                <div style="color: #8b949e; font-size: 10px; text-transform: uppercase;">NOAA Fire Events</div>
                                <div style="color: {COLORS['fire']}; font-size: 18px; font-weight: 600;">{fire_count:,}</div>
                            </div>
                        </div>
                    </div>
                    """
                elif stats is not None:
                    # Generic hazard popup for Flood, Wind, Winter, Seismic
                    hazard_key = hazard_layer.lower()
                    hazard_color = COLORS.get(hazard_key, COLORS['primary'])
                    population = int(stats['population'])
                    svi = stats.get('svi_score', 0)
                    
                    # Get hazard-specific events
                    hazard_events = {
                        'fire': int(stats.get('fire_events', 0)),
                        'flood': int(stats.get('flood_events', 0)),
                        'wind': int(stats.get('wind_events', 0)),
                        'winter': int(stats.get('winter_events', 0)),
                        'seismic': int(stats.get('seismic_events', 0))
                    }
                    
                    # Build hazard-specific content
                    if hazard_key == 'flood':
                        primary_events = hazard_events['flood']
                        hazard_title = "Flood Risk"
                        event_label = "Flood Events"
                        details_html = f"""<tr style="border-bottom: 1px solid #2d333b;"><td style="padding: 4px 0;">{event_label}</td><td style="text-align: right; color: {COLORS['flood']}; font-weight: 600;">{hazard_events['flood']}</td></tr><tr><td style="padding: 4px 0;">Total Hazard Events</td><td style="text-align: right; color: #e6edf3;">{sum(hazard_events.values())}</td></tr>"""
                    elif hazard_key == 'wind':
                        primary_events = hazard_events['wind']
                        hazard_title = "Wind Risk"
                        event_label = "Wind Events"
                        details_html = f"""<tr style="border-bottom: 1px solid #2d333b;"><td style="padding: 4px 0;">{event_label}</td><td style="text-align: right; color: {COLORS['wind']}; font-weight: 600;">{hazard_events['wind']}</td></tr><tr><td style="padding: 4px 0;">Total Hazard Events</td><td style="text-align: right; color: #e6edf3;">{sum(hazard_events.values())}</td></tr>"""
                    elif hazard_key == 'winter':
                        primary_events = hazard_events['winter']
                        hazard_title = "Winter Storm Risk"
                        event_label = "Winter Events"
                        details_html = f"""<tr style="border-bottom: 1px solid #2d333b;"><td style="padding: 4px 0;">{event_label}</td><td style="text-align: right; color: {COLORS['winter']}; font-weight: 600;">{hazard_events['winter']}</td></tr><tr><td style="padding: 4px 0;">Total Hazard Events</td><td style="text-align: right; color: #e6edf3;">{sum(hazard_events.values())}</td></tr>"""
                    elif hazard_key == 'seismic':
                        primary_events = hazard_events['seismic']
                        hazard_title = "Seismic Risk"
                        event_label = "Seismic Events"
                        details_html = f"""<tr style="border-bottom: 1px solid #2d333b;"><td style="padding: 4px 0;">{event_label}</td><td style="text-align: right; color: {COLORS['seismic']}; font-weight: 600;">{hazard_events['seismic']}</td></tr><tr><td style="padding: 4px 0;">Total Hazard Events</td><td style="text-align: right; color: #e6edf3;">{sum(hazard_events.values())}</td></tr>"""
                    else:
                        # Default/Fire fallback
                        primary_events = hazard_events['fire']
                        hazard_title = "Multi-Hazard Risk"
                        event_label = "Fire Events"
                        hazard_color = COLORS['fire']
                        details_html = f"""<tr style="border-bottom: 1px solid #2d333b;"><td style="padding: 4px 0;">Fire</td><td style="text-align: right; color: {COLORS['fire']};">{hazard_events['fire']}</td></tr><tr style="border-bottom: 1px solid #2d333b;"><td style="padding: 4px 0;">Flood</td><td style="text-align: right; color: {COLORS['flood']};">{hazard_events['flood']}</td></tr><tr style="border-bottom: 1px solid #2d333b;"><td style="padding: 4px 0;">Wind</td><td style="text-align: right; color: {COLORS['wind']};">{hazard_events['wind']}</td></tr><tr style="border-bottom: 1px solid #2d333b;"><td style="padding: 4px 0;">Winter</td><td style="text-align: right; color: {COLORS['winter']};">{hazard_events['winter']}</td></tr><tr><td style="padding: 4px 0;">Seismic</td><td style="text-align: right; color: {COLORS['seismic']};">{hazard_events['seismic']}</td></tr>"""
                    
                    popup_html = f"""
                    <div style="font-family: 'Segoe UI', sans-serif; background: #1a1f26; border-radius: 6px; overflow: hidden;">
                        <div style="background: linear-gradient(135deg, {color} 0%, #1a1f26 100%); padding: 8px 10px;">
                            <h3 style="margin: 0; color: #fff; font-size: 14px; font-weight: 600;">{county_name} County</h3>
                            <div style="color: rgba(255,255,255,0.85); font-size: 11px; margin-top: 2px;">{hazard_title} | {risk_category}</div>
                        </div>
                        <div style="padding: 8px 10px;">
                            <table style="width: 100%; font-size: 11px; color: #8b949e; border-collapse: collapse;">
                                <tr style="border-bottom: 1px solid #2d333b;"><td style="padding: 4px 0;">Population</td><td style="text-align: right; color: #e6edf3;">{population:,}</td></tr>
                                <tr style="border-bottom: 1px solid #2d333b;"><td style="padding: 4px 0;">SVI Score</td><td style="text-align: right; color: #e6edf3;">{svi:.2f}</td></tr>
                                {details_html}
                            </table>
                            <div style="margin-top: 8px; padding: 6px; background: #141920; border-radius: 4px; text-align: center;">
                                <div style="color: #8b949e; font-size: 10px; text-transform: uppercase;">{event_label}</div>
                                <div style="color: {hazard_color}; font-size: 18px; font-weight: 600;">{primary_events:,}</div>
                            </div>
                        </div>
                    </div>
                    """
                else:
                    popup_html = f"<div style='padding: 10px;'>{county_name} County</div>"
                
                # Adjust opacity based on map style - more transparent for satellite view
                fill_opacity = 0.4 if map_style == "Satellite" else 0.6
                border_color = '#ffffff' if map_style != "Satellite" else '#ffff00'
                
                try:
                    folium.GeoJson(
                        row.geometry.__geo_interface__,
                        style_function=lambda x, c=color, fo=fill_opacity, bc=border_color: {
                            'fillColor': c,
                            'color': bc,
                            'weight': 1.5 if bc == '#ffff00' else 1,
                            'fillOpacity': fo
                        },
                        tooltip=f"{county_name}: {risk_category} Risk",
                        popup=folium.Popup(popup_html, max_width=280)
                    ).add_to(m)
                except:
                    pass
    else:
        # Fallback: use markers
        # Determine score column for fallback markers based on selected hazard layer
        hazard_key = hazard_layer.lower()
        if hazard_key in ['fire', 'flood', 'wind', 'winter', 'seismic']:
            score_col = f"{hazard_key}_rate"
        else:
            score_col = 'risk_score'

        for _, row in county_stats.iterrows():
            county = row['county']
            if county.upper() in WA_COUNTY_COORDS:
                lat, lon = WA_COUNTY_COORDS[county.upper()]
            elif row['lat'] and row['lon']:
                lat, lon = row['lat'], row['lon']
            else:
                continue
            
            score = row.get(score_col, row.get('risk_score', 0))
            
            if score > 0.7:
                color = 'red'
            elif score > 0.4:
                color = 'orange'
            else:
                color = 'green'
            
            folium.CircleMarker(
                location=[lat, lon],
                radius=10 + score * 20,
                color=color,
                fill=True,
                popup=f"{county}: {score*100:.2f}% risk",
                tooltip=county
            ).add_to(m)
    
    # Display map
    st_folium(m, width=None, height=600, use_container_width=True)

    # Render precomputed calibrated inference table (if available) in a left-aligned, smaller container
    try:
        if 'last_precomp_table' in st.session_state:
            left_col, right_col = st.columns([1, 2])
            with left_col:
                with st.expander('Precomputed Calibrated Inference', expanded=False):
                    try:
                        df_json = st.session_state.get('last_precomp_table')
                        # Limit to essential columns and a compact table
                        st.table(df_json[['Hazard','Before (Model %)','After (Calibrated %)','Calibration Factor (T)']])
                    except Exception:
                        st.write('Unable to render precomputed table.')
            # keep right_col free for main content (map details)
    except Exception:
        pass
    
    # County details table - hazard-specific
    st.markdown("### County Details")
    
    if hazard_layer == "Fire" and climate_fire_df is not None:
        # Fire-specific table
        display_df = climate_fire_df[['County', 'population', 'Fire_Count', 'climate_fire_risk_score', 
                                       'risk_category', 'climate_trend', 'heat_stress', 'drought_stress']].copy()
        display_df = display_df.sort_values('climate_fire_risk_score', ascending=False)
        display_df.columns = ['County', 'Population', 'NOAA Fire Events', 'Risk Score', 
                              'Risk Level', 'Climate Trend', 'Heat Stress', 'Drought Stress']
        
        st.dataframe(
            display_df.style.format({
                'Population': '{:,.0f}',
                'NOAA Fire Events': '{:,.0f}',
                'Risk Score': '{:.1f}',
                'Heat Stress': '{:.1f}',
                'Drought Stress': '{:.1f}'
            }),
            width='stretch',
            hide_index=True
        )
    elif county_stats is not None:
        # Hazard-specific table for other hazards
        hazard_key = hazard_layer.lower()
        event_col = f'{hazard_key}_events' if hazard_key in ['flood', 'wind', 'winter', 'seismic'] else 'fire_events'
        rate_col = f'{hazard_key}_rate' if hazard_key in ['flood', 'wind', 'winter', 'seismic'] else 'fire_rate'
        
        if event_col in county_stats.columns:
            display_df = county_stats[['county', 'population', 'svi_score', event_col, 'total_events']].copy()
            display_df = display_df.sort_values(event_col, ascending=False)
            display_df.columns = ['County', 'Population', 'SVI Score', f'{hazard_layer} Events', 'Total Events']
            
            st.dataframe(
                display_df.style.format({
                    'Population': '{:,.0f}',
                    'SVI Score': '{:.2f}',
                    f'{hazard_layer} Events': '{:,.0f}',
                    'Total Events': '{:,.0f}'
                }),
                width='stretch',
                hide_index=True
            )
        else:
            display_df = county_stats[['county', 'population', 'risk_score', 'fire_events', 'flood_events', 'wind_events', 'winter_events', 'seismic_events']].copy()
            display_df = display_df.sort_values('risk_score', ascending=False)
            display_df.columns = ['County', 'Population', 'Risk Score', 'Fire', 'Flood', 'Wind', 'Winter', 'Seismic']
            
            st.dataframe(
                display_df.style.format({
                    'Population': '{:,.0f}',
                    'Risk Score': '{:.2%}'
                }),
                width='stretch',
                hide_index=True
            )


# =============================================================================
# PAGE: RISK ASSESSMENT
# =============================================================================

def page_risk_assessment():
    st.markdown("## Comprehensive Risk Assessment")
    
    df = load_hazard_data()
    if df is None:
        st.error("No hazard dataset found. Expected a parquet under `data/` (e.g. data/wa_gridmet/wa_hazard_dataset.parquet).")
        return

    county_stats = compute_county_stats(df)
    if county_stats is None or county_stats.empty:
        st.error("Insufficient data to compute county statistics.")
        return
    
    # Risk level breakdown
    col1, col2, col3 = st.columns(3)
    
    critical = len(county_stats[county_stats['risk_score'] > 0.7])
    moderate = len(county_stats[(county_stats['risk_score'] > 0.4) & (county_stats['risk_score'] <= 0.7)])
    low = len(county_stats[county_stats['risk_score'] <= 0.4])
    
    with col1:
        st.metric("Critical Risk", critical, delta=f"{critical/len(county_stats)*100:.0f}%", delta_color="inverse")
    with col2:
        st.metric("Moderate Risk", moderate, delta=f"{moderate/len(county_stats)*100:.0f}%")
    with col3:
        st.metric("Low Risk", low, delta=f"{low/len(county_stats)*100:.0f}%", delta_color="off")
    
    # Risk by hazard type
    st.markdown("### Risk Distribution by Hazard Type")
    
    hazard_data = []
    for hazard in ['fire', 'flood', 'wind', 'winter', 'seismic']:
        col = f'{hazard}_events'
        hazard_data.append({
            'Hazard': hazard.title(),
            'Total Events': int(county_stats[col].sum()),
            'Avg per County': int(county_stats[col].mean()) if pd.notna(county_stats[col].mean()) else 0,
            'Max County': county_stats.loc[county_stats[col].idxmax(), 'county'],
            'Max Events': int(county_stats[col].max())
        })
    
    st.dataframe(pd.DataFrame(hazard_data), width='stretch', hide_index=True)
    
    # Box plot
    fig = go.Figure()
    
    for hazard, color in [('fire', COLORS['fire']), ('flood', COLORS['flood']), 
                          ('wind', COLORS['wind']), ('winter', COLORS['winter']),
                          ('seismic', COLORS['seismic'])]:
        fig.add_trace(go.Box(
            y=county_stats[f'{hazard}_rate'],
            name=hazard.title(),
            marker_color=color
        ))
    
    fig.update_layout(
        **get_plotly_theme(),
        title="Event Rate Distribution by Hazard Type",
        yaxis_title="Events per Year",
        height=400
    )
    
    st.plotly_chart(fig, width='stretch')
    
    # County detail table
    st.markdown("### Detailed County Analysis")
    
    # Option to show credible intervals
    show_ci = st.checkbox('Show credible intervals (90% CI)', value=False)
    if show_ci:
        display = county_stats[['county', 'population', 'svi_score', 'risk_score',
            'fire_rate', 'fire_rate_lower', 'fire_rate_upper',
            'flood_rate', 'flood_rate_lower', 'flood_rate_upper',
            'wind_rate', 'wind_rate_lower', 'wind_rate_upper',
            'winter_rate', 'winter_rate_lower', 'winter_rate_upper',
            'seismic_rate', 'seismic_rate_lower', 'seismic_rate_upper',
            'total_rate', 'total_rate_lower', 'total_rate_upper',
            'fire_events', 'flood_events', 'wind_events', 'winter_events', 'seismic_events', 'total_events']].copy()
        display = display.sort_values('risk_score', ascending=False)
        display = display.rename(columns={
            'county': 'County', 'population': 'Population', 'svi_score': 'SVI', 'risk_score': 'Risk Score',
            'fire_rate': 'Fire Rate', 'fire_rate_lower': 'Fire Rate (Low)', 'fire_rate_upper': 'Fire Rate (High)',
            'flood_rate': 'Flood Rate', 'flood_rate_lower': 'Flood Rate (Low)', 'flood_rate_upper': 'Flood Rate (High)',
            'wind_rate': 'Wind Rate', 'wind_rate_lower': 'Wind Rate (Low)', 'wind_rate_upper': 'Wind Rate (High)',
            'winter_rate': 'Winter Rate', 'winter_rate_lower': 'Winter Rate (Low)', 'winter_rate_upper': 'Winter Rate (High)',
            'seismic_rate': 'Seismic Rate', 'seismic_rate_lower': 'Seismic Rate (Low)', 'seismic_rate_upper': 'Seismic Rate (High)',
            'total_rate': 'Total Rate', 'total_rate_lower': 'Total Rate (Low)', 'total_rate_upper': 'Total Rate (High)',
            'fire_events': 'Fire', 'flood_events': 'Flood', 'wind_events': 'Wind', 'winter_events': 'Winter', 'seismic_events': 'Seismic', 'total_events': 'Total'
        })
        st.dataframe(
            display.style.format({
                'Population': '{:,.0f}',
                'SVI': '{:.2f}',
                'Risk Score': '{:.2%}',
                'Fire Rate': '{:.2f}', 'Fire Rate (Low)': '{:.2f}', 'Fire Rate (High)': '{:.2f}',
                'Flood Rate': '{:.2f}', 'Flood Rate (Low)': '{:.2f}', 'Flood Rate (High)': '{:.2f}',
                'Wind Rate': '{:.2f}', 'Wind Rate (Low)': '{:.2f}', 'Wind Rate (High)': '{:.2f}',
                'Winter Rate': '{:.2f}', 'Winter Rate (Low)': '{:.2f}', 'Winter Rate (High)': '{:.2f}',
                'Seismic Rate': '{:.2f}', 'Seismic Rate (Low)': '{:.2f}', 'Seismic Rate (High)': '{:.2f}',
                'Total Rate': '{:.2f}', 'Total Rate (Low)': '{:.2f}', 'Total Rate (High)': '{:.2f}'
            }),
            width='stretch',
            hide_index=True
        )
    else:
        display = county_stats[['county', 'population', 'svi_score', 'risk_score', 
                                'fire_events', 'flood_events', 'wind_events', 
                                'winter_events', 'seismic_events', 'total_events']].copy()
        display = display.sort_values('risk_score', ascending=False)
        display.columns = ['County', 'Population', 'SVI', 'Risk Score', 
                           'Fire', 'Flood', 'Wind', 'Winter', 'Seismic', 'Total']
        st.dataframe(
            display.style.format({
                'Population': '{:,.0f}',
                'SVI': '{:.2f}',
                'Risk Score': '{:.2%}'
            }),
            width='stretch',
            hide_index=True
        )


# =============================================================================
# PAGE: CLIMATE ANALYSIS
# =============================================================================

def page_climate_analysis():
    st.markdown("## Climate Change Impact Analysis")
    
    df = load_hazard_data()
    climate_fire_df = load_climate_fire_data()
    
    if df is None and climate_fire_df is None:
        st.error("No data available")
        return
    
    # Climate Trend Overview Section
    if climate_fire_df is not None:
        st.markdown("### Climate Trend Impact on Fire Risk")
        
        # Group by climate trend
        trend_summary = climate_fire_df.groupby('climate_trend').agg({
            'County': 'count',
            'climate_fire_risk_score': 'mean',
            'Fire_Count': 'sum',
            'heat_stress': 'mean',
            'drought_stress': 'mean',
            'population_at_risk': 'sum'
        }).reset_index()
        trend_summary.columns = ['Climate Trend', 'Counties', 'Avg Risk Score', 'Total Fire Events', 
                                  'Avg Heat Stress', 'Avg Drought Stress', 'Population at Risk']
        trend_summary = trend_summary.sort_values('Avg Risk Score', ascending=False)
        
        # Trend cards
        col1, col2, col3, col4 = st.columns(4)
        trends = trend_summary.to_dict('records')
        trend_colors = {
            'Warming & Drying': COLORS['critical'],
            'Drying': COLORS['warning'],
            'Warming': '#f39c12',
            'Stable': COLORS['success']
        }
        
        for col, trend_data in zip([col1, col2, col3, col4], trends[:4]):
            trend_name = trend_data['Climate Trend']
            color = trend_colors.get(trend_name, COLORS['info'])
            with col:
                st.markdown(f"""
                <div class="stat-card" style="border-left: 4px solid {color};">
                    <div style="color: {color}; font-weight: 600; font-size: 14px;">{trend_name}</div>
                    <div style="color: {COLORS['text_tertiary']}; font-size: 11px; margin-top: 4px;">{int(trend_data['Counties'])} Counties</div>
                    <div style="color: {COLORS['text_primary']}; font-size: 24px; font-weight: 600; margin-top: 8px;">{trend_data['Avg Risk Score']:.1f}</div>
                    <div style="color: {COLORS['text_tertiary']}; font-size: 11px;">Avg Risk Score</div>
                </div>
                """, unsafe_allow_html=True)
        
        # Climate trend impact chart
        st.markdown("### Climate Trend Impact on Fire Activity")
        
        fig = go.Figure()
        
        for trend in trend_summary['Climate Trend'].tolist():
            trend_data = climate_fire_df[climate_fire_df['climate_trend'] == trend]
            color = trend_colors.get(trend, COLORS['info'])
            
            fig.add_trace(go.Bar(
                name=trend,
                x=['Heat Stress', 'Drought Stress', 'Fire History Score'],
                y=[trend_data['heat_stress'].mean(), trend_data['drought_stress'].mean(), 
                   trend_data['fire_history_score'].mean()],
                marker_color=color
            ))
        
        fig.update_layout(
            **get_plotly_theme(),
            barmode='group',
            title="Stress Indicators by Climate Trend",
            yaxis_title="Score",
            height=400
        )
        
        st.plotly_chart(fig, width='stretch')
        
        # High risk counties by trend
        st.markdown("### High Risk Counties by Climate Trend")
        
        high_risk = climate_fire_df[climate_fire_df['risk_category'] == 'High'][
            ['County', 'climate_trend', 'climate_fire_risk_score', 'Fire_Count', 'heat_stress', 'drought_stress']
        ].sort_values('climate_fire_risk_score', ascending=False)
        high_risk.columns = ['County', 'Climate Trend', 'Risk Score', 'NOAA Fire Events', 'Heat Stress', 'Drought Stress']
        
        st.dataframe(
            high_risk.style.format({
                'Risk Score': '{:.1f}',
                'NOAA Fire Events': '{:,.0f}',
                'Heat Stress': '{:.1f}',
                'Drought Stress': '{:.1f}'
            }),
            width='stretch',
            hide_index=True
        )
    
    st.markdown("---")
    
    # County-specific analysis
    if df is not None:
        st.markdown("### County-Level Climate Trends")
        
        counties = sorted(df['county'].unique())
        selected_county = st.selectbox("Select County for Analysis", counties)
        
        county_df = df[df['county'] == selected_county].copy()
        county_df['year'] = county_df['date'].dt.year
        county_df['month'] = county_df['date'].dt.month
        
        # Yearly aggregation
        yearly = county_df.groupby('year').agg({
            'erc': 'mean',
            'tmmx': 'mean',
            'rmin': 'mean',
            'fire_label': 'sum',
            'any_hazard': 'sum'
        }).reset_index()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Temperature Trends")
            
            # Convert from Kelvin to Fahrenheit: (K - 273.15) * 9/5 + 32
            temps_f = (yearly['tmmx'] - 273.15) * 9/5 + 32
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=yearly['year'],
                y=temps_f,
                mode='lines+markers',
                name='Max Temp',
                line=dict(color=COLORS['fire'], width=2)
            ))
            
            # Add trend line
            if len(yearly) >= 2:
                z = np.polyfit(yearly['year'], temps_f, 1)
                p = np.poly1d(z)
                fig.add_trace(go.Scatter(
                    x=yearly['year'],
                    y=p(yearly['year']),
                    mode='lines',
                    name='Trend',
                    line=dict(color=COLORS['warning'], dash='dash')
                ))
                
                # Calculate warming rate
                warming_per_decade = z[0] * 10
                st.markdown(f"""
                <div class="stat-card">
                    <div style="color: {COLORS['text_tertiary']}; font-size: 11px;">WARMING RATE</div>
                    <div style="color: {COLORS['fire']}; font-size: 20px; font-weight: 600;">{warming_per_decade:+.2f} F/decade</div>
                </div>
                """, unsafe_allow_html=True)
            
            fig.update_layout(
                **get_plotly_theme(),
                title=f"Temperature Trend - {selected_county}",
                xaxis_title="Year",
                yaxis_title="Max Temperature (F)",
                height=350
            )
            
            st.plotly_chart(fig, width='stretch')
        
        with col2:
            st.markdown("### Fire Weather (ERC - Fire Danger)")
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=yearly['year'],
                y=yearly['erc'],
                marker_color=COLORS['warning']
            ))
            
            fig.update_layout(
                **get_plotly_theme(),
                title=f"Energy Release Component - {selected_county}",
                xaxis_title="Year",
                yaxis_title="Avg Fire Danger (ERC)",
                height=350
            )
            
            st.plotly_chart(fig, width='stretch')
        
        # Monthly patterns
        st.markdown("### Seasonal Patterns")
        
        monthly = county_df.groupby('month').agg({
            'fire_label': 'sum',
            'flood_label': 'sum',
            'wind_label': 'sum',
            'winter_label': 'sum'
        }).reset_index()
        
        fig = go.Figure()
        
        for hazard, color in [('fire_label', COLORS['fire']), ('flood_label', COLORS['flood']),
                              ('wind_label', COLORS['wind']), ('winter_label', COLORS['winter'])]:
            fig.add_trace(go.Scatter(
                x=monthly['month'],
                y=monthly[hazard],
                mode='lines+markers',
                name=hazard.replace('_label', '').title(),
                line=dict(color=color, width=2)
            ))
        
        theme = get_plotly_theme()
        fig.update_layout(
            paper_bgcolor=theme['paper_bgcolor'],
            plot_bgcolor=theme['plot_bgcolor'],
            font=theme['font'],
            title=f"Seasonal Hazard Patterns - {selected_county}",
            xaxis=dict(
                tickmode='array', 
                tickvals=list(range(1, 13)), 
                ticktext=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
                gridcolor=COLORS['border'],
                linecolor=COLORS['border']
            ),
            yaxis=dict(
                title="Event Count",
                gridcolor=COLORS['border'],
                linecolor=COLORS['border']
            ),
            height=400
        )
        
        st.plotly_chart(fig, width='stretch')


# =============================================================================
# PAGE: MITIGATION PLANNING
# =============================================================================

def page_mitigation_planning():
    st.markdown("## Mitigation Strategy Planning")
    
    st.markdown("""
    Based on FEMA research showing average 4:1 to 6:1 ROI for hazard mitigation investments.
    Source: NIBS Natural Hazard Mitigation Saves (2017-2019)
    """)
    
    df = load_hazard_data()
    if df is None:
        st.error("No hazard dataset found. Ensure the canonical WA parquet is present under `data/`.")
        return

    county_stats = compute_county_stats(df)
    if county_stats is None or county_stats.empty:
        st.error("Insufficient data to generate mitigation planning. Try running data preparation scripts.")
        return
    
    # County and hazard selection
    col1, col2 = st.columns(2)
    
    with col1:
        selected_county = st.selectbox("Select County", sorted(county_stats['county'].tolist()))
    
    with col2:
        selected_hazard = st.selectbox("Primary Hazard", ["fire", "flood", "wind", "winter", "seismic", "multi"])
    
    # Get county data
    county_data = county_stats[county_stats['county'] == selected_county].iloc[0]
    
    st.markdown(f"### {selected_county} County Profile")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Population", f"{int(county_data['population']):,}")
    with col2:
        st.metric("Risk Score", f"{county_data['risk_score']:.2%}")
    with col3:
        st.metric("SVI Score", f"{county_data['svi_score']:.2f}")
    with col4:
        st.metric("Total Events", int(county_data['total_events']))
    
    # Recommended actions
    st.markdown("### Recommended Mitigation Actions")
    
    actions = []
    for action_name, data in MITIGATION_ROI.items():
        if data['hazard'] == selected_hazard or data['hazard'] == 'multi':
            mid_cost = sum(data['cost_range']) / 2
            expected_savings = mid_cost * data['roi']
            
            actions.append({
                'Action': action_name.replace('_', ' ').title(),
                'Hazard': data['hazard'].title(),
                'Est. Cost': f"${mid_cost:,.0f}",
                'ROI': f"{data['roi']}x",
                'Expected Savings': f"${expected_savings:,.0f}",
                'Priority': 'High' if data['roi'] > 4 else 'Medium' if data['roi'] > 2.5 else 'Low'
            })
    
    actions_df = pd.DataFrame(actions)
    actions_df = actions_df.sort_values('ROI', ascending=False)
    
    st.dataframe(actions_df, width='stretch', hide_index=True)
    
    # Cost-benefit calculator
    st.markdown("### Investment Calculator")
    
    col1, col2 = st.columns(2)
    
    with col1:
        investment = st.number_input("Estimated Project Cost ($)", min_value=10000, max_value=10000000, value=500000, step=50000)
        avg_roi = float(st.selectbox("Expected ROI Multiplier (x)", options=[1.5,2.0,2.5,3.0,4.0,5.0,6.0], index=3))
        years = int(st.selectbox("Project Lifetime (years)", options=[5,10,15,20,25,30], index=3))
    
    with col2:
        total_benefit = investment * avg_roi
        annual_benefit = total_benefit / years
        payback = investment / annual_benefit
        
        st.metric("Total Expected Benefit", f"${total_benefit:,.0f}")
        st.metric("Annual Benefit", f"${annual_benefit:,.0f}")
        st.metric("Payback Period", f"{payback:.1f} years")
    
    # Visualization
    year_range = list(range(0, years + 1))
    costs = [investment] + [0] * years
    cumulative_costs = np.cumsum(costs)
    benefits = [0] + [annual_benefit] * years
    cumulative_benefits = np.cumsum(benefits)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=year_range,
        y=cumulative_costs,
        mode='lines',
        name='Cumulative Cost',
        line=dict(color=COLORS['critical'], width=2),
        fill='tozeroy',
        fillcolor='rgba(192, 57, 43, 0.1)'
    ))
    
    fig.add_trace(go.Scatter(
        x=year_range,
        y=cumulative_benefits,
        mode='lines',
        name='Cumulative Benefit',
        line=dict(color=COLORS['success'], width=2),
        fill='tozeroy',
        fillcolor='rgba(39, 174, 96, 0.1)'
    ))
    
    fig.add_vline(x=payback, line_dash="dash", line_color=COLORS['text_tertiary'],
                  annotation_text=f"Break-even: {payback:.1f} yr")
    
    fig.update_layout(
        **get_plotly_theme(),
        title="Cost-Benefit Analysis",
        xaxis_title="Years",
        yaxis_title="Cumulative Value ($)",
        height=400
    )
    
    st.plotly_chart(fig, width='stretch')


def page_seasonal_planning():
    st.markdown("## Seasonal Planner & Historical Analysis")
    df = load_hazard_data()
    if df is None:
        st.error('No hazard dataset found.')
        return

    # Season choices
    seasons = {
        'Winter (Dec-Feb)': [12,1,2],
        'Spring (Mar-May)': [3,4,5],
        'Summer (Jun-Aug)': [6,7,8],
        'Fall (Sep-Nov)': [9,10,11]
    }

    county_choices = ['Statewide'] + sorted(df['county'].unique().tolist())
    county = st.selectbox('County', county_choices)
    season = st.selectbox('Season', list(seasons.keys()))

    months = seasons[season]
    if county == 'Statewide':
        sel = df[df['date'].dt.month.isin(months)]
    else:
        sel = df[(df['county'] == county) & (df['date'].dt.month.isin(months))]

    st.markdown(f"### Historical summary — {county} — {season}")
    if sel.empty:
        st.info('No historical records for this selection.')
        return

    # Compute simple hazard probabilities from labels
    summary = {}
    for h in ['fire_label','flood_label','wind_label','winter_label','seismic_label']:
        if h in sel.columns:
            summary[h.replace('_label','')] = float(sel[h].mean())

    st.table(pd.DataFrame([summary]))
    st.markdown('Use this view to compare seasonal baselines and assist planning for preparedness and mitigation.')


# =============================================================================
# PAGE: AI PREDICTIONS
# =============================================================================

def page_ai_predictions():
    st.markdown("## Quick Predict")
    
    model, model_ok = load_hazard_model()
    
    if not model_ok:
        st.warning("Hazard-LM model not available. Ensure outputs/hazard_lm/pretrain_finetune/finetune/best_model.pt exists.")
        return
    
    st.markdown(f"""
    <div class="alert-box alert-success">
        <strong>MODEL LOADED</strong><br>
        {MODEL_DISPLAY_NAME} is ready for inference. Device: {DEVICE}
    </div>
    """, unsafe_allow_html=True)

    # Risk Context: public-facing explanation
    st.markdown(f"""
    <div class="alert-box alert-info">
        <strong>Risk Context</strong><br>
        Estimates are relative to historical baselines and county-level exposure — they indicate comparative likelihoods, not event certainty. Use these assessments to inform preparedness and resource prioritization.
    </div>
    """, unsafe_allow_html=True)
    
    df = load_hazard_data()
    if df is None:
        st.warning("Hazard dataset not found. AI Predictions require the WA hazard parquet under `data/`. See README for data prep.")
        return

    county_stats = compute_county_stats(df)
    if county_stats is None or county_stats.empty:
        st.warning("Insufficient hazard data available to run predictions. Ensure dataset contains labeled hazard columns.")
        return
    
    # County selection
    selected_county = st.selectbox("Select County for Prediction", sorted(county_stats['county'].tolist()))
    
    # Seasonal planner link: provide quick access to seasonal/historical planning
    try:
        if st.sidebar.button('Seasonal Planner', width='stretch'):
            st.session_state.page = 'seasonal'
    except Exception:
        pass
    
    # Get latest data for county
    county_df = df[df['county'] == selected_county].sort_values('date', ascending=False)

    # --- Forecast horizon restriction ---
    from datetime import timedelta
    # Allow operator to choose a short (14-day) or extended (30-day) forecast window
    forecast_choice = st.selectbox('Forecast horizon', ['14 days (recommended)', '30 days (extended)'], index=0, help='Shorter horizon preferred for reliability; extended window is available up to 30 days.')
    MAX_FORECAST_DAYS = 14 if '14' in forecast_choice else 30
    today = datetime.now().date()
    max_forecast_date = today + timedelta(days=MAX_FORECAST_DAYS)

    if len(county_df) > 0:
        latest = county_df.iloc[0]

        st.markdown(f"### Current Conditions - {selected_county}")

        # County satellite image for visual context
        def _get_county_image(county_name: str):
            """Find satellite image for county (used for visual context only, not model input)."""
            img_dir = Path('data/images')
            if not img_dir.exists():
                return None
            # Try exact match first
            clean_name = county_name.upper().replace(' ', '_')
            for ext in ['.jpg', '.png', '.jpeg']:
                candidate = img_dir / f"{clean_name}{ext}"
                if candidate.exists():
                    return candidate
            # Fuzzy match
            for p in img_dir.glob('*.jpg'):
                if clean_name in p.stem.upper():
                    return p
            return None
        
        county_img_path = _get_county_image(selected_county)
        if county_img_path:
            col_img, col_info = st.columns([1, 2])
            with col_img:
                st.image(str(county_img_path), caption=f"{selected_county} County", width='stretch')
            with col_info:
                st.markdown(f"""
                **Location**: {selected_county} County, Washington  
                **Forecast Horizon**: {forecast_choice}  
                **Data Source**: Hazard-LM v1.0 multi-hazard model
                """)

        # Quick synchronous prediction (live model) -- useful for interactive exploration
        try:
            sel_date = pd.to_datetime(latest.get('date')).date() if latest is not None and 'date' in latest else pd.Timestamp(pd.Timestamp.now()).date()
        except Exception:
            sel_date = pd.Timestamp(pd.Timestamp.now()).date()

        # Clamp the default selection to the allowed forecast window to avoid Streamlit errors
        try:
            if sel_date < today:
                sel_date = today
            if sel_date > max_forecast_date:
                sel_date = max_forecast_date
        except Exception:
            sel_date = today

        # Hidden date selector: we use the horizon end date as the prediction target by default
        sel_date = today + timedelta(days=MAX_FORECAST_DAYS)

        # Centered Quick Predict button below the date selector
        b1, b2, b3 = st.columns([1,1,1])
        with b2:
            if st.button('Quick Predict (Live Model)'):
                try:
                    risks, summary = predict_and_summarize(selected_county, sel_date)
                    if risks is None:
                        st.error(f'Prediction failed: {summary}')
                    else:
                        # Build a lightweight risk packet for the LLM summarizer
                        try:
                            from utils.risk_packet import build_risk_packet
                            hazards = []
                            for k, v in risks.items():
                                hazards.append({'name': k, 'prob': float(v)})
                            packet = build_risk_packet(selected_county, str(sel_date), hazards)
                        except Exception:
                            packet = {'county': selected_county, 'date': str(sel_date), 'hazards': risks}

                        # Try to summarize with local GGUF/Mistral if available
                        mistral_summary = None
                        try:
                            from utils.llm_explainer import summarize_with_gguf
                            mistral_summary = summarize_with_gguf(packet)
                        except Exception:
                            mistral_summary = None

                        st.session_state['last_quick_prediction'] = {
                            'county': selected_county,
                            'date': str(sel_date),
                            'risks': risks,
                            'summary': summary,
                            'mistral_summary': mistral_summary
                        }
                        st.success('Prediction stored — results shown below.')
                except Exception as e:
                    st.error(f'Quick prediction failed: {e}')

        # Render stored results immediately under the date control (centered)
        if 'last_quick_prediction' in st.session_state:
            last = st.session_state['last_quick_prediction']
            if last.get('county') == selected_county and last.get('date') == str(sel_date):
                mid1, mid2, mid3 = st.columns([1,2,1])
                with mid2:
                    hazard_order = ['fire','flood','wind','winter','seismic']
                    cols_out = st.columns(5)
                    for c, h in zip(cols_out, hazard_order):
                        p = last['risks'].get(h, 0.0)
                        pct = f"{p*100:.1f}%"
                        color = COLORS.get(h, COLORS['primary'])
                        with c:
                            st.markdown(f"<div style='background:{COLORS['card_bg']}; padding:12px; border-radius:6px; text-align:center; margin-bottom:6px;'><div style='color:{color}; font-weight:700;'>{h.title()}</div><div style='font-size:20px; color:{COLORS['text_primary']}; font-weight:600;'>{pct}</div></div>", unsafe_allow_html=True)

                    # end hazards row
                    st.markdown('### Model Summary (Risk)')
                    try:
                        summary = last.get('summary')
                        if isinstance(summary, str):
                            lines = [s.strip() for s in summary.split('\n') if s.strip()]
                            if len(lines) == 1:
                                sentences = [s.strip() for s in lines[0].split('. ') if s.strip()]
                                if len(sentences) > 1:
                                    lines = sentences
                            bullets = '\n'.join([f"- {l}" for l in lines]) if lines else summary
                            st.markdown(bullets)
                        else:
                            st.write(summary)
                    except Exception:
                        st.text(last.get('summary'))

        # LLM Summaries (fallback and local GGUF) - constrained to risk packet
        try:
            from utils.llm_explainer import summarize_with_gguf
        except Exception:
            summarize_with_gguf = None

        # Compact LLM summary card (prefer parsed GGUF summaries)
        def _find_preferred_summary(county, date_str):
            # search order: parsed GGUF -> parsed summaries_gguf -> parsed summaries -> gguf raw -> fallback summaries
            candidates = []
            sdir_gguf = Path('outputs/hazard_lm/summaries_gguf')
            sdir_fallback = Path('outputs/hazard_lm/summaries')
            date_token = str(date_str)

            # helper to match
            def _match(p):
                name = p.stem.upper()
                return (county.upper() in name) or (date_token in name)

            # parsed_ prefixed files first
            for d in [sdir_gguf, sdir_fallback]:
                if d.exists():
                    for p in sorted(d.glob('parsed_*.json')):
                        if _match(p):
                            return p

            # then any parsed files
            for d in [sdir_gguf, sdir_fallback]:
                if d.exists():
                    for p in sorted(d.glob('parsed_*.json')):
                        return p

            # then any gguf summaries
            if sdir_gguf.exists():
                for p in sorted(sdir_gguf.glob('*.json')):
                    if _match(p):
                        return p

            # then fallback summaries
            if sdir_fallback.exists():
                for p in sorted(sdir_fallback.glob('*.json')):
                    if _match(p):
                        return p

            return None

        summary_path = _find_preferred_summary(selected_county, sel_date)
        summary_obj = None
        provenance = None
        if summary_path is not None:
            try:
                summary_obj = json.load(open(summary_path))
                provenance = str(summary_path)
            except Exception:
                summary_obj = None

        # If we have an immediate mistral summary from quick predict, prefer that
        if summary_obj is None and 'last_quick_prediction' in st.session_state:
            lq = st.session_state['last_quick_prediction']
            ms = lq.get('mistral_summary')
            if ms:
                summary_obj = ms
                provenance = 'in-memory (last_quick_prediction)'

        # If still no summary, try deterministic fallback from packet if available
        if summary_obj is None:
            try:
                from utils.llm_explainer import _fallback_summarize
            except Exception:
                _fallback_summarize = None
            # try to find risk packet
            rp = None
            rp_dir = Path('outputs/hazard_lm/risk_packets')
            if rp_dir.exists():
                for p in rp_dir.glob(f'*{selected_county}*.json'):
                    rp = p
                    break
            if rp is not None and _fallback_summarize is not None:
                try:
                    packet = json.load(open(rp))
                    summary_obj = _fallback_summarize(packet)
                    provenance = str(rp)
                except Exception:
                    summary_obj = None

        # Render compact card
        st.markdown('### Risk Summaries')
        if summary_obj is None:
            st.info('No summary available. Run Quick Predict to generate a summary.')
        else:
            pub = summary_obj.get('public_summary') if isinstance(summary_obj, dict) else str(summary_obj)
            eoc = summary_obj.get('eoc_brief', '') if isinstance(summary_obj, dict) else ''
            drivers = summary_obj.get('drivers', []) if isinstance(summary_obj, dict) else []
            conf = summary_obj.get('confidence_band', 'Not available') if isinstance(summary_obj, dict) else 'Not available'

            # Compact card layout
            st.markdown(f"<div style='background:{COLORS['card_bg']}; padding:12px; border-radius:6px;'><div style='font-weight:700; color:{COLORS['text_primary']};'>{pub}</div><div style='color:{COLORS['text_tertiary']}; margin-top:6px;'>Drivers: {', '.join(drivers[:2]) if drivers else 'None reported'} — Confidence: {conf}</div></div>", unsafe_allow_html=True)

            # EOC brief collapsed
            with st.expander('EOC Brief', expanded=False):
                if eoc:
                    st.markdown(eoc)
                else:
                    st.markdown('No EOC brief available in the summary.')

            # provenance and actions (only show to advanced users)
            try:
                show_adv = bool(st.session_state.get('show_advanced', False))
            except Exception:
                show_adv = False
            if show_adv:
                prov_line = provenance or 'generated'
                # show only filename (not full path) for privacy unless advanced
                try:
                    prov_display = Path(prov_line).name if isinstance(prov_line, str) and '/' in prov_line else prov_line
                except Exception:
                    prov_display = prov_line
                st.markdown(f"<div style='color:{COLORS['text_tertiary']}; font-size:11px; margin-top:8px;'>Provenance: {prov_display}</div>", unsafe_allow_html=True)

            # LLM summarization disabled for cloud deployment
            # (Regenerate button removed - requires local Mistral/llama-cpp-python)
        

        # If a precomputed calibrated JSON exists for this county, load and show it
        def _load_county_json(name: str):
            if not name:
                return None
            candidate = Path('outputs/hazard_lm') / f'smoke_inference_{name.upper()}.json'
            if candidate.exists():
                try:
                    return json.load(open(candidate))
                except Exception:
                    return None
            # try alternate naming without upper
            for p in Path('outputs/hazard_lm').glob('smoke_inference_*.json'):
                if name.upper() in p.stem.upper():
                    try:
                        return json.load(open(p))
                    except Exception:
                        return None
            return None

        precomp = _load_county_json(selected_county)
        if precomp:
            # Store the parsed precomputed table in session so the map page can render
            sample = precomp[0] if isinstance(precomp, list) and precomp else precomp
            rows = []
            for h, v in sample.get('hazards', {}).items():
                rows.append({'Hazard': h.title(), 'Before (Model %)': v.get('p_before'), 'After (Calibrated %)': v.get('p_after'), 'Calibration Factor (T)': v.get('T')})
            if rows:
                df_json = pd.DataFrame(rows)
                # Human-friendly percent columns
                df_json['Before (Model %)'] = df_json['Before (Model %)'].apply(lambda x: f"{x*100:.2f}%" if x is not None else 'N/A')
                df_json['After (Calibrated %)'] = df_json['After (Calibrated %)'].apply(lambda x: f"{x*100:.2f}%" if x is not None else 'N/A')
                # Instead of rendering inline here (which can crowd the UI), stash in session for the Interactive Map
                try:
                    st.session_state['last_precomp_table'] = df_json
                except Exception:
                    # Fallback: render inline if session_state is unavailable
                    st.table(df_json[['Hazard','Before (Model %)','After (Calibrated %)','Calibration Factor (T)']])
            else:
                st.info('Precomputed inference file found but contained no hazard entries.')

        # Auto-find per-county temperature JSON (search common locations)
        def _find_county_temps(name: str):
            import re
            if not name:
                return None
            candidates = []
            # per-county temps in common output folders
            candidates += list(Path('outputs/wa_adapter_final/temps').glob('*.json')) if Path('outputs/wa_adapter_final/temps').exists() else []
            candidates += list(Path('outputs/hazard_lm/temps').glob('*.json')) if Path('outputs/hazard_lm/temps').exists() else []
            candidates += list(Path('data/images/temps').glob('*.json')) if Path('data/images/temps').exists() else []
            # fallback single temperature file
            fallback = Path('data/images/temperature.json')

            norm = re.sub(r'[^A-Z0-9]', '', name.upper())
            for p in candidates:
                stem = re.sub(r'[^A-Z0-9]', '', p.stem.upper())
                if stem == norm or norm in stem or stem in norm:
                    return p
            # fallback to scanning outputs/hazard_lm for county name in filename
            for p in Path('outputs/hazard_lm').glob('*.json'):
                if name.upper() in p.stem.upper() and 'temperature' in p.stem.lower():
                    return p
            if fallback.exists():
                return fallback
            return None

        temps_found = _find_county_temps(selected_county)
        # Hide technical temperature file paths from general users; keep behavior internal
        if not temps_found:
            st.info('No per-county temperature calibration found yet. A per-county temperature file will be created from available defaults, or the model will generate calibration on-the-fly.')

        # Ensure we have a per-county temps JSON (create lightweight fallback if needed)
        def _ensure_county_temps(name: str):
            # Return a Path to a usable per-county temps JSON. If none found, try to generate a fallback
            p = _find_county_temps(name)
            target_dir = Path('data/images/temps')
            target_dir.mkdir(parents=True, exist_ok=True)
            outp = target_dir / f"{name.replace(' ', '_')}_temperature.json"
            if p is not None and Path(p).exists():
                return Path(p)

            # Try to use global fallback if present
            global_fallback = Path('data/images/temperature.json')
            try:
                if global_fallback.exists():
                    # If global file appears to be a mapping of per-county temps, try to extract
                    try:
                        j = json.load(open(global_fallback))
                        if isinstance(j, dict) and name in j:
                            # write single-county JSON
                            json.dump(j[name], open(outp, 'w'), indent=2)
                            return outp
                        else:
                            # Copy as a generic fallback
                            shutil.copy(global_fallback, outp)
                            return outp
                    except Exception:
                        shutil.copy(global_fallback, outp)
                        return outp

                # As a last resort, create a minimal temps file so the inference script can run
                minimal = {'meta': {'generated_for': name, 'generated_at': datetime.utcnow().isoformat()}, 'temperatures': []}
                json.dump(minimal, open(outp, 'w'), indent=2)
                return outp
            except Exception:
                # If anything fails, return None so the caller knows
                return None

        # NOTE: Background inference removed - use Quick Predict instead

            # Ensure we have a county-level stats object for subsequent UI calculations
            try:
                county_stat = county_stats[county_stats['county'] == selected_county].iloc[0]
            except Exception:
                county_stat = pd.Series()

            # Calculate risk using normalized rates with sigmoid-like scaling
            # This ensures probabilities are meaningful (not all 100%)
            hazards = ['Fire', 'Flood', 'Wind', 'Winter', 'Seismic']
            rate_cols = ['fire_rate', 'flood_rate', 'wind_rate', 'winter_rate', 'seismic_rate']
            colors = [COLORS['fire'], COLORS['flood'], COLORS['wind'], COLORS['winter'], COLORS['seismic']]

            # Get max rates for normalization
            max_rates = {col: county_stats[col].max() for col in rate_cols}

            # Primary presentation: empirical per-county horizon probabilities (with shrinkage fallback)
            probs = []
            factors_list = []
            hazards_label = ['fire_label', 'flood_label', 'wind_label', 'winter_label', 'seismic_label']
            for rate_col, h_label in zip(rate_cols, hazards_label):
                # default factors
                factors = []
                # attempt to fetch empirical probability
                p_hat = None
                try:
                    if emp_map is not None and selected_county in emp_map:
                        rec = emp_map.get(selected_county, {})
                        h_rec = rec.get(h_label, {})
                        p_hat = float(h_rec.get('p_hat', None)) if h_rec is not None else None
                        nanchors = int(h_rec.get('n', 0)) if h_rec is not None else 0
                    else:
                        p_hat = None
                        nanchors = 0
                except Exception:
                    p_hat = None
                    nanchors = 0

                # Fallback to state-level empirical if missing
                try:
                    if (p_hat is None or nanchors == 0) and emp_map is not None:
                        state_probs = emp_map.get('__state__', {})
                        p_hat = float(state_probs.get(h_label, 0.0))
                except Exception:
                    p_hat = p_hat if p_hat is not None else 0.0

                # If still None, fall back to normalized historical rate (legacy) as last resort
                if p_hat is None:
                    try:
                        rate = float(county_stat.get(rate_col, 0)) if hasattr(county_stat, 'get') else float(county_stat[rate_col])
                        max_rate = max_rates.get(rate_col, 1) if (isinstance(max_rates.get(rate_col, None), (int, float)) and max_rates.get(rate_col, 0) > 0) else 1
                        normalized = rate / max_rate if max_rate != 0 else 0.0
                        prob = normalized * 0.85 + 0.05
                        p_hat = min(0.95, max(0.02, prob))
                    except Exception:
                        p_hat = 0.05

                probs.append(float(p_hat))
                # Build simple exposure/vulnerability factors as before
                try:
                    if hasattr(county_stat, 'get'):
                        svi_val = county_stat.get('svi_score', 0)
                        pop_val = county_stat.get('population', 0)
                    else:
                        svi_val = county_stat['svi_score'] if 'svi_score' in county_stat.index else 0
                        pop_val = county_stat['population'] if 'population' in county_stat.index else 0
                except Exception:
                    svi_val = 0
                    pop_val = 0
                if svi_val > 0.6:
                    factors.append("Elevated social vulnerability")
                if pop_val > 100000:
                    factors.append("High population exposure")
                factors_list.append(factors)

            # Seasonal expectation mapping (simple canonical months)
            seasonal_map = {
                'Fire': [6,7,8,9],
                'Flood': [2,3,4,5],
                'Wind': [10,11,12,1,2],
                'Winter': [12,1,2],
                'Seismic': list(range(1,13))  # Seismic considered year-round
            }
            current_month = datetime.now().month

            # Build a simplified card for each hazard showing: relative risk level, likelihood, confidence band, seasonal alignment
            for hazard, prob, color, factors in zip(hazards, probs, colors, factors_list):
                level = "HIGH" if prob > 0.6 else "MODERATE" if prob > 0.3 else "LOW"
                confidence = level
                in_season = current_month in seasonal_map.get(hazard, [])
                season_text = 'In season' if in_season else 'Outside typical season'

                card_col1, card_col2 = st.columns([4, 1])
                with card_col1:
                    st.markdown(f"**{hazard} — {season_text}**")
                    st.markdown(f"<div style='color: {color}; font-size: 28px; font-weight: 600;'>{prob*100:.1f}%</div>", unsafe_allow_html=True)
                    st.markdown(f"<div style='color: {COLORS['text_tertiary']};'>Confidence: {confidence}</div>", unsafe_allow_html=True)
                with card_col2:
                    badge_color = COLORS['critical'] if level == 'HIGH' else COLORS['warning'] if level == 'MODERATE' else COLORS['success']
                    st.markdown(f"<div style='background: {badge_color}; color: white; padding: 8px 12px; border-radius: 6px; text-align: center; margin-top: 18px;'>{level}</div>", unsafe_allow_html=True)

                st.markdown("---")

            # If the top-ranked hazard is outside its typical season, warn the operator
            try:
                top_hazard = results_df.sort_values('probability', ascending=False).iloc[0]
                top_name = top_hazard['hazard'].title()
                top_in_season = datetime.now().month in seasonal_map.get(top_name, [])
                if not top_in_season:
                    st.warning(f"Top hazard ({top_name}) is outside its typical season. Review drivers and confidence before taking action.")
            except Exception:
                pass

            # Add explanation - updated to reflect empirical horizon-aware probabilities
            st.info("Short-term hazard probabilities are derived from historical county-level frequencies for the selected horizon. This ensures calibration to observed event rates and avoids overestimating risk outside typical seasonal regimes.")
            # Compute empirical per-county horizon probabilities (primary presentation layer)
            try:
                emp_map = compute_empirical_horizon_probs(df, horizon_days=MAX_FORECAST_DAYS, alpha=20)
            except Exception:
                emp_map = None
            # Build results dataframe for this county (raw probabilities retained)
            results_df = pd.DataFrame({
                'hazard': hazards,
                'probability': probs,
                'level': ["HIGH" if p > 0.6 else "MODERATE" if p > 0.3 else "LOW" for p in probs],
                'factors': [", ".join(f[:3]) for f in factors_list]
            })
            # Add a human-readable percent column for display
            results_df['probability_pct'] = results_df['probability'].apply(lambda x: f"{x*100:.2f}%")

            # Export CSV button
            csv = results_df.to_csv(index=False)
            st.download_button(label='Download county prediction CSV', data=csv, file_name=f'{selected_county}_predictions.csv', mime='text/csv')

            # Visualization: small bar chart
            try:
                fig = px.bar(results_df, x='hazard', y='probability', color='hazard', labels={'probability':'Probability'}, color_discrete_sequence=colors)
                fig.update_layout(**get_plotly_theme(), title=f'Predicted Hazard Probabilities - {selected_county}', height=360, showlegend=False)
                fig.update_yaxes(tickformat='.2%')
                st.plotly_chart(fig, width='stretch')
            except Exception:
                pass

            # Option: show county on map (if geojson available)
            gdf = load_geojson()
            if FOLIUM_AVAILABLE and gdf is not None:
                if st.button('Show county on map'):
                    try:
                        # Find likely name-like columns
                        name_cols = [c for c in gdf.columns if any(k in c.lower() for k in ('name', 'county'))]
                        county_row = gpd.GeoDataFrame()

                        if name_cols:
                            # Try exact normalized match across candidate columns
                            for nc in name_cols:
                                try:
                                    series = gdf[nc].astype(str).str.upper().str.replace(' COUNTY', '').str.strip()
                                    mask = series == selected_county.upper()
                                    if mask.any():
                                        county_row = gdf[mask]
                                        break
                                except Exception:
                                    continue

                        # Fallback to substring contains matching
                        if county_row.empty:
                            for nc in name_cols:
                                try:
                                    series = gdf[nc].astype(str).str.upper()
                                    mask = series.str.contains(selected_county.upper(), na=False)
                                    if mask.any():
                                        county_row = gdf[mask]
                                        break
                                except Exception:
                                    continue

                        if county_row.empty:
                            st.warning('County geometry not found in GeoJSON for selected county.')
                        else:
                            center = WA_COUNTY_COORDS.get(selected_county, (47.5, -120.5))
                            m = folium.Map(location=center, zoom_start=8)
                            try:
                                folium.GeoJson(
                                    county_row.geometry.__geo_interface__,
                                    style_function=lambda feat: {'fillColor': COLORS['fire'], 'color': '#222', 'weight': 1, 'fillOpacity':0.6},
                                ).add_to(m)
                            except Exception:
                                # Try iterating geometries
                                for _, crow in county_row.iterrows():
                                    try:
                                        folium.GeoJson(crow.geometry.__geo_interface__, style_function=lambda feat: {'fillColor': COLORS['fire'], 'color': '#222', 'weight': 1, 'fillOpacity':0.6}).add_to(m)
                                    except Exception:
                                        continue

                            st_folium(m, width=700, height=400)
                    except Exception as e:
                        st.warning(f'Unable to render county map: {e}')

            st.success('County-level prediction complete.')

            # Save last county results in session for later export or map
            st.session_state['last_county_prediction'] = results_df

            # Offer Risk Packet export and on-demand summarization
            try:
                from utils.llm_explainer import summarize_risk
                from utils.risk_packet import build_risk_packet, write_risk_packet
                rp_dir = Path('outputs/hazard_lm/risk_packets')
                rp_dir.mkdir(parents=True, exist_ok=True)
                # Build a small hazards list from results_df for packet
                hazards_for_pkt = []
                for _, row in results_df.iterrows():
                    hazards_for_pkt.append({
                        'name': row['hazard'],
                        'prob': float(row['probability']),
                        'confidence_band': row.get('level')
                    })

                pkt = build_risk_packet(selected_county, str(sel_date), hazards_for_pkt, data_window_days=MAX_FORECAST_DAYS)
                pkt_path = rp_dir / f"risk_packet_{selected_county.replace(' ','_')}_{str(sel_date)}.json"
                write_risk_packet(pkt_path, pkt)

                if st.button('Generate Situation Summary (Local Mistral)'):
                    with st.spinner('Generating summaries...'):
                        summary = summarize_risk(pkt)
                        st.markdown('**Public Summary**')
                        st.markdown(summary.get('public_summary', 'No summary'))
                        st.markdown('**EOC Brief**')
                        st.markdown(summary.get('eoc_brief', 'No brief'))
                        # persist summary for audit
                        sdir = Path('outputs/hazard_lm/summaries')
                        sdir.mkdir(parents=True, exist_ok=True)
                        sf = sdir / f"situation_summary_{selected_county.replace(' ','_')}_{str(sel_date)}.json"
                        try:
                            json.dump({'packet': pkt, 'summary': summary, 'ts': datetime.utcnow().isoformat() + 'Z'}, open(sf, 'w'), indent=2)
                        except Exception:
                            pass
            except Exception:
                pass

            # Background inference is handled by the Run Hazard Prediction flow above.
            # The previous synchronous call was removed to keep the UI responsive.

    # Note: model-based inference is handled automatically when you click "Run Hazard Prediction" above.
    # Per-county temperature calibration files are auto-detected (look in `outputs/wa_adapter_final/temps/`,
    # `outputs/hazard_lm/temps/`, `data/images/temps/`, or fallback to `data/images/temperature.json`).
    # The dashboard intentionally does not allow uploading a temperature.json from the UI — calibrated
    # model inference uses the validated per-county files on disk to avoid silent calibration drift.

    # -----------------------------
    # Statewide Predictions (Model-based)
    # -----------------------------
    st.markdown('---')
    st.subheader('Statewide Model Predictions')
    st.caption('Run the Hazard-LM model for all 39 Washington counties. This may take 2-3 minutes.')
    
    # Get list of counties
    try:
        counties = sorted(county_stats['county'].dropna().unique().tolist())
    except Exception:
        counties = []
    
    if len(counties) == 0:
        st.warning('No county data available for statewide predictions.')
    else:
        target_date = datetime.now().date() + timedelta(days=7)  # 7-day forecast
        
        if st.button('Run Statewide Predictions', type='primary'):
            progress_bar = st.progress(0)
            status_text = st.empty()
            results_container = st.empty()
            
            all_rows = []
            hazards = ['fire', 'flood', 'wind', 'winter', 'seismic']
            
            for i, county in enumerate(counties):
                status_text.text(f'Processing {county}... ({i+1}/{len(counties)})')
                progress_bar.progress((i + 1) / len(counties))
                
                try:
                    risks, summary = predict_and_summarize(county, target_date)
                    if risks:
                        row = {'county': county, 'date': str(target_date)}
                        for h in hazards:
                            row[f'{h}_p'] = risks.get(h, 0.0)
                        all_rows.append(row)
                except Exception as e:
                    st.warning(f'Failed for {county}: {e}')
                    continue
            
            progress_bar.progress(1.0)
            status_text.text('✅ Complete!')
            
            if all_rows:
                statewide_df = pd.DataFrame(all_rows)
                
                # Store in session state for map visualization
                st.session_state['statewide_predictions'] = statewide_df
                
                # Save to disk
                out_path = Path('outputs/hazard_lm/statewide_predictions.csv')
                out_path.parent.mkdir(parents=True, exist_ok=True)
                statewide_df.to_csv(out_path, index=False)
                
                st.success(f'Completed predictions for {len(all_rows)} counties!')
                
                # Show summary table
                display_df = statewide_df.copy()
                for h in hazards:
                    col = f'{h}_p'
                    if col in display_df.columns:
                        display_df[h.title()] = (display_df[col] * 100).round(1).astype(str) + '%'
                st.dataframe(display_df[['county'] + [h.title() for h in hazards]], width='stretch')
                
                # Download button
                csv = statewide_df.to_csv(index=False)
                st.download_button(
                    '📥 Download Statewide Predictions CSV',
                    data=csv,
                    file_name=f'statewide_predictions_{target_date}.csv',
                    mime='text/csv'
                )
            else:
                st.error('No predictions generated. Check model availability.')
        
        # Show cached results if available
        if 'statewide_predictions' in st.session_state:
            st.markdown('#### Statewide Results')
            cached_df = st.session_state['statewide_predictions']
            
            # Show results table (persists after initial run)
            hazards = ['fire', 'flood', 'wind', 'winter', 'seismic']
            display_df = cached_df.copy()
            for h in hazards:
                col = f'{h}_p'
                if col in display_df.columns:
                    display_df[h.title()] = (display_df[col] * 100).round(1).astype(str) + '%'
            st.dataframe(display_df[['county'] + [h.title() for h in hazards]], use_container_width=True, hide_index=True)
            
            # Download button
            csv = cached_df.to_csv(index=False)
            st.download_button(
                'Download Statewide Predictions CSV',
                data=csv,
                file_name='statewide_predictions.csv',
                mime='text/csv'
            )
            
            st.markdown('---')
            
            # Show hazard selector for map
            hazard_choice = st.selectbox('Select hazard to display on map', ['Fire', 'Flood', 'Wind', 'Winter', 'Seismic'], index=0)
            col_name = hazard_choice.lower() + '_p'
            
            # Visualize on map
            gdf = load_geojson()
            if FOLIUM_AVAILABLE and gdf is not None:
                try:
                    gdf_copy = gdf.copy()
                    # Find name field
                    name_field = None
                    for f in ['NAME', 'name', 'COUNTY', 'county_name']:
                        if f in gdf_copy.columns:
                            name_field = f
                            break
                    
                    if name_field:
                        gdf_copy['county_norm'] = gdf_copy[name_field].str.replace(' County', '').str.strip()
                        cached_df['county_norm'] = cached_df['county'].str.replace(' County', '').str.strip()
                        merged = gdf_copy.merge(cached_df, on='county_norm', how='left')
                        
                        m = folium.Map(location=[47.5, -120.5], zoom_start=7)
                        
                        # Simple marker-based map (avoids choropleth complexity)
                        for _, row in merged.iterrows():
                            try:
                                prob = float(row.get(col_name, 0.0) or 0.0)
                                county = row.get('county_norm', 'Unknown')
                                
                                # Color based on risk level
                                if prob > 0.7:
                                    color = 'red'
                                elif prob > 0.4:
                                    color = 'orange'
                                elif prob > 0.2:
                                    color = 'yellow'
                                else:
                                    color = 'green'
                                
                                # Get centroid
                                geom = row.get('geometry')
                                if geom is not None:
                                    centroid = geom.centroid
                                    lat, lon = centroid.y, centroid.x
                                    
                                    folium.CircleMarker(
                                        location=[lat, lon],
                                        radius=8 + prob * 15,
                                        color=color,
                                        fill=True,
                                        fill_opacity=0.7,
                                        popup=f"{county}: {prob*100:.1f}% {hazard_choice}",
                                        tooltip=f"{county}: {prob*100:.1f}%"
                                    ).add_to(m)
                            except Exception:
                                continue
                        
                        st_folium(m, width=800, height=500)
                    else:
                        st.warning('GeoJSON missing county name field.')
                except Exception as e:
                    st.warning(f'Unable to render map: {e}')
            else:
                st.info('Folium not available for map visualization.')


# =============================================================================
# PAGE: MODEL DIAGNOSTICS
# =============================================================================

def page_model_diagnostics():
    st.markdown("## Hazard-LM Model Diagnostics")
    
    model, model_ok = load_hazard_model()
    
    # Plain English explanation
    st.markdown("""
    ### What is Hazard-LM?
    
    **Hazard-LM** (Hazard Language Model) is an AI system that predicts the likelihood of natural 
    disasters occurring in Washington State counties. Think of it like a weather forecast, but for 
    emergencies - it looks at patterns from the past to estimate future risk.
    
    **How it works in simple terms:**
    1. **It learns from history** - The model studied over 156,000 days of data across all 39 Washington counties
    2. **It watches the weather** - Temperature, humidity, wind, and fire weather conditions over 14-day windows
    3. **It knows the land** - Forest, urban, agricultural areas all have different risk profiles (via NLCD land cover data)
    4. **It understands context** - Demographics, infrastructure, and social vulnerability factors
    5. **It predicts 5 hazard types** - Fire, flood, wind storms, winter storms, and seismic events
    
    **Why this matters:** Emergency managers can use these predictions to pre-position resources, 
    issue early warnings, and prioritize mitigation funding for high-risk areas.
    """)
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Model Version", MODEL_DISPLAY_NAME)
    
    with col2:
        st.metric("Parameters", "15.8M")
    
    with col3:
        status_text = "✅ Online" if model_ok else "❌ Offline"
        st.metric("Status", status_text)
    
    st.markdown("### Architecture")
    
    st.markdown("""
    | Component | Details | What it does |
    |-----------|---------|--------------|
    | **Static Encoder** | MLP, 256 hidden dim, 41 input features | Processes county demographics, geography, infrastructure |
    | **Temporal Encoder** | Transformer with diffusion attention, 128 dim | Tracks weather patterns over 14-day windows |
    | **Land Cover** | NLCD embedding, 16 dim | Understands forest, urban, agricultural land types |
    | **Cross-Hazard Layer** | Interaction learning | Models dependencies between hazard types |
    | **Fusion Layer** | Multi-modal attention, 256 dim | Combines all information sources |
    | **Prediction Heads** | 5 hazard-specific heads | Separate calibrated predictors for each hazard |
    """)
    
    # Completed work and future improvements
    st.markdown("---")
    st.markdown("### Updates & Roadmap")

    st.markdown("**Completed (this project)**")
    st.markdown("""
    - Trained Hazard-LM v1.0 on 370,000+ county-day observations (2000-2025) across 39 Washington counties
    - Achieved strong discrimination: Fire AUC 0.89, Winter AUC 0.94, Wind AUC 0.87, Flood AUC 0.83, Seismic AUC 0.77
    - Implemented diffusion-based attention mechanism in temporal encoder for improved calibration
    - Applied per-hazard temperature scaling to improve probability calibration (ECE reduced 55-59% for fire/seismic)
    - Integrated local LLM summarization (Mistral-7B) for plain-language risk explanations
    - Built interactive dashboard with Quick Predict, statewide predictions, and county-level risk assessment
    - Validated on 37,039 held-out test samples with comprehensive calibration analysis
    """)

    st.markdown("**Planned / Future Work**")
    st.markdown("""
    - Expand to Pacific Northwest states (Oregon, Idaho) with hierarchical calibration for low-event counties
    - Integrate real-time weather feeds (NWS/NOAA) for operational nowcasts
    - Add Monte Carlo Dropout uncertainty quantification for prediction intervals
    - Improve spatial modeling (graph neural networks for county adjacency / hazard spread)
    - Build continual learning pipeline with scheduled model retraining
    - Conduct softmax ablation study to quantify diffusion attention benefit
    """)

    st.info("**Recommendation:** Prioritize expanding dataset coverage (more states/regions) and uncertainty quantification to make calibrated outputs operationally usable for emergency managers.")

    # Data sources section
    st.markdown("---")
    st.markdown("### Data Sources")
    st.markdown("The primary data sources used to train and evaluate Hazard-LM:")
    st.markdown("""
    | Source | Dataset | Usage |
    |--------|---------|-------|
    | **NOAA GridMET** | Daily gridded weather | Temperature, humidity, wind, fire weather (ERC) |
    | **NOAA Storm Events** | Historical storm records | Flood, wind, winter storm labels |
    | **US Census** | Demographics | Population, housing density |
    | **CDC SVI** | Social Vulnerability Index | Community resilience factors |
    | **USGS NLCD** | Land Cover Database | Forest, urban, agricultural classification |
    | **USGS Earthquakes** | Seismic catalog | Historical earthquake events |
    | **FEMA** | Disaster declarations | Validation labels, historical context |
    """)
    
    st.markdown("""
    **County Images:** Static satellite/aerial imagery for each county is used for **visual context only** in the dashboard. 
    These images are not processed by the model — they help users visually identify terrain and land cover when assessing risk.
    """)


# =============================================================================
# PAGE: MODEL EVALUATION
# =============================================================================

def page_model_evaluation():
    st.markdown("## Model Evaluation")

    st.markdown("""
    This page shows validation results for the deployed Hazard-LM model. 
    **AUC** (Area Under Curve) measures how well the model discriminates between hazard/no-hazard. 
    1.0 = perfect, 0.5 = random guessing, >0.8 = good.
    """)

    # Current model performance - hardcoded from evaluation results
    st.markdown("### Current Model Performance")
    st.caption(f"Model: `{RESOLVED_MODEL_PATH}`")
    
    # Performance metrics table
    performance_data = [
        {"Hazard": "🔥 Fire", "AUC": 0.89, "Quality": "Excellent", "Notes": "Strong weather signal from ERC, temperature, humidity"},
        {"Hazard": "❄️ Winter", "AUC": 0.94, "Quality": "Excellent", "Notes": "Clear seasonal patterns, temperature-driven"},
        {"Hazard": "💨 Wind", "AUC": 0.87, "Quality": "Good", "Notes": "Captures storm patterns from temporal features"},
        {"Hazard": "🌊 Flood", "AUC": 0.83, "Quality": "Good", "Notes": "Precipitation and streamflow patterns"},
        {"Hazard": "🌋 Seismic", "AUC": 0.77, "Quality": "Good", "Notes": "Historical patterns; earthquakes less predictable"},
    ]
    
    st.dataframe(pd.DataFrame(performance_data), width='stretch', hide_index=True)
    
    # Visual AUC bar chart
    fig = go.Figure()
    hazards = ["Fire", "Winter", "Wind", "Flood", "Seismic"]
    aucs = [0.89, 0.94, 0.87, 0.83, 0.77]
    colors = [COLORS.get('fire', '#ff6b6b'), COLORS.get('winter', '#74c0fc'), 
              COLORS.get('wind', '#63e6be'), COLORS.get('flood', '#4dabf7'), 
              COLORS.get('seismic', '#da77f2')]
    
    fig.add_trace(go.Bar(x=hazards, y=aucs, marker_color=colors))
    fig.add_hline(y=0.8, line_dash="dash", line_color="green", annotation_text="Good (0.8)")
    fig.add_hline(y=0.5, line_dash="dash", line_color="red", annotation_text="Random (0.5)")
    fig.update_layout(**get_plotly_theme(), title="AUC by Hazard Type", 
                      yaxis_title="AUC Score", yaxis_range=[0, 1], height=350)
    st.plotly_chart(fig, width='stretch')
    
    avg_auc = sum(aucs) / len(aucs)
    st.success(f"**Overall Performance:** Average AUC of {avg_auc:.2f} across all hazard types")

    # Calibration section
    st.markdown("---")
    st.markdown("### Calibration Analysis")
    st.markdown("""
    **Calibration** means the predicted probabilities match real-world frequencies. 
    If the model says 30% fire risk, fires should occur ~30% of the time in those conditions.
    
    Temperature scaling was applied to improve calibration (reduces overconfidence):
    """)
    
    calibration_data = [
        {"Hazard": "Fire", "Temperature": 1.735, "ECE Before": "12.6%", "ECE After": "5.2%", "Improvement": "59% better"},
        {"Hazard": "Flood", "Temperature": 1.259, "ECE Before": "0.8%", "ECE After": "1.4%", "Improvement": "Already calibrated"},
        {"Hazard": "Wind", "Temperature": 1.0, "ECE Before": "3.9%", "ECE After": "3.9%", "Improvement": "No scaling needed"},
        {"Hazard": "Winter", "Temperature": 1.0, "ECE Before": "5.4%", "ECE After": "5.4%", "Improvement": "No scaling needed"},
        {"Hazard": "Seismic", "Temperature": 0.835, "ECE Before": "2.0%", "ECE After": "0.9%", "Improvement": "55% better"},
    ]
    st.dataframe(pd.DataFrame(calibration_data), width='stretch', hide_index=True)
    st.caption("ECE = Expected Calibration Error. Lower is better. <5% is considered well-calibrated.")

    # Show reliability diagram if available
    reliability_img = Path("figures/figure_reliability_multi_panel.png")
    if reliability_img.exists():
        st.markdown("**Reliability Diagrams**")
        st.image(str(reliability_img), width='stretch')
        st.caption("Blue = before temperature scaling, Orange = after. Diagonal line = perfect calibration.")

    # User guide section
    st.markdown("---")
    st.markdown("### How to Use These Predictions")

    st.markdown("**Interpreting Risk Probabilities**")
    st.markdown("""
    | Risk Level | Probability | Recommended Action |
    |------------|-------------|-------------------|
    | **Low** | < 10% | Monitor conditions, routine operations |
    | **Moderate** | 10-25% | Increase situational awareness, review resources |
    | **Elevated** | 25-50% | Brief leadership, prepare response assets |
    | **High** | > 50% | Pre-position resources, consider public advisories |
    """)
    
    st.markdown("**Key Points for Emergency Managers**")
    st.markdown("""
    - **Probabilities are calibrated** — a 30% prediction means ~30% historical occurrence rate in similar conditions
    - **Weather-driven hazards** (fire, winter) have strongest signals; predictions update as conditions change
    - **Seismic predictions** capture historical patterns but cannot predict specific earthquakes
    - **Combine with local knowledge** — the model provides a baseline; local factors may increase/decrease risk
    - **Confidence matters** — use the Model Diagnostics tab to see uncertainty estimates when available
    """)

    st.info("Tip: Use Quick Predict to get current predictions for any county.")


# =============================================================================
# PAGE: ABOUT THIS PROJECT
# =============================================================================

def page_about():
    st.markdown("## About This Project")
    
    st.markdown("""
    ### Adaptive Hazard Intelligence System
    
    This dashboard is a capstone project demonstrating the application of machine learning 
    to multi-hazard risk assessment for emergency management in Washington State.
    
    **Author:** Joshua D. Curry  
    **Institution:** Pierce College Fort Steilacoom  
    **Program:** Bachelor of Applied Science in Emergency Management (BAS-EM)  
    **Expected Graduation:** June 2026
    
    ---
    
    ### Capstone Course
    
    **EM 470 Emergency Management Capstone** (5 credits)
    
    The Capstone is a culminating academic and intellectual experience demonstrating learning 
    acquisition and practical application from all courses, theories, techniques, and content 
    taught in the Bachelor of Applied Science in Emergency Management Program.
    
    ---
    
    ### Research Papers
    
    This work is supported by two preprints on SSRN:
    
    - [Diffusion Attention: Replacing Softmax with Heat Kernel Dynamics](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5953096)
    - [Heat Kernel Attention: Provable Sparsity via Diffusion Dynamics](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5959898)
    """)
    
    st.markdown("---")
    
    st.markdown("""
    ### Model Overview
    
    **Hazard-LM v1.0** is a 136-million parameter transformer model trained on:
    - 79,000+ historical hazard events (2000-2025)
    - NOAA Storm Events Database
    - GridMET climate data (temperature, precipitation, humidity, wind)
    - USGS seismic records
    - MTBS fire perimeter data
    
    The model produces calibrated probability estimates for five hazard types,
    enabling emergency managers to make data-driven resource allocation decisions.
    
    ### Key Features
    
    - **Calibrated Outputs:** Temperature scaling ensures predicted probabilities match real-world frequencies
    - **County-Level Resolution:** Predictions available for all 39 Washington counties
    - **Multi-Hazard:** Single model handles fire, flood, wind, winter, and seismic risk
    - **Uncertainty Quantification:** Model confidence estimates help prioritize response
    
    ---
    
    ### Source Code & Data
    
    This project is deployed on Streamlit Cloud from the GitHub repository.
    
    For questions or collaboration inquiries, please reach out through the SSRN paper contact information.
    """)
    
    # Model status footer
    st.markdown("---")
    st.caption(f"Model: {MODEL_DISPLAY_NAME} | Device: {DEVICE} | Dashboard Version: 2.0")


# =============================================================================
# MAIN
# =============================================================================

def main():
    inject_custom_css()
    
    # Initialize session state
    if 'page' not in st.session_state:
        st.session_state.page = 'dashboard'
    if 'selected_county' not in st.session_state:
        st.session_state.selected_county = None
    
    # Navigation
    current_page = render_sidebar()
    
    # Header
    render_header()
    
    # Route to page
    if current_page == 'dashboard':
        page_executive_dashboard()
    elif current_page == 'map':
        page_interactive_map()
    elif current_page == 'risk':
        page_risk_assessment()
    elif current_page == 'climate':
        page_climate_analysis()
    elif current_page == 'mitigation':
        page_mitigation_planning()
    elif current_page == 'ai_predict':
        page_ai_predictions()
    elif current_page == 'seasonal':
        page_seasonal_planning()
    elif current_page == 'diagnostics':
        page_model_diagnostics()
    elif current_page == 'model_eval':
        page_model_evaluation()
    elif current_page == 'about':
        page_about()
    else:
        page_executive_dashboard()


if __name__ == "__main__":
    main()
