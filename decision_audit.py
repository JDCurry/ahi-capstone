"""
Decision Audit Module for Adaptive Hazard Intelligence
=======================================================
Generates evidentiary basis for risk predictions to support defensible decisions.

This module provides transparency about what data underlies each prediction,
enabling emergency managers to cite specific evidence in briefings and reports.

Calibration transparency: shows what pipeline adjustments (temperature scaling,
seasonal priors, base-rate ceilings) were applied to each prediction.
"""

from datetime import datetime
from typing import Dict, Optional, Tuple
import pandas as pd


# Model performance metrics (validated against held-out test data)
# Source: HazardLM-Diffusion v2.0 test evaluation on clean labels
# These are the ACTUAL diffusion model AUCs used by Quick Predict via inference_core.py
MODEL_METRICS = {
    'fire':    {'auc': 0.731, 'training_obs': 370000, 'data_range': '2000-2025'},
    'flood':   {'auc': 0.648, 'training_obs': 370000, 'data_range': '2000-2025'},
    'wind':    {'auc': 0.585, 'training_obs': 370000, 'data_range': '2000-2025'},
    'winter':  {'auc': 0.742, 'training_obs': 370000, 'data_range': '2000-2025'},
    'seismic': {'auc': 0.499, 'training_obs': 370000, 'data_range': '2000-2025'},
}

# XGBoost baseline AUCs (shown for comparison only)
XGBOOST_METRICS = {
    'fire':    {'auc': 0.870},
    'flood':   {'auc': 0.714},
    'wind':    {'auc': 0.713},
    'winter':  {'auc': 0.885},
    'seismic': {'auc': 0.721},
}

# Data source citations
DATA_SOURCES = {
    'weather': 'NOAA GridMET (daily temperature, precipitation, humidity, wind)',
    'events': 'NOAA Storm Events Database',
    'population': 'US Census Bureau',
    'vulnerability': 'CDC Social Vulnerability Index (SVI)',
    'land_cover': 'USGS National Land Cover Database (NLCD)',
    'seismic': 'USGS Earthquake Hazards Program',
    'disasters': 'FEMA Disaster Declarations & NIFC Wildfire Data',
}

# Seasonal context
SEASONAL_CONTEXT = {
    1: {'season': 'Winter', 'primary_hazards': ['winter', 'flood'], 'note': 'Peak winter storm season; flood risk from rain-on-snow events'},
    2: {'season': 'Winter', 'primary_hazards': ['winter', 'flood'], 'note': 'Continued winter storm risk; early snowmelt flooding possible'},
    3: {'season': 'Spring', 'primary_hazards': ['flood', 'wind'], 'note': 'Snowmelt flooding; transitional wind patterns; winter storms declining'},
    4: {'season': 'Spring', 'primary_hazards': ['flood', 'wind'], 'note': 'Peak snowmelt period; spring wind events'},
    5: {'season': 'Spring', 'primary_hazards': ['flood', 'fire'], 'note': 'Late snowmelt; early fire season in eastern WA'},
    6: {'season': 'Summer', 'primary_hazards': ['fire', 'wind'], 'note': 'Fire season beginning; dry conditions developing'},
    7: {'season': 'Summer', 'primary_hazards': ['fire'], 'note': 'Peak fire season; critical fire weather days likely'},
    8: {'season': 'Summer', 'primary_hazards': ['fire'], 'note': 'Peak fire season continues; highest historical fire frequency'},
    9: {'season': 'Fall', 'primary_hazards': ['fire', 'wind'], 'note': 'Fire season waning; fall wind events increasing'},
    10: {'season': 'Fall', 'primary_hazards': ['wind', 'flood'], 'note': 'Atmospheric river season begins; wind events common'},
    11: {'season': 'Fall', 'primary_hazards': ['winter', 'flood', 'wind'], 'note': 'Transition to winter; early snow possible at elevation'},
    12: {'season': 'Winter', 'primary_hazards': ['winter', 'flood'], 'note': 'Winter storm season; holiday travel impacts likely'},
}

# Calibration parameters (mirrors inference_core.py for display purposes)
_SEASONAL_LOGIT_BIAS = {
    'fire': {1: -3.0, 2: -3.0, 3: -2.0, 4: -1.0, 5: -0.5, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: -0.5, 11: -2.0, 12: -3.0},
    'winter': {1: 0.0, 2: 0.0, 3: -0.5, 4: -0.5, 5: -1.5, 6: -3.0, 7: -3.0, 8: -3.0, 9: -2.0, 10: -0.5, 11: 0.0, 12: 0.0},
    'wind': {1: 0.0, 2: 0.0, 3: 0.0, 4: -0.3, 5: -0.5, 6: -0.5, 7: -0.5, 8: -0.3, 9: 0.0, 10: 0.0, 11: 0.0, 12: 0.0},
    'flood': {m: 0.0 for m in range(1, 13)},
    'seismic': {m: 0.0 for m in range(1, 13)},
}

_TEMPERATURE_SCALES = {
    'fire': 0.436, 'flood': 0.272, 'wind': 1.000, 'winter': 0.660, 'seismic': 1.000,
}

_WEAK_HEAD_BIAS = {'wind': -1.5, 'seismic': -2.5}

_SEASONAL_CEILING = {
    'winter': {1: 0.35, 2: 0.35, 3: 0.25, 4: 0.15, 5: 0.08, 6: 0.05, 7: 0.05, 8: 0.05, 9: 0.08, 10: 0.20, 11: 0.35, 12: 0.35},
}

_BASE_RATE_CEILING = {'fire': 0.35, 'flood': 0.25, 'wind': 0.15, 'winter': 0.35, 'seismic': 0.08}


def get_seasonal_context(date: datetime) -> dict:
    """Get seasonal context for a given date."""
    month = date.month if hasattr(date, 'month') else datetime.now().month
    return SEASONAL_CONTEXT.get(month, SEASONAL_CONTEXT[1])


def build_decision_audit(
    county: str,
    hazard: str,
    probability: float,
    county_stats: Optional[pd.DataFrame] = None,
    forecast_date: Optional[datetime] = None,
    horizon_days: int = 14
) -> dict:
    """
    Build a decision audit packet for a single hazard prediction.

    Args:
        county: County name
        hazard: Hazard type (fire, flood, wind, winter, seismic)
        probability: Predicted probability (0-1)
        county_stats: DataFrame with county-level statistics (from compute_county_stats)
        forecast_date: Date of the forecast
        horizon_days: Forecast horizon in days

    Returns:
        Dictionary containing audit information with calibration transparency
    """
    if forecast_date is None:
        forecast_date = datetime.now()

    month = forecast_date.month if hasattr(forecast_date, 'month') else 1

    # Get model metrics
    metrics = MODEL_METRICS.get(hazard, MODEL_METRICS['fire'])
    xgb_metrics = XGBOOST_METRICS.get(hazard, {})

    # Get seasonal context
    seasonal = get_seasonal_context(forecast_date)

    # --- Calibration pipeline details ---
    T = _TEMPERATURE_SCALES.get(hazard, 1.0)
    seasonal_bias = _SEASONAL_LOGIT_BIAS.get(hazard, {}).get(month, 0.0)
    weak_bias = _WEAK_HEAD_BIAS.get(hazard, 0.0)

    # Determine effective ceiling
    if hazard in _SEASONAL_CEILING and month in _SEASONAL_CEILING[hazard]:
        ceiling = _SEASONAL_CEILING[hazard][month]
        ceiling_source = 'seasonal'
    else:
        ceiling = _BASE_RATE_CEILING.get(hazard, 1.0)
        ceiling_source = 'base'
    ceiling_hit = probability >= (ceiling - 0.001)

    # Build calibration trace
    calibration_steps = []
    calibration_steps.append(
        f"Temperature scaling: T={T:.3f}"
        + (" (sharpens predictions)" if T < 1.0 else " (softens predictions)" if T > 1.0 else " (clamped, weak head)")
    )
    if weak_bias != 0.0:
        calibration_steps.append(
            f"Weak-head recalibration: {weak_bias:+.1f} logit bias (anchors near-random head to base rate)"
        )
    if seasonal_bias != 0.0:
        calibration_steps.append(
            f"Seasonal prior: {seasonal_bias:+.1f} logit ({'suppression' if seasonal_bias < 0 else 'boost'} for {forecast_date.strftime('%B') if hasattr(forecast_date, 'strftime') else 'this month'})"
        )
    else:
        calibration_steps.append(
            f"Seasonal prior: 0.0 (no adjustment - within peak/normal season for {forecast_date.strftime('%B') if hasattr(forecast_date, 'strftime') else 'this month'})"
        )
    if ceiling_hit:
        calibration_steps.append(
            f"Probability ceiling applied: capped at {ceiling*100:.0f}% (max plausible for {forecast_date.strftime('%B') if hasattr(forecast_date, 'strftime') else 'this month'})"
        )

    # Build audit packet
    audit = {
        'county': county,
        'hazard': hazard,
        'probability': probability,
        'probability_pct': f"{probability * 100:.1f}%",
        'forecast_date': forecast_date.strftime('%Y-%m-%d') if hasattr(forecast_date, 'strftime') else str(forecast_date),
        'horizon_days': horizon_days,
        'model': {
            'name': 'HazardLM-Diffusion v2.0',
            'auc': metrics['auc'],
            'auc_interpretation': _interpret_auc(metrics['auc']),
            'xgboost_auc': xgb_metrics.get('auc', 0),
            'training_observations': metrics['training_obs'],
            'data_range': metrics['data_range'],
            'calibration_method': 'Temperature scaling + seasonal prior + base-rate ceiling',
            'head_quality': _head_quality(metrics['auc']),
        },
        'seasonal_context': {
            'season': seasonal['season'],
            'month': forecast_date.strftime('%B') if hasattr(forecast_date, 'strftime') else '',
            'primary_hazards': seasonal['primary_hazards'],
            'is_primary': hazard in seasonal['primary_hazards'],
            'note': seasonal['note'],
        },
        'calibration': {
            'temperature': T,
            'seasonal_bias': seasonal_bias,
            'weak_head_bias': weak_bias,
            'ceiling': ceiling,
            'ceiling_source': ceiling_source,
            'ceiling_hit': ceiling_hit,
            'steps': calibration_steps,
        },
        'data_sources': DATA_SOURCES,
    }

    # Add county-specific statistics if available
    if county_stats is not None and not county_stats.empty:
        county_row = county_stats[county_stats['county'].str.upper() == county.upper()]
        if not county_row.empty:
            row = county_row.iloc[0]
            event_col = f'{hazard}_events'
            rate_col = f'{hazard}_rate'

            audit['county_data'] = {
                'historical_events': int(row.get(event_col, 0)),
                'days_observed': int(row.get('days_observed', 0)),
                'annualized_rate': float(row.get(rate_col, 0)) if pd.notna(row.get(rate_col)) else 0,
                'population': int(row.get('population', 0)),
                'svi_score': float(row.get('svi_score', 0)) if pd.notna(row.get('svi_score')) else 0,
            }

    # Generate plain-language calibration statement
    audit['calibration_statement'] = _generate_calibration_statement(
        hazard, probability, county, audit.get('county_data', {}), audit['calibration']
    )

    return audit


def _interpret_auc(auc: float) -> str:
    """Interpret AUC score in plain language.

    Thresholds calibrated for the actual model performance range (0.50-0.74).
    """
    if auc >= 0.80:
        return "Strong discrimination"
    elif auc >= 0.70:
        return "Good discrimination"
    elif auc >= 0.60:
        return "Fair discrimination"
    elif auc >= 0.55:
        return "Weak discrimination"
    else:
        return "Near-random (base-rate estimate only)"


def _head_quality(auc: float) -> str:
    """Return a short quality label for the prediction head."""
    if auc >= 0.70:
        return "Good"
    elif auc >= 0.60:
        return "Fair"
    else:
        return "Weak"


def _generate_calibration_statement(
    hazard: str, probability: float, county: str,
    county_data: dict, calibration: dict
) -> str:
    """Generate a plain-language statement about what the probability means,
    including calibration transparency."""
    pct = probability * 100
    events = county_data.get('historical_events', 0)
    days = county_data.get('days_observed', 0)

    # Core probability statement
    if events > 0 and days > 0:
        statement = (
            f"This {pct:.0f}% {hazard} risk estimate is based on statewide learned patterns "
            f"from 25 years of Washington State hazard data. "
            f"{county} County has experienced {events:,} {hazard} events over {days:,} observed days "
            f"in the training data (2000-2025)."
        )
    elif events == 0 and days > 0:
        statement = (
            f"This {pct:.0f}% {hazard} risk estimate is driven by statewide learned patterns "
            f"and seasonal priors, not county-specific history. "
            f"{county} County has 0 recorded {hazard} events in the training data (2000-2025). "
            f"The prediction reflects conditions that preceded {hazard} events in other WA counties."
        )
    else:
        statement = (
            f"This {pct:.0f}% {hazard} risk estimate is calibrated against 25 years of "
            f"Washington State hazard data using statewide learned patterns."
        )

    # Add ceiling note if hit
    if calibration.get('ceiling_hit'):
        statement += (
            f" Note: this value is capped at the maximum plausible probability "
            f"({calibration['ceiling']*100:.0f}%) for this hazard in this month."
        )

    return statement


def render_decision_audit_html(audit: dict, colors: dict) -> str:
    """
    Render the decision audit as markdown for Streamlit display.

    Args:
        audit: Decision audit dictionary from build_decision_audit
        colors: Color scheme dictionary from dashboard

    Returns:
        Markdown string for st.markdown()
    """
    hazard = audit['hazard']
    prob_pct = audit['probability_pct']
    county = audit['county']

    # Get county data if available
    county_data = audit.get('county_data', {})
    events = county_data.get('historical_events', 0)
    days = county_data.get('days_observed', 0)

    # Format for display
    events_str = f"{events:,}" if isinstance(events, (int, float)) else str(events)
    days_str = f"{days:,}" if isinstance(days, (int, float)) else str(days)

    # Model info
    model = audit['model']
    auc = model['auc']
    auc_interp = model['auc_interpretation']
    head_quality = model.get('head_quality', '')
    xgb_auc = model.get('xgboost_auc', 0)

    # Seasonal
    seasonal = audit['seasonal_context']
    is_primary = "Yes" if seasonal['is_primary'] else "No"

    # Calibration
    cal = audit.get('calibration', {})
    ceiling_note = ""
    if cal.get('ceiling_hit'):
        ceiling_note = f" *(capped at {cal['ceiling']*100:.0f}% ceiling)*"

    # Historical context note for 0 events
    hist_note = ""
    if events == 0:
        hist_note = " Prediction based on statewide patterns, not county-specific history."

    # Build calibration steps as markdown list
    cal_steps = cal.get('steps', [])
    cal_steps_md = ""
    if cal_steps:
        cal_steps_md = "\n\n**Calibration pipeline applied:**\n"
        for i, step in enumerate(cal_steps, 1):
            cal_steps_md += f"- {step}\n"

    # Build markdown table (renders better in Streamlit)
    markdown = f"""
**Decision Audit: {county} County - {hazard.title()} Risk ({prob_pct}){ceiling_note}**

| Factor | Value |
|--------|-------|
| Historical events (2000-2025) | {events_str} {hazard} events |
| Training observations | {days_str} county-days |
| Current season | {seasonal['season']} ({seasonal['month']}) |
| Primary hazard this season? | {is_primary} |
| Model AUC (Diffusion) | {auc:.3f} ({auc_interp}) |
| Baseline AUC (XGBoost) | {xgb_auc:.3f} |
| Head quality | {head_quality} |

*{audit['calibration_statement']}*
{cal_steps_md}
---
"""

    return markdown


def render_decision_audit_compact(audit: dict, colors: dict) -> str:
    """
    Render a compact single-line audit citation.

    Useful for embedding in reports or briefings.
    """
    hazard = audit['hazard']
    prob_pct = audit['probability_pct']
    county = audit['county']
    model = audit['model']
    county_data = audit.get('county_data', {})
    events = county_data.get('historical_events', 0)
    cal = audit.get('calibration', {})
    ceiling_tag = ", ceiling applied" if cal.get('ceiling_hit') else ""

    return (
        f"**Audit:** {prob_pct} {hazard} risk | "
        f"{events:,} historical events in {county} County (2000-2025) | "
        f"Model AUC: {model['auc']:.2f} ({model.get('head_quality', '')}){ceiling_tag}"
    )


# Export main functions
__all__ = [
    'build_decision_audit',
    'render_decision_audit_html',
    'render_decision_audit_compact',
    'get_seasonal_context',
]
