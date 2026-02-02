"""
Decision Audit Module for Adaptive Hazard Intelligence
=======================================================
Generates evidentiary basis for risk predictions to support defensible decisions.

This module provides transparency about what data underlies each prediction,
enabling emergency managers to cite specific evidence in briefings and reports.
"""

from datetime import datetime
from typing import Dict, Optional, Tuple
import pandas as pd


# Model performance metrics (validated against held-out test data)
MODEL_METRICS = {
    'fire': {'auc': 0.96, 'training_obs': 370000, 'data_range': '2000-2025'},
    'flood': {'auc': 0.90, 'training_obs': 370000, 'data_range': '2000-2025'},
    'wind': {'auc': 0.90, 'training_obs': 370000, 'data_range': '2000-2025'},
    'winter': {'auc': 0.96, 'training_obs': 370000, 'data_range': '2000-2025'},
    'seismic': {'auc': 0.85, 'training_obs': 370000, 'data_range': '2000-2025'},
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
    3: {'season': 'Spring', 'primary_hazards': ['flood', 'wind'], 'note': 'Snowmelt flooding; transitional wind patterns'},
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
        Dictionary containing audit information
    """
    if forecast_date is None:
        forecast_date = datetime.now()
    
    # Get model metrics
    metrics = MODEL_METRICS.get(hazard, MODEL_METRICS['fire'])
    
    # Get seasonal context
    seasonal = get_seasonal_context(forecast_date)
    
    # Build audit packet
    audit = {
        'county': county,
        'hazard': hazard,
        'probability': probability,
        'probability_pct': f"{probability * 100:.1f}%",
        'forecast_date': forecast_date.strftime('%Y-%m-%d') if hasattr(forecast_date, 'strftime') else str(forecast_date),
        'horizon_days': horizon_days,
        'model': {
            'name': 'Hazard-LM v1.0',
            'auc': metrics['auc'],
            'auc_interpretation': _interpret_auc(metrics['auc']),
            'training_observations': metrics['training_obs'],
            'data_range': metrics['data_range'],
            'calibration_method': 'Diffusion attention + temperature scaling',
        },
        'seasonal_context': {
            'season': seasonal['season'],
            'month': forecast_date.strftime('%B') if hasattr(forecast_date, 'strftime') else '',
            'primary_hazards': seasonal['primary_hazards'],
            'is_primary': hazard in seasonal['primary_hazards'],
            'note': seasonal['note'],
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
        hazard, probability, county, audit.get('county_data', {})
    )
    
    return audit


def _interpret_auc(auc: float) -> str:
    """Interpret AUC score in plain language."""
    if auc >= 0.95:
        return "Excellent discrimination"
    elif auc >= 0.90:
        return "Strong discrimination"
    elif auc >= 0.85:
        return "Good discrimination"
    elif auc >= 0.80:
        return "Acceptable discrimination"
    else:
        return "Limited discrimination"


def _generate_calibration_statement(hazard: str, probability: float, county: str, county_data: dict) -> str:
    """Generate a plain-language statement about what the probability means."""
    pct = probability * 100
    
    # Base statement
    if pct < 15:
        level = "low"
        historical_context = "rarely"
    elif pct < 30:
        level = "moderate"
        historical_context = "occasionally"
    elif pct < 50:
        level = "elevated"
        historical_context = "with some frequency"
    elif pct < 70:
        level = "high"
        historical_context = "frequently"
    else:
        level = "severe"
        historical_context = "very frequently"
    
    # Build statement
    events = county_data.get('historical_events', 0)
    days = county_data.get('days_observed', 0)
    
    if events > 0 and days > 0:
        statement = (
            f"This {pct:.0f}% {hazard} risk estimate indicates conditions similar to those "
            f"that preceded {hazard} events approximately {pct:.0f}% of the time historically. "
            f"{county} County has experienced {events:,} {hazard} events over {days:,} observed days "
            f"in the training data (2000-2025)."
        )
    else:
        statement = (
            f"This {pct:.0f}% {hazard} risk estimate indicates conditions similar to those "
            f"that preceded {hazard} events approximately {pct:.0f}% of the time historically. "
            f"The probability is calibrated against 25 years of Washington State hazard data."
        )
    
    return statement


def render_decision_audit_html(audit: dict, colors: dict) -> str:
    """
    Render the decision audit as HTML for Streamlit display.
    
    Args:
        audit: Decision audit dictionary from build_decision_audit
        colors: Color scheme dictionary from dashboard
    
    Returns:
        HTML string for st.markdown(unsafe_allow_html=True)
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
    
    # Seasonal
    seasonal = audit['seasonal_context']
    is_primary = "Yes" if seasonal['is_primary'] else "No"
    
    # Build HTML table
    html = f"""
    <div style="background: {colors.get('card_bg', '#1a1f26')}; border-radius: 8px; padding: 16px; margin: 16px 0; border-left: 4px solid {colors.get(hazard, colors.get('primary', '#0d7dc1'))};">
        <h4 style="color: {colors.get('text_primary', '#ffffff')}; margin: 0 0 12px 0;">
            Decision Audit: {county} County - {hazard.title()} Risk ({prob_pct})
        </h4>
        
        <table style="width: 100%; border-collapse: collapse; font-size: 14px;">
            <tr style="border-bottom: 1px solid {colors.get('border', '#2d3748')};">
                <td style="padding: 8px 0; color: {colors.get('text_secondary', '#a0aec0')}; width: 40%;">Historical events (2000-2025)</td>
                <td style="padding: 8px 0; color: {colors.get('text_primary', '#ffffff')};">{events_str} {hazard} events</td>
            </tr>
            <tr style="border-bottom: 1px solid {colors.get('border', '#2d3748')};">
                <td style="padding: 8px 0; color: {colors.get('text_secondary', '#a0aec0')};">Training observations</td>
                <td style="padding: 8px 0; color: {colors.get('text_primary', '#ffffff')};">{days_str} county-days</td>
            </tr>
            <tr style="border-bottom: 1px solid {colors.get('border', '#2d3748')};">
                <td style="padding: 8px 0; color: {colors.get('text_secondary', '#a0aec0')};">Current season</td>
                <td style="padding: 8px 0; color: {colors.get('text_primary', '#ffffff')};">{seasonal['season']} ({seasonal['month']})</td>
            </tr>
            <tr style="border-bottom: 1px solid {colors.get('border', '#2d3748')};">
                <td style="padding: 8px 0; color: {colors.get('text_secondary', '#a0aec0')};">Primary hazard this season?</td>
                <td style="padding: 8px 0; color: {colors.get('text_primary', '#ffffff')};">{is_primary}</td>
            </tr>
            <tr style="border-bottom: 1px solid {colors.get('border', '#2d3748')};">
                <td style="padding: 8px 0; color: {colors.get('text_secondary', '#a0aec0')};">Model discrimination (AUC)</td>
                <td style="padding: 8px 0; color: {colors.get('text_primary', '#ffffff')};">{auc:.2f} ({auc_interp})</td>
            </tr>
            <tr>
                <td style="padding: 8px 0; color: {colors.get('text_secondary', '#a0aec0')};">Calibration method</td>
                <td style="padding: 8px 0; color: {colors.get('text_primary', '#ffffff')};">{model['calibration_method']}</td>
            </tr>
        </table>
        
        <p style="color: {colors.get('text_secondary', '#a0aec0')}; font-style: italic; margin: 12px 0 0 0; font-size: 13px; line-height: 1.5;">
            {audit['calibration_statement']}
        </p>
        
        <p style="color: {colors.get('text_tertiary', '#718096')}; font-size: 11px; margin: 12px 0 0 0;">
            Data sources: {', '.join(list(audit['data_sources'].keys())[:4])} | Model: {model['name']}
        </p>
    </div>
    """
    
    return html


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
    
    return f"""
    <p style="color: {colors.get('text_tertiary', '#718096')}; font-size: 12px; margin: 8px 0; padding: 8px; background: {colors.get('elevated_bg', '#1f2430')}; border-radius: 4px;">
        <strong>Audit:</strong> {prob_pct} {hazard} risk based on {events:,} historical events in {county} County (2000-2025). 
        Model AUC: {model['auc']:.2f}. Calibration: diffusion attention + temperature scaling.
    </p>
    """


# Export main functions
__all__ = [
    'build_decision_audit',
    'render_decision_audit_html',
    'render_decision_audit_compact',
    'get_seasonal_context',
]
