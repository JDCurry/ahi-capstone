# Decision Audit Integration for AHI Dashboard
# =============================================
# 
# This file shows how to integrate the decision_audit.py module into
# vasta_dashboard.py to add evidentiary basis for risk predictions.

"""
STEP 1: Add import at the top of vasta_dashboard.py (around line 100)
------------------------------------------------------------------------
"""

# Add this import near the other local imports:
try:
    from decision_audit import build_decision_audit, render_decision_audit_html, render_decision_audit_compact
    DECISION_AUDIT_AVAILABLE = True
except ImportError:
    DECISION_AUDIT_AVAILABLE = False
    build_decision_audit = None
    render_decision_audit_html = None


"""
STEP 2: Modify the Quick Predict results rendering (around line 2470)
------------------------------------------------------------------------
After the "How to interpret these numbers" expander, add the Decision Audit section.

Find this section (approximately line 2450):

                    # Interpretation guide
                    with st.expander("How to interpret these numbers", expanded=False):
                        st.markdown(f\"\"\"
                        **What the percentages mean:**
                        ...
                        \"\"\")

Then ADD this new section right after the expander closes:
"""

# === DECISION AUDIT INTEGRATION CODE ===
# Insert this after the "How to interpret these numbers" expander (around line 2470)

DECISION_AUDIT_CODE = '''
                    # Decision Audit - Evidentiary basis for top hazard
                    if DECISION_AUDIT_AVAILABLE and sorted_risks:
                        st.markdown("---")
                        with st.expander("Decision Audit (Evidentiary Basis)", expanded=False):
                            st.markdown("""
                            **What is this?** The Decision Audit provides the evidentiary basis for 
                            each risk prediction. Use this information to support defensible decisions 
                            and cite specific data sources in briefings.
                            """)
                            
                            # Get county stats for the audit
                            audit_county_stats = compute_county_stats(df) if df is not None else None
                            
                            # Build audit for top 2 hazards
                            for hazard, prob in sorted_risks[:2]:
                                try:
                                    audit = build_decision_audit(
                                        county=selected_county,
                                        hazard=hazard,
                                        probability=prob,
                                        county_stats=audit_county_stats,
                                        forecast_date=sel_date,
                                        horizon_days=MAX_FORECAST_DAYS
                                    )
                                    
                                    # Render the audit
                                    audit_html = render_decision_audit_html(audit, COLORS)
                                    st.markdown(audit_html, unsafe_allow_html=True)
                                    
                                except Exception as e:
                                    st.warning(f"Could not generate audit for {hazard}: {e}")
                            
                            # Data sources reference
                            st.markdown("**Data Sources:**")
                            st.markdown("""
                            | Source | Data Provided |
                            |--------|---------------|
                            | NOAA GridMET | Daily weather variables |
                            | NOAA Storm Events | Historical hazard events |
                            | US Census | Population demographics |
                            | CDC SVI | Social vulnerability scores |
                            | USGS NLCD | Land cover classification |
                            | USGS Earthquakes | Seismic event catalog |
                            | FEMA | Disaster declarations |
                            """)
                            
                            st.markdown(f"""
                            <p style="color: {COLORS['text_tertiary']}; font-size: 11px; margin-top: 12px;">
                            Model: Hazard-LM v1.0 | Training: 370,000+ observations | 
                            Validation: Held-out test set (37,039 samples) | 
                            Calibration: Diffusion attention (Curry, 2025)
                            </p>
                            """, unsafe_allow_html=True)
'''


"""
STEP 3: Also add a compact audit line below each hazard card (optional)
------------------------------------------------------------------------
This provides a quick citation without needing to expand the full audit.

In the hazard rendering loop (around line 2440), after the suggested action,
add the compact audit:
"""

COMPACT_AUDIT_CODE = '''
                    # After the "Suggested action" line for each hazard:
                    if DECISION_AUDIT_AVAILABLE:
                        try:
                            audit = build_decision_audit(
                                county=selected_county,
                                hazard=hazard,
                                probability=prob,
                                county_stats=audit_county_stats,
                                forecast_date=sel_date,
                                horizon_days=MAX_FORECAST_DAYS
                            )
                            compact_html = render_decision_audit_compact(audit, COLORS)
                            st.markdown(compact_html, unsafe_allow_html=True)
                        except Exception:
                            pass
'''


"""
ALTERNATIVE: Simpler integration without the full module
------------------------------------------------------------------------
If you want a simpler approach without importing the module, you can add
this inline function directly in the dashboard:
"""

INLINE_AUDIT_FUNCTION = '''
def _render_inline_audit(county: str, hazard: str, prob: float, county_stats, sel_date, horizon_days: int):
    """Render a simple inline decision audit."""
    
    # Model AUC scores
    auc_scores = {'fire': 0.96, 'flood': 0.90, 'wind': 0.90, 'winter': 0.96, 'seismic': 0.85}
    auc = auc_scores.get(hazard, 0.90)
    
    # Get county events
    events = 0
    days = 0
    if county_stats is not None:
        row = county_stats[county_stats['county'].str.upper() == county.upper()]
        if not row.empty:
            events = int(row.iloc[0].get(f'{hazard}_events', 0))
            days = int(row.iloc[0].get('days_observed', 0))
    
    # Seasonal context
    month = sel_date.month if hasattr(sel_date, 'month') else datetime.now().month
    seasons = {
        (12,1,2): ('Winter', ['winter', 'flood']),
        (3,4,5): ('Spring', ['flood', 'wind']),
        (6,7,8): ('Summer', ['fire']),
        (9,10,11): ('Fall', ['wind', 'flood', 'fire']),
    }
    season_name = 'Unknown'
    is_primary = False
    for months, (name, hazards) in seasons.items():
        if month in months:
            season_name = name
            is_primary = hazard in hazards
            break
    
    pct = prob * 100
    
    return f"""
    <div style="background: {COLORS['elevated_bg']}; border-radius: 6px; padding: 12px; margin: 12px 0; border-left: 3px solid {COLORS.get(hazard, COLORS['primary'])};">
        <strong style="color: {COLORS['text_primary']};">Decision Audit: {hazard.title()} ({pct:.1f}%)</strong>
        <table style="width: 100%; margin-top: 8px; font-size: 13px;">
            <tr>
                <td style="color: {COLORS['text_secondary']}; padding: 4px 0;">Historical events</td>
                <td style="color: {COLORS['text_primary']};">{events:,} events ({days:,} days observed)</td>
            </tr>
            <tr>
                <td style="color: {COLORS['text_secondary']}; padding: 4px 0;">Season</td>
                <td style="color: {COLORS['text_primary']};">{season_name} (primary hazard: {'Yes' if is_primary else 'No'})</td>
            </tr>
            <tr>
                <td style="color: {COLORS['text_secondary']}; padding: 4px 0;">Model AUC</td>
                <td style="color: {COLORS['text_primary']};">{auc:.2f} ({"Excellent" if auc >= 0.95 else "Strong" if auc >= 0.90 else "Good"})</td>
            </tr>
        </table>
        <p style="color: {COLORS['text_tertiary']}; font-size: 12px; margin: 8px 0 0 0; font-style: italic;">
            This {pct:.0f}% estimate reflects conditions similar to those preceding {hazard} events 
            approximately {pct:.0f}% of the time in historical data (2000-2025).
        </p>
    </div>
    """
'''


"""
USAGE EXAMPLE
-------------
Once integrated, the Quick Predict results will show:

1. Hazard cards with percentages (existing)
2. "Top Hazards for This Period" with guidance (existing)
3. "How to interpret these numbers" expander (existing)
4. NEW: "Decision Audit (Evidentiary Basis)" expander containing:
   - Table with historical events, training observations, season, AUC
   - Calibration statement explaining what the probability means
   - Data sources reference

This gives emergency managers a citable, defensible basis for their decisions.
"""


# Quick test
if __name__ == "__main__":
    # Test the module
    from datetime import datetime
    from decision_audit import build_decision_audit, render_decision_audit_html
    
    # Mock colors
    colors = {
        'card_bg': '#1a1f26',
        'elevated_bg': '#1f2430',
        'text_primary': '#ffffff',
        'text_secondary': '#a0aec0',
        'text_tertiary': '#718096',
        'border': '#2d3748',
        'primary': '#0d7dc1',
        'fire': '#ef4444',
    }
    
    # Build a test audit
    audit = build_decision_audit(
        county="King",
        hazard="fire",
        probability=0.34,
        county_stats=None,  # Would normally pass the DataFrame
        forecast_date=datetime.now(),
        horizon_days=14
    )
    
    print("Audit packet:")
    for k, v in audit.items():
        print(f"  {k}: {v}")
    
    print("\nHTML output:")
    html = render_decision_audit_html(audit, colors)
    print(html[:500] + "...")
