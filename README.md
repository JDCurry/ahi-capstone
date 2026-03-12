# Adaptive Hazard Intelligence (AHI)

> **Research Snapshot** — This repository is archived as a research artifact from the Pierce College BAS-EM capstone project. Active development continues in a private repository. Training code is available upon request for research collaboration.

A calibrated, uncertainty-aware multi-hazard risk engine for Washington State emergency managers.

### Live Dashboard

The Streamlit dashboard remains fully operational for capstone demonstration purposes.

```bash
pip install -r requirements.txt
streamlit run vasta_dashboard.py
```

### Model Architecture

**AHI v2 — Stacked Mesh** (1,294,547 parameters) uses a dual-mesh transformer that separates temporal and spatial processing into dedicated attention stages:

- **Temporal Mesh** (3 layers): Heat kernel diffusion attention processes 14-day weather sequences, learning per-hazard memory horizons
- **Spatial Mesh** (2 layers): Standard softmax attention with k-nearest-neighbor county adjacency masking captures cross-county correlations
- **Gated Coupling**: Learned gate controls spatial contribution to final predictions

Trained on 370,000+ county-day observations across all 39 Washington counties (2000-2025).

### Performance (AUC Scores)

| Hazard | AHI v2 | XGBoost Baseline |
|--------|--------|------------------|
| Fire | 0.848 | 0.872 |
| Flood | 0.818 | 0.714 |
| Wind | 0.823 | 0.711 |
| Winter | 0.904 | 0.890 |
| Seismic | 0.703 | 0.719 |
| **Mean** | **0.819** | **0.781** |

AHI v2 surpasses the XGBoost baseline on aggregate (+0.038) and on 3 of 5 hazard types, with the largest gains in spatially correlated hazards (Flood +0.10, Wind +0.11).

### Published Research

- Curry, J.D. (2025). Heat Kernel Attention. SSRN.
- Curry, J.D. (2026). Meta-Meta Attention. SSRN 6316718.
- Curry, J.D. (2026). Simplicial Computation. SSRN 6037977.

### Data Sources

- NOAA GridMET (weather)
- NOAA Storm Events (hazard labels)
- US Census (demographics)
- CDC SVI (social vulnerability)
- USGS NLCD (land cover)
- USGS Earthquakes (seismic catalog)
- FEMA Disaster Declarations

### Author

Joshua D. Curry
Resilience Analytics Lab, LLC
Pierce College Fort Steilacoom — Emergency Management Department
April 2026
