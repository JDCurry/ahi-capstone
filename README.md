# Adaptive Hazard Intelligence (AHI)
## Capstone Project - Pierce College BAS-EM

A calibrated, uncertainty-aware hazard risk engine for Washington State emergency managers.

### Features
- **Multi-hazard prediction**: Fire, Flood, Wind, Winter, Seismic
- **County-level resolution**: All 39 Washington counties
- **Calibrated probabilities**: Diffusion-based attention for reliable risk estimates
- **Interactive dashboard**: Quick Predict for single-county analysis

### Model Performance (AUC Scores)
| Hazard | AUC | Quality |
|--------|-----|---------|
| Fire | 0.89 | Excellent |
| Winter | 0.94 | Excellent |
| Wind | 0.87 | Good |
| Flood | 0.83 | Good |
| Seismic | 0.77 | Good |

### Running Locally
```bash
pip install -r requirements.txt
streamlit run vasta_dashboard.py
```

### Data Sources
- NOAA GridMET (weather)
- NOAA Storm Events (hazard labels)
- US Census (demographics)
- CDC SVI (social vulnerability)
- USGS NLCD (land cover)
- USGS Earthquakes (seismic catalog)
- FEMA Disaster Declarations

### Architecture
Hazard-LM uses a transformer architecture with diffusion-based attention:
- Static Encoder (41 features): Demographics, geography, infrastructure
- Temporal Encoder (14-day sequences): Weather patterns with diffusion attention
- Land Cover Embedding: NLCD classification
- Cross-hazard interaction layer
- 5 calibrated prediction heads

### Author
Joshua D. Curry  
Pierce College Fort Steilacoom  
Emergency Management Department  
April 2026
