"""
HAZARD-LM v2.0: Coupled Hazard Graph Architecture

A transformer-based model for multi-hazard emergency management with:
- Independent convergence per hazard domain
- Cross-hazard causal interaction modeling
- Physics-informed regularization
- Adaptive checkpointing
"""

from .model import (
    HazardLM,
    HazardLMConfig,
    BackboneConfig,
    LoRAConfig,
    InteractionConfig,
    HeadConfig,
    create_hazard_lm,
    HazardAdapter,
    CrossHazardInteractionLayer,
    HazardHead,
    CompoundEventHead,
)

# Trainer is optional - not needed for inference/deployment
try:
    from .trainer import (
        HazardLMTrainer,
        TrainingConfig,
        TrainingPhaseConfig,
        AdaptiveCheckpointer,
        MetricsCalculator,
        PhaseTrainer,
        SampleHazardDataset,
    )
    _TRAINER_AVAILABLE = True
except ImportError:
    _TRAINER_AVAILABLE = False
    HazardLMTrainer = None
    TrainingConfig = None
    TrainingPhaseConfig = None
    AdaptiveCheckpointer = None
    MetricsCalculator = None
    PhaseTrainer = None
    SampleHazardDataset = None

from .data import (
    HazardSample,
    HazardTaxonomy,
    HazardDataset,
    SimpleTokenizer,
    NWSDataLoader,
    FEMADataLoader,
    stratified_split,
    per_hazard_split,
    extract_compound_events,
    generate_sample_data,
)

from .inference import (
    HazardLMInference,
    HazardPrediction,
    InferenceResult,
    VASTAIntegration,
    ModelLoader,
    ModelEvaluator,
)

__version__ = "2.0.0"
__author__ = "Josh + Claude"

__all__ = [
    # Model
    "HazardLM",
    "HazardLMConfig", 
    "BackboneConfig",
    "LoRAConfig",
    "InteractionConfig",
    "HeadConfig",
    "create_hazard_lm",
    "HazardAdapter",
    "CrossHazardInteractionLayer",
    "HazardHead",
    "CompoundEventHead",
    
    # Training
    "HazardLMTrainer",
    "TrainingConfig",
    "TrainingPhaseConfig",
    "AdaptiveCheckpointer",
    "MetricsCalculator",
    "PhaseTrainer",
    "SampleHazardDataset",
    
    # Data
    "HazardSample",
    "HazardTaxonomy",
    "HazardDataset",
    "SimpleTokenizer",
    "NWSDataLoader",
    "FEMADataLoader",
    "stratified_split",
    "per_hazard_split",
    "extract_compound_events",
    "generate_sample_data",
    
    # Inference
    "HazardLMInference",
    "HazardPrediction",
    "InferenceResult",
    "VASTAIntegration",
    "ModelLoader",
    "ModelEvaluator",
]
