"""
HAZARD-LM v2.0: Inference and API Utilities

Provides:
- Fast inference for single/batch predictions
- Multi-mode inference (known hazard, unknown, compound)
- Explainability and interpretation
- VASTA integration endpoints
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# Inference Results
# =============================================================================

@dataclass
class HazardPrediction:
    """Single hazard prediction result."""
    hazard: str
    probability: float
    confidence: float
    severity: Optional[int] = None
    
    def to_dict(self) -> Dict:
        return {
            'hazard': self.hazard,
            'probability': self.probability,
            'confidence': self.confidence,
            'severity': self.severity
        }


@dataclass
class InferenceResult:
    """Complete inference result."""
    primary_hazard: str
    predictions: Dict[str, HazardPrediction]
    compound_predictions: Dict[str, float]
    interaction_effects: Dict[Tuple[str, str], float]
    explanation: Optional[Dict] = None
    
    def to_dict(self) -> Dict:
        return {
            'primary_hazard': self.primary_hazard,
            'predictions': {k: v.to_dict() for k, v in self.predictions.items()},
            'compound_predictions': self.compound_predictions,
            'interaction_effects': {f"{k[0]}->{k[1]}": v for k, v in self.interaction_effects.items()},
            'explanation': self.explanation
        }


# =============================================================================
# Inference Engine
# =============================================================================

class HazardLMInference:
    """
    Inference engine for HAZARD-LM.
    
    Supports multiple inference modes:
    1. Known hazard: Fast single-adapter inference
    2. Unknown hazard: Full multi-adapter with routing
    3. Compound event: Interaction-aware prediction
    """
    
    def __init__(
        self,
        model: Any,  # HazardLM
        tokenizer: Any,
        device: str = "cuda",
        confidence_threshold: float = 0.5
    ):
        self.model = model.to(device)
        self.model.eval()
        self.tokenizer = tokenizer
        self.device = device
        self.confidence_threshold = confidence_threshold
        
        # Precompute hazard info
        self.hazards = model.config.hazards
        self.compound_events = list(model.compound_heads.keys())
    
    @torch.no_grad()
    def predict(
        self,
        text: str,
        mode: str = "auto",
        hazard: Optional[str] = None,
        return_explanation: bool = False
    ) -> InferenceResult:
        """
        Main prediction method.
        
        Args:
            text: Input text to classify
            mode: "auto", "known", or "compound"
            hazard: Specific hazard type (for mode="known")
            return_explanation: Whether to generate explanation
        
        Returns:
            InferenceResult with predictions and metadata
        """
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        if mode == "known" and hazard:
            return self._predict_known_hazard(input_ids, attention_mask, hazard, text, return_explanation)
        elif mode == "compound":
            return self._predict_compound(input_ids, attention_mask, text, return_explanation)
        else:  # auto mode
            return self._predict_auto(input_ids, attention_mask, text, return_explanation)
    
    def _predict_known_hazard(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        hazard: str,
        text: str,
        return_explanation: bool
    ) -> InferenceResult:
        """Fast prediction when hazard type is known."""
        outputs = self.model(
            input_ids,
            attention_mask,
            hazard=hazard,
            use_interactions=False,
            return_all_hazards=False
        )
        
        logits = outputs['logits']
        probs = F.softmax(logits, dim=-1)
        
        prediction = HazardPrediction(
            hazard=hazard,
            probability=probs[0, 1].item(),
            confidence=self._compute_confidence(probs[0])
        )
        
        explanation = None
        if return_explanation:
            explanation = self._generate_explanation(text, {hazard: prediction}, {}, {})
        
        return InferenceResult(
            primary_hazard=hazard,
            predictions={hazard: prediction},
            compound_predictions={},
            interaction_effects={},
            explanation=explanation
        )
    
    def _predict_auto(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        text: str,
        return_explanation: bool
    ) -> InferenceResult:
        """Full prediction with all hazards and interactions."""
        outputs = self.model(
            input_ids,
            attention_mask,
            use_interactions=True,
            return_all_hazards=True
        )
        
        # Process hazard predictions
        predictions = {}
        for hazard in self.hazards:
            logits = outputs['predictions'][hazard]
            probs = F.softmax(logits, dim=-1)
            
            predictions[hazard] = HazardPrediction(
                hazard=hazard,
                probability=probs[0, 1].item(),
                confidence=self._compute_confidence(probs[0])
            )
        
        # Process compound predictions
        compound_predictions = {}
        for compound_key in self.compound_events:
            if compound_key in outputs['compound_predictions']:
                logits = outputs['compound_predictions'][compound_key]
                probs = F.softmax(logits, dim=-1)
                compound_predictions[compound_key] = probs[0, 1].item()
        
        # Get interaction effects
        interaction_effects = self.model.interaction_layer.get_interaction_weights()
        
        # Determine primary hazard
        primary_hazard = max(predictions.items(), key=lambda x: x[1].probability)[0]
        
        explanation = None
        if return_explanation:
            explanation = self._generate_explanation(text, predictions, compound_predictions, interaction_effects)
        
        return InferenceResult(
            primary_hazard=primary_hazard,
            predictions=predictions,
            compound_predictions=compound_predictions,
            interaction_effects=interaction_effects,
            explanation=explanation
        )
    
    def _predict_compound(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        text: str,
        return_explanation: bool
    ) -> InferenceResult:
        """Prediction focused on compound events."""
        return self._predict_auto(input_ids, attention_mask, text, return_explanation)
    
    def _compute_confidence(self, probs: torch.Tensor) -> float:
        """Compute prediction confidence (entropy-based)."""
        entropy = -(probs * torch.log(probs + 1e-9)).sum().item()
        max_entropy = torch.log(torch.tensor(float(probs.shape[0]))).item()
        confidence = 1.0 - (entropy / max_entropy)
        return confidence
    
    def _generate_explanation(
        self,
        text: str,
        predictions: Dict[str, HazardPrediction],
        compound_predictions: Dict[str, float],
        interaction_effects: Dict[Tuple[str, str], float]
    ) -> Dict:
        """Generate human-readable explanation."""
        explanation = {
            'summary': '',
            'hazard_factors': [],
            'interaction_insights': [],
            'compound_risks': [],
            'confidence_assessment': ''
        }
        
        # Get top hazards
        sorted_hazards = sorted(
            predictions.items(),
            key=lambda x: x[1].probability,
            reverse=True
        )
        
        top_hazard = sorted_hazards[0]
        explanation['summary'] = (
            f"Primary hazard: {top_hazard[0].upper()} "
            f"(probability: {top_hazard[1].probability:.1%})"
        )
        
        # Hazard factors
        for hazard, pred in sorted_hazards:
            if pred.probability > 0.1:
                explanation['hazard_factors'].append({
                    'hazard': hazard,
                    'probability': f"{pred.probability:.1%}",
                    'confidence': f"{pred.confidence:.1%}"
                })
        
        # Interaction insights
        for (source, target), weight in interaction_effects.items():
            if weight > 0.1:
                explanation['interaction_insights'].append({
                    'source': source,
                    'target': target,
                    'influence': f"{weight:.2f}",
                    'interpretation': self._interpret_interaction(source, target)
                })
        
        # Compound risks
        for compound, prob in compound_predictions.items():
            if prob > 0.3:
                explanation['compound_risks'].append({
                    'event': compound,
                    'probability': f"{prob:.1%}",
                    'description': self._describe_compound(compound)
                })
        
        # Confidence assessment
        avg_confidence = sum(p.confidence for p in predictions.values()) / len(predictions)
        if avg_confidence > 0.8:
            explanation['confidence_assessment'] = "HIGH - Model is confident in predictions"
        elif avg_confidence > 0.5:
            explanation['confidence_assessment'] = "MODERATE - Some uncertainty in predictions"
        else:
            explanation['confidence_assessment'] = "LOW - Predictions should be verified"
        
        return explanation
    
    def _interpret_interaction(self, source: str, target: str) -> str:
        """Get human-readable interpretation of interaction."""
        interpretations = {
            ('wind', 'fire'): "Wind conditions amplify fire spread rate",
            ('drought', 'fire'): "Drought conditions increase fire ignition risk",
            ('earthquake', 'flood'): "Seismic activity may trigger dam failure or tsunami",
            ('heat', 'fire'): "High temperatures increase fire weather conditions",
            ('freeze', 'flood'): "Ice dam breakup can cause sudden flooding",
        }
        return interpretations.get((source, target), f"{source} influences {target}")
    
    def _describe_compound(self, compound: str) -> str:
        """Get description of compound event."""
        descriptions = {
            'wind_fire': "Wind-driven wildfire with rapid spread potential",
            'earthquake_flood': "Earthquake-triggered flooding (dam failure or tsunami)",
            'drought_fire': "Drought-preconditioned wildfire conditions",
            'freeze_flood': "Ice dam breakup flooding",
            'heat_fire': "Heat wave fire conditions",
        }
        return descriptions.get(compound, compound)
    
    @torch.no_grad()
    def batch_predict(
        self,
        texts: List[str],
        mode: str = "auto"
    ) -> List[InferenceResult]:
        """Batch prediction for efficiency."""
        results = []
        
        batch_size = 32
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            encoding = self.tokenizer(
                batch_texts,
                max_length=512,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            
            outputs = self.model(
                input_ids,
                attention_mask,
                use_interactions=True,
                return_all_hazards=True
            )
            
            # Cache interaction weights once for this batch (avoid repeated calls)
            cached_interaction_weights = self.model.interaction_layer.get_interaction_weights()
            
            for j in range(len(batch_texts)):
                predictions = {}
                for hazard in self.hazards:
                    logits = outputs['predictions'][hazard][j:j+1]
                    probs = F.softmax(logits, dim=-1)
                    
                    predictions[hazard] = HazardPrediction(
                        hazard=hazard,
                        probability=probs[0, 1].item(),
                        confidence=self._compute_confidence(probs[0])
                    )
                
                compound_predictions = {}
                for compound_key in self.compound_events:
                    if compound_key in outputs['compound_predictions']:
                        logits = outputs['compound_predictions'][compound_key][j:j+1]
                        probs = F.softmax(logits, dim=-1)
                        compound_predictions[compound_key] = probs[0, 1].item()
                
                primary = max(predictions.items(), key=lambda x: x[1].probability)[0]
                
                results.append(InferenceResult(
                    primary_hazard=primary,
                    predictions=predictions,
                    compound_predictions=compound_predictions,
                    interaction_effects=cached_interaction_weights
                ))
        
        return results


# =============================================================================
# Model Loading Utilities
# =============================================================================

class ModelLoader:
    """Utilities for loading trained models."""
    
    @staticmethod
    def load_from_checkpoints(
        checkpoint_dir: str,
        model_class: Any,
        config: Any,
        device: str = "cuda"
    ) -> Any:
        """Load model from checkpoint directory."""
        checkpoint_dir = Path(checkpoint_dir)
        
        model = model_class(config)
        
        # Load backbone
        backbone_path = checkpoint_dir / "backbone" / "backbone_final.pt"
        if backbone_path.exists():
            model.load_backbone(str(backbone_path))
            logger.info("Loaded backbone")
        
        # Load best adapters
        history_path = checkpoint_dir / "training_history.json"
        if history_path.exists():
            with open(history_path) as f:
                history = json.load(f)
            
            for hazard in config.hazards:
                best_epoch = history.get('best_epochs', {}).get(hazard)
                if best_epoch is not None:
                    adapter_path = checkpoint_dir / "adapters" / hazard / f"adapter_epoch{best_epoch}_best.pt"
                    if adapter_path.exists():
                        model.load_adapter(hazard, str(adapter_path))
                        logger.info(f"Loaded adapter: {hazard} (epoch {best_epoch})")
        
        # Load interaction layer
        interaction_path = checkpoint_dir / "interaction" / "interaction_final.pt"
        if interaction_path.exists():
            model.load_interaction_layer(str(interaction_path))
            logger.info("Loaded interaction layer")
        
        return model.to(device)


# =============================================================================
# VASTA Integration API
# =============================================================================

class VASTAIntegration:
    """
    Integration layer for VASTA emergency management platform.
    
    Provides standardized API for:
    - Real-time hazard prediction
    - Batch historical analysis
    - Alert generation
    """
    
    def __init__(self, inference_engine: HazardLMInference):
        self.engine = inference_engine
    
    def process_alert(self, alert_text: str, source: str = "unknown") -> Dict:
        """
        Process incoming emergency alert.
        
        Args:
            alert_text: Raw alert text
            source: Source system (NWS, FEMA, etc.)
        
        Returns:
            Structured alert analysis
        """
        result = self.engine.predict(alert_text, mode="auto", return_explanation=True)
        
        return {
            'status': 'processed',
            'source': source,
            'analysis': {
                'primary_hazard': result.primary_hazard,
                'all_hazards': {
                    h: {
                        'probability': p.probability,
                        'confidence': p.confidence
                    }
                    for h, p in result.predictions.items()
                    if p.probability > 0.1
                },
                'compound_events': {
                    k: v for k, v in result.compound_predictions.items()
                    if v > 0.3
                },
                'explanation': result.explanation
            },
            'recommended_actions': self._generate_recommendations(result)
        }
    
    def _generate_recommendations(self, result: InferenceResult) -> List[Dict]:
        """Generate action recommendations based on predictions."""
        recommendations = []
        
        for hazard, pred in result.predictions.items():
            if pred.probability > 0.7:
                recommendations.append({
                    'priority': 'HIGH',
                    'hazard': hazard,
                    'action': f"Activate {hazard.upper()} response protocols",
                    'reason': f"High probability ({pred.probability:.1%}) with {pred.confidence:.1%} confidence"
                })
            elif pred.probability > 0.4:
                recommendations.append({
                    'priority': 'MEDIUM',
                    'hazard': hazard,
                    'action': f"Monitor {hazard.upper()} conditions",
                    'reason': f"Moderate probability ({pred.probability:.1%})"
                })
        
        for compound, prob in result.compound_predictions.items():
            if prob > 0.5:
                recommendations.append({
                    'priority': 'HIGH',
                    'hazard': compound,
                    'action': f"Prepare for compound {compound.replace('_', '+')} event",
                    'reason': f"Compound event probability: {prob:.1%}"
                })
        
        return sorted(recommendations, key=lambda x: {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}[x['priority']])
    
    def batch_analyze_historical(
        self,
        incidents: List[Dict],
        include_trends: bool = True
    ) -> Dict:
        """
        Batch analyze historical incidents.
        
        Args:
            incidents: List of incident records with 'text' field
            include_trends: Whether to compute trend analysis
        
        Returns:
            Batch analysis results with optional trends
        """
        texts = [inc.get('text', inc.get('description', '')) for inc in incidents]
        results = self.engine.batch_predict(texts)
        
        analysis = {
            'total_incidents': len(incidents),
            'individual_results': [],
            'summary': {
                'hazard_counts': {},
                'compound_counts': {},
                'avg_confidence': 0.0
            }
        }
        
        hazard_counts = {}
        compound_counts = {}
        total_confidence = 0.0
        
        for inc, result in zip(incidents, results):
            individual = {
                'incident_id': inc.get('id'),
                'primary_hazard': result.primary_hazard,
                'predictions': {h: p.probability for h, p in result.predictions.items()},
                'compound_events': result.compound_predictions
            }
            analysis['individual_results'].append(individual)
            
            # Aggregate stats
            hazard_counts[result.primary_hazard] = hazard_counts.get(result.primary_hazard, 0) + 1
            
            for compound, prob in result.compound_predictions.items():
                if prob > 0.5:
                    compound_counts[compound] = compound_counts.get(compound, 0) + 1
            
            total_confidence += sum(p.confidence for p in result.predictions.values()) / len(result.predictions)
        
        analysis['summary']['hazard_counts'] = hazard_counts
        analysis['summary']['compound_counts'] = compound_counts
        analysis['summary']['avg_confidence'] = total_confidence / len(results) if results else 0
        
        if include_trends:
            analysis['trends'] = self._compute_trends(incidents, results)
        
        return analysis
    
    def _compute_trends(
        self,
        incidents: List[Dict],
        results: List[InferenceResult]
    ) -> Dict:
        """Compute trend analysis from historical data."""
        # Group by time period if dates available
        trends = {
            'hazard_frequency': {},
            'compound_event_frequency': {},
            'notes': []
        }
        
        # Simple frequency analysis
        for result in results:
            h = result.primary_hazard
            trends['hazard_frequency'][h] = trends['hazard_frequency'].get(h, 0) + 1
            
            for compound, prob in result.compound_predictions.items():
                if prob > 0.5:
                    trends['compound_event_frequency'][compound] = \
                        trends['compound_event_frequency'].get(compound, 0) + 1
        
        # Generate insights
        if trends['hazard_frequency']:
            top_hazard = max(trends['hazard_frequency'].items(), key=lambda x: x[1])
            trends['notes'].append(f"Most frequent hazard: {top_hazard[0]} ({top_hazard[1]} incidents)")
        
        if trends['compound_event_frequency']:
            top_compound = max(trends['compound_event_frequency'].items(), key=lambda x: x[1])
            trends['notes'].append(f"Most frequent compound event: {top_compound[0]} ({top_compound[1]} incidents)")
        
        return trends


# =============================================================================
# Evaluation Utilities
# =============================================================================

class ModelEvaluator:
    """Evaluation utilities for HAZARD-LM."""
    
    def __init__(self, inference_engine: HazardLMInference):
        self.engine = inference_engine
    
    def evaluate_dataset(
        self,
        texts: List[str],
        labels: Dict[str, List[int]],
        compound_labels: Optional[Dict[str, List[int]]] = None
    ) -> Dict:
        """
        Evaluate model on labeled dataset.
        
        Args:
            texts: List of input texts
            labels: Dict mapping hazard name to list of binary labels
            compound_labels: Optional compound event labels
        
        Returns:
            Evaluation metrics
        """
        results = self.engine.batch_predict(texts)
        
        metrics = {
            'per_hazard': {},
            'overall': {},
            'compound': {}
        }
        
        for hazard in self.engine.hazards:
            if hazard in labels:
                preds = [r.predictions[hazard].probability for r in results]
                true = labels[hazard]
                
                # Compute metrics
                metrics['per_hazard'][hazard] = self._compute_metrics(preds, true)
        
        # Overall metrics (average)
        if metrics['per_hazard']:
            avg_auc = sum(m.get('auc', 0) for m in metrics['per_hazard'].values()) / len(metrics['per_hazard'])
            avg_f1 = sum(m.get('f1', 0) for m in metrics['per_hazard'].values()) / len(metrics['per_hazard'])
            metrics['overall'] = {'avg_auc': avg_auc, 'avg_f1': avg_f1}
        
        # Compound event metrics
        if compound_labels:
            for compound, true in compound_labels.items():
                preds = [r.compound_predictions.get(compound, 0) for r in results]
                metrics['compound'][compound] = self._compute_metrics(preds, true)
        
        return metrics
    
    def _compute_metrics(self, predictions: List[float], labels: List[int]) -> Dict:
        """Compute classification metrics."""
        try:
            from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
            import numpy as np
            
            preds_binary = [1 if p > 0.5 else 0 for p in predictions]
            labels_np = np.array(labels)
            
            metrics = {
                'accuracy': sum(p == l for p, l in zip(preds_binary, labels)) / len(labels),
            }
            
            if len(set(labels)) > 1:
                metrics['auc'] = roc_auc_score(labels, predictions)
                metrics['f1'] = f1_score(labels, preds_binary, zero_division=0)
                metrics['precision'] = precision_score(labels, preds_binary, zero_division=0)
                metrics['recall'] = recall_score(labels, preds_binary, zero_division=0)
            
            return metrics
        except ImportError:
            return {'accuracy': sum(1 if (p > 0.5) == l else 0 for p, l in zip(predictions, labels)) / len(labels)}
    
    def cross_hazard_analysis(self, texts: List[str]) -> Dict:
        """
        Analyze cross-hazard relationships in predictions.
        
        Useful for understanding model behavior and interaction effects.
        """
        results = self.engine.batch_predict(texts)
        
        # Co-occurrence matrix
        hazards = self.engine.hazards
        co_occurrence = {h1: {h2: 0 for h2 in hazards} for h1 in hazards}
        
        for result in results:
            high_prob_hazards = [
                h for h, p in result.predictions.items()
                if p.probability > 0.5
            ]
            
            for h1 in high_prob_hazards:
                for h2 in high_prob_hazards:
                    co_occurrence[h1][h2] += 1
        
        # Interaction effect analysis
        interaction_weights = results[0].interaction_effects if results else {}
        
        return {
            'co_occurrence_matrix': co_occurrence,
            'learned_interactions': interaction_weights,
            'sample_count': len(results)
        }


# =============================================================================
# Entry Point for Testing
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # This would require a trained model - just testing imports
    print("Inference module loaded successfully")
    print(f"Available classes: HazardLMInference, VASTAIntegration, ModelEvaluator, ModelLoader")
