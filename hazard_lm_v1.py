#!/usr/bin/env python3
"""
Hazard-LM v1.0 - Enhanced Model Architecture
============================================
Improvements over v0.5:
1. Transformer-based temporal encoder (replaces LSTM)
2. Feature Pyramid Network for multi-scale visual features
3. Cross-attention between modalities
4. Monte Carlo Dropout for uncertainty quantification
5. Regional embeddings for spatial context
6. Improved hazard heads with residual connections

Part of Hazard-LM v1.0 Development Roadmap
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import math
import torchvision.models as models


@dataclass
class HazardConfigV1:
    """Enhanced configuration for Hazard-LM v1.0."""
    
    # Input dimensions
    static_cont_dim: int = 41  # Expanded feature set
    temporal_seq_len: int = 14
    temporal_feat_dim: int = 8  # Added more weather vars
    nlcd_vocab_size: int = 100
    nlcd_embed_dim: int = 32  # Increased
    
    # Regional embeddings (WA counties)
    num_regions: int = 40
    region_embed_dim: int = 32
    
    # Encoder dimensions
    static_hidden: int = 256
    temporal_hidden: int = 128
    visual_embed_dim: int = 512
    
    # Transformer config (for temporal)
    num_attention_heads: int = 4
    transformer_layers: int = 2
    transformer_ff_dim: int = 256
    
    # Fusion
    fusion_dim: int = 512
    shared_repr_dim: int = 256
    cross_attention_heads: int = 4
    
    # Hazard heads
    hazard_types: List[str] = field(default_factory=lambda: ['fire', 'flood', 'wind', 'winter', 'seismic'])
    head_hidden_dim: int = 128
    
    # Uncertainty quantification
    dropout: float = 0.15  # Lower base dropout
    mc_dropout: float = 0.1  # Separate dropout for MC sampling
    
    # Training
    use_focal_loss: bool = True
    focal_gamma: float = 2.0
    label_smoothing: float = 0.05


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer."""
    
    def __init__(self, d_model: int, max_len: int = 100):
        super().__init__()
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_len, d_model]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input."""
        return x + self.pe[:, :x.size(1), :]


class TransformerTemporalEncoder(nn.Module):
    """
    Transformer-based temporal encoder.
    Better at capturing long-range dependencies than LSTM.
    """
    
    def __init__(self, config: HazardConfigV1):
        super().__init__()
        
        self.config = config
        d_model = config.temporal_hidden
        
        # Project input features to model dimension
        self.input_proj = nn.Linear(config.temporal_feat_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=config.temporal_seq_len + 10)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=config.num_attention_heads,
            dim_feedforward=config.transformer_ff_dim,
            dropout=config.dropout,
            activation='gelu',
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.transformer_layers
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU()
        )
        
        # Aggregation: learnable [CLS] token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [Batch, SeqLen, Features]
        
        Returns:
            [Batch, hidden_dim] temporal representation
        """
        batch_size = x.size(0)
        
        # Project to model dimension
        x = self.input_proj(x)  # [B, T, D]
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Prepend CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # [B, T+1, D]
        
        # Transformer encoding
        x = self.transformer(x)
        
        # Use CLS token output as summary
        cls_output = x[:, 0, :]  # [B, D]
        
        return self.output_proj(cls_output)


class EnhancedStaticEncoder(nn.Module):
    """Static encoder with residual connections."""
    
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.15):
        super().__init__()
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        self.block1 = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.block2 = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = x + self.block1(x)  # Residual
        x = x + self.block2(x)  # Residual
        return x


class EnhancedVisualEncoder(nn.Module):
    """
    Visual encoder with Feature Pyramid Network for multi-scale features.
    """
    
    def __init__(self, output_dim: int = 512, freeze_backbone: bool = True):
        super().__init__()
        
        # Load pretrained ResNet18
        weights = models.ResNet18_Weights.DEFAULT
        resnet = models.resnet18(weights=weights)
        
        # Extract feature layers for FPN
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1  # 64 channels
        self.layer2 = resnet.layer2  # 128 channels
        self.layer3 = resnet.layer3  # 256 channels
        self.layer4 = resnet.layer4  # 512 channels
        
        if freeze_backbone:
            for layer in [self.layer0, self.layer1, self.layer2]:
                for param in layer.parameters():
                    param.requires_grad = False
        
        # FPN lateral connections
        self.lateral4 = nn.Conv2d(512, 256, kernel_size=1)
        self.lateral3 = nn.Conv2d(256, 256, kernel_size=1)
        self.lateral2 = nn.Conv2d(128, 256, kernel_size=1)
        
        # FPN smooth layers
        self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        
        # Global average pool and projection
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(256 * 3, output_dim)  # Concat 3 FPN levels
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Bottom-up
        c1 = self.layer0(x)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        
        # Top-down with lateral connections
        p5 = self.lateral4(c5)
        p4 = self.lateral3(c4) + F.interpolate(p5, size=c4.shape[2:], mode='nearest')
        p4 = self.smooth3(p4)
        p3 = self.lateral2(c3) + F.interpolate(p4, size=c3.shape[2:], mode='nearest')
        p3 = self.smooth2(p3)
        
        # Global pool each level
        f5 = self.global_pool(p5).flatten(1)  # [B, 256]
        f4 = self.global_pool(p4).flatten(1)  # [B, 256]
        f3 = self.global_pool(p3).flatten(1)  # [B, 256]
        
        # Concat and project
        features = torch.cat([f5, f4, f3], dim=1)  # [B, 768]
        return self.proj(features)


class CrossModalAttention(nn.Module):
    """Cross-attention between two modalities."""
    
    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        
        self.attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim)
        )
    
    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query: [B, dim] or [B, L, dim]
            key_value: [B, dim] or [B, L, dim]
        """
        # Ensure 3D
        if query.dim() == 2:
            query = query.unsqueeze(1)
        if key_value.dim() == 2:
            key_value = key_value.unsqueeze(1)
        
        # Cross attention
        attn_out, _ = self.attention(query, key_value, key_value)
        query = self.norm1(query + attn_out)
        
        # FFN
        out = self.norm2(query + self.ffn(query))
        
        return out.squeeze(1) if out.size(1) == 1 else out


class EnhancedMultiModalFusion(nn.Module):
    """
    Enhanced fusion with cross-attention between modalities.
    """
    
    def __init__(self, config: HazardConfigV1):
        super().__init__()
        
        self.config = config
        
        # Project all modalities to same dimension
        fusion_in_dim = config.shared_repr_dim
        
        self.static_proj = nn.Linear(config.static_hidden, fusion_in_dim)
        self.temporal_proj = nn.Linear(config.temporal_hidden, fusion_in_dim)
        self.visual_proj = nn.Linear(config.visual_embed_dim, fusion_in_dim)
        self.nlcd_proj = nn.Linear(config.nlcd_embed_dim, fusion_in_dim)
        self.region_proj = nn.Linear(config.region_embed_dim, fusion_in_dim)
        
        # Cross-modal attention
        self.visual_to_temporal = CrossModalAttention(fusion_in_dim, config.cross_attention_heads)
        self.temporal_to_static = CrossModalAttention(fusion_in_dim, config.cross_attention_heads)
        
        # Final fusion MLP
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_in_dim * 5, config.fusion_dim),
            nn.LayerNorm(config.fusion_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.fusion_dim, config.shared_repr_dim),
            nn.LayerNorm(config.shared_repr_dim)
        )
        
        # MC Dropout layer (always on during inference for uncertainty)
        self.mc_dropout = nn.Dropout(config.mc_dropout)
    
    def forward(self, static: torch.Tensor, temporal: torch.Tensor,
                visual: torch.Tensor, nlcd: torch.Tensor,
                region: torch.Tensor) -> torch.Tensor:
        
        # Project to common dimension
        s = self.static_proj(static)
        t = self.temporal_proj(temporal)
        v = self.visual_proj(visual)
        n = self.nlcd_proj(nlcd)
        r = self.region_proj(region)
        
        # Cross-modal attention
        vt = self.visual_to_temporal(v, t)  # Visual attends to temporal
        ts = self.temporal_to_static(t, s)  # Temporal attends to static
        
        # Combine all
        combined = torch.cat([vt, ts, v, n, r], dim=1)
        fused = self.fusion_mlp(combined)
        
        # MC Dropout for uncertainty (stays active during eval if enabled)
        fused = self.mc_dropout(fused)
        
        return fused


class ResidualHazardHead(nn.Module):
    """Enhanced hazard head with residual connections and gating."""
    
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.15):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.residual = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Gating mechanism
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        
        self.output = nn.Linear(hidden_dim, 1)
        
        # MC Dropout
        self.mc_dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.net(x)
        
        # Gated residual
        residual = self.residual(h)
        gate = self.gate(h)
        h = h + gate * residual
        
        # MC Dropout
        h = self.mc_dropout(h)
        
        return self.output(h)


class HazardLMv1(nn.Module):
    """
    Hazard-LM v1.0: Enhanced Multi-Hazard Prediction Model
    
    Key improvements:
    - Transformer temporal encoder
    - Feature Pyramid Network visual encoder
    - Cross-modal attention fusion
    - Regional embeddings
    - Built-in uncertainty quantification via MC Dropout
    """
    
    def __init__(self, config: HazardConfigV1 = None):
        super().__init__()
        
        self.config = config or HazardConfigV1()
        
        # Encoders
        self.static_encoder = EnhancedStaticEncoder(
            input_dim=self.config.static_cont_dim,
            hidden_dim=self.config.static_hidden,
            dropout=self.config.dropout
        )
        
        self.temporal_encoder = TransformerTemporalEncoder(self.config)
        
        self.visual_encoder = EnhancedVisualEncoder(
            output_dim=self.config.visual_embed_dim,
            freeze_backbone=True
        )
        
        self.nlcd_embedding = nn.Embedding(
            num_embeddings=self.config.nlcd_vocab_size,
            embedding_dim=self.config.nlcd_embed_dim
        )
        
        self.region_embedding = nn.Embedding(
            num_embeddings=self.config.num_regions,
            embedding_dim=self.config.region_embed_dim
        )
        
        # Multi-Modal Fusion
        self.fusion = EnhancedMultiModalFusion(self.config)
        
        # Hazard-Specific Heads
        self.hazard_heads = nn.ModuleDict({
            hazard: ResidualHazardHead(
                input_dim=self.config.shared_repr_dim,
                hidden_dim=self.config.head_hidden_dim,
                dropout=self.config.dropout
            )
            for hazard in self.config.hazard_types
        })
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier/Kaiming."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0, std=0.02)
    
    def forward(
        self,
        static_features: torch.Tensor,
        temporal_seq: torch.Tensor,
        image: torch.Tensor,
        nlcd_code: torch.Tensor,
        region_id: Optional[torch.Tensor] = None,
        hazard_types: Optional[List[str]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            static_features: [B, static_dim]
            temporal_seq: [B, seq_len, temporal_dim]
            image: [B, 3, 224, 224]
            nlcd_code: [B]
            region_id: [B] optional regional ID
            hazard_types: subset of hazards to predict
        
        Returns:
            Dict with logits and embeddings
        """
        # Encode each modality
        static_embed = self.static_encoder(static_features)
        temporal_embed = self.temporal_encoder(temporal_seq)
        visual_embed = self.visual_encoder(image)
        nlcd_embed = self.nlcd_embedding(nlcd_code)
        
        # Regional embedding (default to 0 if not provided)
        if region_id is None:
            region_id = torch.zeros(static_features.size(0), dtype=torch.long, 
                                   device=static_features.device)
        region_embed = self.region_embedding(region_id)
        
        # Fuse modalities
        shared_repr = self.fusion(
            static_embed, temporal_embed, visual_embed, nlcd_embed, region_embed
        )
        
        # Predict each hazard
        hazards = hazard_types or self.config.hazard_types
        outputs = {
            'shared_repr': shared_repr,
            'static_embed': static_embed,
            'temporal_embed': temporal_embed,
            'visual_embed': visual_embed
        }
        
        for hazard in hazards:
            if hazard in self.hazard_heads:
                logits = self.hazard_heads[hazard](shared_repr)
                outputs[hazard] = logits
                outputs[f'{hazard}_prob'] = torch.sigmoid(logits)
        
        return outputs
    
    def enable_mc_dropout(self):
        """Enable dropout for MC sampling during inference."""
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()
    
    def predict_with_uncertainty(
        self,
        static_features: torch.Tensor,
        temporal_seq: torch.Tensor,
        image: torch.Tensor,
        nlcd_code: torch.Tensor,
        region_id: Optional[torch.Tensor] = None,
        n_samples: int = 30
    ) -> Dict[str, Dict[str, float]]:
        """
        Predict with uncertainty using MC Dropout.
        
        Returns mean, std, and 95% CI for each hazard.
        """
        import numpy as np
        
        self.eval()
        self.enable_mc_dropout()
        
        predictions = {h: [] for h in self.config.hazard_types}
        
        with torch.no_grad():
            for _ in range(n_samples):
                outputs = self.forward(
                    static_features, temporal_seq, image, nlcd_code, region_id
                )
                
                for hazard in self.config.hazard_types:
                    prob = outputs[f'{hazard}_prob'].cpu().numpy()
                    predictions[hazard].append(prob)
        
        results = {}
        for hazard in self.config.hazard_types:
            preds = np.array(predictions[hazard])
            mean = np.mean(preds, axis=0)
            std = np.std(preds, axis=0)
            
            results[hazard] = {
                'mean': float(mean[0]) if len(mean) == 1 else mean.tolist(),
                'std': float(std[0]) if len(std) == 1 else std.tolist(),
                'lower_95': float(np.percentile(preds, 2.5, axis=0)[0]),
                'upper_95': float(np.percentile(preds, 97.5, axis=0)[0]),
                'n_samples': n_samples
            }
        
        return results
    
    @classmethod
    def from_pretrained(cls, path: str, device: str = 'cpu') -> 'HazardLMv1':
        """Load pretrained model."""
        checkpoint = torch.load(path, map_location=device)
        
        config = checkpoint.get('config', HazardConfigV1())
        model = cls(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        
        return model
    
    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_hazard_lm_v1(config: Optional[HazardConfigV1] = None) -> HazardLMv1:
    """Factory function to create HazardLM v1.0."""
    model = HazardLMv1(config)
    print(f"Created HazardLM v1.0 with {model.count_parameters():,} trainable parameters")
    return model


if __name__ == "__main__":
    # Test the model
    print("Testing HazardLM v1.0...")
    
    config = HazardConfigV1()
    model = create_hazard_lm_v1(config)
    
    # Create dummy inputs
    batch_size = 4
    static = torch.randn(batch_size, config.static_cont_dim)
    temporal = torch.randn(batch_size, config.temporal_seq_len, config.temporal_feat_dim)
    image = torch.randn(batch_size, 3, 224, 224)
    nlcd = torch.randint(0, 95, (batch_size,))
    region = torch.randint(0, 39, (batch_size,))
    
    # Forward pass
    outputs = model(static, temporal, image, nlcd, region)
    
    print(f"\nOutput keys: {list(outputs.keys())}")
    print(f"Shared repr shape: {outputs['shared_repr'].shape}")
    
    for hazard in config.hazard_types:
        print(f"{hazard} logits shape: {outputs[hazard].shape}")
    
    # Test uncertainty prediction
    print("\nTesting uncertainty prediction...")
    uncertainty = model.predict_with_uncertainty(
        static[:1], temporal[:1], image[:1], nlcd[:1], region[:1], n_samples=10
    )
    
    for hazard, data in uncertainty.items():
        print(f"  {hazard}: {data['mean']:.3f} +/- {data['std']:.3f} "
              f"(95% CI: {data['lower_95']:.3f} - {data['upper_95']:.3f})")
