"""
HAZARD-LM v2.0: Coupled Hazard Graph Architecture

"""
"""
HAZARD-LM v2.0: Coupled Hazard Graph Architecture

Core model implementation with:
- Shared transformer backbone
- Hazard-specific LoRA adapters
- Cross-hazard interaction layer
- Hazard-specific and compound event heads
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import math
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration Classes
# =============================================================================

class HazardType(Enum):
    """Primary hazard types."""
    FIRE = "fire"
    FLOOD = "flood"
    EARTHQUAKE = "earthquake"
    WIND = "wind"
    FREEZE = "freeze"
    HEAT = "heat"
    DROUGHT = "drought"
    LANDSLIDE = "landslide"


@dataclass
class BackboneConfig:
    """Configuration for shared transformer backbone."""
    hidden_dim: int = 768
    num_layers: int = 12
    num_heads: int = 12
    intermediate_dim: int = 3072
    max_seq_length: int = 2048
    vocab_size: int = 50257  # GPT-2 default
    dropout: float = 0.1
    layer_norm_eps: float = 1e-6
    pad_token_id: int = 0


@dataclass
class LoRAConfig:
    """Configuration for LoRA adapters."""
    rank: int = 16
    alpha: float = 32.0
    dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    
    @property
    def scaling(self) -> float:
        return self.alpha / self.rank


@dataclass
class InteractionConfig:
    """Configuration for cross-hazard interaction layer."""
    hidden_dim: int = 768
    bottleneck_dim: int = 64
    influence_threshold: float = 0.05
    use_physics_prior: bool = True
    sparsity_lambda: float = 0.01


@dataclass 
class HeadConfig:
    """Configuration for prediction heads."""
    hidden_dim: int = 768
    num_classes: int = 2  # binary by default
    dropout: float = 0.1
    head_type: str = "classification"  # classification, regression, multi_label


@dataclass
class HazardLMConfig:
    """Master configuration for HAZARD-LM v2."""
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    interaction: InteractionConfig = field(default_factory=InteractionConfig)
    
    hazards: List[str] = field(default_factory=lambda: [
        "fire", "flood", "earthquake", "wind", "freeze"
    ])
    
    compound_events: List[Tuple[str, ...]] = field(default_factory=lambda: [
        ("wind", "fire"),
        ("earthquake", "flood"),
        ("drought", "fire"),
        ("freeze", "flood"),
    ])


# =============================================================================
# LoRA Implementation
# =============================================================================

class LoRALinear(nn.Module):
    """
    Linear layer with Low-Rank Adaptation.
    
    Implements: output = W @ x + (A @ B) @ x
    Where A and B are low-rank matrices learned per-adapter.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        config: LoRAConfig,
        bias: bool = True
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.config = config
        
        # Original weight (frozen during adapter training)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        
        # LoRA matrices
        self.lora_A = nn.Parameter(torch.empty(config.rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, config.rank))
        self.lora_dropout = nn.Dropout(config.dropout)
        
        # Initialize
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x: torch.Tensor, use_lora: bool = True) -> torch.Tensor:
        # Original linear transformation
        result = F.linear(x, self.weight, self.bias)
        
        if use_lora:
            # LoRA delta: (A @ B) @ x, scaled
            lora_out = self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T
            result = result + lora_out * self.config.scaling
        
        return result
    
    def merge_weights(self):
        """Merge LoRA weights into main weights (for deployment)."""
        self.weight.data += (self.lora_B @ self.lora_A) * self.config.scaling
    
    def get_lora_state(self) -> Dict[str, torch.Tensor]:
        """Extract just the LoRA weights for checkpointing."""
        return {
            'lora_A': self.lora_A.data.clone(),
            'lora_B': self.lora_B.data.clone(),
        }
    
    def load_lora_state(self, state: Dict[str, torch.Tensor]):
        """Load LoRA weights from checkpoint."""
        self.lora_A.data = state['lora_A']
        self.lora_B.data = state['lora_B']


# =============================================================================
# Transformer Components
# =============================================================================

class MultiHeadAttention(nn.Module):
    """Multi-head attention with optional LoRA on Q and V projections."""
    
    def __init__(self, config: BackboneConfig, lora_config: Optional[LoRAConfig] = None):
        super().__init__()
        self.config = config
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_dim // config.num_heads
        
        assert config.hidden_dim % config.num_heads == 0
        
        # Projections (Q and V get LoRA if configured)
        if lora_config and "q_proj" in lora_config.target_modules:
            self.q_proj = LoRALinear(config.hidden_dim, config.hidden_dim, lora_config)
        else:
            self.q_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        
        self.k_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        
        if lora_config and "v_proj" in lora_config.target_modules:
            self.v_proj = LoRALinear(config.hidden_dim, config.hidden_dim, lora_config)
        else:
            self.v_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        
        self.out_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_lora: bool = True
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        
        # Project Q, K, V
        if isinstance(self.q_proj, LoRALinear):
            q = self.q_proj(hidden_states, use_lora=use_lora)
        else:
            q = self.q_proj(hidden_states)
        
        k = self.k_proj(hidden_states)
        
        if isinstance(self.v_proj, LoRALinear):
            v = self.v_proj(hidden_states, use_lora=use_lora)
        else:
            v = self.v_proj(hidden_states)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if attention_mask is not None:
            scores = scores + attention_mask
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        
        return self.out_proj(context)


class FeedForward(nn.Module):
    """Feed-forward network with optional LoRA."""
    
    def __init__(self, config: BackboneConfig, lora_config: Optional[LoRAConfig] = None):
        super().__init__()
        
        if lora_config and "ffn.up" in lora_config.target_modules:
            self.up_proj = LoRALinear(config.hidden_dim, config.intermediate_dim, lora_config, bias=False)
        else:
            self.up_proj = nn.Linear(config.hidden_dim, config.intermediate_dim, bias=False)
        
        if lora_config and "ffn.down" in lora_config.target_modules:
            self.down_proj = LoRALinear(config.intermediate_dim, config.hidden_dim, lora_config, bias=False)
        else:
            self.down_proj = nn.Linear(config.intermediate_dim, config.hidden_dim, bias=False)
        
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x: torch.Tensor, use_lora: bool = True) -> torch.Tensor:
        if isinstance(self.up_proj, LoRALinear):
            h = self.up_proj(x, use_lora=use_lora)
        else:
            h = self.up_proj(x)
        
        h = F.gelu(h)
        
        if isinstance(self.down_proj, LoRALinear):
            h = self.down_proj(h, use_lora=use_lora)
        else:
            h = self.down_proj(h)
        
        return self.dropout(h)


class TransformerBlock(nn.Module):
    """Single transformer block with attention and FFN."""
    
    def __init__(self, config: BackboneConfig, lora_config: Optional[LoRAConfig] = None):
        super().__init__()
        self.attention = MultiHeadAttention(config, lora_config)
        self.ffn = FeedForward(config, lora_config)
        self.ln1 = nn.LayerNorm(config.hidden_dim, eps=config.layer_norm_eps)
        self.ln2 = nn.LayerNorm(config.hidden_dim, eps=config.layer_norm_eps)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_lora: bool = True
    ) -> torch.Tensor:
        # Self-attention with residual
        residual = hidden_states
        hidden_states = self.ln1(hidden_states)
        hidden_states = self.attention(hidden_states, attention_mask, use_lora=use_lora)
        hidden_states = residual + hidden_states
        
        # FFN with residual
        residual = hidden_states
        hidden_states = self.ln2(hidden_states)
        hidden_states = self.ffn(hidden_states, use_lora=use_lora)
        hidden_states = residual + hidden_states
        
        return hidden_states


# =============================================================================
# Hazard Adapter
# =============================================================================

class HazardAdapter(nn.Module):
    """
    Hazard-specific adapter wrapping LoRA modifications.
    
    Each hazard gets its own adapter that can be:
    - Trained independently
    - Early-stopped at its optimal epoch
    - Checkpointed separately
    """
    
    def __init__(
        self,
        hazard_name: str,
        backbone_blocks: nn.ModuleList,
        lora_config: LoRAConfig
    ):
        super().__init__()
        self.hazard_name = hazard_name
        self.lora_config = lora_config
        self.is_frozen = False
        self.best_epoch = None
        self.best_score = None
        
        # Create LoRA layers for this adapter
        # These are stored separately and applied during forward pass
        self.lora_layers = nn.ModuleDict()
        
        for block_idx, block in enumerate(backbone_blocks):
            # Q projection LoRA
            if "q_proj" in lora_config.target_modules:
                self.lora_layers[f"block_{block_idx}_q"] = nn.ModuleDict({
                    'A': nn.Linear(block.attention.q_proj.in_features, lora_config.rank, bias=False),
                    'B': nn.Linear(lora_config.rank, block.attention.q_proj.out_features, bias=False),
                })
                nn.init.kaiming_uniform_(self.lora_layers[f"block_{block_idx}_q"]['A'].weight)
                nn.init.zeros_(self.lora_layers[f"block_{block_idx}_q"]['B'].weight)
            
            # V projection LoRA
            if "v_proj" in lora_config.target_modules:
                self.lora_layers[f"block_{block_idx}_v"] = nn.ModuleDict({
                    'A': nn.Linear(block.attention.v_proj.in_features, lora_config.rank, bias=False),
                    'B': nn.Linear(lora_config.rank, block.attention.v_proj.out_features, bias=False),
                })
                nn.init.kaiming_uniform_(self.lora_layers[f"block_{block_idx}_v"]['A'].weight)
                nn.init.zeros_(self.lora_layers[f"block_{block_idx}_v"]['B'].weight)
        
        self.dropout = nn.Dropout(lora_config.dropout)
    
    def get_lora_delta(self, layer_key: str, x: torch.Tensor) -> torch.Tensor:
        """Compute LoRA delta for a specific layer."""
        if layer_key not in self.lora_layers:
            return torch.zeros_like(x)
        
        lora = self.lora_layers[layer_key]
        delta = self.dropout(x) @ lora['A'].weight.T @ lora['B'].weight.T
        return delta * self.lora_config.scaling
    
    def freeze(self, epoch: int, score: float):
        """Freeze this adapter at its best performance."""
        self.is_frozen = True
        self.best_epoch = epoch
        self.best_score = score
        for param in self.parameters():
            param.requires_grad = False
        logger.info(f"Adapter [{self.hazard_name}] frozen at epoch {epoch} with score {score:.4f}")
    
    def unfreeze(self):
        """Unfreeze for continued training."""
        self.is_frozen = False
        for param in self.parameters():
            param.requires_grad = True
    
    def get_state(self) -> Dict[str, Any]:
        """Get adapter state for checkpointing."""
        return {
            'hazard_name': self.hazard_name,
            'lora_state_dict': self.lora_layers.state_dict(),
            'is_frozen': self.is_frozen,
            'best_epoch': self.best_epoch,
            'best_score': self.best_score,
        }
    
    def load_state(self, state: Dict[str, Any]):
        """Load adapter state from checkpoint."""
        self.lora_layers.load_state_dict(state['lora_state_dict'])
        self.is_frozen = state['is_frozen']
        self.best_epoch = state['best_epoch']
        self.best_score = state['best_score']


# =============================================================================
# Cross-Hazard Interaction Layer
# =============================================================================

class CrossHazardInteractionLayer(nn.Module):
    """
    Models causal influences between hazards.
    
    Key features:
    - Asymmetric interaction matrix (wind→fire ≠ fire→wind)
    - Physics-informed initialization
    - Sparse regularization
    - Interpretable weights
    """
    
    # Physics-informed prior: which hazards influence which
    PHYSICS_PRIOR = {
        ('wind', 'fire'): 0.5,      # Wind strongly amplifies fire
        ('drought', 'fire'): 0.4,   # Drought preconditions fire
        ('heat', 'fire'): 0.2,      # Heat contributes to fire risk
        ('earthquake', 'flood'): 0.3,  # Quake can cause dam failure
        ('earthquake', 'fire'): 0.1,   # Quake can cause ignition
        ('freeze', 'flood'): 0.2,   # Ice dam breakup
        ('wind', 'flood'): 0.2,     # Storm surge
    }
    
    def __init__(self, config: InteractionConfig, hazards: List[str]):
        super().__init__()
        self.config = config
        self.hazards = hazards
        self.n_hazards = len(hazards)
        self.hazard_to_idx = {h: i for i, h in enumerate(hazards)}
        
        # Learnable interaction matrix
        init_matrix = torch.zeros(self.n_hazards, self.n_hazards)
        
        if config.use_physics_prior:
            for (source, target), value in self.PHYSICS_PRIOR.items():
                if source in self.hazard_to_idx and target in self.hazard_to_idx:
                    src_idx = self.hazard_to_idx[source]
                    tgt_idx = self.hazard_to_idx[target]
                    init_matrix[src_idx, tgt_idx] = value
        
        self.interaction_matrix = nn.Parameter(init_matrix)
        
        # Interaction projections (source → target transformation)
        self.projections = nn.ModuleDict()
        for source in hazards:
            for target in hazards:
                if source != target:
                    key = f"{source}_to_{target}"
                    self.projections[key] = nn.Sequential(
                        nn.Linear(config.hidden_dim, config.bottleneck_dim),
                        nn.GELU(),
                        nn.Linear(config.bottleneck_dim, config.hidden_dim),
                    )
    
    def forward(
        self,
        hazard_representations: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Apply cross-hazard interactions.
        
        Args:
            hazard_representations: Dict mapping hazard name to representation tensor
                                   Each tensor is [batch, seq_len, hidden_dim]
        
        Returns:
            Modified representations incorporating causal influences
        """
        output_reprs = {}
        
        for target in self.hazards:
            if target not in hazard_representations:
                continue
            
            target_idx = self.hazard_to_idx[target]
            target_repr = hazard_representations[target].clone()
            
            # Accumulate influences from other hazards
            for source in self.hazards:
                if source == target or source not in hazard_representations:
                    continue
                
                source_idx = self.hazard_to_idx[source]
                influence = self.interaction_matrix[source_idx, target_idx]
                
                # Only apply if influence is above threshold
                if influence.abs() > self.config.influence_threshold:
                    source_repr = hazard_representations[source]
                    proj_key = f"{source}_to_{target}"
                    projected = self.projections[proj_key](source_repr)
                    target_repr = target_repr + influence * projected
            
            output_reprs[target] = target_repr
        
        return output_reprs
    
    def get_interaction_weights(self) -> Dict[Tuple[str, str], float]:
        """Get current interaction weights as interpretable dict."""
        weights = {}
        matrix = self.interaction_matrix.detach().cpu()
        
        for source in self.hazards:
            for target in self.hazards:
                if source != target:
                    src_idx = self.hazard_to_idx[source]
                    tgt_idx = self.hazard_to_idx[target]
                    weights[(source, target)] = matrix[src_idx, tgt_idx].item()
        
        return weights
    
    def physics_regularization_loss(self) -> torch.Tensor:
        """
        Compute physics regularization loss.
        
        Penalizes:
        - Physically impossible influences (fire→wind)
        - Dense matrices (encourage sparsity)
        """
        loss = torch.tensor(0.0, device=self.interaction_matrix.device)
        
        # Penalize impossible directions (fire doesn't cause wind)
        impossible_pairs = [
            ('fire', 'wind'),
            ('fire', 'earthquake'),
            ('flood', 'earthquake'),
            ('freeze', 'heat'),
            ('heat', 'freeze'),
        ]
        
        for source, target in impossible_pairs:
            if source in self.hazard_to_idx and target in self.hazard_to_idx:
                src_idx = self.hazard_to_idx[source]
                tgt_idx = self.hazard_to_idx[target]
                loss = loss + F.relu(self.interaction_matrix[src_idx, tgt_idx])
        
        # L1 sparsity regularization
        loss = loss + self.config.sparsity_lambda * self.interaction_matrix.abs().sum()
        
        return loss


# =============================================================================
# Prediction Heads
# =============================================================================

class HazardHead(nn.Module):
    """Prediction head for a single hazard."""
    
    def __init__(self, config: HeadConfig):
        super().__init__()
        self.config = config
        
        self.dense = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(config.hidden_dim, config.num_classes)
    
    def forward(self, pooled_output: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pooled_output: [batch, hidden_dim] - pooled representation
        
        Returns:
            logits: [batch, num_classes]
        """
        x = self.dropout(pooled_output)
        x = self.dense(x)
        x = F.gelu(x)
        x = self.dropout(x)
        logits = self.classifier(x)
        
        return logits


class CompoundEventHead(nn.Module):
    """Prediction head for compound/cascading events."""
    
    def __init__(
        self,
        involved_hazards: Tuple[str, ...],
        hidden_dim: int,
        num_classes: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        self.involved_hazards = involved_hazards
        
        # Fuse representations from involved hazards
        input_dim = hidden_dim * len(involved_hazards)
        self.fusion = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, hazard_outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            hazard_outputs: Dict mapping hazard name to pooled representation
        
        Returns:
            logits: [batch, num_classes]
        """
        # Concatenate involved hazard representations
        reprs = [hazard_outputs[h] for h in self.involved_hazards]
        fused = torch.cat(reprs, dim=-1)
        fused = self.fusion(fused)
        return self.classifier(fused)


# =============================================================================
# Main Model
# =============================================================================

class HazardLM(nn.Module):
    """
    HAZARD-LM v2.0: Coupled Hazard Graph Architecture
    
    Complete model with:
    - Shared transformer backbone
    - Per-hazard LoRA adapters
    - Cross-hazard interaction layer
    - Hazard-specific and compound event heads
    """
    
    def __init__(self, config: HazardLMConfig):
        super().__init__()
        self.config = config
        
        # Embeddings
        self.token_embedding = nn.Embedding(
            config.backbone.vocab_size,
            config.backbone.hidden_dim,
            padding_idx=config.backbone.pad_token_id
        )
        self.position_embedding = nn.Embedding(
            config.backbone.max_seq_length,
            config.backbone.hidden_dim
        )
        self.embedding_dropout = nn.Dropout(config.backbone.dropout)
        
        # Shared transformer backbone
        self.backbone = nn.ModuleList([
            TransformerBlock(config.backbone)
            for _ in range(config.backbone.num_layers)
        ])
        self.backbone_ln = nn.LayerNorm(
            config.backbone.hidden_dim,
            eps=config.backbone.layer_norm_eps
        )
        
        # Hazard adapters
        self.adapters = nn.ModuleDict({
            hazard: HazardAdapter(hazard, self.backbone, config.lora)
            for hazard in config.hazards
        })
        
        # Cross-hazard interaction layer
        self.interaction_layer = CrossHazardInteractionLayer(
            config.interaction,
            config.hazards
        )
        
        # Hazard-specific heads
        self.heads = nn.ModuleDict({
            hazard: HazardHead(HeadConfig(
                hidden_dim=config.backbone.hidden_dim,
                num_classes=2,  # binary classification by default
                dropout=0.1
            ))
            for hazard in config.hazards
        })
        
        # Compound event heads
        self.compound_heads = nn.ModuleDict()
        for hazard_tuple in config.compound_events:
            key = "_".join(hazard_tuple)
            self.compound_heads[key] = CompoundEventHead(
                involved_hazards=hazard_tuple,
                hidden_dim=config.backbone.hidden_dim
            )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    
    def get_embeddings(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get token + position embeddings."""
        batch_size, seq_len = input_ids.shape
        
        # Token embeddings
        token_embeds = self.token_embedding(input_ids)
        
        # Position embeddings
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        position_embeds = self.position_embedding(positions)
        
        # Combine and dropout
        hidden_states = self.embedding_dropout(token_embeds + position_embeds)
        
        # Create attention mask for transformer
        if attention_mask is not None:
            # Convert [batch, seq] to [batch, 1, 1, seq] for broadcasting
            extended_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            extended_mask = (1.0 - extended_mask) * -10000.0
        else:
            extended_mask = None
        
        return hidden_states, extended_mask
    
    def forward_backbone(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward through shared backbone (no adapters)."""
        for block in self.backbone:
            hidden_states = block(hidden_states, attention_mask, use_lora=False)
        
        return self.backbone_ln(hidden_states)
    
    def forward_with_adapter(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        hazard: str = None
    ) -> torch.Tensor:
        """
        Forward through backbone with a specific hazard's adapter applied.
        
        This is the key innovation: adapter deltas are added during forward pass,
        allowing independent training and checkpointing per hazard.
        """
        adapter = self.adapters[hazard] if hazard in self.adapters else None
        
        for block_idx, block in enumerate(self.backbone):
            # Standard attention computation
            residual = hidden_states
            hidden_states = block.ln1(hidden_states)
            
            # Q projection with adapter delta
            q = block.attention.q_proj(hidden_states)
            if adapter:
                q = q + adapter.get_lora_delta(f"block_{block_idx}_q", hidden_states)
            
            k = block.attention.k_proj(hidden_states)
            
            # V projection with adapter delta
            v = block.attention.v_proj(hidden_states)
            if adapter:
                v = v + adapter.get_lora_delta(f"block_{block_idx}_v", hidden_states)
            
            # Complete attention
            batch_size, seq_len, _ = hidden_states.shape
            head_dim = block.attention.head_dim
            num_heads = block.attention.num_heads
            
            q = q.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
            k = k.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
            v = v.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
            
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
            if attention_mask is not None:
                scores = scores + attention_mask
            
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = block.attention.dropout(attn_weights)
            
            context = torch.matmul(attn_weights, v)
            context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
            attn_output = block.attention.out_proj(context)
            
            hidden_states = residual + attn_output
            
            # FFN
            residual = hidden_states
            hidden_states = block.ln2(hidden_states)
            hidden_states = block.ffn(hidden_states, use_lora=False)
            hidden_states = residual + hidden_states
        
        return self.backbone_ln(hidden_states)
    
    def pool(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Pool sequence representation (mean pooling over non-padding tokens)."""
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (hidden_states * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        else:
            pooled = hidden_states.mean(1)
        return pooled
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        hazard: Optional[str] = None,
        use_interactions: bool = True,
        return_all_hazards: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Main forward pass.
        
        Args:
            input_ids: [batch, seq_len] token IDs
            attention_mask: [batch, seq_len] attention mask
            hazard: Specific hazard to predict (if None, predicts all)
            use_interactions: Whether to apply cross-hazard interactions
            return_all_hazards: Return predictions for all hazards
        
        Returns:
            Dictionary with predictions and optional intermediate representations
        """
        # Get embeddings
        hidden_states, extended_mask = self.get_embeddings(input_ids, attention_mask)
        
        outputs = {}
        
        if hazard and not return_all_hazards:
            # Single hazard mode (most efficient)
            hazard_hidden = self.forward_with_adapter(hidden_states, extended_mask, hazard)
            pooled = self.pool(hazard_hidden, attention_mask)
            outputs['logits'] = self.heads[hazard](pooled)
            outputs['hazard'] = hazard
        
        else:
            # All hazards mode
            hazard_reprs = {}
            for h in self.config.hazards:
                hazard_hidden = self.forward_with_adapter(hidden_states.clone(), extended_mask, h)
                hazard_reprs[h] = hazard_hidden
            
            # Apply interactions if requested
            if use_interactions:
                hazard_reprs = self.interaction_layer(hazard_reprs)
            
            # Pool and predict
            predictions = {}
            pooled_reprs = {}
            for h in self.config.hazards:
                pooled = self.pool(hazard_reprs[h], attention_mask)
                pooled_reprs[h] = pooled
                predictions[h] = self.heads[h](pooled)
            
            outputs['predictions'] = predictions
            outputs['pooled_representations'] = pooled_reprs
            
            # Compound event predictions
            compound_preds = {}
            for key, head in self.compound_heads.items():
                compound_preds[key] = head(pooled_reprs)
            outputs['compound_predictions'] = compound_preds
        
        return outputs
    
    def predict_single_hazard(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        hazard: str
    ) -> torch.Tensor:
        """Efficient prediction for a known hazard type."""
        return self.forward(input_ids, attention_mask, hazard=hazard)['logits']
    
    def predict_all_hazards(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        use_interactions: bool = True
    ) -> Dict[str, torch.Tensor]:
        """Predict all hazards (for unknown/mixed scenarios)."""
        return self.forward(
            input_ids, attention_mask,
            use_interactions=use_interactions,
            return_all_hazards=True
        )
    
    # =========================================================================
    # Checkpointing Methods
    # =========================================================================
    
    def save_backbone(self, path: str):
        """Save backbone weights."""
        state = {
            'backbone': self.backbone.state_dict(),
            'backbone_ln': self.backbone_ln.state_dict(),
            'token_embedding': self.token_embedding.state_dict(),
            'position_embedding': self.position_embedding.state_dict(),
        }
        torch.save(state, path)
        logger.info(f"Backbone saved to {path}")
    
    def load_backbone(self, path: str):
        """Load backbone weights."""
        state = torch.load(path, map_location='cpu')
        self.backbone.load_state_dict(state['backbone'])
        self.backbone_ln.load_state_dict(state['backbone_ln'])
        self.token_embedding.load_state_dict(state['token_embedding'])
        self.position_embedding.load_state_dict(state['position_embedding'])
        logger.info(f"Backbone loaded from {path}")
    
    def save_adapter(self, hazard: str, path: str):
        """Save a single hazard adapter."""
        adapter = self.adapters[hazard]
        torch.save(adapter.get_state(), path)
        logger.info(f"Adapter [{hazard}] saved to {path}")
    
    def load_adapter(self, hazard: str, path: str):
        """Load a single hazard adapter."""
        state = torch.load(path, map_location='cpu')
        self.adapters[hazard].load_state(state)
        logger.info(f"Adapter [{hazard}] loaded from {path}")
    
    def save_interaction_layer(self, path: str):
        """Save interaction layer."""
        torch.save(self.interaction_layer.state_dict(), path)
        logger.info(f"Interaction layer saved to {path}")
    
    def load_interaction_layer(self, path: str):
        """Load interaction layer."""
        self.interaction_layer.load_state_dict(torch.load(path, map_location='cpu'))
        logger.info(f"Interaction layer loaded from {path}")
    
    def save_heads(self, path: str):
        """Save all prediction heads."""
        state = {
            'hazard_heads': self.heads.state_dict(),
            'compound_heads': self.compound_heads.state_dict(),
        }
        torch.save(state, path)
        logger.info(f"Heads saved to {path}")
    
    def load_heads(self, path: str):
        """Load all prediction heads."""
        state = torch.load(path, map_location='cpu')
        self.heads.load_state_dict(state['hazard_heads'])
        self.compound_heads.load_state_dict(state['compound_heads'])
        logger.info(f"Heads loaded from {path}")
    
    def freeze_backbone(self):
        """Freeze backbone for adapter training."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.backbone_ln.parameters():
            param.requires_grad = False
        for param in self.token_embedding.parameters():
            param.requires_grad = False
        for param in self.position_embedding.parameters():
            param.requires_grad = False
        logger.info("Backbone frozen")
    
    def unfreeze_backbone(self):
        """Unfreeze backbone."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        for param in self.backbone_ln.parameters():
            param.requires_grad = True
        for param in self.token_embedding.parameters():
            param.requires_grad = True
        for param in self.position_embedding.parameters():
            param.requires_grad = True
        logger.info("Backbone unfrozen")


# =============================================================================
# Model Factory
# =============================================================================

def create_hazard_lm(
    hazards: List[str] = None,
    hidden_dim: int = 768,
    num_layers: int = 12,
    lora_rank: int = 16,
    **kwargs
) -> HazardLM:
    """
    Factory function to create HazardLM with common configurations.
    
    Args:
        hazards: List of hazard names (default: 5 core hazards)
        hidden_dim: Transformer hidden dimension
        num_layers: Number of transformer layers
        lora_rank: Rank for LoRA adapters
        **kwargs: Additional config overrides
    
    Returns:
        Configured HazardLM model
    """
    if hazards is None:
        hazards = ["fire", "flood", "earthquake", "wind", "freeze"]
    
    # Filter default compound events to only include hazards present in this model
    default_compounds = [
        ("wind", "fire"),
        ("earthquake", "flood"),
        ("drought", "fire"),
        ("freeze", "flood"),
    ]
    compound_events = [t for t in default_compounds if set(t).issubset(set(hazards))]

    config = HazardLMConfig(
        backbone=BackboneConfig(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=hidden_dim // 64,
            intermediate_dim=hidden_dim * 4,
        ),
        lora=LoRAConfig(rank=lora_rank),
        interaction=InteractionConfig(hidden_dim=hidden_dim, bottleneck_dim=max(32, hidden_dim // 12)),
        hazards=hazards,
        compound_events=compound_events,
    )
    
    return HazardLM(config)


if __name__ == "__main__":
    # Quick test
    model = create_hazard_lm()
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Test forward pass
    batch = torch.randint(0, 1000, (2, 128))
    mask = torch.ones_like(batch)
    
    # Single hazard
    out = model(batch, mask, hazard="fire")
    print(f"Single hazard output shape: {out['logits'].shape}")
    
    # All hazards
    out = model(batch, mask, return_all_hazards=True)
    print(f"All hazards predictions: {list(out['predictions'].keys())}")
    print(f"Compound predictions: {list(out['compound_predictions'].keys())}")
