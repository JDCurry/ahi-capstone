"""
Diffusion Attention Training Script

Train and compare:
1. Standard transformer (softmax attention)
2. Diffusion transformer (fixed t)
3. Diffusion transformer (adaptive t)

Metrics:
- Perplexity
- ECE (Expected Calibration Error)
- Brier Score
- Entropy distribution

Usage:
    python train_diffusion_attention.py --model diffusion_adaptive --epochs 10

"""

import os
import sys
import math
import json
import argparse
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Optional, Dict, List, Tuple
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

import numpy as np
from tqdm import tqdm

# Import our attention modules
from diffusion_attention_torch import DiffusionAttention, StandardAttention


# ============================================================
# Configuration
# ============================================================

@dataclass
class ModelConfig:
    """Model configuration."""
    vocab_size: int = 50257  # GPT-2 vocab size
    max_seq_len: int = 256
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 4
    d_ff: int = 1024
    dropout: float = 0.1
    
    # Diffusion-specific
    attention_type: str = "standard"  # "standard", "diffusion_fixed", "diffusion_adaptive"
    fixed_t: float = 1.0
    t_min: float = 0.1
    t_max: float = 10.0


@dataclass
class TrainConfig:
    """Training configuration."""
    batch_size: int = 32
    epochs: int = 10
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0
    
    # Data
    dataset: str = "wikitext"  # or path to custom data
    eval_interval: int = 500
    save_interval: int = 1000
    
    # Logging
    log_dir: str = "./logs"
    experiment_name: str = "diffusion_attention"


# ============================================================
# Simple GPT-style Transformer
# ============================================================

class TransformerBlock(nn.Module):
    """Single transformer block with configurable attention."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        
        self.ln1 = nn.LayerNorm(config.d_model)
        self.ln2 = nn.LayerNorm(config.d_model)
        
        # Attention layer based on config
        if config.attention_type == "standard":
            self.attn = StandardAttention(
                d_model=config.d_model,
                n_heads=config.n_heads,
                dropout=config.dropout,
            )
        elif config.attention_type == "diffusion_fixed":
            self.attn = DiffusionAttention(
                d_model=config.d_model,
                n_heads=config.n_heads,
                dropout=config.dropout,
                adaptive_time=False,
                fixed_t=config.fixed_t,
            )
        elif config.attention_type == "diffusion_adaptive":
            self.attn = DiffusionAttention(
                d_model=config.d_model,
                n_heads=config.n_heads,
                dropout=config.dropout,
                adaptive_time=True,
                t_min=config.t_min,
                t_max=config.t_max,
            )
        else:
            raise ValueError(f"Unknown attention type: {config.attention_type}")
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.d_model),
            nn.Dropout(config.dropout),
        )
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention with residual
        x_norm = self.ln1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, mask=mask)
        x = x + attn_out
        
        # FFN with residual
        x = x + self.ffn(self.ln2(x))
        
        return x


class DiffusionTransformer(nn.Module):
    """
    GPT-style causal language model with configurable attention.
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Embeddings
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])
        
        # Output
        self.ln_f = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Weight tying
        self.head.weight = self.token_emb.weight
        
        # Causal mask (cached)
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.max_seq_len, config.max_seq_len))
            .unsqueeze(0).unsqueeze(0)  # (1, 1, seq, seq)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            input_ids: (batch, seq_len)
            labels: (batch, seq_len) - shifted targets for LM loss
        
        Returns:
            Dict with 'logits', 'loss' (if labels provided)
        """
        batch, seq_len = input_ids.shape
        device = input_ids.device
        
        # Embeddings
        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        x = self.token_emb(input_ids) + self.pos_emb(positions)
        x = self.dropout(x)
        
        # Causal mask
        mask = self.causal_mask[:, :, :seq_len, :seq_len]
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x, mask=mask)
        
        # Output
        x = self.ln_f(x)
        logits = self.head(x)  # (batch, seq, vocab)
        
        result = {"logits": logits}
        
        if labels is not None:
            # Shift for causal LM
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            result["loss"] = loss
        
        return result
    
    def get_diffusion_times(self) -> Optional[torch.Tensor]:
        """Get diffusion times from all layers (if using adaptive diffusion)."""
        if self.config.attention_type != "diffusion_adaptive":
            return None
        
        times = []
        for block in self.blocks:
            if hasattr(block.attn, '_diffusion_times'):
                times.append(block.attn._diffusion_times)
        
        if times:
            return torch.stack(times)  # (n_layers, ...)
        return None


# ============================================================
# Metrics
# ============================================================

class CalibrationMetrics:
    """Compute calibration metrics for language models."""
    
    def __init__(self, n_bins: int = 15):
        self.n_bins = n_bins
        self.reset()
    
    def reset(self):
        self.confidences = []
        self.accuracies = []
        self.predictions = []
        self.targets = []
    
    def update(
        self,
        logits: torch.Tensor,  # (batch, seq, vocab)
        labels: torch.Tensor,  # (batch, seq)
    ):
        """Update metrics with a batch of predictions."""
        # Get probabilities
        probs = F.softmax(logits, dim=-1)
        
        # Get predictions and confidences
        confidence, predictions = probs.max(dim=-1)  # (batch, seq)
        
        # Flatten and filter out padding (-100)
        mask = labels != -100
        confidence = confidence[mask].cpu().numpy()
        predictions = predictions[mask].cpu().numpy()
        targets = labels[mask].cpu().numpy()
        
        self.confidences.extend(confidence.tolist())
        self.predictions.extend(predictions.tolist())
        self.targets.extend(targets.tolist())
    
    def compute(self) -> Dict[str, float]:
        """Compute all calibration metrics."""
        if not self.confidences:
            return {}
        
        confidences = np.array(self.confidences)
        predictions = np.array(self.predictions)
        targets = np.array(self.targets)
        accuracies = (predictions == targets).astype(float)
        
        # ECE (Expected Calibration Error)
        ece = self._compute_ece(confidences, accuracies)
        
        # Brier Score
        brier = self._compute_brier(confidences, accuracies)
        
        # Mean confidence and accuracy
        mean_conf = confidences.mean()
        mean_acc = accuracies.mean()
        
        # Overconfidence
        overconfidence = mean_conf - mean_acc
        
        return {
            "ece": ece,
            "brier": brier,
            "mean_confidence": mean_conf,
            "mean_accuracy": mean_acc,
            "overconfidence": overconfidence,
        }
    
    def _compute_ece(self, confidences: np.ndarray, accuracies: np.ndarray) -> float:
        """Expected Calibration Error."""
        bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
        ece = 0.0
        
        for i in range(self.n_bins):
            in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                avg_confidence = confidences[in_bin].mean()
                avg_accuracy = accuracies[in_bin].mean()
                ece += prop_in_bin * abs(avg_accuracy - avg_confidence)
        
        return ece
    
    def _compute_brier(self, confidences: np.ndarray, accuracies: np.ndarray) -> float:
        """Brier Score (mean squared error of confidence vs accuracy)."""
        return ((confidences - accuracies) ** 2).mean()


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Compute entropy of prediction distribution."""
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1)
    return entropy


# ============================================================
# Data Loading
# ============================================================

class SimpleTextDataset(Dataset):
    """Simple dataset for language modeling."""
    
    def __init__(
        self,
        data: torch.Tensor,
        seq_len: int,
    ):
        self.data = data
        self.seq_len = seq_len
    
    def __len__(self):
        return len(self.data) - self.seq_len
    
    def __getitem__(self, idx):
        chunk = self.data[idx:idx + self.seq_len + 1]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y


def load_wikitext(
    split: str = "train",
    tokenizer_name: str = "gpt2",
    max_length: Optional[int] = None,
    dataset_name: str = "wikitext-2",
) -> torch.Tensor:
    """Load WikiText-2 or WikiText-103."""
    try:
        from datasets import load_dataset
        from transformers import AutoTokenizer
        
        # Map friendly names to HuggingFace dataset names
        dataset_map = {
            "wikitext-2": "wikitext-2-raw-v1",
            "wikitext-103": "wikitext-103-raw-v1",
        }
        hf_name = dataset_map.get(dataset_name, dataset_name)
        
        print(f"Loading {dataset_name} ({split} split)...")
        dataset = load_dataset("wikitext", hf_name, split=split)
        
        print(f"Loading tokenizer: {tokenizer_name}...")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # Tokenize
        print("Tokenizing...")
        text = "\n\n".join(dataset["text"])
        tokens = tokenizer.encode(text)
        
        if max_length:
            tokens = tokens[:max_length]
        
        print(f"Total tokens: {len(tokens):,}")
        return torch.tensor(tokens, dtype=torch.long)
    
    except ImportError:
        print("datasets/transformers not installed. Using dummy data.")
        return create_dummy_data(max_length or 100000)


def create_dummy_data(length: int, vocab_size: int = 50257) -> torch.Tensor:
    """Create random dummy data for testing."""
    return torch.randint(0, vocab_size, (length,))


# ============================================================
# Training Loop
# ============================================================

class Trainer:
    """Training loop with logging and evaluation."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: TrainConfig,
        device: torch.device,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        
        # Scheduler
        total_steps = len(train_loader) * config.epochs
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps,
            eta_min=config.learning_rate / 10,
        )
        
        # Metrics
        self.calibration = CalibrationMetrics()
        
        # Logging
        self.log_dir = Path(config.log_dir) / config.experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.step = 0
        self.best_val_loss = float('inf')
        self.history = defaultdict(list)
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        n_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for batch_idx, (x, y) in enumerate(pbar):
            x, y = x.to(self.device), y.to(self.device)
            
            # Forward
            self.optimizer.zero_grad()
            output = self.model(x, labels=y)
            loss = output["loss"]
            
            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.optimizer.step()
            self.scheduler.step()
            
            # Logging
            total_loss += loss.item()
            n_batches += 1
            self.step += 1
            
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "ppl": f"{math.exp(loss.item()):.2f}",
                "lr": f"{self.scheduler.get_last_lr()[0]:.2e}",
            })
            
            # Periodic evaluation
            if self.step % self.config.eval_interval == 0:
                val_metrics = self.evaluate()
                self.log_metrics(val_metrics, prefix="val")
                
                # Save best model during training (not just at epoch end)
                if val_metrics["val_loss"] < self.best_val_loss:
                    self.best_val_loss = val_metrics["val_loss"]
                    self.save_checkpoint("best_model.pt")
                
                self.model.train()
            
            # Periodic checkpoint save (regardless of val_loss)
            if self.config.save_interval > 0 and self.step % self.config.save_interval == 0:
                self.save_checkpoint(f"checkpoint_step{self.step}.pt")
        
        return {"train_loss": total_loss / n_batches}
    
    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Evaluate on validation set."""
        self.model.eval()
        self.calibration.reset()
        
        total_loss = 0
        n_batches = 0
        all_entropies = []
        all_diffusion_times = []
        
        for x, y in tqdm(self.val_loader, desc="Evaluating", leave=False):
            x, y = x.to(self.device), y.to(self.device)
            
            output = self.model(x, labels=y)
            total_loss += output["loss"].item()
            n_batches += 1
            
            # Calibration
            self.calibration.update(output["logits"], y)
            
            # Entropy
            entropy = compute_entropy(output["logits"])
            all_entropies.append(entropy.cpu())
            
            # Diffusion times (if applicable)
            times = self.model.get_diffusion_times()
            if times is not None:
                all_diffusion_times.append(times.cpu())
        
        # Aggregate metrics
        avg_loss = total_loss / n_batches
        perplexity = math.exp(avg_loss)
        
        calibration_metrics = self.calibration.compute()
        
        entropies = torch.cat(all_entropies, dim=0).flatten()
        entropy_stats = {
            "entropy_mean": entropies.mean().item(),
            "entropy_std": entropies.std().item(),
            "entropy_min": entropies.min().item(),
            "entropy_max": entropies.max().item(),
        }
        
        metrics = {
            "val_loss": avg_loss,
            "perplexity": perplexity,
            **calibration_metrics,
            **entropy_stats,
        }
        
        # Diffusion time stats
        if all_diffusion_times:
            times = torch.cat(all_diffusion_times, dim=1).flatten()
            metrics["diffusion_time_mean"] = times.mean().item()
            metrics["diffusion_time_std"] = times.std().item()
        
        return metrics
    
    def log_metrics(self, metrics: Dict[str, float], prefix: str = ""):
        """Log metrics to console and file."""
        # Console
        metrics_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        print(f"[Step {self.step}] {prefix}: {metrics_str}")
        
        # Save to history
        for k, v in metrics.items():
            self.history[f"{prefix}_{k}"].append((self.step, v))
        
        # Save to file
        log_file = self.log_dir / "metrics.json"
        with open(log_file, "w") as f:
            json.dump(dict(self.history), f, indent=2)
    
    def save_checkpoint(self, name: str = "checkpoint.pt"):
        """Save model checkpoint."""
        path = self.log_dir / name
        torch.save({
            "step": self.step,
            "epoch": getattr(self, 'current_epoch', 0),
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "history": dict(self.history),
        }, path)
        print(f"Saved checkpoint to {path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        print(f"Loading checkpoint from {path}...")
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.step = checkpoint["step"]
        self.best_val_loss = checkpoint.get("best_val_loss", float('inf'))
        if "history" in checkpoint:
            self.history = defaultdict(list, checkpoint["history"])
        start_epoch = checkpoint.get("epoch", self.step // len(self.train_loader)) + 1
        print(f"Resumed from step {self.step}, will start at epoch {start_epoch}")
        return start_epoch
    
    def train(self, start_epoch: int = 1) -> Dict[str, float]:
        """Full training loop."""
        print(f"Starting training for {self.config.epochs} epochs (starting at epoch {start_epoch})...")
        print(f"Log directory: {self.log_dir}")
        
        for epoch in range(start_epoch, self.config.epochs + 1):
            self.current_epoch = epoch
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.evaluate()
            
            self.log_metrics(train_metrics, prefix="train")
            self.log_metrics(val_metrics, prefix="val")
            
            # Save best model
            if val_metrics["val_loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["val_loss"]
                self.save_checkpoint("best_model.pt")
        
        # Final save
        self.save_checkpoint("final_model.pt")
        
        return val_metrics


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Train Diffusion Attention Transformer")
    
    # Model
    parser.add_argument("--model", type=str, default="standard",
                        choices=["standard", "diffusion_fixed", "diffusion_adaptive"],
                        help="Attention type")
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--d_ff", type=int, default=1024,
                        help="Feed-forward hidden dimension (default: 4 * d_model)")
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--fixed_t", type=float, default=1.0)
    parser.add_argument("--t_min", type=float, default=0.1)
    parser.add_argument("--t_max", type=float, default=10.0)
    
    # Training
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--eval_interval", type=int, default=500,
                        help="Evaluate every N steps")
    parser.add_argument("--save_every", type=int, default=None,
                        help="Save checkpoint every N steps (in addition to best model)")
    
    # Data
    parser.add_argument("--dataset", type=str, default="wikitext-2",
                        choices=["wikitext-2", "wikitext-103"],
                        help="Dataset to use (wikitext-2 or wikitext-103)")
    parser.add_argument("--max_tokens", type=int, default=None)
    
    # Misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--log_dir", type=str, default="./logs")
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from (e.g., logs/final_t03/best_model.pt)")
    
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Config
    model_config = ModelConfig(
        d_model=args.d_model,
        d_ff=args.d_ff,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        max_seq_len=args.seq_len,
        dropout=args.dropout,
        attention_type=args.model,
        fixed_t=args.fixed_t,
        t_min=args.t_min,
        t_max=args.t_max,
    )
    
    exp_name = args.exp_name or f"{args.model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    train_config = TrainConfig(
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        log_dir=args.log_dir,
        experiment_name=exp_name,
        eval_interval=args.eval_interval,
        save_interval=args.save_every if args.save_every else 0,
    )
    
    # Save configs
    log_dir = Path(train_config.log_dir) / exp_name
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(log_dir / "config.json", "w") as f:
        json.dump({
            "model": asdict(model_config),
            "train": asdict(train_config),
            "args": vars(args),
        }, f, indent=2)
    
    # Data
    print("Loading data...")
    train_data = load_wikitext("train", max_length=args.max_tokens, dataset_name=args.dataset)
    val_data = load_wikitext("validation", max_length=args.max_tokens // 10 if args.max_tokens else None, dataset_name=args.dataset)
    
    train_dataset = SimpleTextDataset(train_data, args.seq_len)
    val_dataset = SimpleTextDataset(val_data, args.seq_len)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Model
    print(f"Creating model with {args.model} attention...")
    model = DiffusionTransformer(model_config)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    
    # Train
    trainer = Trainer(model, train_loader, val_loader, train_config, device)
    
    start_epoch = 1
    if args.resume:
        start_epoch = trainer.load_checkpoint(args.resume)
    
    final_metrics = trainer.train(start_epoch=start_epoch)
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("Final metrics:")
    for k, v in final_metrics.items():
        print(f"  {k}: {v:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
