#!/usr/bin/env python3
"""
Text generation script for sanity checking trained models.
Loads a checkpoint and generates text to verify model isn't degenerate.
"""

import torch
import torch.nn.functional as F
import argparse
from pathlib import Path

# Import from training script
from train_diffusion_attention import DiffusionTransformer, ModelConfig


def load_model(checkpoint_path: str, device: str = "cuda") -> tuple:
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get config from checkpoint
    config_dict = checkpoint.get("config", {})
    config = ModelConfig(**config_dict)
    
    # Create model
    model = DiffusionTransformer(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    return model, config


def generate(
    model: DiffusionTransformer,
    prompt_ids: torch.Tensor,
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
) -> torch.Tensor:
    """Generate text autoregressively."""
    device = next(model.parameters()).device
    generated = prompt_ids.to(device)
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Get logits for last position
            output = model(generated)
            logits = output["logits"][:, -1, :] / temperature
            
            # Top-k filtering
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')
            
            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')
            
            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            generated = torch.cat([generated, next_token], dim=1)
            
            # Stop at max sequence length
            if generated.shape[1] >= model.config.max_seq_len:
                break
    
    return generated


def main():
    parser = argparse.ArgumentParser(description="Generate text from trained model")
    parser.add_argument("checkpoint", type=str, help="Path to model checkpoint")
    parser.add_argument("--prompt", type=str, default="The meaning of life is",
                       help="Text prompt to start generation")
    parser.add_argument("--max_tokens", type=int, default=100,
                       help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8,
                       help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=50,
                       help="Top-k filtering")
    parser.add_argument("--top_p", type=float, default=0.9,
                       help="Top-p (nucleus) filtering")
    parser.add_argument("--num_samples", type=int, default=3,
                       help="Number of samples to generate")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args()
    
    # Load tokenizer
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
    except ImportError:
        print("Error: transformers library required for tokenization")
        return
    
    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model, config = load_model(args.checkpoint, args.device)
    print(f"Model config: {config.n_layers} layers, {config.d_model} dim, {config.attention_type} attention")
    
    if hasattr(config, 'fixed_t') and config.attention_type == "diffusion_fixed":
        print(f"Diffusion time: t={config.fixed_t}")
    
    # Tokenize prompt
    prompt_ids = tokenizer.encode(args.prompt, return_tensors="pt")
    
    print(f"\nPrompt: {args.prompt}")
    print(f"Generating {args.num_samples} samples with {args.max_tokens} tokens each...")
    print("=" * 60)
    
    for i in range(args.num_samples):
        generated_ids = generate(
            model,
            prompt_ids,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )
        
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
        print(f"\n--- Sample {i+1} ---")
        print(generated_text)
    
    print("\n" + "=" * 60)
    print("Generation complete!")
    
    # Quick quality check
    print("\n[Sanity Check]")
    print("✓ Model loaded successfully")
    print("✓ Generation completed without errors")
    print("→ Review samples above for coherence and fluency")


if __name__ == "__main__":
    main()
