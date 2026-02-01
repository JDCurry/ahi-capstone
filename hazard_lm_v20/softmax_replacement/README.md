# Diffusion-Based Attention

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18004600.svg)](https://doi.org/10.5281/zenodo.18004600)


This repository contains the implementation and experimental results for the paper:

**"Diffusion Attention: Replacing Softmax with Heat Kernel Dynamics"**

We propose replacing the softmax operation in transformer attention with a heat kernel diffusion process. Standard softmax attention is equivalent to the equilibrium distribution of drift-diffusion dynamics on similarity scores. By stopping before equilibrium, diffusion attention provides improved calibration (6-46% ECE reduction) while maintaining competitive perplexity.

## Key Findings

- Diffusion attention with fixed t=0.28 reduces Expected Calibration Error (ECE) by 6-12% at 4 layers
- Calibration improvements **increase with depth**: 24-46% ECE reduction at 12 layers
- Depth scaling law: optimal diffusion time follows t proportional to 1/sqrt(L)
- Perplexity-calibration tradeoff: adaptive t optimizes perplexity, not calibration

## Repository Structure

```
Diffusion-Based-Attention/
|-- train_diffusion_attention.py   # Main training script
|-- diffusion_attention_torch.py   # PyTorch attention modules (required import)
|-- requirements.txt               # Python dependencies
|-- README.md                      # This file
|-- logs/                          # Experiment logs with metrics
|   |-- 4layer_t028/
|   |-- 8layer_t020/
|   |-- 12layer_t016/
|   |-- adaptive_500k/
|   +-- ...
|-- manuscript_images/             # Figures for the paper
|-- scripts/                       # Additional utility scripts
    |-- generate_text.py           # Text generation for sanity checks
    +-- visualize_results.py       # Figure generation
```

## Installation

```bash
git clone https://github.com/jdcurry/diffusion-based-attention.git
cd diffusion-based-attention
pip install -r requirements.txt
```

For GPU support, install PyTorch with CUDA following instructions at:
https://pytorch.org/get-started/locally/

## Usage

### Training Models

Basic training with diffusion attention (4 layers, t=0.28):

```bash
python train_diffusion_attention.py --model diffusion_fixed --fixed_t 0.28 --n_layers 4 --epochs 1 --max_tokens 500000 --exp_name 4layer_t028
```

Training with depth-scaled diffusion time (follows t proportional to 1/sqrt(L)):

```bash
# 8 layers: t = 0.28 * sqrt(4/8) = 0.20
python train_diffusion_attention.py --model diffusion_fixed --fixed_t 0.20 --n_layers 8 --epochs 1 --max_tokens 500000 --exp_name 8layer_t020

# 12 layers: t = 0.28 * sqrt(4/12) = 0.16
python train_diffusion_attention.py --model diffusion_fixed --fixed_t 0.16 --n_layers 12 --epochs 1 --max_tokens 500000 --exp_name 12layer_t016
```

Standard softmax baseline:

```bash
python train_diffusion_attention.py --model standard --n_layers 4 --epochs 1 --max_tokens 500000 --exp_name 4layer_standard
```

Adaptive diffusion time (learns t during training):

```bash
python train_diffusion_attention.py --model diffusion_adaptive --epochs 1 --max_tokens 500000 --exp_name adaptive_500k
```

### Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| --model | standard | Attention type: standard, diffusion_fixed, diffusion_adaptive |
| --fixed_t | 1.0 | Diffusion time for diffusion_fixed model |
| --n_layers | 4 | Number of transformer layers |
| --d_model | 256 | Model dimension |
| --d_ff | 1024 | Feed-forward dimension |
| --n_heads | 4 | Number of attention heads |
| --epochs | 10 | Number of training epochs |
| --max_tokens | None | Maximum tokens to use (None = full dataset) |
| --batch_size | 32 | Batch size |
| --lr | 3e-4 | Learning rate |
| --seq_len | 256 | Sequence length |
| --dataset | wikitext-2 | Dataset: wikitext-2 or wikitext-103 |
| --eval_interval | 500 | Evaluate every N steps |
| --save_every | None | Save checkpoint every N steps |
| --exp_name | None | Experiment name for logging |
| --resume | None | Path to checkpoint to resume from |

### Text Generation (Sanity Check)

```bash
python scripts/generate_text.py logs/12layer_t016/best_model.pt --prompt "The meaning of life is" --max_tokens 100 --num_samples 3
```

### Generating Figures

```bash
python scripts/visualize_results.py
```

This generates the paper figures in the current directory.

## Experiment Logs

Each experiment directory in `logs/` contains:

- `config.json`: Model and training configuration
- `metrics.json`: Per-step metrics including:
  - ECE (Expected Calibration Error)
  - Brier score
  - Perplexity
  - Entropy statistics
  - Learned diffusion times (for adaptive models)

## Depth Scaling Law

The optimal diffusion time scales with model depth:

```
t(L) = t_0 * sqrt(L_0 / L)
```

Where:
- t_0 = 0.28 (optimal at 4 layers)
- L_0 = 4 (reference depth)

Predictions:
- 4 layers: t = 0.28
- 8 layers: t = 0.20
- 12 layers: t = 0.16
- 24 layers (GPT-2 Medium): t = 0.11
- 96 layers (GPT-3 scale): t = 0.06

## Results Summary

| Model | Layers | t | ECE @ 500 | ECE @ 3000 | ECE Final |
|-------|--------|---|-----------|------------|-----------|
| Diffusion | 4 | 0.28 | 0.106 | 0.245 | 0.453 |
| Standard | 4 | - | 0.121 | 0.277 | 0.479 |
| Diffusion | 8 | 0.20 | 0.097 | 0.238 | 0.452 |
| Standard | 8 | - | 0.109 | 0.290 | 0.499 |
| Diffusion | 12 | 0.16 | 0.088 | 0.150 | 0.356 |
| Standard | 12 | - | 0.116 | 0.279 | 0.488 |

## Hardware

Experiments were conducted on:
- Dell Precision Tower 7810
- Dual Intel Xeon E5-2699 v3 (36 cores, 72 threads)
- 128 GB RAM
- NVIDIA RTX A4000 (16GB VRAM)

Typical training times (500k tokens, 1 epoch):
- 4 layers: ~3 hours
- 8 layers: ~4 hours
- 12 layers: ~6 hours

