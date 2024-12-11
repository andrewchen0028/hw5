# Diffusion Model Implementation

A PyTorch implementation of Denoising Diffusion Probabilistic Models (DDPM) for image generation.

## Overview

This implementation includes:
- Basic DDPM training pipeline
- UNet architecture with configurable parameters
- Noise scheduling with linear beta schedule
- Training on ImageNet100 dataset (128x128 resolution)

## Setup

1. **Download Dataset**
```bash
# Download ImageNet100 from provided link
tar -xvf imagenet100_128x128.tar.gz
```

2. **Install Dependencies**
```bash
pip install torch torchvision wandb ruamel.yaml tqdm pillow
```

## Project Structure

```
.
├── models/
│   ├── unet.py             # UNet architecture
│   └── unet_modules.py     # UNet building blocks (ResBlock, Attention)
├── schedulers/
│   └── ddpm.py            # DDPM noise scheduler (linear schedule)
├── utils/
│   ├── metric.py          # Training metrics (AverageMeter)
│   └── misc.py            # Helper functions
├── configs/
│   └── ddpm.yaml          # Training configuration
├── train.py               # Training script with validation
└── inference.py           # Image generation script
```

## Usage

1. **Training**
```bash
python train.py \
--config configs/ddpm.yaml \
--run_name experiment_name \
--batch_size 64 \
--image_size 64 \
--unet_in_size 64 \
--unet_ch 128 \
--num_train_timesteps 128 \
--num_inference_steps 128 \
--learning_rate 2e-4 \
--num_epochs 16
```

Key Parameters:
- `image_size`: Input image resolution
- `unet_ch`: Base channel width of UNet
- `num_train_timesteps`: Number of noise steps
- `batch_size`: Images per batch
- `unet_ch_mult`: Channel multipliers ([1,2,2,4] default)
- `unet_attn`: Attention layer indices ([2,3] default)
- `gradient_accumulation_steps`: Default 1

2. **Inference**
```bash
python inference.py \
--config configs/ddpm.yaml \
--ckpt experiments/exp-name/checkpoints/latest.pt
```

## Model Architecture

The implementation uses a UNet with:
- Configurable channel multipliers
- Attention layers at specified depths
- Time embedding for noise level conditioning
- Residual blocks with optional dropout
- GroupNorm with 32 groups
- Optional class conditioning

## Training Process

1. Data preprocessing:
   - Resize to specified resolution
   - Normalize to [-1, 1] range
   - Random horizontal flip augmentation

2. Training loop:
   - Add noise according to scheduler
   - Predict noise using UNet
   - Optimize with AdamW (weight decay 1e-4)
   - Cosine learning rate schedule
   - Optional gradient clipping at 1.0

3. Validation:
   - Generate 4 samples every epoch
   - Save model checkpoints
   - Log loss metrics and samples to WandB
   - Track running average loss

## Results

The model progressively learns to:
1. Distinguish basic patterns from noise
2. Generate coherent textures
3. Form recognizable features
4. Create thematic compositions

Typical training metrics:
- Initial loss: ~1.0
- Convergence: 1000-2000 steps
- Final loss: ~0.025
- Training time: ~12 hours on V100

## Monitoring

Training progress can be monitored through:
- WandB dashboard (loss curves, generated samples)
- Console output with loss metrics (every 100 steps)
- Generated samples saved each epoch
- Checkpoints saved in experiments/exp-name/checkpoints/

