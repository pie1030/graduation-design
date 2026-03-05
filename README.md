# DeltaVLM: Unified Change Detection and Change Captioning via Shared Difference Representation

A unified vision-language framework for remote sensing change understanding. DeltaVLM performs both **pixel-level Change Detection (CD)** and **language-level Change Captioning (CC)** through a shared Cross-temporal Spatial Reasoning Module (CSRM).

## Architecture Overview

```
Image_A, Image_B (bi-temporal pair)
        │
        ▼
  EVA-ViT-G (frozen backbone, 986M)
        │
        ▼ feat_A, feat_B  (257 tokens, 1408-dim)
        │
        ▼
┌────────────────────────────────────┐
│  CSRM (Cross-temporal Spatial      │
│  Reasoning Module)                 │
│  gate-modulated difference repr.   │
└──────────┬─────────────────────────┘
           │
    ┌──────┴───────┐
    ▼              ▼
 CC Branch      CD Branch (DeltaCD v2)
 Q-Former →     SemanticAdapter →
 Vicuna LLM →   HR Encoder (ResNet-18) →
 Caption        Multi-scale Diff Fusion →
                FPN Decoder → Change Map
```

## CD Branch: DeltaCD v2

Multi-scale change detection aligned with [Change-Agent](https://github.com/Chen-Yang-Liu/Change-Agent) pipeline.

**Key components:**
- **Path 1 (frozen)**: EVA-ViT semantic features → Semantic Adapter (1408d → 256d, 14x14)
- **Path 2 (trainable)**: Pretrained ResNet-18 → multi-scale spatial features (56x56, 28x28, 14x14)
- **Fusion**: Semantic injection at deepest scale + CSRM gate modulation
- **Decoder**: FPN top-down with multi-scale difference fusion (conv_dif + cosine similarity + conv_fuse)
- **Loss**: CrossEntropyLoss with class weights [0.2, 1.0, 1.0]

**Trainable parameters**: 13.6M (1.4% of total model)

## Current Results: CD-Only on LEVIR-MCI

| Metric | Value |
|---|---|
| mIoU (3-class) | **0.8409** |
| mIoU (change-only) | 0.7725 |
| IoU (road) | 0.7547 |
| IoU (building) | 0.7903 |
| IoU (background) | 0.9777 |
| F1 (road) | 0.8602 |
| F1 (building) | 0.8829 |
| OA | 0.9792 |

> Full experiment record: [`experiments/cd_only_baseline_v2/EXPERIMENT_RECORD.md`](experiments/cd_only_baseline_v2/EXPERIMENT_RECORD.md)

## Project Structure

```
DeltaVLM/
├── model/
│   ├── blip2_vicua.py              # Base model with CSRM (CC branch)
│   ├── blip2_vicua_mask.py         # Extended model with CD branch
│   ├── eva_vit.py                  # EVA-ViT-G backbone
│   └── mask_branch/
│       ├── delta_cd.py             # DeltaCD v2 (current CD implementation)
│       └── change_agent_cd.py      # Legacy CD module
├── train_mask.py                   # CD training script
├── train_delta_cd.yaml             # CD training config
├── dataset_mask.py                 # LEVIR-MCI dataset loader
├── processor_mask.py               # Data augmentation
├── analyze_cd.py                   # Visualization & analysis script
├── experiments/
│   └── cd_only_baseline_v2/        # Frozen experiment snapshot
│       ├── EXPERIMENT_RECORD.md    # Complete experiment record
│       ├── config.yaml             # Training config
│       ├── mask_branch_best.pth    # Best checkpoint (epoch 90)
│       ├── mask_branch_final.pth   # Final checkpoint (epoch 100)
│       ├── train.log               # Training log
│       └── *.py.snapshot           # Source code at time of experiment
├── ablation_no_csrm.yaml          # Ablation: w/o CSRM
├── ablation_no_hr.yaml            # Ablation: w/o HR branch
├── ablation_no_sem.yaml           # Ablation: w/o semantic injection
└── docs/
    └── ARCHITECTURE_OVERVIEW_AND_FIGURE_PROMPTS.md
```

## Dataset

**LEVIR-MCI** (Multi-Class Imagery): 3-class semantic change detection (background, road, building)

| Split | Pairs |
|---|---|
| Train | 6,815 |
| Val | 1,333 |
| Test | 1,929 |

## Training

```bash
# CD-only training
python train_mask.py --cfg_path train_delta_cd.yaml

# Ablation experiments
python train_mask.py --cfg_path ablation_no_csrm.yaml
python train_mask.py --cfg_path ablation_no_hr.yaml
python train_mask.py --cfg_path ablation_no_sem.yaml

# Visualization & analysis
python analyze_cd.py --checkpoint experiments/cd_only_baseline_v2/mask_branch_best.pth
```

## Requirements

- Python 3.10+
- PyTorch 2.x
- torchvision
- timm
- transformers
- matplotlib
- numpy
- PyYAML

## Acknowledgements

- [Change-Agent](https://github.com/Chen-Yang-Liu/Change-Agent) (CD architecture reference)
- [BLIP-2](https://github.com/salesforce/LAVIS) (VLM backbone)
- [EVA-CLIP](https://github.com/baaivision/EVA) (Visual encoder)
