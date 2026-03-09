# DeltaVLM

Unified Change Detection and Change Captioning framework built on EVA-ViT-G (frozen backbone) with a self-designed **CSRM** (Cross-temporal Spatial Reasoning Module) for shared difference representation learning.

## Architecture

```
Image A, Image B
     |
  EVA-ViT-G (frozen)          ResNet-18 HR Encoder (trainable)
     |                              |
  Semantic Adapter              Multi-scale features
     |                         (56x56, 28x28, 14x14)
     +----------Fusion-----------+
                 |
              CSRM (gate-modulated difference)
                 |
          FPN Decoder → Change Map (256x256, 3-class)
                 |
          Q-Former + Vicuna → Change Caption (optional)
```

**Core innovation**: CSRM produces a unified difference representation that serves both pixel-level CD and language-level CC tasks.

## Setup

```bash
pip install -r requirements.txt
```

**Pretrained weights** (place in project root):
- `checkpoint_best.pth` — DeltaVLM pretrained (EVA-ViT-G + Q-Former)
- `vicuna-7b-v1.5/` — Vicuna LLM (for CC branch only)

**Dataset**: [LEVIR-MCI](https://github.com/Chen-Yang-Liu/LEVIR-MCI) at `/root/autodl-tmp/LEVIR-MCI-dataset/images/`

```
images/
  ├── train/  (A/, B/, label/)
  ├── val/
  └── test/
```

## Training

### Change Detection (DeltaVLM)

```bash
python train_cd.py --cfg_path configs/cd.yaml
```

### SOTA Baselines

```bash
# Change-Agent decoder (EVA-ViT backbone, fair comparison)
python train_cd.py --cfg_path configs/cd_agent.yaml

# SegformerCD (standalone Segformer-B1 backbone)
python train_segformer.py --epochs 100 --batch_size 16
```

### Ablation Studies

```bash
bash ablations.sh
```

Configs: `configs/abl_nocsrm.yaml`, `configs/abl_nohr.yaml`, `configs/abl_nosem.yaml`

## Evaluation & Visualization

```bash
# Module-level analysis (gate maps, HR features, confusion matrix)
python analyze.py --checkpoint <path_to_best.pth>

# Qualitative comparison (DeltaVLM vs Change-Agent vs SegformerCD)
python compare.py \
    --deltavlm    <deltavlm_ckpt> \
    --change_agent <agent_ckpt> \
    --segformer   <segformer_ckpt> \
    --num_samples 20

# Efficiency benchmark (FLOPs, params, FPS, VRAM)
python benchmark.py
```

## Results on LEVIR-MCI (3-class)

### SOTA Comparison

| Method | mIoU | IoU_road | IoU_bldg | mF1(chg) | OA |
|--------|------|----------|----------|----------|------|
| **DeltaVLM (Ours)** | **0.8409** | 0.7547 | 0.7903 | 0.8716 | 0.9792 |
| Change-Agent decoder | 0.8196 | 0.7185 | 0.7646 | 0.8514 | 0.9771 |
| SegformerCD | 0.7829 | 0.6488 | 0.7287 | 0.8150 | 0.9727 |

### Ablation Study

| Variant | mIoU | mF1(chg) | OA |
|---------|------|----------|------|
| Full model | **0.8409** | **0.8716** | **0.9792** |
| w/o CSRM | 0.8285 | 0.8587 | 0.9776 |
| w/o HR branch | 0.8269 | 0.8601 | 0.9769 |
| w/o Semantic | 0.8341 | 0.8671 | 0.9783 |

## Structure

```
DeltaVLM/
├── configs/              
├── model/
│   ├── blip2_vlm.py      # Unified CD-CC model entry point
│   ├── blip2_vicuna.py   
│   ├── blip2.py           # BLIP-2 backbone (EVA-ViT + Q-Former)
│   ├── eva_vit.py         # EVA-ViT-G visual encoder
│   ├── Qformer.py         
│   └── mask_branch/
│       ├── delta_cd.py    # CSRM + HR + FPN
│       ├── agent_decoder.py  # Change-Agent CD module
│       ├── agent_encoder.py  # HR spatial encoder
│       ├── segformer.py      # Segformer backbone
│       ├── segformer_cd.py   # SegformerCD baseline
│       └── mask_head.py      # Shared mask head
├── train_cd.py           # CD training script
├── train_segformer.py    # SegformerCD baseline training
├── analyze.py            # visualize
├── compare.py            
├── predict.py            
├── benchmark.py         
├── ablations.sh          # Run ablation experiments
├── dataset_cd.py         # CD dataset (LEVIR-MCI)
├── processor_cd.py       # CD augmentation
├── eval_func/            # Caption evaluation metrics
├── utils.py            
├── logger.py             
└── requirements.txt
```
