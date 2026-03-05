# CD-Only Baseline Experiment Record (DeltaCD v2)

> **Purpose**: This snapshot preserves the complete CD-only training state as a stable baseline.
> If subsequent CD+CC joint training degrades CD performance, this record enables exact rollback.

---

## 1. Final Results (Best Checkpoint)

| Metric | Value |
|---|---|
| **mIoU (3-class)** | **0.8409** |
| **mIoU (change-only)** | **0.7725** |
| **Best Epoch** | **90 / 100** |
| OA (Overall Accuracy) | 0.9792 |
| FWIoU | 0.9610 |
| MPA | 0.9368 |
| Binary IoU | 0.7795 |

### Per-Class Breakdown (Best Epoch 90)

| Class | IoU | F1 | Precision | Recall |
|---|---|---|---|---|
| Background | 0.9777 | — | — | — |
| Road | 0.7547 | 0.8602 | 0.8444 | 0.8766 |
| Building | 0.7903 | 0.8829 | 0.8352 | 0.9363 |

### Final Epoch 100 Results (for reference)

| Metric | Value |
|---|---|
| mIoU (3-class) | 0.8393 |
| mIoU (change-only) | 0.7703 |
| OA | 0.9787 |
| IoU_road | 0.7528 |
| IoU_building | 0.7879 |
| F1_road | 0.8590 |
| F1_building | 0.8813 |
| Precision_road | 0.8335 |
| Precision_building | 0.8293 |
| Recall_road | 0.8860 |
| Recall_building | 0.9404 |

---

## 2. Training Configuration

| Parameter | Value |
|---|---|
| Config file | `train_delta_cd.yaml` |
| Decoder type | `delta_cd` (DeltaCD v2) |
| Backbone | EVA-ViT-G (frozen) |
| HR Encoder | ResNet-18 (pretrained, trainable) |
| Num classes | 3 (background, road, building) |
| Input size | 224 x 224 |
| Mask output size | 256 x 256 |
| Loss function | CrossEntropyLoss (weights: [0.2, 1.0, 1.0]) |
| Optimizer | AdamW |
| Learning rate | 1e-4 (cosine decay) |
| Min LR | 1e-6 |
| Weight decay | 0.01 |
| Warmup epochs | 3 |
| Total epochs | 100 |
| Batch size | 8 |
| Random seed | **42** |
| Gradient clipping | max_norm=1.0 |
| Filter no-change samples | False (all samples used) |
| Pretrained checkpoint | `./checkpoint_best.pth` (DeltaVLM CC pretrained) |

---

## 3. Model Architecture

```
DeltaCD v2 (13.6M trainable / 1.0B total)
├── hr_encoder (ResNet-18, pretrained): 2,782,784 params
│   ├── stem → (B, 64, 56, 56)
│   ├── layer1 → (B, 64, 56, 56)
│   ├── layer2 → (B, 128, 28, 28)
│   └── layer3 → (B, 256, 14, 14)
├── sem_adapter (ViT→spatial): 951,552 params
│   ├── Linear(1408→256) + LayerNorm
│   ├── Interpolate(16→14)
│   └── Conv2d(256→256) + BN + ReLU
├── sem_inject (fusion): 1,180,160 params
│   └── Conv2d(512→256) + BN + ReLU
├── csrm (SpatialCSRM, dim=256): 262,656 params
│   ├── gate_diff: Linear(256→256, no bias)
│   ├── gate_feat: Linear(256→256)
│   ├── context_diff: Linear(256→256, no bias)
│   └── context_feat: Linear(256→256)
├── conv_dif (3 scales): 87,360 params
├── conv_fuse (3 scales): 4,992,512 params
├── to_fused (FPN decoder): 3,230,848 params
└── to_seg (seg head): 131,459 params
```

---

## 4. Dataset

| Split | Images (pairs) | Source |
|---|---|---|
| Train | 6,815 | LEVIR-MCI |
| Val | 1,333 | LEVIR-MCI |
| Test | 1,929 | LEVIR-MCI |

- **Data root**: `/root/autodl-tmp/LEVIR-MCI-dataset/images`
- **Image format**: `{split}/A/{split}_{id}.png` and `{split}/B/{split}_{id}.png`
- **Mask format**: `{split}/label/{split}_{id}.png`
- **Mask classes**: 0=background, 1=road, 2=building
- **Augmentation (train)**: Random horizontal flip, random vertical flip, resize to 224x224 / 256x256
- **Augmentation (val)**: Resize only
- **Normalization**: ImageNet (mean=[0.4815, 0.4578, 0.4082], std=[0.2686, 0.2613, 0.2758])

---

## 5. Training Timeline

| Epoch | Train Loss | Val Loss | mIoU (3cls) | OA | Note |
|---|---|---|---|---|---|
| 1 | 0.3070 | 0.1620 | 0.7243 | 0.9542 | |
| 4 | 0.1041 | 0.0979 | 0.8012 | 0.9724 | |
| 7 | 0.0800 | 0.0955 | 0.8151 | 0.9745 | |
| 10 | 0.0692 | 0.0862 | 0.7857 | 0.9682 | Val loss minimum |
| 12 | 0.0640 | 0.0960 | 0.8211 | 0.9760 | |
| 19 | 0.0540 | 0.0939 | 0.8286 | 0.9768 | |
| 25 | 0.0483 | 0.1079 | 0.8337 | 0.9776 | |
| 41 | 0.0401 | 0.1255 | 0.8369 | 0.9784 | |
| 49 | 0.0370 | 0.1201 | 0.8384 | 0.9784 | |
| 60 | 0.0345 | 0.1260 | 0.8391 | 0.9785 | |
| 75 | 0.0324 | 0.1404 | 0.8401 | 0.9789 | |
| **90** | **0.0303** | **0.1608** | **0.8409** | **0.9792** | **Best** |
| 100 | 0.0299 | 0.1501 | 0.8393 | 0.9787 | Final |

---

## 6. Checkpoint Files

| File | Size | MD5 | Description |
|---|---|---|---|
| `mask_branch_best.pth` | 53M | `5b87c1248789d73981b2e24461bdf398` | Best epoch (90), mask_decoder state_dict only |
| `mask_branch_final.pth` | 53M | `900eb9f906aa42dff4df2ea4148b4950` | Final epoch (100), mask_decoder state_dict only |
| `config.yaml` | 0.5K | — | Training configuration |
| `train.log` | 219K | — | Full training log (outer) |
| `train_internal.log` | — | — | Full training log (inner, with verbose model loading) |

Note: Full optimizer-state checkpoints (`checkpoint_10.pth` .. `checkpoint_100.pth`, 157M each)
are stored in `output/delta_cd_v2/20260304_211738/` and can be used to resume training.

---

## 7. Code Snapshots

The following source files are frozen at the state used for this experiment:

| File | Original path |
|---|---|
| `delta_cd.py.snapshot` | `model/mask_branch/delta_cd.py` |
| `blip2_vicua_mask.py.snapshot` | `model/blip2_vicua_mask.py` |
| `train_mask.py.snapshot` | `train_mask.py` |
| `dataset_mask.py.snapshot` | `dataset_mask.py` |

---

## 8. Evaluation Protocol

- **Metric computation**: Global confusion matrix accumulated over entire val set, then compute IoU/F1/Prec/Rec per class (matching Change-Agent protocol)
- **NOT per-batch averaging** (this was a critical fix from earlier versions)
- **Evaluator class**: `ConfusionMatrixEvaluator` in `train_mask.py`

---

## 9. Reproducibility Checklist

- [x] Random seed fixed: 42
- [x] `cudnn.benchmark = False`, `cudnn.deterministic = True`
- [x] Dataset split: standard LEVIR-MCI train/val/test
- [x] All source code snapshotted
- [x] Config file preserved
- [x] Checkpoint MD5 recorded
- [x] Full training log preserved
- [x] Evaluation uses global confusion matrix (not per-batch average)

---

## 10. How to Reproduce

```bash
cd /root/autodl-tmp/DeltaVLM

# Train from scratch
python train_mask.py --cfg_path train_delta_cd.yaml

# Evaluate with saved checkpoint
python analyze_cd.py --checkpoint experiments/cd_only_baseline_v2/mask_branch_best.pth
```

---

## 11. Known Observations

1. **Overfitting**: Val loss increases after epoch 10 (0.0862 → 0.1608 at best epoch), but mIoU continues to improve slowly. The model is well-calibrated in terms of classification boundaries but not in terms of confidence scores.
2. **CSRM gate health**: Gate values are well-distributed (mean ~0.41, 62% in [0.3, 0.7]), not saturated.
3. **Building recall is high (0.94)** but precision is lower (0.83), suggesting some false positives for buildings.
4. **Road IoU (0.75)** is the weakest class, likely due to thin/linear geometry being harder to segment at 256x256 resolution.
