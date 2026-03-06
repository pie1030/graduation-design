#!/bin/bash
# Run all baseline experiments for SOTA comparison.
# Execute this AFTER ablation experiments have finished.
# Usage:  bash runBaselines.sh [--skip-change-agent] [--skip-segformer] [--skip-levircd]

set -e
cd /root/autodl-tmp/DeltaVLM

SKIP_CA=false
SKIP_SEG=false
SKIP_LCD=false
for arg in "$@"; do
    case $arg in
        --skip-change-agent) SKIP_CA=true ;;
        --skip-segformer)    SKIP_SEG=true ;;
        --skip-levircd)      SKIP_LCD=true ;;
    esac
done

echo "================================================"
echo "  DeltaVLM SOTA Baseline Training  $(date)"
echo "================================================"

# --- 1. Change-Agent baseline on LEVIR-MCI (3-class) ---
if [ "$SKIP_CA" = false ]; then
    echo ">>> [1/3] Change-Agent on LEVIR-MCI  start: $(date)"
    python train_mask.py --cfg_path train_mask_change_agent.yaml \
        2>&1 | tee output/changeAgentMci_train.log
    echo "    Done: $(date)"
else
    echo ">>> [1/3] Change-Agent -- SKIPPED"
fi

# --- 2. SegformerCD (ChangeFormer-style) on LEVIR-MCI (3-class) ---
if [ "$SKIP_SEG" = false ]; then
    echo ">>> [2/3] SegformerCD on LEVIR-MCI  start: $(date)"
    python trainSegformerMci.py \
        --data_root /root/autodl-tmp/LEVIR-MCI-dataset/images \
        --output_dir ./output/segformer_mci \
        --epochs 100 --batch_size 16 \
        2>&1 | tee output/segformerMci_train.log
    echo "    Done: $(date)"
else
    echo ">>> [2/3] SegformerCD -- SKIPPED"
fi

# --- 3. DeltaVLM binary CD on LEVIR-CD ---
if [ "$SKIP_LCD" = false ]; then
    echo ">>> [3/3] DeltaVLM on LEVIR-CD (binary)"
    LEVIRCD=/root/autodl-tmp/LEVIR-CD/images
    if [ ! -d "$LEVIRCD/train" ]; then
        echo "    WARNING: LEVIR-CD not found. Run downloadLevircd.sh first. Skipping."
    else
        echo "    Start: $(date)"
        python train_mask.py --cfg_path trainDeltaLevircd.yaml \
            2>&1 | tee output/deltaLevircd_train.log
        echo "    Done: $(date)"
    fi
else
    echo ">>> [3/3] DeltaVLM LEVIR-CD -- SKIPPED"
fi

echo "================================================"
echo "  All baselines complete: $(date)"
echo "================================================"
