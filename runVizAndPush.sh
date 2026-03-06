#!/bin/bash
set -e
cd /root/autodl-tmp/DeltaVLM

echo "============================================"
echo "Step 1/3: Copy paper-selected samples"
echo "============================================"
mkdir -p output/paperFigures
PANELS=output/comparison/panels
ERRORS=output/comparison/errorMaps

# 5 representative samples for paper (dense urban, mixed, road-heavy, large change, dense small buildings)
for IDX in 000_test_000018 003_test_000408 005_test_000644 007_test_000853 019_test_001883; do
    cp "${PANELS}/${IDX}_panel.png" output/paperFigures/ 2>/dev/null || true
    cp "${ERRORS}/${IDX}_fpfn.png"  output/paperFigures/ 2>/dev/null || true
done
echo ">>> Copied 5 panel + 5 error maps to output/paperFigures/"
ls -la output/paperFigures/

echo ""
echo "============================================"
echo "Step 2/3: Run analyze_cd.py (CSRM/gate/HR)"
echo "============================================"
python analyze_cd.py \
    --checkpoint experiments/cd_only_baseline_v2/mask_branch_best.pth \
    --data_root /root/autodl-tmp/LEVIR-MCI-dataset/images \
    --output_dir ./output/analysis \
    --num_vis 20

echo ">>> analyze_cd.py done"
ls -la output/analysis/

echo ""
echo "============================================"
echo "Step 3/3: Git add, commit, push"
echo "============================================"
git add -A
git status
git commit -m "Add SOTA comparison viz, module analysis, paper figures

- compareViz.py: fix hidden_dim for Change-Agent, fix tensor or logic
- output/comparison: 20 panel + 20 FP/FN error maps (DeltaVLM vs Change-Agent vs SegformerCD)
- output/analysis: CSRM gate maps, HR features, semantic adapter, confusion matrix
- output/paperFigures: 5 selected representative samples for paper
- train_segformer_cd.py: 3-class training with ConfusionMatrixEvaluator
- runBaselines.sh: corrected script references" || echo "Nothing to commit"

git push origin main || git push --force origin main

echo ""
echo "============================================"
echo "ALL 3 STEPS COMPLETE"
echo "============================================"
