#!/bin/bash
set -e
cd /root/autodl-tmp/DeltaVLM

echo "=== Ablation 1/3: w/o CSRM (50 epochs) ==="
echo "Started: $(date)"
python train_cd.py --cfg_path configs/abl_nocsrm.yaml 2>&1 | tee output/abl_nocsrm.log
echo "Finished: $(date)"

echo "=== Ablation 2/3: w/o HR branch (50 epochs) ==="
echo "Started: $(date)"
python train_cd.py --cfg_path configs/abl_nohr.yaml 2>&1 | tee output/abl_nohr.log
echo "Finished: $(date)"

echo "=== Ablation 3/3: w/o Semantic injection (50 epochs) ==="
echo "Started: $(date)"
python train_cd.py --cfg_path configs/abl_nosem.yaml 2>&1 | tee output/abl_nosem.log
echo "Finished: $(date)"

echo "=== All 3 ablations complete! ==="
