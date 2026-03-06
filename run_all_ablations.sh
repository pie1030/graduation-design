#!/bin/bash
set -e
cd /root/autodl-tmp/DeltaVLM

echo "=== Ablation 1/3: w/o CSRM (50 epochs) ==="
echo "Started: $(date)"
python train_mask.py --cfg_path ablation_no_csrm.yaml 2>&1 | tee output/ablation_no_csrm_train.log
echo "Finished: $(date)"
echo ""

echo "=== Ablation 2/3: w/o HR branch (50 epochs) ==="
echo "Started: $(date)"
python train_mask.py --cfg_path ablation_no_hr.yaml 2>&1 | tee output/ablation_no_hr_train.log
echo "Finished: $(date)"
echo ""

echo "=== Ablation 3/3: w/o Semantic injection (50 epochs) ==="
echo "Started: $(date)"
python train_mask.py --cfg_path ablation_no_sem.yaml 2>&1 | tee output/ablation_no_sem_train.log
echo "Finished: $(date)"
echo ""

echo "=== All 3 ablations complete! ==="
echo "Finished at: $(date)"
