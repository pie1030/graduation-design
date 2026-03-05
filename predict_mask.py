"""
Inference & Visualization for DeltaVLM Multi-class Change Detection

Generates:
  1. Per-class metrics (IoU/F1 for road, building) and binary IoU on test set
  2. Color change maps: red=building, yellow=road, black=background

Usage:
    python predict_mask.py \
        --cfg_path train_mask_change_agent.yaml \
        --checkpoint ./output/mask_branch_change_agent/<run>/mask_branch_best.pth \
        --save_dir ./results
"""

import argparse
import os
import sys
import logging
import json
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset_mask import build_mask_dataloaders, ChangeMaskDataset
from model.blip2_vicua_mask import Blip2VicunaMask
from processor_mask import MaskEvalTransforms

CLASS_NAMES = {0: 'bg', 1: 'road', 2: 'building'}
# Color palette: 0=black(bg), 1=yellow(road), 2=red(building)
PALETTE = np.array([
    [0,   0,   0],    # background
    [255, 255, 0],    # road (yellow)
    [255, 0,   0],    # building (red)
], dtype=np.uint8)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str,
                        default="/root/autodl-tmp/LEVIR-MCI-dataset/images")
    parser.add_argument("--mask_root", type=str, default=None)
    parser.add_argument("--pretrained", type=str, default="./checkpoint_best.pth")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="mask_branch_best.pth from training")
    parser.add_argument("--save_dir", type=str, default="./results")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--mask_hidden_dim", type=int, default=256)
    parser.add_argument("--mask_num_stages", type=int, default=4)
    parser.add_argument("--mask_size", type=int, default=256)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--cfg_path", type=str, default=None)
    parser.add_argument("--max_vis", type=int, default=50,
                        help="Max number of visualization samples to save")
    args, _ = parser.parse_known_args()

    import sys as _sys
    explicit = {a[2:].split('=')[0] for a in _sys.argv[1:] if a.startswith('--')}
    if args.cfg_path and os.path.exists(args.cfg_path):
        import yaml
        with open(args.cfg_path) as f:
            cfg = yaml.safe_load(f)
        for k, v in cfg.items():
            if k not in explicit:
                setattr(args, k, v)
    return args


def compute_metrics(pred_cls, gt_cls, num_classes=3):
    pred = pred_cls.view(-1).long()
    gt = gt_cls.view(-1).long()
    results = {}
    iou_sum, f1_sum, n_change = 0.0, 0.0, 0
    for c in range(num_classes):
        p = (pred == c)
        g = (gt == c)
        tp = (p & g).sum().float()
        fp = (p & ~g).sum().float()
        fn = (~p & g).sum().float()
        prec = (tp / (tp + fp + 1e-8)).item()
        rec  = (tp / (tp + fn + 1e-8)).item()
        iou  = (tp / (tp + fp + fn + 1e-8)).item()
        f1   = (2 * tp / (2 * tp + fp + fn + 1e-8)).item()
        name = CLASS_NAMES.get(c, str(c))
        results[f'iou_{name}'] = iou
        results[f'f1_{name}'] = f1
        results[f'prec_{name}'] = prec
        results[f'rec_{name}'] = rec
        if c > 0:
            iou_sum += iou; f1_sum += f1; n_change += 1
    results['mIoU'] = iou_sum / max(n_change, 1)
    results['mF1']  = f1_sum / max(n_change, 1)
    # Binary change IoU
    pb = (pred > 0); gb = (gt > 0)
    tp_b = (pb & gb).sum().float()
    fp_b = (pb & ~gb).sum().float()
    fn_b = (~pb & gb).sum().float()
    results['binary_iou'] = (tp_b / (tp_b + fp_b + fn_b + 1e-8)).item()
    results['binary_f1']  = (2*tp_b / (2*tp_b + fp_b + fn_b + 1e-8)).item()
    return results


def colorize(cls_map: np.ndarray) -> np.ndarray:
    """cls_map: (H, W) int -> (H, W, 3) uint8 color image."""
    h, w = cls_map.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for c, color in enumerate(PALETTE):
        out[cls_map == c] = color
    return out


@torch.no_grad()
def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    vis_dir = os.path.join(args.save_dir, 'vis')
    os.makedirs(vis_dir, exist_ok=True)

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_classes = getattr(args, 'num_classes', 3)
    use_hr = getattr(args, 'use_hr_branch', False)
    hr_dims = tuple(getattr(args, 'hr_dims', [64, 128, 256]))
    ms_layers = getattr(args, 'multiscale_layers', [9, 19, 29, 38])

    mk = dict(
        enable_mask_branch=True,
        mask_hidden_dim=args.mask_hidden_dim,
        mask_num_stages=args.mask_num_stages,
        mask_output_size=(args.mask_size, args.mask_size),
        freeze_for_mask_training=True,
        mask_training_mode=True,
        num_bi3_layers=getattr(args, 'num_bi3_layers', 3),
        num_heads=getattr(args, 'num_heads', 8),
        mlp_ratio=getattr(args, 'mlp_ratio', 4.0),
        num_classes=num_classes,
        use_hr_branch=use_hr,
        hr_dims=hr_dims,
    )
    if not use_hr:
        mk['multiscale_layers'] = ms_layers

    logging.info("Building model (use_hr_branch={})...".format(use_hr))
    try:
        model = Blip2VicunaMask.from_pretrained(args.pretrained, **mk)
    except Exception:
        model = Blip2VicunaMask(**mk)

    logging.info(f"Loading CD weights from {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt, strict=False)
    model = model.to(device).eval()

    logging.info("Building dataset...")
    dataset = ChangeMaskDataset(
        root=args.data_root,
        split=args.split,
        image_size=args.image_size,
        mask_size=args.mask_size,
        is_train=False,
        mask_root=args.mask_root,
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers,
        collate_fn=ChangeMaskDataset.collate_fn,
    )
    logging.info(f"{args.split} set: {len(dataset)} samples")

    all_preds, all_gts = [], []
    vis_count = 0

    for batch_idx, batch in enumerate(loader):
        img_A = batch['image_A'].to(device)
        img_B = batch['image_B'].to(device)
        gt_mask = batch['gt_mask'].to(device)
        names = batch['name']

        outputs = model.forward_mask(img_A, img_B, gt_mask)
        pred_cls = outputs['mask_cls']  # (B, H, W)

        all_preds.append(pred_cls.cpu())
        all_gts.append(gt_mask.cpu())

        # Save visualizations
        if vis_count < args.max_vis:
            for i in range(pred_cls.shape[0]):
                if vis_count >= args.max_vis:
                    break
                name = names[i].replace('.png', '').replace('.jpg', '')
                pred_np = pred_cls[i].cpu().numpy().astype(np.uint8)
                gt_np = gt_mask[i].cpu().numpy().astype(np.uint8)

                pred_color = Image.fromarray(colorize(pred_np))
                gt_color = Image.fromarray(colorize(gt_np))
                pred_color.save(os.path.join(vis_dir, f'{name}_pred.png'))
                gt_color.save(os.path.join(vis_dir, f'{name}_gt.png'))
                vis_count += 1

        if (batch_idx + 1) % 10 == 0:
            logging.info(f"Processed {batch_idx+1}/{len(loader)} batches")

    # Aggregate
    all_preds = torch.cat(all_preds, dim=0)
    all_gts = torch.cat(all_gts, dim=0)

    metrics = compute_metrics(all_preds, all_gts, num_classes)

    # Print results
    print("\n" + "="*60)
    print("  Multi-class Change Detection Results")
    print("="*60)
    for c in range(num_classes):
        n = CLASS_NAMES[c]
        print(f"  {n:>10s}  IoU={metrics[f'iou_{n}']:.4f}  F1={metrics[f'f1_{n}']:.4f}  "
              f"Prec={metrics[f'prec_{n}']:.4f}  Rec={metrics[f'rec_{n}']:.4f}")
    print("-"*60)
    print(f"  {'mIoU':>10s}  {metrics['mIoU']:.4f}  (road + building avg)")
    print(f"  {'mF1':>10s}  {metrics['mF1']:.4f}")
    print(f"  {'binary_IoU':>10s}  {metrics['binary_iou']:.4f}")
    print(f"  {'binary_F1':>10s}  {metrics['binary_f1']:.4f}")
    print("="*60)

    # Save metrics
    metrics_path = os.path.join(args.save_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logging.info(f"Metrics saved to {metrics_path}")
    logging.info(f"Visualizations saved to {vis_dir}/ ({vis_count} samples)")


if __name__ == "__main__":
    main()
