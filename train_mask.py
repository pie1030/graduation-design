"""
Training Script for DeltaVLM CD Branch (Multi-class)

Trains the Change-Agent style CD module (with multi-scale ViT skip connections)
while freezing all other components (ViT, CSRM, Q-Former, LLM).

Supports 3-class semantic change detection: 0=background, 1=road, 2=building
Reports per-class IoU and mIoU (excluding background).

Usage:
    python train_mask.py --cfg_path ./train_mask_change_agent.yaml
    python train_mask.py --cfg_path ./train_mask_change_agent.yaml --warmstart ./output/binary_best.pth
"""

import argparse
import os
import sys
import time
import datetime
import random
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset_mask import build_mask_dataloaders
from model.blip2_vicua_mask import Blip2VicunaMask
from utils import init_distributed_mode, setup_logger, is_main_process


CLASS_NAMES = {0: 'bg', 1: 'road', 2: 'building'}


def parse_args():
    parser = argparse.ArgumentParser(description="Train DeltaVLM Mask Branch")

    # Data
    parser.add_argument("--data_root", type=str,
                       default="/root/autodl-tmp/LEVIR-MCI-dataset/images")
    parser.add_argument("--mask_root", type=str, default=None)

    # Model
    parser.add_argument("--pretrained", type=str,
                       default="./checkpoint_best.pth")
    parser.add_argument("--mask_hidden_dim", type=int, default=256)
    parser.add_argument("--mask_num_stages", type=int, default=4)
    parser.add_argument("--mask_size", type=int, default=256)
    parser.add_argument("--image_size", type=int, default=224)

    # Training
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_epochs", type=int, default=2)
    parser.add_argument("--min_lr", type=float, default=1e-6)

    # Misc
    parser.add_argument("--output_dir", type=str, default="./output/mask_branch")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--log_freq", type=int, default=50)
    parser.add_argument("--save_freq", type=int, default=5)
    parser.add_argument("--eval_freq", type=int, default=1)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--warmstart", type=str, default=None,
                       help="Warm-start from binary checkpoint (loads shared weights, skips head)")
    parser.add_argument("--reset_lr_schedule", action="store_true")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="DeltaVLM-Mask")

    parser.add_argument("--cfg_path", type=str, default=None)

    args, unknown = parser.parse_known_args()

    import sys
    explicit_args = set()
    for arg in sys.argv[1:]:
        if arg.startswith('--'):
            explicit_args.add(arg[2:].split('=')[0])

    if args.cfg_path and os.path.exists(args.cfg_path):
        import yaml
        with open(args.cfg_path, 'r') as f:
            cfg = yaml.safe_load(f)
        for key, value in cfg.items():
            if key not in explicit_args:
                setattr(args, key, value)
        if hasattr(args, 'lr'):
            args.lr = float(args.lr)
        if hasattr(args, 'min_lr'):
            args.min_lr = float(args.min_lr)

    return args


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


def cosine_lr_schedule(optimizer, epoch, max_epoch, init_lr, min_lr, warmup_epochs):
    if epoch < warmup_epochs:
        lr = min_lr + (init_lr - min_lr) * (epoch + 1) / warmup_epochs
    else:
        lr = min_lr + (init_lr - min_lr) * 0.5 * (
            1.0 + np.cos(np.pi * (epoch - warmup_epochs) / (max_epoch - warmup_epochs))
        )
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


class ConfusionMatrixEvaluator:
    """
    Global confusion matrix evaluator (matching Change-Agent).

    Accumulates predictions across ALL batches, then computes metrics once.
    This gives correct IoU (not the biased per-batch average).
    """

    def __init__(self, num_classes=3):
        self.num_classes = num_classes
        self.confusion_matrix = np.zeros((num_classes, num_classes))

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))

    def add_batch(self, gt, pred):
        """gt, pred: numpy arrays of class indices."""
        gt = gt.flatten().astype(int)
        pred = pred.flatten().astype(int)
        mask = (gt >= 0) & (gt < self.num_classes)
        label = self.num_classes * gt[mask] + pred[mask]
        count = np.bincount(label, minlength=self.num_classes ** 2)
        self.confusion_matrix += count.reshape(self.num_classes, self.num_classes)

    def compute_metrics(self):
        cm = self.confusion_matrix
        metrics = {}

        iou_per_class = np.diag(cm) / (
            cm.sum(axis=1) + cm.sum(axis=0) - np.diag(cm) + 1e-8
        )

        for c in range(self.num_classes):
            name = CLASS_NAMES.get(c, str(c))
            metrics[f'iou_{name}'] = iou_per_class[c]
            tp = cm[c, c]
            fp = cm[:, c].sum() - tp
            fn = cm[c, :].sum() - tp
            metrics[f'prec_{name}'] = tp / (tp + fp + 1e-8)
            metrics[f'rec_{name}'] = tp / (tp + fn + 1e-8)
            metrics[f'f1_{name}'] = 2 * tp / (2 * tp + fp + fn + 1e-8)

        metrics['mIoU_3class'] = float(np.nanmean(iou_per_class))
        change_iou = iou_per_class[1:]
        metrics['mIoU_change'] = float(np.nanmean(change_iou))
        metrics['mIoU'] = metrics['mIoU_3class']

        metrics['OA'] = float(np.diag(cm).sum() / (cm.sum() + 1e-8))

        # MPA (mean pixel accuracy per class)
        acc_cls = np.diag(cm) / (cm.sum(axis=1) + 1e-8)
        metrics['MPA'] = float(np.nanmean(acc_cls))

        # FWIoU
        freq = cm.sum(axis=1) / (cm.sum() + 1e-8)
        metrics['FWIoU'] = float((freq * iou_per_class).sum())

        # Binary change IoU
        tp_b = cm[1:, 1:].sum()
        fp_b = cm[0, 1:].sum()
        fn_b = cm[1:, 0].sum()
        metrics['binary_iou'] = float(tp_b / (tp_b + fp_b + fn_b + 1e-8))

        return metrics


@torch.no_grad()
def evaluate(model, val_loader, device, num_classes=3):
    model.eval()
    evaluator = ConfusionMatrixEvaluator(num_classes)
    total_loss = 0
    num_batches = 0

    for batch in val_loader:
        image_A = batch['image_A'].to(device)
        image_B = batch['image_B'].to(device)
        gt_mask = batch['gt_mask'].to(device)

        outputs = model.forward_mask(image_A, image_B, gt_mask)
        total_loss += outputs['loss'].item()

        pred_np = outputs['mask_cls'].cpu().numpy()
        gt_np = gt_mask.cpu().numpy()
        evaluator.add_batch(gt_np, pred_np)
        num_batches += 1

    avg_loss = total_loss / num_batches
    metrics = evaluator.compute_metrics()
    model.train()
    return avg_loss, metrics


def train_one_epoch(model, train_loader, optimizer, device, epoch, args, writer=None):
    model.train()
    total_loss = 0
    num_batches = len(train_loader)

    for batch_idx, batch in enumerate(train_loader):
        image_A = batch['image_A'].to(device)
        image_B = batch['image_B'].to(device)
        gt_mask = batch['gt_mask'].to(device)

        outputs = model.forward_mask(image_A, image_B, gt_mask)
        loss = outputs['loss']

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

        if (batch_idx + 1) % args.log_freq == 0:
            lr = optimizer.param_groups[0]['lr']
            logging.info(
                f"Epoch [{epoch+1}/{args.epochs}] "
                f"Iter [{batch_idx+1}/{num_batches}] "
                f"Loss: {loss.item():.4f} "
                f"LR: {lr:.6f}"
            )
            if writer is not None:
                global_step = epoch * num_batches + batch_idx
                writer.add_scalar('train/loss', loss.item(), global_step)
                writer.add_scalar('train/lr', lr, global_step)

    return total_loss / num_batches


def main():
    args = parse_args()

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    args.output_dir = os.path.join(args.output_dir, timestamp)
    os.makedirs(args.output_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(args.output_dir, 'train.log')),
            logging.StreamHandler()
        ]
    )

    logging.info(f"Arguments: {args}")
    logging.info(f"Output directory: {args.output_dir}")

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, 'tensorboard'))

    if args.use_wandb:
        import wandb
        wandb.init(project=args.wandb_project, config=vars(args), dir=args.output_dir)

    # Build dataloaders
    logging.info("Building dataloaders...")
    filter_nc = getattr(args, 'filter_no_change', False)
    train_loader, val_loader = build_mask_dataloaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        mask_size=args.mask_size,
        mask_root=args.mask_root,
        filter_no_change=filter_nc,
    )
    if filter_nc:
        logging.info("Balanced sampling enabled: all-background samples filtered")
    logging.info(f"Train samples: {len(train_loader.dataset)}")
    logging.info(f"Val samples: {len(val_loader.dataset)}")

    # Build model
    logging.info("Building model...")
    num_classes = getattr(args, 'num_classes', 3)
    decoder_type = getattr(args, 'mask_decoder_type', 'change_agent')
    use_hr = getattr(args, 'use_hr_branch', False)
    hr_dims = tuple(getattr(args, 'hr_dims', [64, 128, 256]))
    ms_layers = getattr(args, 'multiscale_layers', [9, 19, 29, 38])
    cd_kwargs = dict(
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
        mask_decoder_type=decoder_type,
        use_hr_branch=use_hr,
        hr_dims=hr_dims,
        # DeltaCD-specific
        sem_dim=getattr(args, 'sem_dim', 256),
        hr_mid_dim=getattr(args, 'hr_mid_dim', 48),
        hr_out_dim=getattr(args, 'hr_out_dim', 96),
        fused_dim=getattr(args, 'fused_dim', 256),
        fusion_size=getattr(args, 'fusion_size', 56),
    )
    if decoder_type != 'delta_cd' and not use_hr:
        cd_kwargs['multiscale_layers'] = ms_layers
    try:
        model = Blip2VicunaMask.from_pretrained(args.pretrained, **cd_kwargs)
        logging.info("Loaded pretrained DeltaVLM checkpoint")
    except Exception as e:
        logging.warning(f"Cannot load pretrained: {e}")
        logging.info("Using base EVA-ViT-G; CD branch trains from scratch")
        model = Blip2VicunaMask(**cd_kwargs)

    # Warm-start from binary checkpoint (load shared weights, skip mismatched head)
    if args.warmstart and os.path.exists(args.warmstart):
        logging.info(f"Warm-starting from binary checkpoint: {args.warmstart}")
        binary_sd = torch.load(args.warmstart, map_location='cpu', weights_only=False)
        loaded, skipped = 0, 0
        model_sd = model.state_dict()
        for k, v in binary_sd.items():
            if k in model_sd and model_sd[k].shape == v.shape:
                model_sd[k] = v
                loaded += 1
            else:
                skipped += 1
                if k in model_sd:
                    logging.info(f"  Skip (shape mismatch): {k}  binary={list(v.shape)} vs model={list(model_sd[k].shape)}")
        model.load_state_dict(model_sd, strict=False)
        logging.info(f"Warm-start: loaded {loaded} params, skipped {skipped} (shape mismatch = new head)")

    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Total parameters: {total_params:,}")
    logging.info(f"Trainable parameters: {trainable_params:,}")
    logging.info(f"num_classes: {num_classes}")

    # Optimizer
    mask_params = [p for n, p in model.named_parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(mask_params, lr=args.lr, weight_decay=args.weight_decay)

    # Resume from checkpoint (full resume with optimizer state)
    start_epoch = 0
    best_miou = 0
    if args.resume and os.path.exists(args.resume):
        logging.info(f"Resuming from {args.resume}")
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
        if 'mask_decoder' in checkpoint:
            model.load_state_dict(checkpoint['mask_decoder'], strict=False)
            if 'optimizer' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint.get('epoch', 0) + 1
            best_miou = checkpoint.get('best_iou', 0)
        else:
            model.load_state_dict(checkpoint, strict=False)
        logging.info(f"Resuming from epoch {start_epoch}")

    # Training loop
    logging.info("Starting training...")
    for epoch in range(start_epoch, args.epochs):
        lr = cosine_lr_schedule(
            optimizer, epoch - start_epoch, args.epochs - start_epoch,
            args.lr, args.min_lr, args.warmup_epochs
        )

        train_loss = train_one_epoch(
            model, train_loader, optimizer, device, epoch, args, writer
        )
        logging.info(f"Epoch [{epoch+1}/{args.epochs}] Train Loss: {train_loss:.4f}")

        if (epoch + 1) % args.eval_freq == 0:
            val_loss, val_metrics = evaluate(model, val_loader, device, num_classes)
            logging.info(
                f"Epoch [{epoch+1}/{args.epochs}] Val Loss: {val_loss:.4f} "
                f"mIoU(change): {val_metrics['mIoU_change']:.4f} "
                f"mIoU(3cls): {val_metrics['mIoU_3class']:.4f} "
                f"mF1: {val_metrics.get('mF1_change',0):.4f} "
                f"OA: {val_metrics.get('OA',0):.4f}"
            )
            logging.info(
                f"  road  IoU={val_metrics.get('iou_road',0):.4f} "
                f"F1={val_metrics.get('f1_road',0):.4f} "
                f"P={val_metrics.get('prec_road',0):.4f} "
                f"R={val_metrics.get('rec_road',0):.4f}"
            )
            logging.info(
                f"  bldg  IoU={val_metrics.get('iou_building',0):.4f} "
                f"F1={val_metrics.get('f1_building',0):.4f} "
                f"P={val_metrics.get('prec_building',0):.4f} "
                f"R={val_metrics.get('rec_building',0):.4f}"
            )
            logging.info(
                f"  bg    IoU={val_metrics.get('iou_bg',0):.4f} "
                f"binary_iou={val_metrics.get('binary_iou',0):.4f}"
            )

            writer.add_scalar('val/loss', val_loss, epoch)
            for k, v in val_metrics.items():
                writer.add_scalar(f'val/{k}', v, epoch)

            if args.use_wandb:
                wandb.log({
                    'epoch': epoch + 1, 'train_loss': train_loss, 'val_loss': val_loss,
                    **{f'val_{k}': v for k, v in val_metrics.items()},
                })

            cur_miou = val_metrics['mIoU']  # 3-class mIoU (Change-Agent protocol)
            if cur_miou > best_miou:
                best_miou = cur_miou
                model.save_mask_branch(os.path.join(args.output_dir, 'mask_branch_best.pth'))
                logging.info(f"New best mIoU: {best_miou:.4f}")

        if (epoch + 1) % args.save_freq == 0:
            mask_state_dict = {k: v for k, v in model.state_dict().items() if 'mask_decoder' in k}
            ckpt = {
                'mask_decoder': mask_state_dict,
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'best_iou': best_miou,
                'args': args,
            }
            torch.save(ckpt, os.path.join(args.output_dir, f'checkpoint_{epoch+1}.pth'))

    model.save_mask_branch(os.path.join(args.output_dir, 'mask_branch_final.pth'))

    val_loss, val_metrics = evaluate(model, val_loader, device, num_classes)
    logging.info(f"Final Val Loss: {val_loss:.4f}")
    logging.info(f"Final mIoU(change): {val_metrics['mIoU_change']:.4f}")
    logging.info(f"Final mIoU(3class): {val_metrics['mIoU_3class']:.4f}")
    logging.info(f"Final OA: {val_metrics.get('OA', 0):.4f}")
    for k, v in sorted(val_metrics.items()):
        logging.info(f"  {k}: {v:.4f}")

    writer.close()
    if args.use_wandb:
        wandb.finish()

    logging.info("Training complete!")


if __name__ == "__main__":
    main()
