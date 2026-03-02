"""
Training Script for DeltaVLM CD Branch

Trains the Change-Agent style CD module (with multi-scale ViT skip connections)
while freezing all other components (ViT, CSRM, Q-Former, LLM).

Usage:
    python train_mask.py --cfg_path ./train_mask_change_agent.yaml

Or with command line overrides:
    python train_mask.py --data_root /path/to/LEVIR-MCI --pretrained ./checkpoint_best.pth
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

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset_mask import build_mask_dataloaders
from model.blip2_vicua_mask import Blip2VicunaMask
from utils import init_distributed_mode, setup_logger, is_main_process


def parse_args():
    parser = argparse.ArgumentParser(description="Train DeltaVLM Mask Branch")
    
    # Data
    parser.add_argument("--data_root", type=str, 
                       default="/root/autodl-tmp/LEVIR-MCI-dataset/images",
                       help="Path to dataset images directory (LEVIR-MCI or ChangeChat)")
    parser.add_argument("--mask_root", type=str, default=None,
                       help="Path to mask labels (if different from data_root)")
    
    # Model
    parser.add_argument("--pretrained", type=str,
                       default="./checkpoint_best.pth",
                       help="Path to pretrained DeltaVLM checkpoint")
    parser.add_argument("--mask_hidden_dim", type=int, default=256,
                       help="Hidden dimension for mask branch")
    parser.add_argument("--mask_num_stages", type=int, default=4,
                       help="Number of upsampling stages (2^n upscale)")
    parser.add_argument("--mask_size", type=int, default=256,
                       help="Output mask size")
    parser.add_argument("--image_size", type=int, default=224,
                       help="Input image size")
    
    # Training
    parser.add_argument("--epochs", type=int, default=50,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Batch size per GPU")
    parser.add_argument("--lr", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                       help="Weight decay")
    parser.add_argument("--warmup_epochs", type=int, default=2,
                       help="Number of warmup epochs")
    parser.add_argument("--min_lr", type=float, default=1e-6,
                       help="Minimum learning rate")
    
    # Misc
    parser.add_argument("--output_dir", type=str, default="./output/mask_branch",
                       help="Output directory")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--num_workers", type=int, default=4,
                       help="Number of data loading workers")
    parser.add_argument("--log_freq", type=int, default=50,
                       help="Logging frequency (iterations)")
    parser.add_argument("--save_freq", type=int, default=5,
                       help="Checkpoint saving frequency (epochs)")
    parser.add_argument("--eval_freq", type=int, default=1,
                       help="Evaluation frequency (epochs)")
    parser.add_argument("--resume", type=str, default=None,
                       help="Resume from checkpoint")
    parser.add_argument("--reset_lr_schedule", action="store_true",
                       help="Reset LR schedule when resuming (warm restart)")
    parser.add_argument("--use_wandb", action="store_true",
                       help="Use Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="DeltaVLM-Mask",
                       help="W&B project name")
    
    # Config file (optional, overrides defaults)
    parser.add_argument("--cfg_path", type=str, default=None,
                       help="Path to YAML config file")
    
    # First parse to get cfg_path and any explicitly set args
    args, unknown = parser.parse_known_args()
    
    # Track which args were explicitly set on command line
    import sys
    explicit_args = set()
    for arg in sys.argv[1:]:
        if arg.startswith('--'):
            explicit_args.add(arg[2:].split('=')[0])
    
    # Load config file if provided (YAML provides defaults, CLI overrides)
    if args.cfg_path and os.path.exists(args.cfg_path):
        import yaml
        with open(args.cfg_path, 'r') as f:
            cfg = yaml.safe_load(f)
        for key, value in cfg.items():
            # Only use YAML value if not explicitly set on command line
            if hasattr(args, key) and key not in explicit_args:
                setattr(args, key, value)
        
        # Ensure numeric types for learning rates (YAML may parse 1e-5 as string)
        if hasattr(args, 'lr'):
            args.lr = float(args.lr)
        if hasattr(args, 'min_lr'):
            args.min_lr = float(args.min_lr)
    
    return args


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


def cosine_lr_schedule(optimizer, epoch, max_epoch, init_lr, min_lr, warmup_epochs):
    """Cosine learning rate schedule with warmup."""
    if epoch < warmup_epochs:
        # Ensure lr starts at min_lr and ramps up to init_lr
        lr = min_lr + (init_lr - min_lr) * (epoch + 1) / warmup_epochs
    else:
        lr = min_lr + (init_lr - min_lr) * 0.5 * (
            1.0 + np.cos(np.pi * (epoch - warmup_epochs) / (max_epoch - warmup_epochs))
        )
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


def compute_metrics(pred_mask, gt_mask, threshold=0.5):
    """Compute evaluation metrics for binary segmentation."""
    pred_binary = (pred_mask > threshold).float()
    gt_binary = (gt_mask > 0.5).float()
    
    # Flatten
    pred_flat = pred_binary.view(-1)
    gt_flat = gt_binary.view(-1)
    
    # TP, FP, FN, TN
    tp = (pred_flat * gt_flat).sum()
    fp = (pred_flat * (1 - gt_flat)).sum()
    fn = ((1 - pred_flat) * gt_flat).sum()
    tn = ((1 - pred_flat) * (1 - gt_flat)).sum()
    
    # Metrics
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    
    return {
        'precision': precision.item(),
        'recall': recall.item(),
        'f1': f1.item(),
        'iou': iou.item(),
        'accuracy': accuracy.item(),
    }


@torch.no_grad()
def evaluate(model, val_loader, device):
    """Evaluate model on validation set."""
    model.eval()
    
    total_loss = 0
    total_metrics = {'precision': 0, 'recall': 0, 'f1': 0, 'iou': 0, 'accuracy': 0}
    num_batches = 0
    
    for batch in val_loader:
        image_A = batch['image_A'].to(device)
        image_B = batch['image_B'].to(device)
        gt_mask = batch['gt_mask'].to(device)
        
        # Forward
        outputs = model.forward_mask(image_A, image_B, gt_mask)
        
        # Loss
        total_loss += outputs['loss'].item()
        
        # Metrics
        batch_metrics = compute_metrics(outputs['mask_pred'], gt_mask)
        for k, v in batch_metrics.items():
            total_metrics[k] += v
        
        num_batches += 1
    
    # Average
    avg_loss = total_loss / num_batches
    avg_metrics = {k: v / num_batches for k, v in total_metrics.items()}
    
    model.train()
    return avg_loss, avg_metrics


def train_one_epoch(model, train_loader, optimizer, device, epoch, args, writer=None):
    """Train for one epoch."""
    model.train()
    
    total_loss = 0
    num_batches = len(train_loader)
    
    for batch_idx, batch in enumerate(train_loader):
        image_A = batch['image_A'].to(device)
        image_B = batch['image_B'].to(device)
        gt_mask = batch['gt_mask'].to(device)
        
        # Forward
        outputs = model.forward_mask(image_A, image_B, gt_mask)
        loss = outputs['loss']
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        
        # Logging
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
    
    # Create output directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    args.output_dir = os.path.join(args.output_dir, timestamp)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup logging
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
    
    # Set seed
    set_seed(args.seed)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    # Tensorboard
    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, 'tensorboard'))
    
    # W&B (optional)
    if args.use_wandb:
        import wandb
        wandb.init(
            project=args.wandb_project,
            config=vars(args),
            dir=args.output_dir,
        )
    
    # Build dataloaders
    logging.info("Building dataloaders...")
    train_loader, val_loader = build_mask_dataloaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        mask_size=args.mask_size,
        mask_root=args.mask_root,
    )
    logging.info(f"Train samples: {len(train_loader.dataset)}")
    logging.info(f"Val samples: {len(val_loader.dataset)}")
    
    # Build model
    logging.info("Building model...")
    model = Blip2VicunaMask.from_pretrained(
        args.pretrained,
        enable_mask_branch=True,
        mask_hidden_dim=args.mask_hidden_dim,
        mask_num_stages=args.mask_num_stages,
        mask_output_size=(args.mask_size, args.mask_size),
        freeze_for_mask_training=True,
        mask_training_mode=True,  # Delete LLM to save GPU memory
    )
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Total parameters: {total_params:,}")
    logging.info(f"Trainable parameters: {trainable_params:,}")
    
    # Optimizer (only mask branch parameters)
    mask_params = [p for n, p in model.named_parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        mask_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    
    # Resume from checkpoint
    start_epoch = 0
    best_iou = 0
    lr_epoch_offset = 0  # For warm restart: offset epoch in lr schedule
    if args.resume and os.path.exists(args.resume):
        logging.info(f"Resuming from {args.resume}")
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
        
        # Handle different checkpoint formats
        if 'model' in checkpoint:
            # Full checkpoint format
            model.load_state_dict(checkpoint['model'], strict=False)
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch'] + 1
            best_iou = checkpoint.get('best_iou', 0)
        elif 'mask_decoder' in checkpoint:
            # New lightweight checkpoint format (mask_decoder only)
            model.load_state_dict(checkpoint['mask_decoder'], strict=False)
            if 'optimizer' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint.get('epoch', 0) + 1
            best_iou = checkpoint.get('best_iou', 0)
        else:
            # mask_branch_best.pth format (direct mask_decoder state_dict)
            model.load_state_dict(checkpoint, strict=False)
            logging.info("Loaded mask_decoder weights only (no optimizer/epoch info)")
        
        if args.reset_lr_schedule or start_epoch == 0:
            # Warm restart: reset lr schedule but keep model weights
            lr_epoch_offset = start_epoch
            if start_epoch > 0:
                logging.info(f"Warm restart: LR schedule reset, training from epoch {start_epoch} to {args.epochs}")
                logging.info(f"Effective LR schedule: epoch 0 to {args.epochs - start_epoch}")
        else:
            logging.info(f"Resuming training from epoch {start_epoch}")
    
    # Training loop
    logging.info("Starting training...")
    for epoch in range(start_epoch, args.epochs):
        # Update learning rate (with optional offset for warm restart)
        effective_epoch = epoch - lr_epoch_offset
        effective_max_epoch = args.epochs - lr_epoch_offset
        lr = cosine_lr_schedule(
            optimizer, effective_epoch, effective_max_epoch,
            args.lr, args.min_lr, args.warmup_epochs
        )
        
        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, device, epoch, args, writer
        )
        logging.info(f"Epoch [{epoch+1}/{args.epochs}] Train Loss: {train_loss:.4f}")
        
        # Evaluate
        if (epoch + 1) % args.eval_freq == 0:
            val_loss, val_metrics = evaluate(model, val_loader, device)
            logging.info(
                f"Epoch [{epoch+1}/{args.epochs}] Val Loss: {val_loss:.4f} "
                f"IoU: {val_metrics['iou']:.4f} F1: {val_metrics['f1']:.4f}"
            )
            
            # Log to tensorboard
            writer.add_scalar('val/loss', val_loss, epoch)
            for k, v in val_metrics.items():
                writer.add_scalar(f'val/{k}', v, epoch)
            
            # Log to W&B
            if args.use_wandb:
                wandb.log({
                    'epoch': epoch + 1,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    **{f'val_{k}': v for k, v in val_metrics.items()},
                })
            
            # Save best model
            if val_metrics['iou'] > best_iou:
                best_iou = val_metrics['iou']
                model.save_mask_branch(os.path.join(args.output_dir, 'mask_branch_best.pth'))
                logging.info(f"New best IoU: {best_iou:.4f}")
        
        # Save checkpoint (only mask_decoder weights to save disk space)
        if (epoch + 1) % args.save_freq == 0:
            # Only save mask_decoder weights (~190MB) instead of full model (~4.3GB)
            mask_state_dict = {k: v for k, v in model.state_dict().items() if 'mask_decoder' in k}
            checkpoint = {
                'mask_decoder': mask_state_dict,
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'best_iou': best_iou,
                'args': args,
            }
            torch.save(checkpoint, os.path.join(args.output_dir, f'checkpoint_{epoch+1}.pth'))
    
    # Save final model
    model.save_mask_branch(os.path.join(args.output_dir, 'mask_branch_final.pth'))
    
    # Final evaluation
    val_loss, val_metrics = evaluate(model, val_loader, device)
    logging.info(f"Final Val Loss: {val_loss:.4f}")
    logging.info(f"Final Metrics: {val_metrics}")
    
    writer.close()
    if args.use_wandb:
        wandb.finish()
    
    logging.info("Training complete!")


if __name__ == "__main__":
    main()

