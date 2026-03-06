"""
Training script for Segformer-based Change Detection

This is a standalone training script that uses SegformerCD directly,
without the VLM components. This is the recommended approach for 
best change detection performance.

Usage:
    python train_segformer_cd.py --epochs 100 --batch_size 16
"""

import os
import sys
import argparse
import logging
import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from dataset_mask import ChangeMaskDataset
from model.mask_branch.segformer_cd import SegformerCD


def setup_logging(output_dir: str):
    """Setup logging to file and console."""
    os.makedirs(output_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(output_dir, 'train.log')),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


class DiceBCELoss(nn.Module):
    """Combined Dice + BCE loss for better segmentation."""
    def __init__(self, dice_weight: float = 0.5, bce_weight: float = 0.5):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.bce = nn.BCEWithLogitsLoss()
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # BCE
        bce_loss = self.bce(pred, target)
        
        # Dice
        pred_sigmoid = torch.sigmoid(pred)
        intersection = (pred_sigmoid * target).sum(dim=(2, 3))
        union = pred_sigmoid.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = (2 * intersection + 1) / (union + 1)
        dice_loss = 1 - dice.mean()
        
        return self.dice_weight * dice_loss + self.bce_weight * bce_loss


class FocalDiceLoss(nn.Module):
    """Focal + Dice loss for handling class imbalance."""
    def __init__(self, gamma: float = 2.0, alpha: float = 0.75):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Focal loss
        pred_sigmoid = torch.sigmoid(pred)
        pt = target * pred_sigmoid + (1 - target) * (1 - pred_sigmoid)
        focal_weight = (1 - pt) ** self.gamma
        alpha_t = target * self.alpha + (1 - target) * (1 - self.alpha)
        focal_loss = -alpha_t * focal_weight * torch.log(pt + 1e-8)
        focal_loss = focal_loss.mean()
        
        # Dice loss
        intersection = (pred_sigmoid * target).sum(dim=(2, 3))
        union = pred_sigmoid.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = (2 * intersection + 1) / (union + 1)
        dice_loss = 1 - dice.mean()
        
        return focal_loss + dice_loss


def compute_metrics(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5):
    """Compute IoU, F1, Precision, Recall."""
    pred_bin = (torch.sigmoid(pred) > threshold).float()
    target_bin = (target > 0.5).float()
    
    # Flatten
    pred_flat = pred_bin.view(-1)
    target_flat = target_bin.view(-1)
    
    tp = (pred_flat * target_flat).sum()
    fp = (pred_flat * (1 - target_flat)).sum()
    fn = ((1 - pred_flat) * target_flat).sum()
    tn = ((1 - pred_flat) * (1 - target_flat)).sum()
    
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


def train_epoch(model, loader, criterion, optimizer, scaler, device, epoch):
    model.train()
    total_loss = 0
    pbar = tqdm(loader, desc=f'Epoch {epoch+1} [Train]')
    
    for batch in pbar:
        image_a = batch['image_A'].to(device)
        image_b = batch['image_B'].to(device)
        gt_mask = batch['gt_mask'].to(device)
        
        optimizer.zero_grad()
        
        with autocast():
            outputs = model(image_a, image_b, output_size=gt_mask.shape[-2:])
            loss = criterion(outputs['mask_logits'], gt_mask)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_metrics = {'precision': 0, 'recall': 0, 'f1': 0, 'iou': 0, 'accuracy': 0}
    
    for batch in tqdm(loader, desc='Validating'):
        image_a = batch['image_A'].to(device)
        image_b = batch['image_B'].to(device)
        gt_mask = batch['gt_mask'].to(device)
        
        with autocast():
            outputs = model(image_a, image_b, output_size=gt_mask.shape[-2:])
            loss = criterion(outputs['mask_logits'], gt_mask)
        
        total_loss += loss.item()
        
        metrics = compute_metrics(outputs['mask_logits'], gt_mask)
        for k, v in metrics.items():
            all_metrics[k] += v
    
    n = len(loader)
    return total_loss / n, {k: v / n for k, v in all_metrics.items()}


def main():
    parser = argparse.ArgumentParser(description='Train SegformerCD')
    parser.add_argument('--data_root', type=str, default='/root/autodl-tmp/LEVIR-MCI-dataset/images')
    parser.add_argument('--output_dir', type=str, default='./output/segformer_cd')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=6e-5)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--pretrained', action='store_true', default=True)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--loss', type=str, default='focal_dice', choices=['bce', 'dice_bce', 'focal_dice'])
    args = parser.parse_args()
    
    # Setup
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(args.output_dir, timestamp)
    logger = setup_logging(output_dir)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')
    logger.info(f'Arguments: {args}')
    
    # Data
    train_dataset = ChangeMaskDataset(
        root=args.data_root,
        split='train',
        image_size=args.image_size,
        mask_size=args.image_size,
        is_train=True,
    )
    val_dataset = ChangeMaskDataset(
        root=args.data_root,
        split='val',
        image_size=args.image_size,
        mask_size=args.image_size,
        is_train=False,
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )
    
    logger.info(f'Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}')
    
    # Model
    model = SegformerCD(
        pretrained=args.pretrained,
        num_bi3_layers=3,
        num_heads=8,
        hidden_dim=256,
        num_classes=1,
    )
    model = model.to(device)
    
    # Loss
    if args.loss == 'bce':
        criterion = nn.BCEWithLogitsLoss()
    elif args.loss == 'dice_bce':
        criterion = DiceBCELoss()
    else:
        criterion = FocalDiceLoss()
    
    logger.info(f'Loss function: {args.loss}')
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Scheduler: Cosine with warmup
    warmup_epochs = 5
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / (args.epochs - warmup_epochs)
        return 0.5 * (1 + math.cos(math.pi * progress))
    
    import math
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    scaler = GradScaler()
    
    # Resume
    start_epoch = 0
    best_iou = 0
    if args.resume and os.path.exists(args.resume):
        logger.info(f'Resuming from {args.resume}')
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        best_iou = checkpoint.get('best_iou', 0)
    
    # Training loop
    logger.info('Starting training...')
    for epoch in range(start_epoch, args.epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch)
        val_loss, metrics = validate(model, val_loader, criterion, device)
        scheduler.step()
        
        lr = optimizer.param_groups[0]['lr']
        logger.info(f'Epoch [{epoch+1}/{args.epochs}] LR: {lr:.6f}')
        logger.info(f'  Train Loss: {train_loss:.4f}')
        logger.info(f'  Val Loss: {val_loss:.4f} IoU: {metrics["iou"]:.4f} F1: {metrics["f1"]:.4f}')
        
        # Save best
        if metrics['iou'] > best_iou:
            best_iou = metrics['iou']
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'best_iou': best_iou,
                'metrics': metrics,
            }, os.path.join(output_dir, 'best.pth'))
            logger.info(f'  *** New best IoU: {best_iou:.4f} ***')
        
        # Save periodic checkpoint
        if (epoch + 1) % 20 == 0:
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'best_iou': best_iou,
            }, os.path.join(output_dir, f'checkpoint_{epoch+1}.pth'))
    
    logger.info(f'Training complete! Best IoU: {best_iou:.4f}')
    logger.info(f'Model saved to: {output_dir}')


if __name__ == '__main__':
    main()

