"""
Mask Head for Binary Change Mask Prediction

Simple yet effective head for predicting binary change masks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class ConvBNReLU(nn.Module):
    """Conv-BN-ReLU block."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            padding=kernel_size // 2, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class MaskHead(nn.Module):
    """
    Binary mask prediction head.
    
    Takes upsampled features and predicts binary change mask.
    """
    def __init__(
        self,
        in_channels: int = 128,       # From AnyUpStyleUpsampler output
        hidden_channels: int = 64,
        num_classes: int = 1,          # Binary mask
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # Feature refinement
        self.refine = nn.Sequential(
            ConvBNReLU(in_channels, hidden_channels, 3),
            nn.Dropout2d(dropout),
            ConvBNReLU(hidden_channels, hidden_channels, 3),
        )
        
        # Multi-scale context aggregation (ASPP-lite)
        self.aspp = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(hidden_channels, hidden_channels // 4, 1, bias=False),
                nn.BatchNorm2d(hidden_channels // 4),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(hidden_channels, hidden_channels // 4, 3, padding=3, dilation=3, bias=False),
                nn.BatchNorm2d(hidden_channels // 4),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(hidden_channels, hidden_channels // 4, 3, padding=6, dilation=6, bias=False),
                nn.BatchNorm2d(hidden_channels // 4),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(hidden_channels, hidden_channels // 4, 1, bias=False),
                nn.BatchNorm2d(hidden_channels // 4),
                nn.ReLU(inplace=True),
            ),
        ])
        
        # Fusion
        self.fuse = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, 1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
        )
        
        # Final prediction
        self.head = nn.Conv2d(hidden_channels, num_classes, 1)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, in_channels, H, W) upsampled features
        Returns:
            (B, 1, H, W) binary mask logits (before sigmoid)
        """
        # Refine features
        x = self.refine(x)
        
        # Multi-scale context
        aspp_outs = []
        for aspp_module in self.aspp[:-1]:
            aspp_outs.append(aspp_module(x))
        
        # Global context
        global_ctx = self.aspp[-1](x)
        global_ctx = F.interpolate(global_ctx, size=x.shape[-2:], mode='bilinear', align_corners=False)
        aspp_outs.append(global_ctx)
        
        # Concatenate and fuse
        x = torch.cat(aspp_outs, dim=1)
        x = self.fuse(x)
        
        # Predict mask
        mask_logits = self.head(x)
        
        return mask_logits


class DiceBCELoss(nn.Module):
    """Combined Dice Loss and BCE Loss for binary segmentation."""
    def __init__(self, dice_weight: float = 0.5, bce_weight: float = 0.5, smooth: float = 1.0):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (B, 1, H, W) predicted logits
            target: (B, 1, H, W) ground truth mask (0 or 1)
        Returns:
            Combined loss scalar
        """
        # BCE Loss
        bce_loss = self.bce(pred, target)
        
        # Dice Loss
        pred_sigmoid = torch.sigmoid(pred)
        pred_flat = pred_sigmoid.view(-1)
        target_flat = target.view(-1)
        
        intersection = (pred_flat * target_flat).sum()
        dice = (2. * intersection + self.smooth) / (pred_flat.sum() + target_flat.sum() + self.smooth)
        dice_loss = 1 - dice
        
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


class FocalDiceLoss(nn.Module):
    """Focal Loss + Dice Loss for handling class imbalance."""
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        dice_weight: float = 0.5,
        focal_weight: float = 0.5,
        smooth: float = 1.0,
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.smooth = smooth
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (B, 1, H, W) predicted logits
            target: (B, 1, H, W) ground truth mask
        """
        # Focal Loss
        pred_sigmoid = torch.sigmoid(pred)
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        pt = torch.where(target == 1, pred_sigmoid, 1 - pred_sigmoid)
        focal_weight = (self.alpha * target + (1 - self.alpha) * (1 - target)) * ((1 - pt) ** self.gamma)
        focal_loss = (focal_weight * bce).mean()
        
        # Dice Loss
        pred_flat = pred_sigmoid.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        dice = (2. * intersection + self.smooth) / (pred_flat.sum() + target_flat.sum() + self.smooth)
        dice_loss = 1 - dice
        
        return self.focal_weight * focal_loss + self.dice_weight * dice_loss

