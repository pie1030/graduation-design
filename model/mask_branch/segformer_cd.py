"""
Segformer-based Change Detection Module

Uses Siamese Segformer-B1 backbone (same as Change-Agent) for proper multi-scale
feature extraction, instead of trying to adapt EVA-ViT features.

This is the recommended approach for change detection as it matches the
original Change-Agent paper's design.

Key advantages:
- Multi-scale features: 1/4, 1/8, 1/16, 1/32 of input resolution
- Designed for dense prediction tasks
- Pre-trained on ImageNet-1K
- Small parameter count (~14M for B1)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from einops import rearrange

# Import Segformer from copied file
from .segformer import mit_b1


class LPE(nn.Module):
    """Local Perceptual Enhancement - exactly as in Change-Agent."""
    def __init__(self, dim: int):
        super().__init__()
        self.conv3x3 = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
        self.conv5x1 = nn.Conv2d(dim, dim, (5, 1), 1, (2, 0), groups=dim)
        self.conv1x5 = nn.Conv2d(dim, dim, (1, 5), 1, (0, 2), groups=dim)
        self.norm = nn.GroupNorm(1, dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        out = self.conv3x3(x) + self.conv5x1(x) + self.conv1x5(x)
        return self.norm(out) + x


class GDFA(nn.Module):
    """Global Difference-guided Feature Attention - exactly as in Change-Agent."""
    def __init__(self, dim: int, num_heads: int = 8, mlp_ratio: float = 4.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        # Q, K, V projections
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)
        
        # Difference feature projection
        self.diff_proj = nn.Linear(dim, dim)
        
        # FFN
        hidden_dim = int(dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
        
    def forward(self, x: torch.Tensor, diff: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, C) feature sequence
            diff: (B, N, C) difference feature
        """
        B, N, C = x.shape
        
        # Normalize
        x_norm = self.norm1(x)
        diff_norm = self.norm1(diff)
        
        # Compute Q, K, V
        q = self.q(x_norm)
        k = self.k(x_norm)
        v = self.v(x_norm)
        
        # Reshape for multi-head attention
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.num_heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.num_heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.num_heads)
        
        # Attention with difference modulation
        diff_weight = self.diff_proj(diff_norm).sigmoid()  # (B, N, C)
        diff_weight = rearrange(diff_weight, 'b n (h d) -> b h n d', h=self.num_heads)
        
        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        # Apply attention with difference weighting
        out = attn @ (v * diff_weight)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.proj(out)
        
        # Residual + FFN
        x = x + out
        x = x + self.ffn(self.norm2(x))
        
        return x


class BI3Block(nn.Module):
    """Bidirectional Iterative Interaction Block."""
    def __init__(self, dim: int, num_heads: int = 8, mlp_ratio: float = 4.0):
        super().__init__()
        self.lpe = LPE(dim)
        self.gdfa = GDFA(dim, num_heads, mlp_ratio)
        
    def forward(self, feat_a: torch.Tensor, feat_b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            feat_a: (B, C, H, W) features from time A
            feat_b: (B, C, H, W) features from time B
        Returns:
            Enhanced features for both times
        """
        B, C, H, W = feat_a.shape
        
        # LPE
        feat_a = self.lpe(feat_a)
        feat_b = self.lpe(feat_b)
        
        # Compute difference
        diff = torch.abs(feat_a - feat_b)
        
        # Flatten for attention
        feat_a_flat = rearrange(feat_a, 'b c h w -> b (h w) c')
        feat_b_flat = rearrange(feat_b, 'b c h w -> b (h w) c')
        diff_flat = rearrange(diff, 'b c h w -> b (h w) c')
        
        # GDFA
        feat_a_flat = self.gdfa(feat_a_flat, diff_flat)
        feat_b_flat = self.gdfa(feat_b_flat, diff_flat)
        
        # Reshape back
        feat_a = rearrange(feat_a_flat, 'b (h w) c -> b c h w', h=H, w=W)
        feat_b = rearrange(feat_b_flat, 'b (h w) c -> b c h w', h=H, w=W)
        
        return feat_a, feat_b


class CBF(nn.Module):
    """Change Bi-temporal Fusion module."""
    def __init__(self, in_channels: List[int], out_channels: int = 256):
        super().__init__()
        self.out_channels = out_channels
        
        # Per-scale projection
        self.projs = nn.ModuleList([
            nn.Conv2d(c, out_channels, 1) for c in in_channels
        ])
        
        # Fusion: concat(A, B, |A-B|, A*B) -> 4 * out_channels
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels * 4, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        
    def forward(self, feats_a: List[torch.Tensor], feats_b: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Fuse bi-temporal features at each scale.
        
        Args:
            feats_a: List of features from time A at different scales
            feats_b: List of features from time B at different scales
        Returns:
            List of fused change features
        """
        fused = []
        for proj, fa, fb in zip(self.projs, feats_a, feats_b):
            fa = proj(fa)
            fb = proj(fb)
            
            # Fusion: concat, diff, product
            diff = torch.abs(fa - fb)
            prod = fa * fb
            
            x = torch.cat([fa, fb, diff, prod], dim=1)
            x = self.fusion(x)
            fused.append(x)
            
        return fused


class SegformerCDDecoder(nn.Module):
    """Decoder for change detection using multi-scale features."""
    def __init__(self, in_channels: int = 256, num_classes: int = 1):
        super().__init__()
        
        # Progressive upsampling with skip connections
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels, 4, 2, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )
        self.fuse3 = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 3, 1, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )
        
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels, 4, 2, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )
        self.fuse2 = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 3, 1, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )
        
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels // 2, 4, 2, 1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
        )
        self.fuse1 = nn.Sequential(
            nn.Conv2d(in_channels + in_channels // 2, in_channels // 2, 3, 1, 1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
        )
        
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels // 2, in_channels // 4, 4, 2, 1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
        )
        
        # Final prediction head
        self.head = nn.Sequential(
            nn.Conv2d(in_channels // 4, in_channels // 4, 3, 1, 1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, num_classes, 1),
        )
        
    def forward(self, features: List[torch.Tensor], output_size: Tuple[int, int] = None) -> torch.Tensor:
        """
        Args:
            features: [s1, s2, s3, s4] multi-scale features from CBF
                     s1: 1/4, s2: 1/8, s3: 1/16, s4: 1/32
        """
        s1, s2, s3, s4 = features
        
        # Bottom-up path
        x = self.up4(s4)  # 1/32 -> 1/16
        x = self.fuse3(torch.cat([x, s3], dim=1))
        
        x = self.up3(x)  # 1/16 -> 1/8
        x = self.fuse2(torch.cat([x, s2], dim=1))
        
        x = self.up2(x)  # 1/8 -> 1/4
        x = self.fuse1(torch.cat([x, s1], dim=1))
        
        x = self.up1(x)  # 1/4 -> 1/2
        
        # Final prediction
        x = self.head(x)
        
        # Upsample to target size
        if output_size is not None:
            x = F.interpolate(x, size=output_size, mode='bilinear', align_corners=False)
            
        return x


class SegformerCD(nn.Module):
    """
    Complete Segformer-based Change Detection Module.
    
    This is the recommended CD implementation that matches Change-Agent's design.
    Uses Siamese Segformer-B1 for multi-scale feature extraction.
    """
    
    def __init__(
        self,
        pretrained: bool = True,
        num_bi3_layers: int = 3,
        num_heads: int = 8,
        hidden_dim: int = 256,
        num_classes: int = 1,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        
        # Siamese Segformer backbone
        self.backbone = mit_b1(pretrained=pretrained)
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Get backbone output channels: [64, 128, 320, 512] for B1
        self.backbone_channels = [64, 128, 320, 512]
        
        # BI3 blocks for each scale (except the smallest)
        self.bi3_blocks = nn.ModuleList()
        for i, c in enumerate(self.backbone_channels[:-1]):  # Skip last scale
            layers = nn.ModuleList([
                BI3Block(c, num_heads=min(num_heads, c // 32))
                for _ in range(num_bi3_layers)
            ])
            self.bi3_blocks.append(layers)
        
        # For the last scale (512), use simpler processing
        self.bi3_last = nn.ModuleList([
            BI3Block(512, num_heads=8)
            for _ in range(num_bi3_layers)
        ])
        
        # CBF for multi-scale fusion
        self.cbf = CBF(self.backbone_channels, hidden_dim)
        
        # Decoder
        self.decoder = SegformerCDDecoder(hidden_dim, num_classes)
        
        # Parameter count
        self._count_params()
        
    def _count_params(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"SegformerCD: {total/1e6:.1f}M params, {trainable/1e6:.1f}M trainable")
        
    def forward(
        self,
        image_a: torch.Tensor,
        image_b: torch.Tensor,
        output_size: Tuple[int, int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            image_a: (B, 3, H, W) image from time A
            image_b: (B, 3, H, W) image from time B
            output_size: Target output size for the mask
        """
        B = image_a.shape[0]
        
        # Extract multi-scale features using Siamese backbone
        feats_a = self.backbone(image_a)  # List of 4 tensors
        feats_b = self.backbone(image_b)
        
        # Apply BI3 at each scale
        enhanced_a = []
        enhanced_b = []
        
        for i, (fa, fb) in enumerate(zip(feats_a[:-1], feats_b[:-1])):
            for bi3 in self.bi3_blocks[i]:
                fa, fb = bi3(fa, fb)
            enhanced_a.append(fa)
            enhanced_b.append(fb)
        
        # Last scale
        fa, fb = feats_a[-1], feats_b[-1]
        for bi3 in self.bi3_last:
            fa, fb = bi3(fa, fb)
        enhanced_a.append(fa)
        enhanced_b.append(fb)
        
        # CBF fusion
        fused = self.cbf(enhanced_a, enhanced_b)
        
        # Decode
        if output_size is None:
            output_size = (image_a.shape[2], image_a.shape[3])
        
        mask_logits = self.decoder(fused, output_size)
        
        return {
            'mask_logits': mask_logits,
            'features_a': enhanced_a,
            'features_b': enhanced_b,
            'fused_features': fused,
        }


def build_segformer_cd(pretrained: bool = True, **kwargs) -> SegformerCD:
    """Build SegformerCD model."""
    return SegformerCD(pretrained=pretrained, **kwargs)

