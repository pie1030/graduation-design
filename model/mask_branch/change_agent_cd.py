"""
Change-Agent Style Change Detection Module for DeltaVLM

Migrated from: https://github.com/Chen-Yang-Liu/Change-Agent
Paper: "Change-Agent: Toward Interactive Comprehensive Remote Sensing
        Change Interpretation and Analysis"

Core Components:
- LPE (Local Perceptual Enhancement): Multi-kernel depthwise convolutions
- GDFA (Global Difference-guided Feature Attention): Difference-modulated attention
- BI3 (Bidirectional Iterative Interaction): Iterative cross-attention
- CBF (Change Bi-temporal Fusion): Feature fusion with cosine similarity
- MultiScaleSkipAdapter: Converts multi-level ViT features into decoder skip
  connections for high-resolution change map generation

Adaptation for DeltaVLM:
- Input: EVA-ViT-G features (B, 257, 1408) + optional multi-scale intermediates
- Output: Binary change mask (B, 1, H, W)
"""

import math
from typing import Optional, Tuple, List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from einops import rearrange

from .mask_head import FocalDiceLoss


# ============================================================================
# LPE: Local Perceptual Enhancement (from Change-Agent)
# ============================================================================

class LPE(nn.Module):
    """
    Local Perceptual Enhancement module.
    
    Uses multiple depthwise convolutions with different kernel sizes:
    - 3×3: captures local context
    - 1×5: horizontal edge features
    - 5×1: vertical edge features
    
    Adapted from Change-Agent's Dynamic_conv class.
    """
    def __init__(self, dim: int):
        super().__init__()
        # Depthwise convolutions with different kernel sizes
        self.d_conv_3x3 = nn.Conv2d(
            dim, dim, kernel_size=3, stride=1, padding=1, groups=dim
        )
        self.d_conv_1x5 = nn.Conv2d(
            dim, dim, kernel_size=(1, 5), padding=(0, 2), groups=dim
        )
        self.d_conv_5x1 = nn.Conv2d(
            dim, dim, kernel_size=(5, 1), padding=(2, 0), groups=dim
        )
        
        self.bn = nn.BatchNorm2d(3 * dim)
        self.activation = nn.GELU()
        self.fusion = nn.Conv2d(3 * dim, dim, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) spatial features
        Returns:
            (B, C, H, W) enhanced features
        """
        x1 = self.d_conv_3x3(x)
        x2 = self.d_conv_1x5(x)
        x3 = self.d_conv_5x1(x)
        
        x_cat = torch.cat([x1, x2, x3], dim=1)
        x_cat = self.bn(x_cat)
        x_cat = self.activation(x_cat)
        out = self.fusion(x_cat)
        
        return out


# ============================================================================
# GDFA: Global Difference-guided Feature Attention
# ============================================================================

class GDFA(nn.Module):
    """
    Global Difference-guided Feature Attention.
    
    Computes cross-attention where the difference between bi-temporal features
    modulates the key/value features.
    
    Key insight: dif = x2 - x1, then K/V = conv(x1 * dif)
    
    Adapted from Change-Agent's MultiHeadAtt class.
    """
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Q, K, V projections
        self.to_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_v = nn.Linear(dim, dim, bias=qkv_bias)
        
        # Difference-guided feature modulation
        self.diff_modulate = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
        )
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        is_cross: bool = True,
    ) -> torch.Tensor:
        """
        Args:
            query: (B, N, C) query features (e.g., from time A)
            key: (B, N, C) key features (e.g., from time B)
            value: (B, N, C) value features (same as key typically)
            is_cross: Whether this is cross-attention (enables difference modulation)
        Returns:
            (B, N, C) attended features
        """
        B, N, C = query.shape
        H = W = int(math.sqrt(N))
        
        # Reshape to spatial for difference computation
        if is_cross:
            q_spatial = query.transpose(1, 2).view(B, C, H, W)
            k_spatial = key.transpose(1, 2).view(B, C, H, W)
            
            # Difference-guided modulation
            diff = k_spatial - q_spatial
            modulated = self.diff_modulate(q_spatial * diff)
            
            # Update key and value with modulated features
            key = modulated.view(B, C, -1).transpose(1, 2)
            value = key
        
        # Multi-head attention
        q = self.to_q(query).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.to_k(key).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.to_v(value).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        
        return out


# ============================================================================
# BI3: Bidirectional Iterative Interaction Block
# ============================================================================

class BI3Block(nn.Module):
    """
    Bidirectional Iterative Interaction Block.
    
    Combines LPE (local enhancement) and GDFA (global attention) for
    bi-directional feature interaction between two temporal images.
    
    Structure:
    1. LPE enhancement for both temporal features
    2. GDFA cross-attention: A -> B and B -> A
    3. FFN with residual connection
    
    Adapted from Change-Agent's Transformer class.
    """
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        norm_first: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.norm_first = norm_first
        
        # LPE for local enhancement
        self.lpe_q = LPE(dim)
        self.lpe_k = LPE(dim)
        
        # GDFA for global attention
        self.gdfa = GDFA(dim, num_heads, attn_drop=attn_drop, proj_drop=drop)
        
        # FFN
        mlp_hidden = int(dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(drop),
        )
        
        # Layer norms
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
    
    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        x3: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x1: (B, N, C) query features (typically from time A)
            x2: (B, N, C) key features (typically from time B)
            x3: (B, N, C) value features (typically same as x2)
        Returns:
            (B, N, C) output features
        """
        B, N, C = x1.shape
        H = W = int(math.sqrt(N))
        
        # Reshape to spatial for LPE
        x1_spatial = x1.transpose(1, 2).view(B, C, H, W)
        x2_spatial = x2.transpose(1, 2).view(B, C, H, W)
        x3_spatial = x3.transpose(1, 2).view(B, C, H, W)
        
        # LPE enhancement (residual)
        x1_spatial = x1_spatial + self.lpe_q(x1_spatial)
        x2_spatial = x2_spatial + self.lpe_k(x2_spatial)
        x3_spatial = x3_spatial + self.lpe_k(x3_spatial)
        
        # Reshape back to sequence
        x1_seq = x1_spatial.view(B, C, -1).transpose(1, 2)
        x2_seq = x2_spatial.view(B, C, -1).transpose(1, 2)
        x3_seq = x3_spatial.view(B, C, -1).transpose(1, 2)
        
        # Store residual
        residual = x1_seq
        
        # GDFA cross-attention with normalization
        if self.norm_first:
            x_attn = self.gdfa(self.norm1(x1_seq), self.norm1(x2_seq), self.norm1(x3_seq)) + residual
            x_out = self.ffn(self.norm2(x_attn)) + x_attn
        else:
            x_attn = self.norm1(self.gdfa(x1_seq, x2_seq, x3_seq) + residual)
            x_out = self.norm2(self.ffn(x_attn) + x_attn)
        
        return x_out


class BI3Neck(nn.Module):
    """
    BI3 Neck: Multiple BI3 blocks for bidirectional iterative interaction.
    
    Processes the deepest features from both temporal images through
    multiple iterations of cross-attention.
    """
    def __init__(
        self,
        dim: int,
        num_layers: int = 3,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_layers = num_layers
        
        # Positional embeddings (learnable)
        self.pos_embed = None  # Will be created dynamically based on input size
        
        # BI3 layers (each layer has A->B and B->A blocks)
        self.layers = nn.ModuleList([
            nn.ModuleList([
                BI3Block(dim, num_heads, mlp_ratio, drop),
                BI3Block(dim, num_heads, mlp_ratio, drop),
            ])
            for _ in range(num_layers)
        ])
    
    def _get_pos_embed(self, H: int, W: int, device: torch.device) -> torch.Tensor:
        """Generate learnable 2D positional embeddings."""
        if self.pos_embed is None or self.pos_embed.shape[1] != H * W:
            # Create new positional embedding
            self.pos_embed = nn.Parameter(
                torch.zeros(1, H * W, self.dim, device=device)
            )
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        return self.pos_embed
    
    def forward(
        self,
        feat_A: torch.Tensor,
        feat_B: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            feat_A: (B, C, H, W) features from time A
            feat_B: (B, C, H, W) features from time B
        Returns:
            feat_A_out: (B, C, H, W) enhanced features from time A
            feat_B_out: (B, C, H, W) enhanced features from time B
        """
        B, C, H, W = feat_A.shape
        
        # Flatten to sequence
        feat_A_seq = feat_A.view(B, C, -1).transpose(1, 2)  # (B, H*W, C)
        feat_B_seq = feat_B.view(B, C, -1).transpose(1, 2)
        
        # Add positional embedding
        # Note: For simplicity, we skip positional embedding here
        # In practice, you can add it like: feat_A_seq = feat_A_seq + pos_embed
        
        # Iterative BI3 interaction
        for layer_A, layer_B in self.layers:
            # A attends to B
            feat_A_new = layer_A(feat_A_seq, feat_B_seq, feat_B_seq) + feat_A_seq
            # B attends to A (using updated A)
            feat_B_new = layer_B(feat_B_seq, feat_A_new, feat_A_new) + feat_B_seq
            
            feat_A_seq = feat_A_new
            feat_B_seq = feat_B_new
        
        # Reshape back to spatial
        feat_A_out = feat_A_seq.transpose(1, 2).view(B, C, H, W)
        feat_B_out = feat_B_seq.transpose(1, 2).view(B, C, H, W)
        
        return feat_A_out, feat_B_out


# ============================================================================
# CBF: Change Bi-temporal Fusion
# ============================================================================

class CBF(nn.Module):
    """
    Change Bi-temporal Fusion module.
    
    Fuses bi-temporal features using:
    1. Difference feature: B - A
    2. Cosine similarity
    3. Concatenation: [A, diff, B]
    
    Adapted from Change-Agent's change_detection method.
    """
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        # Difference convolution
        self.conv_diff = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, kernel_size=1),
            nn.BatchNorm2d(in_dim),
        )
        
        # Fusion convolution: [A, diff+cos, B] -> out_dim
        # Input: 3 * in_dim (because diff and cos are combined differently in original)
        # For simplicity, we use: [A, diff_enhanced, B] where diff_enhanced = conv(diff) + cos
        self.conv_fuse = nn.Sequential(
            nn.Conv2d(3 * in_dim, 2 * out_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(2 * out_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(2 * out_dim, out_dim, kernel_size=1),
        )
        
        # Cosine similarity
        self.cos = nn.CosineSimilarity(dim=1)
    
    def forward(
        self,
        feat_A: torch.Tensor,
        feat_B: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            feat_A: (B, C, H, W) features from time A
            feat_B: (B, C, H, W) features from time B
        Returns:
            (B, out_dim, H, W) fused features
        """
        # Difference with convolution
        diff = self.conv_diff(feat_B - feat_A)
        
        # Cosine similarity (expand to match channels)
        cos_sim = self.cos(feat_A, feat_B).unsqueeze(1)  # (B, 1, H, W)
        
        # Enhanced difference: diff + cosine similarity broadcast
        diff_enhanced = diff + cos_sim
        
        # Concatenate: [A, diff_enhanced, B]
        fused = torch.cat([feat_A, diff_enhanced, feat_B], dim=1)
        
        # Fuse
        out = self.conv_fuse(fused)
        
        return out


# ============================================================================
# Multi-Scale Skip Adapter: ViT intermediates -> decoder skip connections
# ============================================================================

class MultiScaleSkipAdapter(nn.Module):
    """
    Converts multi-level ViT features into spatial skip connections for the
    decoder.  Each level's (B, N, C_vit) sequence is stripped of its CLS token,
    reshaped to (B, C_vit, H, W), then projected to the decoder's expected
    channel width at that stage.

    The adapter computes per-level *difference features* from the bi-temporal
    pair so that spatial detail and change signal are both preserved.
    """

    def __init__(
        self,
        vit_dims: List[int],
        decoder_dims: List[int],
    ):
        super().__init__()
        assert len(vit_dims) == len(decoder_dims)
        self.num_levels = len(vit_dims)
        self.projs = nn.ModuleList()
        for vit_d, dec_d in zip(vit_dims, decoder_dims):
            self.projs.append(nn.Sequential(
                nn.Conv2d(vit_d, dec_d, kernel_size=1, bias=False),
                nn.BatchNorm2d(dec_d),
                nn.ReLU(inplace=True),
            ))

    @staticmethod
    def _to_spatial(x: torch.Tensor) -> torch.Tensor:
        """(B, N, C) -> (B, C, H, W), dropping CLS token."""
        B, N, C = x.shape
        x = x[:, 1:, :]  # drop CLS
        H = W = int(math.sqrt(N - 1))
        return x.transpose(1, 2).reshape(B, C, H, W)

    def forward(
        self,
        ms_bef: List[torch.Tensor],
        ms_aft: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        """
        Returns list of (B, dec_dim_i, H, W) difference skip features,
        one per level, from shallow (early ViT layer) to deep.
        """
        skips = []
        for i, proj in enumerate(self.projs):
            bef_sp = self._to_spatial(ms_bef[i])
            aft_sp = self._to_spatial(ms_aft[i])
            diff = aft_sp - bef_sp  # change signal at this semantic level
            skips.append(proj(diff))
        return skips


# ============================================================================
# CD Decoder: Change Detection Decoder with Upsampling + Skip Connections
# ============================================================================

class CDDecoder(nn.Module):
    """
    Change Detection Decoder with progressive upsampling.

    Each upsample stage doubles spatial resolution via ConvTranspose2d.
    Optional skip connections (from MultiScaleSkipAdapter) are added at
    matching stages to inject multi-level change detail.

    When skip_dims is provided, skip features are concatenated with the
    decoder feature at each stage and fused via a 1x1 conv.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 128,
        num_classes: int = 1,
        num_upsample_stages: int = 4,
        skip_dims: Optional[List[int]] = None,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_upsample_stages = num_upsample_stages
        self.has_skips = skip_dims is not None and len(skip_dims) > 0

        self.input_proj = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, kernel_size=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )

        self.upsample_stages = nn.ModuleList()
        self.skip_fuse = nn.ModuleList() if self.has_skips else None

        current_dim = hidden_dim
        for i in range(num_upsample_stages):
            next_dim = max(hidden_dim // (2 ** (i + 1)), 32)
            self.upsample_stages.append(nn.Sequential(
                nn.ConvTranspose2d(
                    current_dim, next_dim,
                    kernel_size=4, stride=2, padding=1,
                ),
                nn.BatchNorm2d(next_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(next_dim, next_dim, kernel_size=3, padding=1),
                nn.BatchNorm2d(next_dim),
                nn.ReLU(inplace=True),
            ))
            if self.has_skips and i < len(skip_dims):
                self.skip_fuse.append(nn.Sequential(
                    nn.Conv2d(next_dim + skip_dims[i], next_dim, 1, bias=False),
                    nn.BatchNorm2d(next_dim),
                    nn.ReLU(inplace=True),
                ))
            elif self.has_skips:
                self.skip_fuse.append(None)
            current_dim = next_dim

        self.head = nn.Conv2d(current_dim, num_classes, kernel_size=1)

    def forward(
        self,
        x: torch.Tensor,
        output_size: Optional[Tuple[int, int]] = None,
        skip_features: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) fused features from CBF
            output_size: target spatial size
            skip_features: list of (B, C_i, H_skip, W_skip) from
                           MultiScaleSkipAdapter, ordered shallow -> deep
        """
        x = self.input_proj(x)

        for i, stage in enumerate(self.upsample_stages):
            x = stage(x)
            # Inject skip connection (upsampled to match decoder resolution)
            if (skip_features is not None
                    and self.skip_fuse is not None
                    and i < len(skip_features)
                    and self.skip_fuse[i] is not None):
                skip = skip_features[i]
                if skip.shape[-2:] != x.shape[-2:]:
                    skip = F.interpolate(
                        skip, size=x.shape[-2:],
                        mode='bilinear', align_corners=False,
                    )
                x = self.skip_fuse[i](torch.cat([x, skip], dim=1))

        if output_size is not None and x.shape[-2:] != tuple(output_size):
            x = F.interpolate(
                x, size=output_size,
                mode='bilinear', align_corners=False,
            )

        return self.head(x)


# ============================================================================
# Feature Adapter: EVA-ViT -> Change-Agent format
# ============================================================================

class EVAToSpatialAdapter(nn.Module):
    """
    Adapter to convert EVA-ViT-G features to spatial format for Change-Agent modules.
    
    EVA-ViT-G output: (B, 257, 1408) where 257 = 16x16 + 1 (CLS token)
    Change-Agent expects: (B, C, H, W) spatial features
    
    This adapter:
    1. Removes CLS token
    2. Reshapes to spatial format
    3. Projects to target dimension
    """
    def __init__(
        self,
        in_dim: int = 1408,  # EVA-ViT-G dim
        out_dim: int = 512,  # Change-Agent dim
        spatial_size: int = 16,  # 16x16 for EVA-ViT-G with 224x224 input
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.spatial_size = spatial_size
        
        # Linear projection from EVA dim to target dim
        self.proj = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
        )
    
    def forward(
        self,
        x: torch.Tensor,
        has_cls_token: bool = True,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, N, C) sequence features from EVA-ViT
            has_cls_token: Whether the first token is CLS token
        Returns:
            (B, out_dim, H, W) spatial features
        """
        B, N, C = x.shape
        
        # Remove CLS token
        if has_cls_token:
            x = x[:, 1:, :]  # (B, 256, 1408)
            N = N - 1
        
        # Project to target dimension
        x = self.proj(x)  # (B, 256, out_dim)
        
        # Reshape to spatial
        H = W = int(math.sqrt(N))
        x = x.transpose(1, 2).view(B, self.out_dim, H, W)
        
        return x


# ============================================================================
# Complete Change Detection Module for DeltaVLM
# ============================================================================

class ChangeAgentCD(nn.Module):
    """
    Complete Change Detection module using Change-Agent architecture,
    enhanced with multi-scale ViT skip connections for high-resolution output.

    Pipeline:
    1. EVAToSpatialAdapter: Convert EVA-ViT features to spatial format
    2. BI3Neck: Bidirectional iterative interaction (cross-temporal)
    3. CBF: Change bi-temporal fusion
    4. MultiScaleSkipAdapter (optional): diff features from ViT intermediates
    5. CDDecoder: Progressive upsampling with skip injection -> mask
    """

    def __init__(
        self,
        eva_dim: int = 1408,
        hidden_dim: int = 512,
        num_bi3_layers: int = 3,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        num_classes: int = 1,
        num_upsample_stages: int = 4,
        output_size: Tuple[int, int] = (256, 256),
        dropout: float = 0.1,
        # Multi-scale: list of ViT dim per tapped layer (e.g. [1408]*4)
        # When provided, enables skip connections in the decoder.
        multiscale_dims: Optional[List[int]] = None,
    ):
        super().__init__()
        self.eva_dim = eva_dim
        self.hidden_dim = hidden_dim
        self.output_size = output_size
        self.num_classes = num_classes

        self.adapter = EVAToSpatialAdapter(
            in_dim=eva_dim,
            out_dim=hidden_dim,
        )

        self.bi3_neck = BI3Neck(
            dim=hidden_dim,
            num_layers=num_bi3_layers,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            drop=dropout,
        )

        self.cbf = CBF(
            in_dim=hidden_dim,
            out_dim=hidden_dim,
        )

        # Multi-scale skip adapter (converts ViT intermediates -> decoder skips)
        self.use_multiscale = multiscale_dims is not None and len(multiscale_dims) > 0
        skip_dims_for_decoder = None
        if self.use_multiscale:
            n_levels = min(len(multiscale_dims), num_upsample_stages)
            decoder_stage_dims = [
                max(hidden_dim // (2 ** (i + 1)), 32)
                for i in range(n_levels)
            ]
            self.ms_adapter = MultiScaleSkipAdapter(
                vit_dims=multiscale_dims[:n_levels],
                decoder_dims=decoder_stage_dims,
            )
            skip_dims_for_decoder = decoder_stage_dims

        self.decoder = CDDecoder(
            in_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            num_upsample_stages=num_upsample_stages,
            skip_dims=skip_dims_for_decoder,
        )

        self.loss_fn = FocalDiceLoss()
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(
        self,
        feat_bef: torch.Tensor,
        feat_aft: torch.Tensor,
        gt_mask: Optional[torch.Tensor] = None,
        output_size: Optional[Tuple[int, int]] = None,
        ms_feats_bef: Optional[List[torch.Tensor]] = None,
        ms_feats_aft: Optional[List[torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            feat_bef/aft: (B, 257, 1408) final-layer EVA-ViT features
            gt_mask: (B, 1, H, W) optional ground truth
            output_size: target spatial size for the mask
            ms_feats_bef/aft: optional lists of (B, 257, 1408) from
                intermediate ViT layers, used for skip connections
        """
        output_size = output_size or self.output_size

        # 1. Adapt to spatial  (B, hidden_dim, 16, 16)
        feat_A = self.adapter(feat_bef, has_cls_token=True)
        feat_B = self.adapter(feat_aft, has_cls_token=True)

        # 2. BI3 bidirectional interaction
        feat_A_enh, feat_B_enh = self.bi3_neck(feat_A, feat_B)

        # 3. CBF fusion  (B, hidden_dim, 16, 16)
        fused = self.cbf(feat_A_enh, feat_B_enh)

        # 4. Multi-scale skip features (if available)
        skip_features = None
        if (self.use_multiscale
                and ms_feats_bef is not None
                and ms_feats_aft is not None):
            skip_features = self.ms_adapter(ms_feats_bef, ms_feats_aft)

        # 5. Decode
        mask_logits = self.decoder(
            fused, output_size=output_size,
            skip_features=skip_features,
        )
        mask_pred = torch.sigmoid(mask_logits)

        outputs = {'mask_logits': mask_logits, 'mask_pred': mask_pred}

        if gt_mask is not None:
            if gt_mask.shape[-2:] != tuple(output_size):
                gt_mask = F.interpolate(gt_mask, size=output_size, mode='nearest')
            outputs['loss'] = self.loss_fn(mask_logits, gt_mask)

        return outputs

    def predict(
        self,
        feat_bef: torch.Tensor,
        feat_aft: torch.Tensor,
        threshold: float = 0.5,
        output_size: Optional[Tuple[int, int]] = None,
        ms_feats_bef: Optional[List[torch.Tensor]] = None,
        ms_feats_aft: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        with torch.no_grad():
            outputs = self.forward(
                feat_bef, feat_aft,
                output_size=output_size,
                ms_feats_bef=ms_feats_bef,
                ms_feats_aft=ms_feats_aft,
            )
            return (outputs['mask_pred'] > threshold).float()
