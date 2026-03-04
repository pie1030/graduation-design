"""
DeltaCD: Clean Change Detection Module for DeltaVLM

Architecture:
  Path 1 (frozen): EVA-ViT → semantic features (16x16, 1408d)
  Path 2 (trainable): Lightweight HR Encoder → spatial detail features (56x56)
  Fusion: Upsample semantic + HR → fused features (56x56)
  CSRM: Difference-gated features in fused high-resolution space
  MaskDecoder: concat(F_A', diff, F_B') → mask (256x256)

Key design principles:
  - HR is not a skip connection; it fuses as an equal partner with semantic features
  - Difference modeling happens in the fused high-resolution space
  - CSRM provides per-position gating for change-aware feature selection
  - Clean gradient path: all trainable components receive direct gradients
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .mask_head import MultiClassFocalDiceLoss, FocalDiceLoss


class SemanticAdapter(nn.Module):
    """Adapts frozen EVA-ViT output (B, 257, 1408) → spatial features (B, C, H, H)."""

    def __init__(self, vit_dim=1408, out_dim=256, target_size=56):
        super().__init__()
        self.target_size = target_size
        self.proj = nn.Sequential(
            nn.Linear(vit_dim, out_dim),
            nn.LayerNorm(out_dim),
        )
        self.refine = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.GELU(),
            nn.Conv2d(out_dim, out_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.GELU(),
        )

    def forward(self, x):
        """(B, 257, 1408) → (B, out_dim, target_size, target_size)"""
        B = x.shape[0]
        x = x[:, 1:, :]  # drop CLS token → (B, 256, 1408)
        x = self.proj(x)  # (B, 256, out_dim)
        H = W = int(math.sqrt(x.shape[1]))  # 16
        x = x.transpose(1, 2).reshape(B, -1, H, W)  # (B, out_dim, 16, 16)
        x = F.interpolate(
            x, size=(self.target_size, self.target_size),
            mode='bilinear', align_corners=False,
        )
        x = x + self.refine(x)  # residual refinement
        return x


class HREncoder(nn.Module):
    """
    Lightweight CNN for high-resolution spatial feature extraction.
    Siamese (shared weights for both temporal images).

    224x224 → stem(s=2) → 112x112 → stage1(s=2) → 56x56
    """

    def __init__(self, in_channels=3, mid_dim=48, out_dim=96):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, mid_dim, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(mid_dim),
            nn.ReLU(inplace=True),
        )
        self.stage1 = nn.Sequential(
            nn.Conv2d(mid_dim, out_dim, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_dim, out_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_dim),
        )
        self.downsample = nn.Sequential(
            nn.Conv2d(mid_dim, out_dim, 1, stride=2, bias=False),
            nn.BatchNorm2d(out_dim),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """(B, 3, 224, 224) → (B, out_dim, 56, 56)"""
        x = self.stem(x)              # (B, 48, 112, 112)
        identity = self.downsample(x)  # (B, out_dim, 56, 56)
        x = self.stage1(x)            # (B, out_dim, 56, 56)
        return self.relu(x + identity)


class FusionModule(nn.Module):
    """Fuses upsampled semantic features and HR spatial features."""

    def __init__(self, sem_dim=256, hr_dim=96, out_dim=256):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(sem_dim + hr_dim, out_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.GELU(),
            nn.Conv2d(out_dim, out_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.GELU(),
        )

    def forward(self, sem_feat, hr_feat):
        return self.fuse(torch.cat([sem_feat, hr_feat], dim=1))


class SpatialCSRM(nn.Module):
    """
    Cross-temporal Spatial Reasoning Module in fused feature space.

    For each spatial position independently:
      diff = B - A
      C_A = tanh(W_cd · diff + W_cf · A)      (context)
      G_A = sigmoid(W_gd · diff + W_gf · A)    (gate)
      F_A' = G_A * C_A                          (gated output)
    Symmetric for B.

    Operates on (B, C, H, W) spatial features via per-position linear layers.
    """

    def __init__(self, dim):
        super().__init__()
        self.context_diff = nn.Linear(dim, dim, bias=False)
        self.context_feat = nn.Linear(dim, dim)
        self.gate_diff = nn.Linear(dim, dim, bias=False)
        self.gate_feat = nn.Linear(dim, dim)

    def forward(self, feat_A, feat_B):
        """
        Args:
            feat_A, feat_B: (B, C, H, W) fused features
        Returns:
            F_A': gated feature for A  (B, C, H, W)
            F_B': gated feature for B  (B, C, H, W)
            diff: difference feature    (B, C, H, W)
        """
        B, C, H, W = feat_A.shape

        # Flatten spatial dims for linear layers: (B, H*W, C)
        a = feat_A.flatten(2).transpose(1, 2)
        b = feat_B.flatten(2).transpose(1, 2)
        diff = b - a

        # Gate and context for A
        ctx_a = torch.tanh(self.context_diff(diff) + self.context_feat(a))
        gate_a = torch.sigmoid(self.gate_diff(diff) + self.gate_feat(a))
        fa_prime = gate_a * ctx_a

        # Gate and context for B
        ctx_b = torch.tanh(self.context_diff(diff) + self.context_feat(b))
        gate_b = torch.sigmoid(self.gate_diff(diff) + self.gate_feat(b))
        fb_prime = gate_b * ctx_b

        # Reshape back to spatial
        fa_prime = fa_prime.transpose(1, 2).reshape(B, C, H, W)
        fb_prime = fb_prime.transpose(1, 2).reshape(B, C, H, W)
        diff_out = diff.transpose(1, 2).reshape(B, C, H, W)

        return fa_prime, fb_prime, diff_out


class MaskDecoder(nn.Module):
    """
    Progressive upsampling decoder: (3*C, 56, 56) → (num_classes, 256, 256).

    Input: concat(F_A', diff, F_B') from CSRM.
    Two ConvTranspose2d stages (56→112→224), then interpolate to 256.
    """

    def __init__(self, in_dim, num_classes=3, output_size=(256, 256)):
        super().__init__()
        self.output_size = output_size

        self.compress = nn.Sequential(
            nn.Conv2d(in_dim, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.GELU(),
        )

        # 56 → 112
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
        )

        # 112 → 224
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
        )

        self.head = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        """(B, in_dim, 56, 56) → (B, num_classes, 256, 256)"""
        x = self.compress(x)  # (B, 256, 56, 56)
        x = self.up1(x)       # (B, 128, 112, 112)
        x = self.up2(x)       # (B, 64, 224, 224)
        x = self.head(x)      # (B, num_classes, 224, 224)
        if x.shape[-2:] != tuple(self.output_size):
            x = F.interpolate(
                x, size=self.output_size,
                mode='bilinear', align_corners=False,
            )
        return x


class DeltaCD(nn.Module):
    """
    Complete Change Detection module for DeltaVLM.

    Pipeline (per image pair):
      1. SemanticAdapter: EVA-ViT (frozen) feat → upsample to 56x56
      2. HREncoder: raw image → spatial detail at 56x56
      3. Fusion: concat semantic + HR → fused features
      4. SpatialCSRM: difference-gated features in fused space
      5. MaskDecoder: concat(F_A', diff, F_B') → class mask at 256x256

    Trainable parameters: ~3.5M (vs ~50M in old ChangeAgentCD)
    """

    def __init__(
        self,
        vit_dim=1408,
        sem_dim=256,
        hr_mid_dim=48,
        hr_out_dim=96,
        fused_dim=256,
        fusion_size=56,
        num_classes=3,
        output_size=(256, 256),
    ):
        super().__init__()
        self.num_classes = num_classes
        self.output_size = output_size

        # Path 1: Semantic adapter (ViT tokens → spatial)
        self.sem_adapter = SemanticAdapter(vit_dim, sem_dim, fusion_size)

        # Path 2: HR spatial encoder
        self.hr_encoder = HREncoder(
            in_channels=3, mid_dim=hr_mid_dim, out_dim=hr_out_dim,
        )

        # Fusion
        self.fusion = FusionModule(sem_dim, hr_out_dim, fused_dim)

        # CSRM in fused space
        self.csrm = SpatialCSRM(fused_dim)

        # Decoder: input is concat(F_A', diff, F_B') = 3 * fused_dim
        self.mask_decoder = MaskDecoder(
            3 * fused_dim, num_classes, output_size,
        )

        # Loss
        if num_classes > 1:
            self.loss_fn = MultiClassFocalDiceLoss(
                num_classes=num_classes,
                class_weights=[0.2, 1.0, 1.0][:num_classes],
            )
        else:
            self.loss_fn = FocalDiceLoss()

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        feat_A,        # (B, 257, 1408) frozen EVA-ViT output for image A
        feat_B,        # (B, 257, 1408) frozen EVA-ViT output for image B
        img_A,         # (B, 3, 224, 224) normalized image A
        img_B,         # (B, 3, 224, 224) normalized image B
        gt_mask=None,  # (B, H, W) long tensor with class indices
    ):
        # --- Path 1: Semantic (frozen ViT → spatial) ---
        sem_A = self.sem_adapter(feat_A.float())  # (B, sem_dim, 56, 56)
        sem_B = self.sem_adapter(feat_B.float())

        # --- Path 2: HR spatial detail ---
        hr_A = self.hr_encoder(img_A.float())     # (B, hr_dim, 56, 56)
        hr_B = self.hr_encoder(img_B.float())

        # --- Fusion ---
        fused_A = self.fusion(sem_A, hr_A)        # (B, fused_dim, 56, 56)
        fused_B = self.fusion(sem_B, hr_B)

        # --- CSRM: difference-gated features ---
        fa_prime, fb_prime, diff = self.csrm(fused_A, fused_B)

        # --- Mask prediction ---
        mask_input = torch.cat([fa_prime, diff, fb_prime], dim=1)
        mask_logits = self.mask_decoder(mask_input)  # (B, C, 256, 256)

        # Build outputs
        if self.num_classes > 1:
            mask_pred = F.softmax(mask_logits, dim=1)
            mask_cls = mask_logits.argmax(dim=1)
        else:
            mask_pred = torch.sigmoid(mask_logits)
            mask_cls = (mask_pred > 0.5).long().squeeze(1)

        outputs = {
            'mask_logits': mask_logits,
            'mask_pred': mask_pred,
            'mask_cls': mask_cls,
        }

        if gt_mask is not None:
            if self.num_classes > 1:
                if gt_mask.dim() == 4:
                    gt_mask = gt_mask.squeeze(1)
                if gt_mask.shape[-2:] != tuple(self.output_size):
                    gt_mask = F.interpolate(
                        gt_mask.unsqueeze(1).float(),
                        size=self.output_size, mode='nearest',
                    ).squeeze(1).long()
            outputs['loss'] = self.loss_fn(mask_logits, gt_mask)

        return outputs

    def predict(self, feat_A, feat_B, img_A, img_B, threshold=0.5):
        with torch.no_grad():
            return self.forward(feat_A, feat_B, img_A, img_B)['mask_cls']
