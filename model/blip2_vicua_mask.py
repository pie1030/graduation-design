"""
DeltaVLM with Mask Branch

Extends Blip2VicunaInstruct with a mask prediction branch for change detection.

Uses Change-Agent style architecture: BI3/GDFA/CBF based CD module.

Training Strategy:
- Freeze: Visual Encoder, CSRM, Q-Former, LLM
- Train: Mask Branch only (ChangeAgentCD)

Change-Agent Reference:
- Paper: "Change-Agent: Toward Interactive Comprehensive Remote Sensing
         Change Interpretation and Analysis"
- GitHub: https://github.com/Chen-Yang-Liu/Change-Agent
"""

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.cuda.amp import autocast as autocast

from model.blip2_vicua import Blip2VicunaInstruct
from model.mask_branch.change_agent_cd import ChangeAgentCD


class Blip2VicunaMask(Blip2VicunaInstruct):
    """
    DeltaVLM with Mask Prediction Branch.

    Adds a Change-Agent style CD branch after the visual encoder.
    Supports multi-scale ViT feature extraction for HR decoding.
    """

    def __init__(
        self,
        vit_model="eva_clip_g",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        num_query_token=32,
        llm_model="",
        prompt="",
        max_txt_len=128,
        max_output_txt_len=256,
        apply_lemmatizer=False,
        qformer_text_input=True,
        enable_mask_branch=True,
        mask_hidden_dim=512,
        mask_num_stages=4,
        mask_output_size=(256, 256),
        freeze_for_mask_training=True,
        mask_training_mode=False,
        num_bi3_layers=3,
        num_heads=8,
        mlp_ratio=4.0,
        # Multi-scale ViT feature tapping points (0-indexed block indices)
        # EVA-ViT-G has 39 blocks; default taps at 1/4, 1/2, 3/4, final
        multiscale_layers=None,
        # Legacy compat
        mask_decoder_type="change_agent",
    ):
        if mask_training_mode:
            self._skip_llm = True

        super().__init__(
            vit_model=vit_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            num_query_token=num_query_token,
            llm_model=llm_model,
            prompt=prompt,
            max_txt_len=max_txt_len,
            max_output_txt_len=max_output_txt_len,
            apply_lemmatizer=apply_lemmatizer,
            qformer_text_input=qformer_text_input,
        )

        self.enable_mask_branch = enable_mask_branch
        self.freeze_for_mask_training = freeze_for_mask_training
        self.mask_training_mode = mask_training_mode
        self.multiscale_layers = multiscale_layers or [9, 19, 29, 38]

        if enable_mask_branch:
            eva_dim = self.visual_encoder.num_features  # 1408
            self.mask_decoder = ChangeAgentCD(
                eva_dim=eva_dim,
                hidden_dim=mask_hidden_dim,
                num_bi3_layers=num_bi3_layers,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                num_classes=1,
                num_upsample_stages=mask_num_stages,
                output_size=mask_output_size,
                dropout=0.1,
                multiscale_dims=[eva_dim] * len(self.multiscale_layers),
            )
            logging.info(
                f"CD branch: hidden_dim={mask_hidden_dim}, "
                f"bi3_layers={num_bi3_layers}, heads={num_heads}, "
                f"ms_layers={self.multiscale_layers}"
            )

            if freeze_for_mask_training:
                self._freeze_for_mask_training()

        if mask_training_mode:
            self._delete_llm_for_memory()

    # ------------------------------------------------------------------
    # Freeze / memory management
    # ------------------------------------------------------------------

    def _freeze_for_mask_training(self):
        """Freeze all components except mask branch."""
        for param in self.visual_encoder.parameters():
            param.requires_grad = False
        for param in self.ln_vision.parameters():
            param.requires_grad = False

        for name in ['context1', 'context2', 'context3', 'gate1', 'gate2']:
            mod = getattr(self, name, None)
            if mod is not None:
                for param in mod.parameters():
                    param.requires_grad = False

        if hasattr(self, 'Qformer') and self.Qformer is not None:
            for param in self.Qformer.parameters():
                param.requires_grad = False
        if hasattr(self, 'query_tokens') and self.query_tokens is not None:
            self.query_tokens.requires_grad = False

        if hasattr(self, 'llm_model') and self.llm_model is not None:
            for param in self.llm_model.parameters():
                param.requires_grad = False
        if hasattr(self, 'llm_proj') and self.llm_proj is not None:
            for param in self.llm_proj.parameters():
                param.requires_grad = False

        for name in ['vision_proj', 'text_proj']:
            mod = getattr(self, name, None)
            if mod is not None:
                for param in mod.parameters():
                    param.requires_grad = False

    def _delete_llm_for_memory(self):
        """Delete LLM and related components to save GPU memory."""
        import gc
        for attr in ['llm_model', 'llm_tokenizer', 'llm_proj',
                      'Qformer', 'query_tokens']:
            if hasattr(self, attr):
                delattr(self, attr)
                setattr(self, attr, None)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logging.info("LLM/Q-Former deleted for mask training")
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logging.info(f"  {trainable:,}/{total:,} trainable ({100*trainable/total:.2f}%)")

    def unfreeze_all(self):
        for param in self.parameters():
            param.requires_grad = True

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def extract_visual_features(
        self,
        image_A: torch.Tensor,
        image_B: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract final-layer features: each (B, 257, 1408)."""
        with self.maybe_autocast():
            feat_bef = self.ln_vision(self.visual_encoder(image_A))
            feat_aft = self.ln_vision(self.visual_encoder(image_B))
        return feat_bef, feat_aft

    def extract_multiscale_features(
        self,
        image_A: torch.Tensor,
        image_B: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor,
               List[torch.Tensor], List[torch.Tensor]]:
        """
        Extract multi-scale features from EVA-ViT intermediate layers.

        Returns:
            feat_bef: final layer output for A  (B, 257, 1408)
            feat_aft: final layer output for B
            ms_bef: list of intermediate features for A
            ms_aft: list of intermediate features for B
        """
        with self.maybe_autocast():
            all_bef = self.visual_encoder.get_intermediate_layers(image_A)
            all_aft = self.visual_encoder.get_intermediate_layers(image_B)

        ms_bef = [self.ln_vision(all_bef[i]) for i in self.multiscale_layers]
        ms_aft = [self.ln_vision(all_aft[i]) for i in self.multiscale_layers]

        feat_bef = ms_bef[-1]  # final tapped layer
        feat_aft = ms_aft[-1]

        return feat_bef, feat_aft, ms_bef, ms_aft

    # ------------------------------------------------------------------
    # Forward passes
    # ------------------------------------------------------------------

    def forward_mask(
        self,
        image_A: torch.Tensor,
        image_B: torch.Tensor,
        gt_mask: Optional[torch.Tensor] = None,
        use_multiscale: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for mask prediction only.

        Args:
            image_A/B: (B, 3, H, W)
            gt_mask: (B, 1, H, W) optional
            use_multiscale: use multi-level ViT features for HR decoding
        """
        if not self.enable_mask_branch:
            raise RuntimeError("Mask branch not enabled")

        if use_multiscale:
            feat_bef, feat_aft, ms_bef, ms_aft = \
                self.extract_multiscale_features(image_A, image_B)
            return self.mask_decoder(
                feat_bef, feat_aft, gt_mask=gt_mask,
                ms_feats_bef=ms_bef, ms_feats_aft=ms_aft,
            )
        else:
            feat_bef, feat_aft = self.extract_visual_features(image_A, image_B)
            return self.mask_decoder(feat_bef, feat_aft, gt_mask=gt_mask)

    def forward(self, samples, return_mask=False):
        """Forward pass supporting both captioning and mask prediction."""
        outputs = super().forward(samples)

        if return_mask and self.enable_mask_branch:
            mask_outputs = self.forward_mask(
                samples["image_A"], samples["image_B"],
                samples.get("gt_mask", None),
            )
            outputs.update({
                'mask_logits': mask_outputs['mask_logits'],
                'mask_pred': mask_outputs['mask_pred'],
            })
            if 'loss' in mask_outputs:
                outputs['mask_loss'] = mask_outputs['loss']

        return outputs

    def predict_mask(
        self,
        image_A: torch.Tensor,
        image_B: torch.Tensor,
        threshold: float = 0.5,
    ) -> torch.Tensor:
        """Inference-only mask prediction with multi-scale features."""
        if not self.enable_mask_branch:
            raise RuntimeError("Mask branch not enabled")
        with torch.no_grad():
            feat_bef, feat_aft, ms_bef, ms_aft = \
                self.extract_multiscale_features(image_A, image_B)
            return self.mask_decoder.predict(
                feat_bef, feat_aft, threshold=threshold,
                ms_feats_bef=ms_bef, ms_feats_aft=ms_aft,
            )

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(cls, checkpoint_path: str, **kwargs):
        """Load model from pretrained DeltaVLM checkpoint."""
        config = {
            'vit_model': 'eva_clip_g',
            'img_size': 224,
            'freeze_vit': True,
            'num_query_token': 32,
            'qformer_text_input': True,
            'enable_mask_branch': True,
            'mask_hidden_dim': 512,
            'mask_num_stages': 4,
            'mask_output_size': (256, 256),
            'freeze_for_mask_training': True,
            'num_bi3_layers': 3,
            'num_heads': 8,
            'mlp_ratio': 4.0,
            'multiscale_layers': [9, 19, 29, 38],
        }
        config.update(kwargs)

        model = cls(**config)

        checkpoint = torch.load(
            checkpoint_path, map_location='cpu', weights_only=False,
        )
        state_dict = checkpoint.get('model', checkpoint)

        if config.get('mask_training_mode', False):
            skip = ['llm_model', 'llm_proj', 'Qformer', 'query_tokens']
            state_dict = {
                k: v for k, v in state_dict.items()
                if not any(s in k for s in skip)
            }
            logging.info(f"Filtered state_dict: {len(state_dict)} keys")

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        mask_missing = [k for k in missing if 'mask_decoder' in k]
        other_missing = [k for k in missing if 'mask_decoder' not in k]
        if mask_missing:
            logging.info(f"New mask branch: {len(mask_missing)} keys to train")
        if other_missing:
            logging.warning(f"Missing: {other_missing}")
        if unexpected:
            logging.warning(f"Unexpected: {unexpected}")

        return model

    def save_mask_branch(self, path: str):
        state_dict = {
            k: v for k, v in self.state_dict().items()
            if 'mask_decoder' in k
        }
        torch.save(state_dict, path)
        logging.info(f"Mask branch saved to {path}")

    def load_mask_branch(self, path: str):
        state_dict = torch.load(path, map_location='cpu')
        self.load_state_dict(state_dict, strict=False)
        logging.info(f"Mask branch loaded from {path}")


def build_model(cfg):
    """Build Blip2VicunaMask model from config."""
    model = Blip2VicunaMask(
        vit_model=cfg.get("vit_model", "eva_clip_g"),
        img_size=cfg.get("image_size", 224),
        drop_path_rate=cfg.get("drop_path_rate", 0),
        use_grad_checkpoint=cfg.get("use_grad_checkpoint", False),
        vit_precision=cfg.get("vit_precision", "fp16"),
        freeze_vit=cfg.get("freeze_vit", True),
        num_query_token=cfg.get("num_query_token", 32),
        prompt=cfg.get("prompt", ""),
        max_txt_len=cfg.get("max_txt_len", 128),
        max_output_txt_len=cfg.get("max_output_txt_len", 256),
        qformer_text_input=cfg.get("qformer_text_input", True),
        enable_mask_branch=cfg.get("enable_mask_branch", True),
        mask_hidden_dim=cfg.get("mask_hidden_dim", 512),
        mask_num_stages=cfg.get("mask_num_stages", 4),
        mask_output_size=tuple(cfg.get("mask_output_size", [256, 256])),
        freeze_for_mask_training=cfg.get("freeze_for_mask_training", True),
        num_bi3_layers=cfg.get("num_bi3_layers", 3),
        num_heads=cfg.get("num_heads", 8),
        mlp_ratio=cfg.get("mlp_ratio", 4.0),
        multiscale_layers=cfg.get("multiscale_layers", [9, 19, 29, 38]),
    )

    pretrained = cfg.get("pretrained", None)
    if pretrained:
        model = Blip2VicunaMask.from_pretrained(pretrained)

    return model
