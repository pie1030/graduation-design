"""
Image Processor for Mask Training

Applies synchronized augmentation to bi-temporal images and their masks.
Augmentation based on DeltaVLM paper:
- Random crop: 0-5% removal (scale=0.95-1.0)
- Random rotation: [-15°, +15°]
- Resize to target size
"""

import random
import torch
from torchvision import transforms
from torchvision.transforms import functional as F
from torchvision.transforms.functional import InterpolationMode
from PIL import Image
import numpy as np


class MaskAwarePairTransforms:
    """
    Synchronized transforms for bi-temporal images AND mask.
    
    Ensures the same geometric transformations are applied to:
    - Image A (before)
    - Image B (after)  
    - Ground truth mask
    """
    
    def __init__(
        self,
        image_size: int = 224,
        mask_size: int = 256,
        crop_scale: tuple = (0.95, 1.0),
        rotation_range: tuple = (-15, 15),
        flip_prob: float = 0.5,
    ):
        self.image_size = image_size
        self.mask_size = mask_size
        self.crop_scale = crop_scale
        self.rotation_range = rotation_range
        self.flip_prob = flip_prob
        
        # ImageNet normalization
        self.normalize = transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711),
        )
    
    def __call__(self, img_A: Image.Image, img_B: Image.Image, mask: Image.Image = None):
        """
        Apply synchronized transforms.
        
        Args:
            img_A: PIL Image (before)
            img_B: PIL Image (after)
            mask: PIL Image (ground truth mask, optional)
        
        Returns:
            img_A, img_B: Transformed tensors (C, H, W)
            mask: Transformed tensor (1, H, W) if provided, else None
        """
        # Generate random parameters (same for all inputs)
        # Random crop parameters
        i, j, h, w = transforms.RandomResizedCrop.get_params(
            img_A, 
            scale=self.crop_scale, 
            ratio=(1.0, 1.0)  # Keep aspect ratio
        )
        
        # Random rotation angle
        angle = random.uniform(*self.rotation_range)
        
        # Random horizontal flip
        do_flip = random.random() < self.flip_prob
        
        # Apply to images
        img_A = self._transform_image(img_A, i, j, h, w, angle, do_flip)
        img_B = self._transform_image(img_B, i, j, h, w, angle, do_flip)
        
        # Apply to mask if provided
        if mask is not None:
            mask = self._transform_mask(mask, i, j, h, w, angle, do_flip)
            return img_A, img_B, mask
        
        return img_A, img_B
    
    def _transform_image(self, img, i, j, h, w, angle, do_flip):
        """Apply transforms to a single image."""
        # Crop and resize
        img = F.resized_crop(
            img, i, j, h, w, 
            (self.image_size, self.image_size),
            interpolation=InterpolationMode.BICUBIC
        )
        
        # Rotate
        img = F.rotate(img, angle, interpolation=InterpolationMode.BILINEAR)
        
        # Horizontal flip
        if do_flip:
            img = F.hflip(img)
        
        # To tensor and normalize
        img = F.to_tensor(img)
        img = self.normalize(img)
        
        return img
    
    def _transform_mask(self, mask, i, j, h, w, angle, do_flip):
        """Apply transforms to mask (no normalization, nearest interpolation)."""
        # Crop and resize
        mask = F.resized_crop(
            mask, i, j, h, w,
            (self.mask_size, self.mask_size),
            interpolation=InterpolationMode.NEAREST
        )
        
        # Rotate
        mask = F.rotate(mask, angle, interpolation=InterpolationMode.NEAREST)
        
        # Horizontal flip
        if do_flip:
            mask = F.hflip(mask)
        
        # Convert to class-index tensor: 0=bg, 1=road(128), 2=building(255)
        mask_arr = np.array(mask)
        if mask_arr.ndim == 3:
            mask_arr = mask_arr[:, :, 0]  # all channels identical for grayscale PNG
        class_mask = np.zeros(mask_arr.shape, dtype=np.int64)
        class_mask[mask_arr == 128] = 1   # road
        class_mask[mask_arr == 255] = 2   # building
        
        return torch.from_numpy(class_mask)  # (H, W) long tensor


class MaskEvalTransforms:
    """
    Evaluation transforms for mask prediction (no augmentation).
    """
    
    def __init__(self, image_size: int = 224, mask_size: int = 256):
        self.image_size = image_size
        self.mask_size = mask_size
        
        self.normalize = transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711),
        )
    
    def __call__(self, img_A: Image.Image, img_B: Image.Image, mask: Image.Image = None):
        """Apply evaluation transforms (resize only)."""
        # Transform images
        img_A = self._transform_image(img_A)
        img_B = self._transform_image(img_B)
        
        if mask is not None:
            mask = self._transform_mask(mask)
            return img_A, img_B, mask
        
        return img_A, img_B
    
    def _transform_image(self, img):
        img = F.resize(img, (self.image_size, self.image_size), interpolation=InterpolationMode.BICUBIC)
        img = F.to_tensor(img)
        img = self.normalize(img)
        return img
    
    def _transform_mask(self, mask):
        mask = F.resize(mask, (self.mask_size, self.mask_size), interpolation=InterpolationMode.NEAREST)
        mask_arr = np.array(mask)
        if mask_arr.ndim == 3:
            mask_arr = mask_arr[:, :, 0]
        class_mask = np.zeros(mask_arr.shape, dtype=np.int64)
        class_mask[mask_arr == 128] = 1   # road
        class_mask[mask_arr == 255] = 2   # building
        return torch.from_numpy(class_mask)  # (H, W) long tensor


if __name__ == "__main__":
    # Test the transforms
    print("Testing MaskAwarePairTransforms...")
    
    # Create dummy images
    img_A = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
    img_B = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
    mask = Image.fromarray(np.random.randint(0, 2, (256, 256), dtype=np.uint8) * 255)
    
    # Train transforms
    train_transform = MaskAwarePairTransforms(image_size=224, mask_size=256)
    out_A, out_B, out_mask = train_transform(img_A, img_B, mask)
    
    print(f"Train - Image A: {out_A.shape}, Image B: {out_B.shape}, Mask: {out_mask.shape}")
    
    # Eval transforms
    eval_transform = MaskEvalTransforms(image_size=224, mask_size=256)
    out_A, out_B, out_mask = eval_transform(img_A, img_B, mask)
    
    print(f"Eval - Image A: {out_A.shape}, Image B: {out_B.shape}, Mask: {out_mask.shape}")
    
    print("✓ All transforms work correctly!")
