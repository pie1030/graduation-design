"""
Mask Dataset for DeltaVLM Change Detection Training

Loads bi-temporal images and corresponding change masks from LEVIR-MCI dataset.
"""

import os
import json
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple, Callable

# Use mask-aware transforms for synchronized augmentation
from processor_cd import MaskAwarePairTransforms, MaskEvalTransforms


class ChangeMaskDataset(Dataset):
    """
    Dataset for change mask prediction.
    
    Loads bi-temporal images (A, B) and their corresponding change masks.
    Supports both LEVIR-MCI direct loading and JSON annotation loading.
    """
    
    def __init__(
        self,
        root: str,
        split: str = "train",
        image_size: int = 224,
        mask_size: int = 256,
        is_train: bool = True,
        annotation_file: Optional[str] = None,
        mask_root: Optional[str] = None,
        label_mode: str = 'levir_mci',
    ):
        """
        Args:
            root: Root directory containing train/val/test splits (images)
            split: One of 'train', 'val', 'test'
            image_size: Size to resize input images
            mask_size: Size of output mask
            is_train: Whether this is training (affects augmentation)
            annotation_file: Optional JSON annotation file
            mask_root: Optional separate root for mask labels (if None, uses root)
                       This allows using ChangeChat images with LEVIR-MCI masks
            label_mode: 'levir_mci' (default) or 'binary' (for LEVIR-CD)
        """
        super().__init__()
        self.root = root
        self.split = split
        self.image_size = image_size
        self.mask_size = mask_size
        self.is_train = is_train
        self.mask_root = mask_root or root
        
        # Set up paths for images
        self.split_dir = os.path.join(root, split)
        self.dir_A = os.path.join(self.split_dir, "A")
        self.dir_B = os.path.join(self.split_dir, "B")
        
        # Set up path for labels (may be from different root)
        self.dir_label = os.path.join(self.mask_root, split, "label")
        
        # Get file list
        if annotation_file and os.path.exists(annotation_file):
            self.samples = self._load_from_annotation(annotation_file)
        else:
            self.samples = self._load_from_directory()
        
        # Image processor with synchronized augmentation for mask training
        # Uses paper settings: crop 0-5%, rotation ±15°
        if is_train:
            self.transform = MaskAwarePairTransforms(
                image_size=image_size,
                mask_size=mask_size,
                crop_scale=(0.95, 1.0),
                rotation_range=(-15, 15),
                flip_prob=0.5,
                label_mode=label_mode,
            )
        else:
            self.transform = MaskEvalTransforms(
                image_size=image_size,
                mask_size=mask_size,
                label_mode=label_mode,
            )
        
        print(f"ChangeMaskDataset[{split}]: {len(self.samples)} samples")
    
    def _load_from_directory(self) -> List[str]:
        """Load samples from label directory."""
        if not os.path.exists(self.dir_label):
            print(f"Warning: Label directory not found: {self.dir_label}")
            return []
        
        files = sorted([
            f for f in os.listdir(self.dir_label)
            if f.endswith('.png') or f.endswith('.jpg')
        ])
        return files
    
    def _load_from_annotation(self, annotation_file: str) -> List[Dict]:
        """Load samples from JSON annotation file."""
        with open(annotation_file, 'r') as f:
            data = json.load(f)
        
        # Filter samples that have change (changeflag == 1) for balanced training
        # Or include all samples based on your needs
        samples = []
        for item in data:
            if isinstance(item.get('image'), list) and len(item['image']) == 2:
                samples.append({
                    'A': item['image'][0],
                    'B': item['image'][1],
                    'changeflag': item.get('changeflag', 1),
                })
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def _load_mask(self, path: str) -> torch.Tensor:
        """Load and process mask image to binary format."""
        mask_img = Image.open(path)
        mask_img = mask_img.resize((self.mask_size, self.mask_size), Image.NEAREST)
        mask_arr = np.array(mask_img)
        
        # Convert to binary mask
        # LEVIR-MCI format: 0=background, 128=road change, 255=building change
        if mask_arr.ndim == 3:
            # RGB mask - any non-zero channel indicates change
            binary_mask = (mask_arr.sum(axis=2) > 0).astype(np.float32)
        else:
            # Grayscale mask
            binary_mask = (mask_arr > 0).astype(np.float32)
        
        return torch.from_numpy(binary_mask).unsqueeze(0)  # (1, H, W)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        # Get file paths
        if isinstance(sample, str):
            # Direct file name
            name = sample
            path_A = os.path.join(self.dir_A, name)
            path_B = os.path.join(self.dir_B, name)
            path_label = os.path.join(self.dir_label, name)
        else:
            # Dict from annotation
            name = os.path.basename(sample['A'])
            # Adjust paths based on annotation format
            path_A = os.path.join(self.root, sample['A'])
            path_B = os.path.join(self.root, sample['B'])
            # Infer label path
            label_name = name
            path_label = os.path.join(self.dir_label, label_name)
        
        try:
            # Load images
            img_A = Image.open(path_A).convert("RGB")
            img_B = Image.open(path_B).convert("RGB")
            
            # Load mask (as PIL Image for synchronized transform)
            if os.path.exists(path_label):
                mask_img = Image.open(path_label)
            else:
                # No mask available - create empty mask
                mask_img = Image.new('L', (self.mask_size, self.mask_size), 0)
            
            # Apply synchronized transforms (same augmentation to images and mask)
            img_A, img_B, gt_mask = self.transform(img_A, img_B, mask_img)
            
            return {
                "image_A": img_A,
                "image_B": img_B,
                "gt_mask": gt_mask,
                "name": name,
            }
        
        except Exception as e:
            print(f"Error loading sample {idx} ({name}): {e}")
            # Return next sample on error
            return self[(idx + 1) % len(self)]
    
    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """Custom collate function."""
        batch = [b for b in batch if b is not None]
        if not batch:
            return {}
        
        collated = {}
        for key in batch[0].keys():
            values = [b[key] for b in batch]
            if isinstance(values[0], torch.Tensor):
                collated[key] = torch.stack(values, dim=0)
            else:
                collated[key] = values
        return collated


class BalancedChangeMaskDataset(ChangeMaskDataset):
    """
    Balanced dataset that oversamples changed samples.
    
    Useful when change pixels are sparse.
    """
    
    def __init__(
        self,
        root: str,
        split: str = "train",
        image_size: int = 224,
        mask_size: int = 256,
        is_train: bool = True,
        annotation_file: Optional[str] = None,
        change_ratio: float = 0.5,
    ):
        super().__init__(root, split, image_size, mask_size, is_train, annotation_file)
        
        if is_train and annotation_file:
            # Separate changed and unchanged samples
            self.changed_samples = []
            self.unchanged_samples = []
            
            for sample in self.samples:
                if isinstance(sample, dict) and sample.get('changeflag', 1) == 1:
                    self.changed_samples.append(sample)
                else:
                    self.unchanged_samples.append(sample)
            
            # Compute sampling weights
            n_changed = len(self.changed_samples)
            n_unchanged = len(self.unchanged_samples)
            self.change_ratio = change_ratio
            
            print(f"BalancedDataset: {n_changed} changed, {n_unchanged} unchanged")
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.is_train and hasattr(self, 'changed_samples'):
            # Balanced sampling
            if np.random.random() < self.change_ratio and self.changed_samples:
                sample_idx = np.random.randint(len(self.changed_samples))
                sample = self.changed_samples[sample_idx]
            elif self.unchanged_samples:
                sample_idx = np.random.randint(len(self.unchanged_samples))
                sample = self.unchanged_samples[sample_idx]
            else:
                sample = self.samples[idx % len(self.samples)]
            
            # Get paths
            if isinstance(sample, str):
                name = sample
                path_A = os.path.join(self.dir_A, name)
                path_B = os.path.join(self.dir_B, name)
                path_label = os.path.join(self.dir_label, name)
            else:
                name = os.path.basename(sample['A'])
                path_A = os.path.join(self.root, sample['A'])
                path_B = os.path.join(self.root, sample['B'])
                path_label = os.path.join(self.dir_label, name)
            
            try:
                img_A = Image.open(path_A).convert("RGB")
                img_B = Image.open(path_B).convert("RGB")
                
                if os.path.exists(path_label):
                    mask_img = Image.open(path_label)
                else:
                    mask_img = Image.new('L', (self.mask_size, self.mask_size), 0)
                
                img_A, img_B, gt_mask = self.transform(img_A, img_B, mask_img)
                
                return {
                    "image_A": img_A,
                    "image_B": img_B,
                    "gt_mask": gt_mask,
                    "name": name,
                }
            except Exception as e:
                return super().__getitem__(idx)
        else:
            return super().__getitem__(idx)


def _filter_no_change(samples: List[str], label_dir: str) -> List[str]:
    """Remove samples whose label is entirely background (all zeros)."""
    kept = []
    for f in samples:
        path = os.path.join(label_dir, f)
        if os.path.exists(path):
            arr = np.array(Image.open(path))
            if arr.ndim == 3:
                arr = arr[:, :, 0]
            if arr.max() > 0:
                kept.append(f)
        else:
            kept.append(f)
    print(f"filter_no_change: {len(samples)} -> {len(kept)} "
          f"(removed {len(samples)-len(kept)} all-background samples)")
    return kept


def build_mask_dataloaders(
    data_root: str,
    batch_size: int = 8,
    num_workers: int = 4,
    image_size: int = 224,
    mask_size: int = 256,
    annotation_file: Optional[str] = None,
    mask_root: Optional[str] = None,
    filter_no_change: bool = False,
    label_mode: str = 'levir_mci',
) -> Tuple[DataLoader, DataLoader]:
    """
    Build train and validation dataloaders for mask training.
    
    Args:
        data_root: Root directory for images (LEVIR-MCI or ChangeChat)
        batch_size: Batch size
        num_workers: Number of data loading workers
        image_size: Input image size
        mask_size: Output mask size
        annotation_file: Optional JSON annotation for filtering
        mask_root: Optional separate root for mask labels
        filter_no_change: If True, remove all-background samples from train set
    
    Returns:
        (train_loader, val_loader)
    """
    train_dataset = ChangeMaskDataset(
        root=data_root,
        split="train",
        image_size=image_size,
        mask_size=mask_size,
        is_train=True,
        annotation_file=annotation_file,
        mask_root=mask_root,
        label_mode=label_mode,
    )

    if filter_no_change:
        train_dataset.samples = _filter_no_change(
            train_dataset.samples, train_dataset.dir_label,
        )
    
    val_dataset = ChangeMaskDataset(
        root=data_root,
        split="val",
        image_size=image_size,
        mask_size=mask_size,
        is_train=False,
        annotation_file=None,
        mask_root=mask_root,
        label_mode=label_mode,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=ChangeMaskDataset.collate_fn,
        pin_memory=True,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=ChangeMaskDataset.collate_fn,
        pin_memory=True,
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Test dataset loading
    import sys
    
    data_root = "/root/autodl-tmp/LEVIR-MCI-dataset/images"
    
    train_loader, val_loader = build_mask_dataloaders(
        data_root=data_root,
        batch_size=4,
        num_workers=0,
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Test one batch
    batch = next(iter(train_loader))
    print(f"Image A shape: {batch['image_A'].shape}")
    print(f"Image B shape: {batch['image_B'].shape}")
    print(f"GT Mask shape: {batch['gt_mask'].shape}")
    print(f"Names: {batch['name']}")
