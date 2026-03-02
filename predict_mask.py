"""
Inference script for DeltaVLM Mask Branch

Predicts change masks from bi-temporal images.

Usage:
    python predict_mask.py --image_A path/to/before.png --image_B path/to/after.png --output mask.png
"""

import argparse
import os
import sys

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from processor import BlipImageEvalProcessor


def parse_args():
    parser = argparse.ArgumentParser(description="Predict change mask")
    parser.add_argument("--image_A", type=str, required=True,
                       help="Path to before image")
    parser.add_argument("--image_B", type=str, required=True,
                       help="Path to after image")
    parser.add_argument("--output", type=str, default="change_mask.png",
                       help="Output path for predicted mask")
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="Path to model checkpoint")
    parser.add_argument("--mask_branch", type=str, default=None,
                       help="Path to mask branch weights")
    parser.add_argument("--threshold", type=float, default=0.5,
                       help="Binarization threshold")
    parser.add_argument("--visualize", action="store_true",
                       help="Save visualization with original images")
    parser.add_argument("--image_size", type=int, default=224,
                       help="Input image size")
    parser.add_argument("--mask_size", type=int, default=256,
                       help="Output mask size")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use (cuda/cpu)")
    return parser.parse_args()


def load_model(checkpoint_path, mask_branch_path=None, device='cuda'):
    """Load model with mask branch."""
    from model.blip2_vicua_mask import Blip2VicunaMask
    
    # Initialize model
    model = Blip2VicunaMask.from_pretrained(
        checkpoint_path,
        enable_mask_branch=True,
        freeze_for_mask_training=False,
    )
    
    # Load mask branch weights if provided
    if mask_branch_path and os.path.exists(mask_branch_path):
        model.load_mask_branch(mask_branch_path)
        print(f"Loaded mask branch from {mask_branch_path}")
    
    model = model.to(device)
    model.eval()
    return model


def preprocess_images(image_A_path, image_B_path, image_size=224):
    """Load and preprocess images."""
    processor = BlipImageEvalProcessor(image_size=image_size)
    
    img_A = Image.open(image_A_path).convert("RGB")
    img_B = Image.open(image_B_path).convert("RGB")
    
    # Get original size for later resizing
    orig_size = img_A.size  # (W, H)
    
    # Process
    img_A_tensor, img_B_tensor = processor(img_A, img_B)
    
    # Add batch dimension
    img_A_tensor = img_A_tensor.unsqueeze(0)
    img_B_tensor = img_B_tensor.unsqueeze(0)
    
    return img_A_tensor, img_B_tensor, orig_size


def predict_mask(model, image_A, image_B, threshold=0.5, device='cuda'):
    """Predict change mask."""
    with torch.no_grad():
        image_A = image_A.to(device)
        image_B = image_B.to(device)
        
        # Predict
        mask = model.predict_mask(image_A, image_B, threshold=threshold)
        
        # Convert to numpy
        mask = mask[0, 0].cpu().numpy()  # (H, W)
    
    return mask


def save_visualization(image_A_path, image_B_path, mask, output_path):
    """Save visualization with original images and predicted mask."""
    img_A = Image.open(image_A_path).convert("RGB")
    img_B = Image.open(image_B_path).convert("RGB")
    
    # Resize mask to match original image size
    mask_resized = Image.fromarray((mask * 255).astype(np.uint8))
    mask_resized = mask_resized.resize(img_A.size, Image.NEAREST)
    
    # Create figure
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    axes[0].imshow(img_A)
    axes[0].set_title("Before (T1)")
    axes[0].axis('off')
    
    axes[1].imshow(img_B)
    axes[1].set_title("After (T2)")
    axes[1].axis('off')
    
    axes[2].imshow(mask_resized, cmap='gray')
    axes[2].set_title("Predicted Mask")
    axes[2].axis('off')
    
    # Overlay
    overlay = np.array(img_B).copy()
    mask_arr = np.array(mask_resized)
    overlay[mask_arr > 127] = [255, 0, 0]  # Red for changes
    axes[3].imshow(overlay)
    axes[3].set_title("Overlay")
    axes[3].axis('off')
    
    plt.tight_layout()
    
    # Save
    vis_path = output_path.replace('.png', '_vis.png')
    plt.savefig(vis_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Visualization saved to {vis_path}")


def main():
    args = parse_args()
    
    # Check inputs
    if not os.path.exists(args.image_A):
        print(f"Error: Image A not found: {args.image_A}")
        return
    if not os.path.exists(args.image_B):
        print(f"Error: Image B not found: {args.image_B}")
        return
    
    # Device
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load model
    checkpoint = args.checkpoint or "./checkpoint_best.pth"
    if not os.path.exists(checkpoint):
        print(f"Error: Checkpoint not found: {checkpoint}")
        print("Please provide a valid checkpoint path with --checkpoint")
        return
    
    print("Loading model...")
    model = load_model(checkpoint, args.mask_branch, device)
    
    # Preprocess images
    print("Preprocessing images...")
    img_A, img_B, orig_size = preprocess_images(
        args.image_A, args.image_B, args.image_size
    )
    
    # Predict
    print("Predicting mask...")
    mask = predict_mask(model, img_A, img_B, args.threshold, device)
    
    # Resize to mask_size
    mask_img = Image.fromarray((mask * 255).astype(np.uint8))
    
    # Save mask
    mask_img.save(args.output)
    print(f"Mask saved to {args.output}")
    
    # Save visualization if requested
    if args.visualize:
        save_visualization(args.image_A, args.image_B, mask, args.output)
    
    # Print statistics
    change_ratio = mask.mean() * 100
    print(f"Change ratio: {change_ratio:.2f}%")


if __name__ == "__main__":
    main()

