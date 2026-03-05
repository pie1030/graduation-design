"""
Comprehensive CD Analysis Script for DeltaVLM.
Generates: prediction gallery, gate maps, HR features, error maps, confusion matrix.
Usage: python analyze_cd.py
"""
import argparse
import os
import sys
import glob
import logging

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset_mask import build_mask_dataloaders
from model.blip2_vicua_mask import Blip2VicunaMask

CLASS_COLORS = np.array([[0, 0, 0], [255, 255, 0], [255, 0, 0]], dtype=np.uint8)
MEAN = np.array([0.48145466, 0.4578275, 0.40821073])
STD = np.array([0.26862954, 0.26130258, 0.27577711])


def denorm(t):
    img = t.cpu().numpy().transpose(1, 2, 0) * STD + MEAN
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def color_mask(m):
    rgb = np.zeros((*m.shape, 3), dtype=np.uint8)
    for c in range(3):
        rgb[m == c] = CLASS_COLORS[c]
    return rgb


def error_map(pred, gt):
    e = np.zeros((*gt.shape, 3), dtype=np.uint8)
    e[(pred > 0) & (gt > 0)] = [0, 200, 0]
    e[(pred > 0) & (gt == 0)] = [255, 60, 60]
    e[(pred == 0) & (gt > 0)] = [60, 60, 255]
    return e


def vis_sample(idx, iA, iB, gt, pr, gate_a, gate_b, sem_A, sem_B,
               hr_A, hr_B, save_dir):
    A, B = denorm(iA), denorm(iB)
    gn, pn = gt.cpu().numpy(), pr.cpu().numpy()

    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    for ax, title, img in zip(
        axes,
        ["Image A", "Image B", "GT", "Prediction", "Error Map"],
        [A, B, color_mask(gn), color_mask(pn), error_map(pn, gn)],
    ):
        ax.imshow(img)
        ax.set_title(title, fontsize=14)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, f"s{idx:03d}_gallery.png"),
        dpi=150, bbox_inches="tight",
    )
    plt.close()

    if gate_a is not None:
        ga, gb = gate_a[0], gate_b[0]
        H = int(np.sqrt(ga.shape[0]))
        gam = ga.mean(-1).reshape(H, H).numpy()
        gbm = gb.mean(-1).reshape(H, H).numpy()
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        axes[0].imshow(gam, cmap="viridis", vmin=0, vmax=1)
        axes[0].set_title("Gate A")
        axes[1].imshow(gbm, cmap="viridis", vmin=0, vmax=1)
        axes[1].set_title("Gate B")
        axes[2].imshow(np.abs(gbm - gam), cmap="hot")
        axes[2].set_title("|Gate B - A|")
        axes[3].hist(ga.numpy().flatten(), bins=50, alpha=0.6,
                     label="A", color="#4a90d9")
        axes[3].hist(gb.numpy().flatten(), bins=50, alpha=0.6,
                     label="B", color="#e8a055")
        axes[3].legend()
        axes[3].set_xlim(0, 1)
        axes[3].set_title("Gate Distribution")
        for ax in axes[:3]:
            ax.axis("off")
        plt.tight_layout()
        plt.savefig(
            os.path.join(save_dir, f"s{idx:03d}_gates.png"),
            dpi=150, bbox_inches="tight",
        )
        plt.close()

    if sem_A is not None:
        sa = sem_A[0].mean(0).numpy()
        sb = sem_B[0].mean(0).numpy()
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(sa, cmap="viridis")
        axes[0].set_title("Semantic A")
        axes[1].imshow(sb, cmap="viridis")
        axes[1].set_title("Semantic B")
        axes[2].imshow(np.abs(sb - sa), cmap="hot")
        axes[2].set_title("|Sem B - A|")
        for ax in axes:
            ax.axis("off")
        plt.tight_layout()
        plt.savefig(
            os.path.join(save_dir, f"s{idx:03d}_semantic.png"),
            dpi=150, bbox_inches="tight",
        )
        plt.close()

    if hr_A is not None:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        for j, s in enumerate(["56x56", "28x28", "14x14"]):
            ha = hr_A[j][0].mean(0).numpy()
            hd = np.abs(hr_B[j][0].mean(0).numpy() - ha)
            axes[0, j].imshow(ha, cmap="viridis")
            axes[0, j].set_title(f"HR_A {s}")
            axes[0, j].axis("off")
            axes[1, j].imshow(hd, cmap="hot")
            axes[1, j].set_title(f"|HR diff| {s}")
            axes[1, j].axis("off")
        plt.tight_layout()
        plt.savefig(
            os.path.join(save_dir, f"s{idx:03d}_hr.png"),
            dpi=150, bbox_inches="tight",
        )
        plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--data_root",
                        default="/root/autodl-tmp/LEVIR-MCI-dataset/images")
    parser.add_argument("--output_dir", default="./output/analysis")
    parser.add_argument("--num_vis", type=int, default=20)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.checkpoint is None:
        ckpts = sorted(glob.glob("output/delta_cd_v2/*/mask_branch_best.pth"))
        args.checkpoint = ckpts[-1] if ckpts else None
    if not args.checkpoint:
        print("No checkpoint found")
        return

    logging.basicConfig(level=logging.INFO)
    device = torch.device("cuda")

    model = Blip2VicunaMask(
        enable_mask_branch=True,
        mask_decoder_type="delta_cd",
        freeze_for_mask_training=True,
        mask_training_mode=True,
        num_classes=3,
        mask_output_size=(256, 256),
    ).to(device)
    sd = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(sd, strict=False)
    model.eval()
    logging.info(f"Loaded: {args.checkpoint}")

    _, val_loader = build_mask_dataloaders(
        args.data_root, batch_size=1, num_workers=2,
    )

    all_p, all_g = [], []
    vis_count = 0
    cd = model.mask_decoder

    _orig_csrm = type(cd.csrm).forward
    _orig_sem = type(cd.sem_adapter).forward
    _orig_hr = type(cd.hr_encoder).forward

    with torch.no_grad():
        for batch in val_loader:
            iA = batch["image_A"].to(device)
            iB = batch["image_B"].to(device)
            gt_mask = batch["gt_mask"].to(device)

            cap = {}

            def _csrm_hook(self, fA, fB, _c=cap):
                B, C, H, W = fA.shape
                a = fA.flatten(2).transpose(1, 2)
                b = fB.flatten(2).transpose(1, 2)
                d = b - a
                _c["ga"] = torch.sigmoid(
                    self.gate_diff(d) + self.gate_feat(a)
                ).cpu()
                _c["gb"] = torch.sigmoid(
                    self.gate_diff(d) + self.gate_feat(b)
                ).cpu()
                return _orig_csrm(self, fA, fB)

            def _sem_hook(self, x, _c=cap):
                out = _orig_sem(self, x)
                if "sA" not in _c:
                    _c["sA"] = out.cpu()
                else:
                    _c["sB"] = out.cpu()
                return out

            def _hr_hook(self, x, _c=cap):
                out = _orig_hr(self, x)
                if "hA" not in _c:
                    _c["hA"] = [f.cpu() for f in out]
                else:
                    _c["hB"] = [f.cpu() for f in out]
                return out

            type(cd.csrm).forward = _csrm_hook
            type(cd.sem_adapter).forward = _sem_hook
            type(cd.hr_encoder).forward = _hr_hook

            out = model.forward_mask(iA, iB, gt_mask)

            type(cd.csrm).forward = _orig_csrm
            type(cd.sem_adapter).forward = _orig_sem
            type(cd.hr_encoder).forward = _orig_hr

            pc = out["mask_cls"]
            all_p.append(pc.cpu().numpy())
            all_g.append(gt_mask.cpu().numpy())

            if vis_count < args.num_vis:
                vis_sample(
                    vis_count, iA[0], iB[0], gt_mask[0], pc[0],
                    cap.get("ga"), cap.get("gb"),
                    cap.get("sA"), cap.get("sB"),
                    cap.get("hA"), cap.get("hB"),
                    args.output_dir,
                )
                vis_count += 1

    all_p = np.concatenate(all_p)
    all_g = np.concatenate(all_g)

    from train_mask import ConfusionMatrixEvaluator
    ev = ConfusionMatrixEvaluator(3)
    ev.add_batch(all_g, all_p)
    metrics = ev.compute_metrics()
    logging.info("=== Validation Metrics ===")
    for k in sorted(metrics):
        logging.info(f"  {k}: {metrics[k]:.4f}")

    cm = ev.confusion_matrix
    fig, ax = plt.subplots(figsize=(6, 5))
    norm_cm = cm / (cm.sum(axis=1, keepdims=True) + 1e-8)
    ax.imshow(norm_cm, cmap="Blues", vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            pct = cm[i, j] / (cm[i].sum() + 1e-8) * 100
            ax.text(j, i, f"{int(cm[i, j]):,}\n({pct:.1f}%)",
                    ha="center", va="center", fontsize=9)
    labels = ["bg", "road", "bldg"]
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground Truth")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "confusion_matrix.png"), dpi=150)
    plt.close()

    logging.info(f"All outputs saved to {args.output_dir}")


if __name__ == "__main__":
    main()
