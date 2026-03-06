"""Qualitative comparison visualization for DeltaVLM SOTA experiments.

Generates per-sample comparison panels and FP/FN error maps.
Run after baselines (Change-Agent, SegformerCD) have finished training.

Usage:
    python compareViz.py
        --deltavlm    experiments/cd_only_baseline_v2/mask_branch_best.pth
        --change_agent output/mask_branch_change_agent/XXX/mask_branch_best.pth
        --segformer   output/segformer_mci/XXX/best.pth
        --num_samples 20
"""
import argparse, os, sys, numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.patches as mpatches
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset_mask import ChangeMaskDataset

CLASS_COLORS = {0: (220,220,220), 1: (255,165,0), 2: (65,105,225)}
CLASS_NAMES = {0: "bg", 1: "road", 2: "building"}
FPFN = {"tp":(50,205,50), "fp":(220,20,60), "fn":(30,144,255), "tn":(245,245,245)}

def denorm(t):
    mean = np.array([0.48145466, 0.4578275, 0.40821073])
    std  = np.array([0.26862954, 0.26130258, 0.27577711])
    img  = t.permute(1,2,0).numpy() * std + mean
    return np.clip(img * 255, 0, 255).astype(np.uint8)

def classmap_rgb(cm):
    rgb = np.zeros((*cm.shape, 3), dtype=np.uint8)
    for k, v in CLASS_COLORS.items(): rgb[cm==k] = v
    return rgb

def fpfn_rgb(pred, gt, cls):
    rgb = np.full((*pred.shape, 3), FPFN["tn"], dtype=np.uint8)
    rgb[(pred==cls)&(gt==cls)] = FPFN["tp"]
    rgb[(pred==cls)&(gt!=cls)] = FPFN["fp"]
    rgb[(pred!=cls)&(gt==cls)] = FPFN["fn"]
    return rgb

def align_pred(pred, gt):
    if pred.shape == gt.shape: return pred
    img = Image.fromarray(pred.astype(np.uint8))
    img = img.resize((gt.shape[1], gt.shape[0]), Image.NEAREST)
    return np.array(img)

class DeltaVLMPredictor:
    def __init__(self, ckpt, decoder_type="delta_cd", num_classes=3, device="cuda"):
        from model.blip2_vicua_mask import Blip2VicunaMask
        self.device = torch.device(device)
        hidden_dim = 512 if decoder_type == "change_agent" else 256
        self.model = Blip2VicunaMask(
            vit_model="eva_clip_g", img_size=224, drop_path_rate=0,
            use_grad_checkpoint=False, vit_precision="fp16", freeze_vit=True,
            num_query_token=32, llm_model="./vicuna-7b-v1.5", prompt="",
            max_txt_len=128, enable_mask_branch=True,
            mask_hidden_dim=hidden_dim, mask_num_stages=4, mask_output_size=(256, 256),
            freeze_for_mask_training=True, mask_training_mode=True,
            mask_decoder_type=decoder_type, num_classes=num_classes,
        )
        c = torch.load(ckpt, map_location="cpu", weights_only=False)
        if isinstance(c, dict) and "mask_decoder" in c:
            self.model.mask_decoder.load_state_dict(c["mask_decoder"], strict=False)
        else:
            self.model.load_state_dict(c, strict=False)
        self.model = self.model.to(self.device).eval()
    @torch.no_grad()
    def predict(self, a, b):
        out = self.model.forward_mask(a.to(self.device), b.to(self.device), None)
        logits = out["mask_logits"] if "mask_logits" in out else out["logits"]
        return logits.argmax(1).cpu()

class SegformerPredictor:
    def __init__(self, ckpt, num_classes=3, device="cuda"):
        from model.mask_branch.segformer_cd import SegformerCD
        self.device = torch.device(device)
        self.model = SegformerCD(pretrained=False, num_classes=num_classes)
        c = torch.load(ckpt, map_location="cpu")
        self.model.load_state_dict(c.get("model", c), strict=True)
        self.model = self.model.to(self.device).eval()
    @torch.no_grad()
    def predict(self, a, b):
        out = self.model(a.to(self.device), b.to(self.device), output_size=(256, 256))
        return out["mask_logits"].argmax(1).cpu()

def fig_comparison(img_a, img_b, gt, preds, name):
    methods = list(preds.keys())
    n = max(3, len(methods))
    fig, axes = plt.subplots(2, n, figsize=(n*3.5, 7))
    def show(ax, img, t): ax.imshow(img); ax.set_title(t, fontsize=9, fontweight="bold"); ax.axis("off")
    show(axes[0,0], img_a, "Image A (before)")
    show(axes[0,1], img_b, "Image B (after)")
    show(axes[0,2], classmap_rgb(gt), "Ground Truth")
    for i in range(3, n): axes[0,i].axis("off")
    for j, m in enumerate(methods): show(axes[1,j], classmap_rgb(preds[m]), m)
    for j in range(len(methods), n): axes[1,j].axis("off")
    patches = [mpatches.Patch(color=np.array(c)/255, label=CLASS_NAMES[i]) for i,c in CLASS_COLORS.items()]
    fig.legend(handles=patches, loc="lower center", ncol=3, fontsize=9, bbox_to_anchor=(0.5,-0.02))
    fig.suptitle(name, fontsize=11)
    fig.tight_layout()
    return fig

def fig_fpfn(gt, preds, name, change_classes=(1,2)):
    rows = len(change_classes); cols = 1 + len(preds)
    fig, axes = plt.subplots(rows, cols, figsize=(cols*3.5, rows*3.5))
    if rows == 1: axes = axes[np.newaxis, :]
    for r, cls in enumerate(change_classes):
        axes[r,0].imshow(classmap_rgb(gt)); axes[r,0].set_title(f"GT [{CLASS_NAMES.get(cls,cls)}]", fontsize=9); axes[r,0].axis("off")
        for c, (m, pred) in enumerate(preds.items(), 1):
            axes[r,c].imshow(fpfn_rgb(pred, gt, cls)); axes[r,c].set_title(m, fontsize=9); axes[r,c].axis("off")
    legend = [mpatches.Patch(color=np.array(FPFN[k])/255, label=k.upper()) for k in ("tp","fp","fn","tn")]
    fig.legend(handles=legend, loc="lower center", ncol=4, fontsize=9, bbox_to_anchor=(0.5,-0.02))
    fig.suptitle(f"FP/FN Analysis - {name}", fontsize=11)
    fig.tight_layout()
    return fig

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--deltavlm",     default=None)
    p.add_argument("--change_agent", default=None)
    p.add_argument("--segformer",    default=None)
    p.add_argument("--data_root",    default="/root/autodl-tmp/LEVIR-MCI-dataset/images")
    p.add_argument("--split",        default="test")
    p.add_argument("--output_dir",   default="./output/comparison")
    p.add_argument("--num_samples",  type=int, default=20)
    p.add_argument("--image_size",   type=int, default=224)
    p.add_argument("--mask_size",    type=int, default=256)
    p.add_argument("--seed",         type=int, default=0)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds = ChangeMaskDataset(args.data_root, args.split, args.image_size, args.mask_size, is_train=False)
    np.random.seed(args.seed)
    idxs = np.sort(np.random.choice(len(ds), min(args.num_samples, len(ds)), replace=False))
    print(f"Visualising {len(idxs)} / {len(ds)} samples")

    predictors = {}
    if args.deltavlm     and os.path.exists(args.deltavlm):     predictors["DeltaVLM"]     = DeltaVLMPredictor(args.deltavlm,     "delta_cd",     3, device)
    if args.change_agent and os.path.exists(args.change_agent): predictors["Change-Agent"] = DeltaVLMPredictor(args.change_agent, "change_agent", 3, device)
    if args.segformer    and os.path.exists(args.segformer):    predictors["SegformerCD"]  = SegformerPredictor(args.segformer,  3, device)
    if not predictors: print("No checkpoints found."); return
    print(f"Loaded: {list(predictors.keys())}")

    panels_dir  = os.path.join(args.output_dir, "panels")
    errmaps_dir = os.path.join(args.output_dir, "errorMaps")
    os.makedirs(panels_dir, exist_ok=True); os.makedirs(errmaps_dir, exist_ok=True)

    for rank, idx in enumerate(idxs):
        sample = ds[idx]
        a_t = sample["image_A"].unsqueeze(0); b_t = sample["image_B"].unsqueeze(0)
        gt = sample["gt_mask"].numpy(); name = sample["name"]
        preds = {}
        for m, pred_fn in predictors.items():
            pred_np = pred_fn.predict(a_t, b_t)[0].numpy()
            preds[m] = align_pred(pred_np, gt)
        stem = f"{rank:03d}_{os.path.splitext(name)[0]}"
        fig = fig_comparison(denorm(sample["image_A"]), denorm(sample["image_B"]), gt, preds, name)
        fig.savefig(os.path.join(panels_dir, f"{stem}_panel.png"), dpi=120, bbox_inches="tight"); plt.close(fig)
        fig = fig_fpfn(gt, preds, name)
        fig.savefig(os.path.join(errmaps_dir, f"{stem}_fpfn.png"), dpi=120, bbox_inches="tight"); plt.close(fig)
        print(f"  [{rank+1}/{len(idxs)}] {name}")
    print(f"Done. panels->{panels_dir}  errorMaps->{errmaps_dir}")

if __name__ == "__main__":
    main()
