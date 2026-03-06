"""
Standalone SegformerCD training on LEVIR-MCI (3-class).
Fair SOTA baseline: same dataset, same evaluator, same epochs as DeltaVLM.
Usage: python train_segformer_cd.py --epochs 100 --batch_size 16
"""
import os, sys, math, argparse, logging
from datetime import datetime
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import numpy as np
from dataset_mask import ChangeMaskDataset
from model.mask_branch.segformer_cd import SegformerCD

CLASS_NAMES = {0: "background", 1: "road", 2: "building"}

def setup_logging(d):
    os.makedirs(d, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(os.path.join(d,"train.log")), logging.StreamHandler()])
    return logging.getLogger(__name__)

class ConfusionMatrixEvaluator:
    def __init__(s, nc=3):
        s.nc = nc; s.cm = np.zeros((nc,nc))
    def reset(s): s.cm = np.zeros((s.nc,s.nc))
    def add_batch(s, gt, pred):
        g=gt.flatten().astype(int); p=pred.flatten().astype(int)
        m=(g>=0)&(g<s.nc); l=s.nc*g[m]+p[m]
        s.cm += np.bincount(l, minlength=s.nc**2).reshape(s.nc,s.nc)
    def compute_metrics(s):
        cm=s.cm; mt={}
        iou = np.diag(cm)/(cm.sum(1)+cm.sum(0)-np.diag(cm)+1e-8)
        for c in range(s.nc):
            n=CLASS_NAMES.get(c,str(c)); mt[f"iou_{n}"]=iou[c]
            tp=cm[c,c]; fp=cm[:,c].sum()-tp; fn=cm[c,:].sum()-tp
            mt[f"prec_{n}"]=tp/(tp+fp+1e-8); mt[f"rec_{n}"]=tp/(tp+fn+1e-8)
            mt[f"f1_{n}"]=2*tp/(2*tp+fp+fn+1e-8)
        mt["mIoU"]=float(np.nanmean(iou))
        mt["mIoU_change"]=float(np.nanmean(iou[1:]))
        mt["OA"]=float(np.diag(cm).sum()/(cm.sum()+1e-8))
        f1c=[mt[f"f1_{CLASS_NAMES[c]}"] for c in range(1,s.nc)]
        mt["mF1_change"]=float(np.nanmean(f1c))
        return mt

def train_epoch(model, loader, crit, opt, scaler, dev, ep):
    model.train(); tl=0
    pbar=tqdm(loader, desc=f"Epoch {ep+1} [Train]")
    for b in pbar:
        a=b["image_A"].to(dev); bb=b["image_B"].to(dev); g=b["gt_mask"].to(dev).long()
        opt.zero_grad()
        with autocast():
            o=model(a,bb,output_size=g.shape[-2:]); loss=crit(o["mask_logits"],g)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        tl+=loss.item(); pbar.set_postfix({"loss":f"{loss.item():.4f}"})
    return tl/len(loader)

@torch.no_grad()
def validate(model, loader, crit, dev, nc):
    model.eval(); tl=0; ev=ConfusionMatrixEvaluator(nc)
    for b in tqdm(loader, desc="Validating"):
        a=b["image_A"].to(dev); bb=b["image_B"].to(dev); g=b["gt_mask"].to(dev).long()
        with autocast():
            o=model(a,bb,output_size=g.shape[-2:]); loss=crit(o["mask_logits"],g)
        tl+=loss.item()
        ev.add_batch(g.cpu().numpy(), o["mask_logits"].argmax(1).cpu().numpy())
    return tl/len(loader), ev.compute_metrics()

def main():
    p=argparse.ArgumentParser(description="Train SegformerCD (3-class)")
    p.add_argument("--data_root",type=str,default="/root/autodl-tmp/LEVIR-MCI-dataset/images")
    p.add_argument("--output_dir",type=str,default="./output/segformer_mci")
    p.add_argument("--epochs",type=int,default=100)
    p.add_argument("--batch_size",type=int,default=16)
    p.add_argument("--lr",type=float,default=6e-5)
    p.add_argument("--weight_decay",type=float,default=0.01)
    p.add_argument("--image_size",type=int,default=256)
    p.add_argument("--mask_size",type=int,default=256)
    p.add_argument("--num_workers",type=int,default=4)
    p.add_argument("--num_classes",type=int,default=3)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--resume",type=str,default=None)
    a=p.parse_args()

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    ts=datetime.now().strftime("%Y%m%d_%H%M%S")
    od=os.path.join(a.output_dir,ts)
    log=setup_logging(od)
    dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {dev}"); log.info(f"Args: {a}")

    tds=ChangeMaskDataset(root=a.data_root,split="train",image_size=a.image_size,mask_size=a.mask_size,is_train=True,label_mode="levir_mci")
    vds=ChangeMaskDataset(root=a.data_root,split="val",image_size=a.image_size,mask_size=a.mask_size,is_train=False,label_mode="levir_mci")
    tl=DataLoader(tds,batch_size=a.batch_size,shuffle=True,num_workers=a.num_workers,pin_memory=True,drop_last=True)
    vl=DataLoader(vds,batch_size=a.batch_size,shuffle=False,num_workers=a.num_workers,pin_memory=True)
    log.info(f"Train: {len(tds)}, Val: {len(vds)}")

    model=SegformerCD(pretrained=True,num_bi3_layers=3,num_heads=8,hidden_dim=256,num_classes=a.num_classes).to(dev)

    cw=torch.tensor([0.5,2.0,2.0],device=dev)
    crit=nn.CrossEntropyLoss(weight=cw)
    log.info(f"Loss: CrossEntropyLoss, weights={cw.tolist()}")

    opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=a.weight_decay)
    we=5
    def lr_lam(ep):
        if ep<we: return (ep+1)/we
        return 0.5*(1+math.cos(math.pi*(ep-we)/(a.epochs-we)))
    sch=torch.optim.lr_scheduler.LambdaLR(opt,lr_lam)
    sc=GradScaler()

    se=0; bm=0
    if a.resume and os.path.exists(a.resume):
        log.info(f"Resuming from {a.resume}")
        ck=torch.load(a.resume,map_location="cpu")
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["optimizer"])
        se=ck["epoch"]+1; bm=ck.get("best_mIoU",0)

    log.info("Starting training...")
    for ep in range(se,a.epochs):
        trloss=train_epoch(model,tl,crit,opt,sc,dev,ep)
        vloss,mt=validate(model,vl,crit,dev,a.num_classes)
        sch.step()
        lr=opt.param_groups[0]["lr"]
        log.info(f"Epoch [{ep+1}/{a.epochs}] LR: {lr:.6f} Train Loss: {trloss:.4f} Val Loss: {vloss:.4f}")
        log.info(f"  mIoU(3cls): {mt['mIoU']:.4f} | IoU_road: {mt['iou_road']:.4f} | IoU_bldg: {mt['iou_building']:.4f} | mF1(change): {mt['mF1_change']:.4f} | OA: {mt['OA']:.4f}")
        if mt["mIoU"]>bm:
            bm=mt["mIoU"]
            torch.save({"model":model.state_dict(),"optimizer":opt.state_dict(),"epoch":ep,"best_mIoU":bm,"metrics":mt},os.path.join(od,"best.pth"))
            log.info(f"  *** New best mIoU: {bm:.4f} ***")
        if (ep+1)%25==0:
            torch.save({"model":model.state_dict(),"optimizer":opt.state_dict(),"epoch":ep,"best_mIoU":bm},os.path.join(od,f"ckpt_{ep+1}.pth"))
    log.info(f"Done! Best mIoU: {bm:.4f}, saved to {od}")

if __name__=="__main__":
    main()
