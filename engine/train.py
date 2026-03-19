"""
Training and evaluation loops.

Handles mixed-precision, gradient clipping, warmup scheduling, and
metric computation (PLCC at both image and product level).
"""

from collections import defaultdict

import numpy as np
import torch
from tqdm.auto import tqdm

from config import CFG


def safe_plcc(preds, targets) -> float:
    """Pearson correlation with degenerate-case guards."""
    p, t = np.asarray(preds), np.asarray(targets)
    if len(p) < 2 or p.std() < 1e-8 or t.std() < 1e-8:
        return 0.0
    v = np.corrcoef(p, t)[0, 1]
    return 0.0 if np.isnan(v) else float(v)


def product_level_plcc(preds, targets, asins) -> float:
    """
    Compute PLCC after averaging predictions per product.

    Products with multiple images have their per-image scores averaged
    before computing the correlation with the ground-truth rating.
    """
    prod_preds = defaultdict(list)
    prod_targets = {}
    for p, t, a in zip(preds, targets, asins):
        prod_preds[a].append(p)
        prod_targets[a] = t
    asin_list = sorted(prod_preds.keys())
    avg_preds = [np.mean(prod_preds[a]) for a in asin_list]
    tgts = [prod_targets[a] for a in asin_list]
    return safe_plcc(avg_preds, tgts)


@torch.no_grad()
def evaluate(model, loader, loss_fn, device) -> dict:
    """Run a full evaluation pass and return metric dict."""
    model.eval()
    losses, preds_all, targets_all, asins_all = [], [], [], []
    amp = CFG["mixed_precision"] and device.type == "cuda"

    for batch in tqdm(loader, leave=False, desc="eval"):
        imgs = batch["image"].to(device, non_blocking=True)
        tokens = batch["tokens"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
            preds = model(imgs, tokens)
            loss, _ = loss_fn(preds, targets)

        losses.append(loss.item())
        preds_all.extend(preds.cpu().tolist())
        targets_all.extend(targets.cpu().tolist())
        asins_all.extend(batch["asin"])

    plcc_img = safe_plcc(preds_all, targets_all)
    plcc_prod = product_level_plcc(preds_all, targets_all, asins_all)
    p, t = np.asarray(preds_all), np.asarray(targets_all)

    return {
        "loss": float(np.mean(losses)),
        "plcc_img": plcc_img,
        "plcc_prod": plcc_prod,
        "plcc_plus": max(0.0, plcc_prod),
        "mae": float(np.mean(np.abs(p - t))),
        "rmse": float(np.sqrt(np.mean((p - t) ** 2))),
    }


def train_one_epoch(
    model, loader, loss_fn, optimizer, warmup_scheduler, scaler, device, epoch
) -> dict:
    """
    Train for one epoch.

    Keeps frozen encoder layers in eval mode while training unfrozen
    vision blocks and the fusion head.
    """
    model.train()

    # Frozen encoder stays in eval; only unfrozen vision blocks use train mode
    model.clip.eval()
    for attr_path in ["visual.trunk.blocks", "visual.blocks"]:
        obj = model.clip
        try:
            for attr in attr_path.split("."):
                obj = getattr(obj, attr)
            blocks = list(obj)
            n = CFG.get("unfreeze_vision_layers", 2)
            for i in range(max(0, len(blocks) - n), len(blocks)):
                blocks[i].train()
            break
        except AttributeError:
            continue

    losses, preds_all, targets_all = [], [], []
    amp = CFG["mixed_precision"] and device.type == "cuda"

    for batch in tqdm(loader, leave=False, desc=f"train e{epoch}"):
        imgs = batch["image"].to(device, non_blocking=True)
        tokens = batch["tokens"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
            preds = model(imgs, tokens)
            loss, _ = loss_fn(preds, targets)

        if scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CFG["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CFG["grad_clip"])
            optimizer.step()

        warmup_scheduler.step()

        losses.append(loss.item())
        preds_all.extend(preds.detach().cpu().tolist())
        targets_all.extend(targets.detach().cpu().tolist())

    plcc = safe_plcc(preds_all, targets_all)
    p, t = np.asarray(preds_all), np.asarray(targets_all)
    return {
        "loss": float(np.mean(losses)),
        "plcc": plcc,
        "plcc_plus": max(0.0, plcc),
        "mae": float(np.mean(np.abs(p - t))),
        "rmse": float(np.sqrt(np.mean((p - t) ** 2))),
    }
