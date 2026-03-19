"""
Main training script.

Downloads the dataset, builds dataloaders, trains the SigLIP2 fusion model
with warmup + plateau LR scheduling, and saves the best checkpoint by
validation PLCC+. Optionally logs to Weights & Biases.

Usage:
    python train.py
    python train.py --no-wandb
"""

import argparse
import gc

import numpy as np
import pandas as pd
import torch
import open_clip
from datasets import load_dataset
from PIL import ImageFile
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from config import CFG, SAVE_DIR, seed_everything, get_device
from data.download import download_images
from data.dataset import PerImageDataset, collate_fn
from model.fusion import SigLIP2FusionModel
from model.loss import FusionRatingLoss
from engine.train import train_one_epoch, evaluate

ImageFile.LOAD_TRUNCATED_IMAGES = True


def parse_args():
    parser = argparse.ArgumentParser(description="Train LoViF multimodal rating model")
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")
    return parser.parse_args()


def load_and_split_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the Amazon Products dataset and perform a stratified train/val split."""
    ds = load_dataset(CFG["dataset_name"], split=CFG["dataset_split"])
    df = ds.to_pandas()
    n_raw = len(df)
    print(f"[milistu] raw: {n_raw}")

    # Keep rows with valid rating and image URL
    df = df.dropna(subset=[CFG["target_col"], "image"]).reset_index(drop=True)
    df = df[df["image"].str.strip().astype(bool)].reset_index(drop=True)
    df = df[df["image"].str.startswith("http")].reset_index(drop=True)
    print(f"[CLEAN] {n_raw} → {len(df)} products with valid rating + image")

    # Standardize to test set schema
    df = df.rename(columns={"image": "image_url"})
    final_cols = [
        "parent_asin", "image_url", "main_category",
        "title", "description", "features", "average_rating",
    ]
    for col in final_cols:
        if col not in df.columns:
            df[col] = None
    df = df[final_cols].copy()

    # Product-level stratified split
    stratify = df[CFG["target_col"]].round().astype(int)
    train_df, val_df = train_test_split(
        df,
        test_size=CFG["val_ratio"],
        random_state=CFG["seed"],
        shuffle=True,
        stratify=stratify,
    )
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    print(f"\ntrain: {len(train_df)} products/images")
    print(f"val:   {len(val_df)} products/images")

    del ds, df
    gc.collect()
    return train_df, val_df


def build_loaders(train_df, val_df, img_transform, tokenizer):
    """Create train and validation DataLoaders."""
    train_ds = PerImageDataset(train_df, img_transform, tokenizer)
    val_ds = PerImageDataset(val_df, img_transform, tokenizer)

    nw = CFG["num_workers"]
    train_loader = DataLoader(
        train_ds,
        batch_size=CFG["batch_size"],
        shuffle=True,
        num_workers=nw,
        pin_memory=True,
        collate_fn=collate_fn,
        persistent_workers=(nw > 0),
        prefetch_factor=2 if nw > 0 else None,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=CFG["batch_size"],
        shuffle=False,
        num_workers=nw,
        pin_memory=True,
        collate_fn=collate_fn,
        persistent_workers=(nw > 0),
        prefetch_factor=2 if nw > 0 else None,
    )

    print(f"train: {len(train_ds)} samples, {len(train_loader)} batches")
    print(f"val:   {len(val_ds)} samples, {len(val_loader)} batches")
    return train_loader, val_loader


def main():
    args = parse_args()
    seed_everything(CFG["seed"])
    device = get_device()
    print(f"device = {device}")

    # ── Data ──
    train_df, val_df = load_and_split_data()
    print("\n═══ TRAIN ═══")
    train_df = download_images(train_df)
    print("\n═══ VAL ═══")
    val_df = download_images(val_df)

    # ── Encoder ──
    clip_model, img_transform = open_clip.create_model_from_pretrained(CFG["clip_model"])
    clip_tokenizer = open_clip.get_tokenizer(CFG["clip_model"])
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad = False
    clip_model = clip_model.to(device)

    total_params = sum(p.numel() for p in clip_model.parameters())
    print(f"[SigLIP2] {total_params / 1e6:.1f}M params (all frozen)")

    # ── Dataloaders ──
    train_loader, val_loader = build_loaders(train_df, val_df, img_transform, clip_tokenizer)

    # ── Model + Loss ──
    model = SigLIP2FusionModel(clip_model, CFG).to(device)
    loss_fn = FusionRatingLoss(CFG)

    # ── Optimizer: low LR for encoder, high LR for fusion+head ──
    encoder_params, head_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("clip."):
            encoder_params.append(p)
        else:
            head_params.append(p)

    print(f"Encoder trainable: {sum(p.numel() for p in encoder_params) / 1e6:.2f}M")
    print(f"Head trainable:    {sum(p.numel() for p in head_params) / 1e6:.2f}M")

    optimizer = torch.optim.AdamW([
        {"params": encoder_params, "lr": 5e-6, "weight_decay": CFG["weight_decay"]},
        {"params": head_params, "lr": CFG["head_lr"], "weight_decay": CFG["weight_decay"]},
    ])

    warmup_steps = CFG["warmup_epochs"] * len(train_loader)

    def warmup_lambda(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        return 1.0

    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lambda)

    plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=CFG["reduce_lr_factor"],
        patience=CFG["reduce_lr_patience"],
        threshold=CFG["reduce_lr_threshold"],
        threshold_mode="abs",
        min_lr=1e-7,
    )

    scaler = (
        torch.amp.GradScaler("cuda")
        if CFG["mixed_precision"] and device.type == "cuda"
        else None
    )

    # ── W&B ──
    use_wandb = not args.no_wandb
    if use_wandb:
        import wandb
        wandb.login()
        run = wandb.init(
            project=CFG["wandb_project"], name=CFG["wandb_run_name"], config=CFG
        )

    # ── Training loop ──
    best = -1.0
    best_path = SAVE_DIR / "best_model.pt"
    patience_counter = 0

    print(f"\nbatch={CFG['batch_size']}  steps/epoch={len(train_loader)}")
    print(f"warmup: {CFG['warmup_epochs']} epochs ({warmup_steps} steps)")

    for epoch in range(1, CFG["epochs"] + 1):
        train_m = train_one_epoch(
            model, train_loader, loss_fn, optimizer, warmup_scheduler, scaler, device, epoch
        )
        val_m = evaluate(model, val_loader, loss_fn, device)

        plateau_scheduler.step(val_m["plcc_plus"])

        current_lrs = {
            f"lr_g{i}": optimizer.param_groups[i]["lr"]
            for i in range(len(optimizer.param_groups))
        }

        if use_wandb:
            log = {
                "epoch": epoch,
                **{f"train/{k}": v for k, v in train_m.items()},
                **{f"val/{k}": v for k, v in val_m.items()},
                **current_lrs,
            }
            wandb.log(log)

        gap = train_m["plcc_plus"] - val_m["plcc_plus"]
        print(
            f"E{epoch:02d} | train_plcc={train_m['plcc_plus']:.4f} "
            f"val_plcc+={val_m['plcc_plus']:.4f} gap={gap:.4f} | "
            f"val_loss={val_m['loss']:.4f} | "
            f"lr_v={current_lrs['lr_g0']:.1e} lr_h={current_lrs['lr_g1']:.1e}"
        )

        if val_m["plcc_plus"] > best + 1e-5:
            best = val_m["plcc_plus"]
            patience_counter = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": CFG,
                    "epoch": epoch,
                    "val_metrics": val_m,
                },
                best_path,
            )
            print(f"  ★ saved (plcc_plus={best:.4f})")
        else:
            patience_counter += 1
            print(f"  — no improvement ({patience_counter}/{CFG['patience']})")

        if patience_counter >= CFG["patience"]:
            print(f"\nEarly stopping at epoch {epoch}")
            break

    if use_wandb:
        wandb.finish()

    print(f"\n{'=' * 50}")
    print(f"Best val plcc_plus: {best:.4f}")
    print(f"Checkpoint: {best_path}")


if __name__ == "__main__":
    main()
