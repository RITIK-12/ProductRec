"""
Central configuration for the LoViF multimodal rating prediction pipeline.

All hyperparameters, paths, and constants live here. Override via CLI or by
editing this file directly — no config is scattered across modules.
"""

import os
import random
from pathlib import Path

import numpy as np
import torch


CFG = {
    # ── Reproducibility ──
    "seed": 42,

    # ── Dataset ──
    "dataset_name": "milistu/AMAZON-Products-2023",
    "dataset_split": "train",
    "target_col": "average_rating",
    "val_ratio": 0.10,

    # ── Encoder (SigLIP2 ViT-B/16 via open_clip) ──
    "clip_model": "hf-hub:timm/ViT-B-16-SigLIP2",
    "embed_dim": 768,
    "img_size": 224,
    "unfreeze_vision_layers": 2,

    # ── Dataloader ──
    "batch_size": 512,
    "num_workers": 4,
    "download_workers": 64,
    "download_timeout": 10,

    # ── Training ──
    "epochs": 20,
    "head_lr": 3e-4,
    "weight_decay": 1e-3,
    "warmup_epochs": 2,
    "patience": 5,
    "reduce_lr_patience": 2,
    "reduce_lr_factor": 0.5,
    "reduce_lr_threshold": 0.005,
    "dropout": 0.2,
    "grad_clip": 1.0,
    "mixed_precision": True,

    # ── Head ──
    "n_classes": 5,
    "target_min": 1.0,
    "target_max": 5.0,

    # ── Loss weights ──
    "coral_weight": 1.0,
    "reg_weight": 0.2,
    "aux_weight": 0.05,
    "plcc_weight": 2.0,

    # ── W&B ──
    "wandb_project": "lovif-dino-minilm",
    "wandb_run_name": "v6_siglip2_simple_head",

    # ── Test set (HuggingFace-hosted) ──
    "test_hf_repo": "Kirin0010/CVPR_workshop_efficiencyVLM",
    "test_hf_filename": "setB/setB.zip",
}

# ── Paths (default to ./output and ./cache, override with env vars) ──
IMG_CACHE = Path(os.environ.get("LOVIF_IMG_CACHE", "./cache/images"))
SAVE_DIR = Path(os.environ.get("LOVIF_SAVE_DIR", "./output"))
TEST_ROOT = Path(os.environ.get("LOVIF_TEST_ROOT", "./cache/test_setB"))

IMG_CACHE.mkdir(parents=True, exist_ok=True)
SAVE_DIR.mkdir(parents=True, exist_ok=True)


def seed_everything(seed: int = 42) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device() -> torch.device:
    """Return CUDA device if available, else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
