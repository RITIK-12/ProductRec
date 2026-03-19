"""
Evaluation script: loads a trained checkpoint, downloads the test set,
runs inference, and produces a submission CSV.

Usage:
    python evaluate.py --checkpoint output/best_model.pt
    python evaluate.py --checkpoint output/best_model.pt --output submission.csv
"""

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import open_clip
from PIL import ImageFile
from torch.utils.data import DataLoader

from config import CFG, SAVE_DIR, seed_everything, get_device
from data.download import download_test_set
from data.preprocess import stem_to_asin
from data.dataset import TestPerImageDataset, test_collate_fn
from model.fusion import SigLIP2FusionModel
from engine.inference import predict, generate_submission

ImageFile.LOAD_TRUNCATED_IMAGES = True


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate LoViF model on test set")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(SAVE_DIR / "best_model.pt"),
        help="Path to model checkpoint (.pt file)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(SAVE_DIR / "submission.csv"),
        help="Output path for submission CSV",
    )
    parser.add_argument(
        "--compute-flops",
        action="store_true",
        help="Run FLOPs analysis (requires fvcore)",
    )
    return parser.parse_args()


def compute_model_stats(model, test_loader, device) -> tuple[float, float]:
    """Compute total parameters (M) and FLOPs (G) for the submission."""
    total_params_M = sum(p.numel() for p in model.parameters()) / 1e6

    try:
        from fvcore.nn import FlopCountAnalysis, flop_count_table

        model.eval()
        sample_batch = next(iter(test_loader))
        canonical_img = sample_batch["image"][0:1].to(device)
        canonical_tokens = sample_batch["tokens"][0:1].to(device)

        with torch.no_grad():
            flops_counter = FlopCountAnalysis(model, (canonical_img, canonical_tokens))
            total_gflops = flops_counter.total() / 1e9

        print("--- FLOP Analysis ---")
        print(flop_count_table(flops_counter))
    except ImportError:
        print("[WARNING] fvcore not installed, using default FLOPs estimate.")
        total_gflops = 20.676  # Pre-computed for SigLIP2 ViT-B/16

    print(f"\nparams = {total_params_M:.1f}M   flops = {total_gflops:.3f}G")
    return total_params_M, total_gflops


def main():
    args = parse_args()
    seed_everything(CFG["seed"])
    device = get_device()
    print(f"device = {device}")

    # ── Download and prepare test set ──
    test_csv_path, test_images_dir = download_test_set()
    test_df = pd.read_csv(test_csv_path)
    print(f"Test CSV: {test_df.shape}")

    # Map ASINs to their image files on disk
    asin_to_files = defaultdict(list)
    for f in test_images_dir.iterdir():
        if f.is_file():
            asin_to_files[stem_to_asin(f.name)].append(str(f))
    for asin in asin_to_files:
        asin_to_files[asin].sort()

    csv_asins = set(test_df["parent_asin"].astype(str))
    disk_asins = set(asin_to_files.keys())
    missing = csv_asins - disk_asins
    assert len(missing) == 0, f"{len(missing)} products in CSV have no images on disk!"
    print(f"Matched: {len(csv_asins & disk_asins)} products")

    # ── Load encoder + tokenizer ──
    clip_model, img_transform = open_clip.create_model_from_pretrained(CFG["clip_model"])
    clip_tokenizer = open_clip.get_tokenizer(CFG["clip_model"])
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad = False
    clip_model = clip_model.to(device)

    # ── Reconstruct model and load checkpoint ──
    model = SigLIP2FusionModel(clip_model, CFG).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    print(f"Loaded epoch {ckpt['epoch']} | plcc_plus={ckpt['val_metrics']['plcc_plus']:.4f}")

    # ── Build test dataloader ──
    test_ds = TestPerImageDataset(test_df, img_transform, clip_tokenizer, asin_to_files)
    test_loader = DataLoader(
        test_ds,
        batch_size=CFG["batch_size"],
        shuffle=False,
        num_workers=CFG["num_workers"],
        pin_memory=True,
        collate_fn=test_collate_fn,
        persistent_workers=(CFG["num_workers"] > 0),
        prefetch_factor=2 if CFG["num_workers"] > 0 else None,
    )
    print(f"Batches: {len(test_loader)}")

    # ── Compute model stats ──
    if args.compute_flops:
        total_params_M, total_gflops = compute_model_stats(model, test_loader, device)
    else:
        total_params_M = sum(p.numel() for p in model.parameters()) / 1e6
        total_gflops = 20.676  # Pre-computed for this architecture
        print(f"params = {total_params_M:.1f}M   flops = {total_gflops:.3f}G (pre-computed)")

    # ── Inference ──
    raw_ids, raw_scores = predict(model, test_loader, device)
    print(f"Raw predictions: {len(raw_scores)}")

    # ── Generate submission ──
    submission = generate_submission(
        raw_ids, raw_scores, test_df,
        total_params_M, total_gflops,
        args.output,
    )
    print(submission.head(10))


if __name__ == "__main__":
    main()
