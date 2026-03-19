"""
Inference and submission generation for the held-out test set.

Handles multi-image-per-product aggregation and produces a CSV in the
expected submission format (item_id, score, params, flops).
"""

from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from config import CFG


@torch.no_grad()
def predict(model, loader, device) -> tuple[list, list]:
    """
    Run inference over the test loader.

    Returns raw per-image (item_id, score) lists before product-level aggregation.
    """
    model.eval()
    all_ids, all_scores = [], []
    amp = CFG["mixed_precision"] and device.type == "cuda"

    for batch in tqdm(loader, desc="Inference"):
        imgs = batch["image"].to(device, non_blocking=True)
        tokens = batch["tokens"].to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
            scores = model(imgs, tokens).cpu().tolist()
        all_ids.extend(batch["item_id"])
        all_scores.extend(scores)

    return all_ids, all_scores


def generate_submission(
    raw_ids: list,
    raw_scores: list,
    test_df: pd.DataFrame,
    total_params_M: float,
    total_gflops: float,
    output_path: str,
) -> pd.DataFrame:
    """
    Aggregate per-image scores to product level and write submission CSV.

    Each product's final score is the mean of its per-image predictions,
    matching the multi-image test set format.
    """
    # Average scores per product
    score_groups = defaultdict(list)
    for item_id, score in zip(raw_ids, raw_scores):
        score_groups[item_id].append(score)

    final_ids, final_scores = [], []
    for asin in test_df["parent_asin"].astype(str):
        scores_list = score_groups[asin]
        assert len(scores_list) > 0, f"No predictions for {asin}"
        final_ids.append(asin)
        final_scores.append(float(np.mean(scores_list)))

    print(f"Final: {len(final_scores)} products")
    print(f"Range: [{min(final_scores):.3f}, {max(final_scores):.3f}]")
    print(f"Mean:  {np.mean(final_scores):.3f}")

    # Build submission dataframe
    submission = pd.DataFrame({
        "item_id": final_ids,
        "score": [round(s, 4) for s in final_scores],
        "params": [round(total_params_M, 1)] * len(final_scores),
        "flops": [round(total_gflops, 3)] * len(final_scores),
    })

    # Sanity checks
    assert len(submission) == len(test_df), "Row count mismatch!"
    assert submission["score"].notna().all(), "NaN scores!"
    assert submission["score"].between(CFG["target_min"], CFG["target_max"]).all(), "Scores out of range!"

    submission.to_csv(output_path, index=False)
    print(f"Submission saved: {output_path}")

    return submission
