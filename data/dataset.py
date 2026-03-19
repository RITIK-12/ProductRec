"""
PyTorch Dataset classes for training and inference.

PerImageDataset: one sample per product (train/val) with ordinal labels.
TestPerImageDataset: multiple images per product (test set), scores are
averaged at the product level during inference.
"""

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from config import CFG
from data.preprocess import build_text


class PerImageDataset(Dataset):
    """
    Training/validation dataset: one image-text pair per product.

    Pre-tokenises all text at init time so the dataloader only handles
    image I/O and tensor construction at __getitem__.
    """

    def __init__(self, df, transform, tokenizer):
        self.paths = df["local_path"].tolist()
        self.targets = df[CFG["target_col"]].values.astype(np.float32)
        self.transform = transform
        self.asins = df["parent_asin"].astype(str).tolist()

        # Tokenize all texts at once — returns (N, context_length) tensor
        records = df.to_dict("records")
        texts = [build_text(r) for r in records]
        self.tokens = tokenizer(texts)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        rating = self.targets[idx]

        # Ordinal encoding: threshold at each integer boundary
        ordinal = np.zeros(CFG["n_classes"] - 1, dtype=np.float32)
        for k in range(CFG["n_classes"] - 1):
            ordinal[k] = 1.0 if rating > (k + 1) else 0.0

        return {
            "image": self.transform(img),
            "tokens": self.tokens[idx],
            "target": torch.tensor(rating, dtype=torch.float32),
            "ordinal": torch.tensor(ordinal, dtype=torch.float32),
            "asin": self.asins[idx],
        }


class TestPerImageDataset(Dataset):
    """
    Test-time dataset: expands each product into N image-text pairs
    (one per product image). Scores are aggregated to product level
    after inference.
    """

    def __init__(self, df, transform, tokenizer, asin_to_files: dict):
        self.transform = transform
        self.item_ids = []
        self.img_paths = []
        self.tokens_list = []

        records = df.to_dict("records")
        for r in records:
            asin = str(r["parent_asin"])
            text = build_text(r)
            tok = tokenizer([text])[0]
            paths = asin_to_files[asin]
            assert len(paths) > 0, f"No images for {asin}"
            for p in paths:
                self.item_ids.append(asin)
                self.img_paths.append(p)
                self.tokens_list.append(tok)

        print(f"[TEST] {len(df)} products → {len(self)} image-text pairs")

    def __len__(self):
        return len(self.item_ids)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert("RGB")
        return {
            "image": self.transform(img),
            "tokens": self.tokens_list[idx],
            "item_id": self.item_ids[idx],
        }


def collate_fn(batch: list[dict]) -> dict:
    """Collate for train/val batches."""
    return {
        "image": torch.stack([x["image"] for x in batch]),
        "tokens": torch.stack([x["tokens"] for x in batch]),
        "target": torch.stack([x["target"] for x in batch]),
        "ordinal": torch.stack([x["ordinal"] for x in batch]),
        "asin": [x["asin"] for x in batch],
    }


def test_collate_fn(batch: list[dict]) -> dict:
    """Collate for test batches (no target labels)."""
    return {
        "image": torch.stack([x["image"] for x in batch]),
        "tokens": torch.stack([x["tokens"] for x in batch]),
        "item_id": [x["item_id"] for x in batch],
    }
