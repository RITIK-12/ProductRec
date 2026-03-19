"""
Text preprocessing for product metadata fields.

Handles the messy reality of Amazon product data: mixed types, nested JSON
strings, lists-as-strings, and missing values — normalises everything into
clean text suitable for tokenisation.
"""

import json
import re
from pathlib import Path

import pandas as pd


def clean_field(x) -> str:
    """Normalise any metadata field (str | list | dict | None) to a flat string."""
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    if isinstance(x, list):
        return " | ".join(str(v).strip() for v in x if str(v).strip())
    if isinstance(x, dict):
        return " | ".join(f"{k} {v}" for k, v in x.items() if str(v).strip())
    if isinstance(x, str):
        s = x.strip()
        # Attempt to parse stringified JSON containers
        if s and s[0] in "{[":
            try:
                return clean_field(json.loads(s))
            except Exception:
                pass
        return s
    return str(x).strip()


def build_text(row: dict) -> str:
    """
    Construct a single text input from a product row.

    Concatenates category, title, description, and features with [SEP] tokens
    so the tokeniser can leverage structure without needing multiple inputs.
    """
    parts = []
    for col, label in [
        ("main_category", "category"),
        ("title", "title"),
        ("description", "description"),
        ("features", "features"),
    ]:
        val = clean_field(row.get(col))
        if val:
            parts.append(f"{label}: {val}")
    return " [SEP] ".join(parts)


def stem_to_asin(filename: str) -> str:
    """
    Extract the product ASIN from an image filename.

    Test-set images follow the pattern `{ASIN}_{index}.ext`. This strips
    the trailing index to recover the product identifier.
    """
    name = Path(filename).stem
    match = re.match(r"^(.+?)_(\d+)$", name)
    return match.group(1) if match else name
