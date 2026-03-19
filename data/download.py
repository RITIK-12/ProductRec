"""
Download utilities for training images and the held-out test set.

Training images are fetched in parallel from their source URLs and cached
locally by content hash. The test set is pulled from HuggingFace Hub.
"""

import hashlib
import threading
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from tqdm.auto import tqdm
from urllib3.util.retry import Retry

from config import CFG, IMG_CACHE, TEST_ROOT

# Thread-local session so each download worker reuses its own connection pool
_tls = threading.local()


def _get_session() -> requests.Session:
    """Return a thread-local requests session with automatic retries."""
    if not hasattr(_tls, "session"):
        s = requests.Session()
        retries = Retry(
            total=3,
            backoff_factor=0.3,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retries, pool_connections=20, pool_maxsize=20)
        s.mount("https://", adapter)
        s.mount("http://", adapter)
        _tls.session = s
    return _tls.session


def _img_path_for_url(url: str) -> Path:
    """Deterministic local path for a URL, sharded by first 2 hex chars of MD5."""
    h = hashlib.md5(url.encode()).hexdigest()
    d = IMG_CACHE / h[:2]
    d.mkdir(exist_ok=True)
    return d / f"{h}.jpg"


def _download_one(args: tuple) -> tuple:
    """Download a single image. Returns (index, local_path | None, error | None)."""
    idx, url = args
    out = _img_path_for_url(url)
    if out.exists() and out.stat().st_size > 0:
        return idx, str(out), None
    try:
        r = _get_session().get(url, timeout=CFG["download_timeout"])
        r.raise_for_status()
        out.write_bytes(r.content)
        return idx, str(out), None
    except Exception as e:
        return idx, None, str(e)


def download_images(df) -> "pd.DataFrame":
    """
    Download product images in parallel and attach local paths to the dataframe.

    Rows with failed downloads are dropped. A `local_path` column is added
    to the returned dataframe.
    """
    import pandas as pd  # deferred to avoid circular import at module level

    urls = df["image_url"].tolist()
    paths = [None] * len(df)
    failed = set()

    with ThreadPoolExecutor(max_workers=CFG["download_workers"]) as ex:
        futs = [ex.submit(_download_one, (i, urls[i])) for i in range(len(df))]
        for fut in tqdm(as_completed(futs), total=len(futs), desc="downloading"):
            idx, path, err = fut.result()
            if err:
                failed.add(idx)
            else:
                paths[idx] = path

    valid = [i for i in range(len(df)) if i not in failed]
    if failed:
        print(f"  [DROPPED] {len(failed)}/{len(df)} failed downloads")

    out_df = df.iloc[valid].reset_index(drop=True)
    out_df["local_path"] = [paths[i] for i in valid]
    print(f"  Valid: {len(out_df)}")
    return out_df


def download_test_set() -> tuple[Path, Path]:
    """
    Download and extract the test set from HuggingFace Hub.

    Returns (csv_path, images_dir). Skips download if already extracted.
    """
    from huggingface_hub import hf_hub_download

    TEST_ROOT.mkdir(parents=True, exist_ok=True)

    # Check if already extracted
    candidates = [TEST_ROOT / "images", TEST_ROOT / "setB" / "images"]
    for cand in candidates:
        if cand.exists() and len(list(cand.glob("*"))) > 1000:
            print(f"Already extracted: {len(list(cand.glob('*')))} images at {cand}")
            break
    else:
        print("Downloading test set...")
        zip_path = hf_hub_download(
            repo_id=CFG["test_hf_repo"],
            filename=CFG["test_hf_filename"],
            repo_type="dataset",
            local_dir=str(TEST_ROOT),
        )
        print("Extracting...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(TEST_ROOT)
        Path(zip_path).unlink(missing_ok=True)
        print("Done")

    # Locate CSV and images directory
    test_csv = None
    for c in [TEST_ROOT / "input.csv", TEST_ROOT / "setB" / "input.csv"]:
        if c.exists():
            test_csv = c
            break

    test_images = None
    for c in [TEST_ROOT / "setB" / "images", TEST_ROOT / "images"]:
        if c.exists():
            test_images = c
            break

    assert test_csv is not None, f"input.csv not found under {TEST_ROOT}"
    assert test_images is not None, f"images dir not found under {TEST_ROOT}"

    print(f"CSV:    {test_csv}")
    print(f"Images: {test_images} ({len(list(test_images.glob('*')))} files)")
    return test_csv, test_images
