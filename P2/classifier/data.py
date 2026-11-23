from __future__ import annotations

import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold

from .config import CACHE_DIR, CLASSES, PLANTVILLAGE_DIR, PLANTVILLAGE_URL, SPLIT_PLANTVILLAGE_CSV, STORE_SIZE

PV_COLOR_DIR = PLANTVILLAGE_DIR / "raw" / "color"   # the 8 tomato class folders
IMG_EXTS = {".jpg", ".jpeg", ".png"}


# DOWNLOAD

def download(dest: Path = PLANTVILLAGE_DIR) -> Path:
    """sparse clone of only the 8 tomato folders (~150 MB instead of the full repo)"""
    color = dest / "raw" / "color"
    if color.exists() and all((color / folder).is_dir() for _, folder in CLASSES):
        return color

    dest.parent.mkdir(parents=True, exist_ok=True)
    if not (dest / ".git").exists():
        subprocess.run(
            ["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse", PLANTVILLAGE_URL, str(dest)],
            check=True,
        )

    patterns = [f"raw/color/{folder}/" for _, folder in CLASSES]
    subprocess.run(["git", "-C", str(dest), "sparse-checkout", "set", "--no-cone", *patterns], check=True)
    return color


def download_cli() -> None:
    df = index(download())
    print(df["class_name"].value_counts().reindex([name for name, _ in CLASSES]).to_string())
    print("total", len(df))


# LEAF GROUPING

NAME_PATTERN = re.compile(r"^(?P<src>.*?)(?:^|\s)(?P<num>\d+)")
GROUP_BLOCK = 10


def leaf_group(class_name: str, filename: str) -> str:
    stem = Path(filename).stem
    tail = stem.split("___", 1)[1] if "___" in stem else stem

    match = NAME_PATTERN.match(tail)
    if not match:
        return f"{class_name}|{tail.lower()}"

    source = re.sub(r"\s+", " ", match.group("src")).strip().lower()
    return f"{class_name}|{source}|{int(match.group('num')) // GROUP_BLOCK}"


def index(color_dir: Path = PV_COLOR_DIR) -> pd.DataFrame:
    rows = []
    for label, (name, folder) in enumerate(CLASSES):
        files = sorted(path for path in (color_dir / folder).iterdir() if path.suffix.lower() in IMG_EXTS)
        if not files:
            raise FileNotFoundError(f"no images in {color_dir / folder} - run download() first")

        for path in files:
            rows.append(
                {
                    "relpath": f"{folder}/{path.name}",
                    "class_name": name,
                    "label": label,
                    "group": leaf_group(name, path.name),
                }
            )

    return pd.DataFrame(rows)


# SPLIT

def make_split(df: pd.DataFrame, dev_fraction: float = 0.2, seed: int = 0) -> pd.Series:
    # grouped + class-stratified; split seed is fixed and independent of training seeds
    folds = StratifiedGroupKFold(n_splits=round(1 / dev_fraction), shuffle=True, random_state=seed)
    _, dev_index = next(folds.split(df, df["label"], groups=df["group"]))

    split = pd.Series("train", index=df.index)
    split.iloc[dev_index] = "dev"
    return split


def load_or_make_split(df: pd.DataFrame, path: Path = SPLIT_PLANTVILLAGE_CSV, overwrite: bool = False) -> pd.DataFrame:
    """adds a 'split' column, reusing the saved csv so every run sees the same split"""
    if path.exists() and not overwrite:
        saved = pd.read_csv(path)
        merged = df.merge(saved[["relpath", "split"]], on="relpath", how="left")

        missing = merged["split"].isna().sum()
        if missing:
            raise ValueError(f"{missing} images missing from {path} - delete it or pass overwrite=True")
        return merged

    out = df.copy()
    out["split"] = make_split(out)

    path.parent.mkdir(parents=True, exist_ok=True)
    out[["relpath", "class_name", "label", "group", "split"]].to_csv(path, index=False)
    return out


# IMAGE CACHE

def resize_center(image: Image.Image, size: int = STORE_SIZE) -> Image.Image:
    """short side -> size, then center size x size crop"""
    w, h = image.size
    if (w, h) == (size, size):
        return image

    scale = size / min(w, h)
    image = image.resize((max(size, round(w * scale)), max(size, round(h * scale))), Image.BICUBIC)

    w, h = image.size
    left, top = (w - size) // 2, (h - size) // 2
    return image.crop((left, top, left + size, top + size))


def load_resized(path: Path, size: int = STORE_SIZE) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(resize_center(image.convert("RGB"), size), dtype=np.uint8)


def load_images(df: pd.DataFrame, color_dir: Path = PV_COLOR_DIR, cache_name: str = "pv_tomato_256", workers: int = 8) -> np.ndarray:
    """uint8 (N, 256, 256, 3) in df order, cached as .npy (memory-mapped on reload)"""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    array_path = CACHE_DIR / f"{cache_name}.npy"
    index_path = CACHE_DIR / f"{cache_name}_index.csv"

    if array_path.exists() and index_path.exists():
        if pd.read_csv(index_path)["relpath"].tolist() == df["relpath"].tolist():
            return np.load(array_path, mmap_mode="r")

    from tqdm.auto import tqdm

    out = np.lib.format.open_memmap(array_path, mode="w+", dtype=np.uint8, shape=(len(df), STORE_SIZE, STORE_SIZE, 3))
    paths = [color_dir / relpath for relpath in df["relpath"]]
    with ThreadPoolExecutor(workers) as pool:
        for row, image in enumerate(tqdm(pool.map(load_resized, paths), total=len(paths), desc="decode")):
            out[row] = image

    out.flush()
    del out
    df[["relpath"]].to_csv(index_path, index=False)
    return np.load(array_path, mmap_mode="r")
