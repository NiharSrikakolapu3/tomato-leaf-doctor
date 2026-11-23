from __future__ import annotations

import re
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold

from .config import CACHE_DIR, CLASS_NAMES, PLANTDOC_DIR, PLANTDOC_URL, SPLIT_PLANTDOC_CSV, STORE_SIZE
from .data import resize_center

PD_CLASS_MAP = {
    "Tomato leaf bacterial spot": "bacterial-spot",
    "Tomato Early blight leaf": "early-blight",
    "Tomato leaf late blight": "late-blight",
    "Tomato mold leaf": "leaf-mold",
    "Tomato Septoria leaf spot": "septoria",
    "Tomato leaf mosaic virus": "mosaic-virus",
    "Tomato leaf yellow virus": "yellow-leaf-curl",
    "Tomato leaf": "healthy",
}

BOX_MARGIN = 0.10           # 10% margin around labeled boxes
FINAL_PHOTOS_TARGET = 188   # official TEST + held-out TRAIN
DEV_FRACTION = 0.2


# DOWNLOAD / INDEX

def fetch(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    temp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=60) as response, open(temp, "wb") as file:
        file.write(response.read())
    temp.replace(dest)


def local_path(root: Path, official: str, filename: str) -> Path:
    return root / official / re.sub(r'[<>:"/\\|?*]', "_", filename)


def download(dest: Path = PLANTDOC_DIR, workers: int = 16) -> pd.DataFrame:
    """label csvs + only the tomato photos (~750 files instead of the 1 GB repo); returns boxes that exist locally"""
    for name in ("train_labels.csv", "test_labels.csv"):
        fetch(f"{PLANTDOC_URL}/{name}", dest / name)

    boxes = index(dest)
    files = boxes[["official", "filename"]].drop_duplicates()

    from tqdm.auto import tqdm

    def job(row):
        official, filename = row
        try:
            fetch(f"{PLANTDOC_URL}/{official}/{urllib.parse.quote(filename)}", local_path(dest, official, filename))
        except Exception as error:  # noqa: BLE001 - report and keep going
            return f"{official}/{filename}: {error}"
        return None

    with ThreadPoolExecutor(workers) as pool:
        errors = [error for error in tqdm(pool.map(job, files.itertuples(index=False)), total=len(files), desc="plantdoc") if error]
    if errors:
        print(f"couldn't fetch {len(errors)} PlantDoc photos - skipping them:", *errors[:10], sep="\n  ")

    exists = [local_path(dest, official, filename).exists() for official, filename in zip(boxes["official"], boxes["filename"])]
    return boxes[exists].reset_index(drop=True)


def index(root: Path = PLANTDOC_DIR) -> pd.DataFrame:
    """one row per tomato box in our 8 classes"""
    parts = []
    for official, name in (("TRAIN", "train_labels.csv"), ("TEST", "test_labels.csv")):
        part = pd.read_csv(root / name)
        part["official"] = official
        parts.append(part)

    boxes = pd.concat(parts, ignore_index=True)
    boxes = boxes[boxes["class"].isin(PD_CLASS_MAP)].copy()
    boxes["class_name"] = boxes["class"].map(PD_CLASS_MAP)
    boxes["label"] = boxes["class_name"].map({name: label for label, name in enumerate(CLASS_NAMES)}).astype(int)

    boxes = boxes.sort_values(["official", "filename", "xmin", "ymin"]).reset_index(drop=True)
    box_number = boxes.groupby(["official", "filename"]).cumcount().astype(str)
    boxes["box_id"] = boxes["official"] + "/" + boxes["filename"] + "#" + box_number
    return boxes


def photo_table(boxes: pd.DataFrame) -> pd.DataFrame:
    """one row per photo: majority label, box count, mixed = boxes disagree"""
    photos = boxes.groupby(["official", "filename"]).agg(
        n_boxes=("label", "size"),
        n_labels=("label", "nunique"),
        label=("label", lambda labels: labels.value_counts().idxmax()),
    ).reset_index()

    photos["mixed"] = photos["n_labels"] > 1
    photos["class_name"] = [CLASS_NAMES[label] for label in photos["label"]]
    return photos.drop(columns="n_labels")


# IMAGES

def open_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def load_photo(root: Path, official: str, filename: str, size: int = STORE_SIZE) -> np.ndarray:
    # whole photo, same preprocessing as PlantVillage
    return np.asarray(resize_center(open_rgb(local_path(root, official, filename)), size), dtype=np.uint8)


def crop_box(image: Image.Image, row, csv_w: int, csv_h: int, margin: float = BOX_MARGIN) -> Image.Image:
    """labeled box + margin; rescaled when the csv size doesn't match the file"""
    W, H = image.size
    sx, sy = W / csv_w, H / csv_h

    x0, x1 = row.xmin * sx, row.xmax * sx
    y0, y1 = row.ymin * sy, row.ymax * sy
    mx, my = (x1 - x0) * margin, (y1 - y0) * margin
    box = (max(0, x0 - mx), max(0, y0 - my), min(W, x1 + mx), min(H, y1 + my))

    if box[2] - box[0] < 2 or box[3] - box[1] < 2:   # broken annotation -> use the whole photo
        return image
    return image.crop(tuple(round(v) for v in box))


def cached(name: str, keys: list[str], build) -> np.ndarray:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    array_path, index_path = CACHE_DIR / f"{name}.npy", CACHE_DIR / f"{name}_index.csv"

    if array_path.exists() and index_path.exists() and pd.read_csv(index_path)["key"].tolist() == keys:
        return np.load(array_path)

    array = build()
    np.save(array_path, array)
    pd.DataFrame({"key": keys}).to_csv(index_path, index=False)
    return array


def load_photos(photos: pd.DataFrame, root: Path = PLANTDOC_DIR, cache_name: str = "pd_photos_256", workers: int = 8) -> np.ndarray:
    keys = (photos["official"] + "/" + photos["filename"]).tolist()

    def build():
        rows = photos[["official", "filename"]].itertuples(index=False)
        with ThreadPoolExecutor(workers) as pool:
            return np.stack(list(pool.map(lambda row: load_photo(root, row[0], row[1]), rows)))

    return cached(cache_name, keys, build)


def load_crops(boxes: pd.DataFrame, root: Path = PLANTDOC_DIR, cache_name: str = "pd_crops_256", workers: int = 8) -> np.ndarray:
    """uint8 (N, 256, 256, 3): box + 10% margin, short side 256, center crop"""
    keys = boxes["box_id"].tolist()

    def crop_photo(item):
        (official, filename), rows = item
        image = open_rgb(local_path(root, official, filename))
        return [
            (box_index, np.asarray(resize_center(crop_box(image, row, row.width, row.height)), dtype=np.uint8))
            for box_index, row in zip(rows.index, rows.itertuples())
        ]

    def build():
        position = {box_index: row for row, box_index in enumerate(boxes.index)}
        out = np.empty((len(boxes), STORE_SIZE, STORE_SIZE, 3), dtype=np.uint8)
        with ThreadPoolExecutor(workers) as pool:
            for crops in pool.map(crop_photo, boxes.groupby(["official", "filename"])):
                for box_index, crop in crops:
                    out[position[box_index]] = crop
        return out

    return cached(cache_name, keys, build)


# SPLIT

def near_duplicate_groups(hashes: np.ndarray, max_dist: int = 4) -> np.ndarray:
    """union-find over photos whose 64-bit dhashes are within max_dist bits"""
    bits = np.unpackbits(hashes.view(np.uint8).reshape(-1, 8), axis=1).astype(np.int16)
    dist = (bits[:, None, :] != bits[None, :, :]).sum(-1)
    parent = np.arange(len(hashes))

    def find(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for a, b in zip(*np.nonzero(np.triu(dist <= max_dist, k=1))):
        parent[find(a)] = find(b)

    return np.array([find(node) for node in range(len(hashes))])


def make_split(photos: pd.DataFrame, hashes: np.ndarray, seed: int = 0) -> pd.Series:
    # final = official TEST + anything near-duplicating it + held-out TRAIN groups; the rest -> train/dev
    group = near_duplicate_groups(hashes)
    labels = photos["label"].to_numpy()
    split = pd.Series("", index=photos.index, dtype=object)

    test_groups = set(group[photos["official"].to_numpy() == "TEST"])
    forced = np.isin(group, list(test_groups))
    split[forced] = "final"

    rest = np.flatnonzero(~forced)
    n_extra = max(0, FINAL_PHOTOS_TARGET - int(forced.sum()))
    if n_extra:
        folds = StratifiedGroupKFold(max(2, round(len(rest) / n_extra)), shuffle=True, random_state=seed)
        _, held_out = next(folds.split(rest, labels[rest], group[rest]))
        split.iloc[rest[held_out]] = "final"
        rest = np.setdiff1d(rest, rest[held_out])

    folds = StratifiedGroupKFold(round(1 / DEV_FRACTION), shuffle=True, random_state=seed)
    _, dev = next(folds.split(rest, labels[rest], group[rest]))
    split.iloc[rest] = "train"
    split.iloc[rest[dev]] = "dev"
    return split


def load_or_make_split(boxes: pd.DataFrame, root: Path = PLANTDOC_DIR, path: Path = SPLIT_PLANTDOC_CSV,
                       overwrite: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(boxes, photos), both with a 'split' column"""
    photos = photo_table(boxes)
    keys = ["official", "filename"]

    if path.exists() and not overwrite:
        saved = pd.read_csv(path)
        photos = photos.merge(saved[keys + ["split"]], on=keys, how="left")
        if photos["split"].isna().any():
            raise ValueError(f"{photos['split'].isna().sum()} PlantDoc photos missing from {path}")
    else:
        from .diagnostics import dhash

        # 64x64 thumbnails of every photo (TEST too) - only used to group duplicates
        thumbs = np.stack([load_photo(root, official, filename, size=64) for official, filename in photos[keys].itertuples(index=False)])
        photos["split"] = make_split(photos, dhash(thumbs)).to_numpy()

        path.parent.mkdir(parents=True, exist_ok=True)
        photos[keys + ["class_name", "n_boxes", "mixed", "split"]].to_csv(path, index=False)

    boxes = boxes.merge(photos[keys + ["split"]], on=keys, how="left")
    return boxes, photos


def split_table(boxes: pd.DataFrame, photos: pd.DataFrame) -> pd.DataFrame:
    table = pd.concat(
        {
            "photos": pd.crosstab(photos["class_name"], photos["split"]),
            "boxes": pd.crosstab(boxes["class_name"], boxes["split"]),
        },
        axis=1,
    ).reindex(CLASS_NAMES).fillna(0).astype(int)

    table.loc["total"] = table.sum()
    return table
