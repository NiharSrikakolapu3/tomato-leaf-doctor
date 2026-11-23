from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

# dataset paths / loading
P2_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = P2_DIR.parent

DATA_DIR = REPO_ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"

PLANTVILLAGE_DIR = DATA_DIR / "plantvillage"            # sparse clone of spMohanty/PlantVillage-Dataset
PLANTDOC_DIR = DATA_DIR / "plantdoc"                # tomato photos + label CSVs of PlantDoc-Object-Detection-Dataset

SPLIT_PLANTVILLAGE_CSV = P2_DIR / "splits" / "pv-split.csv"
SPLIT_PLANTDOC_CSV = P2_DIR / "splits" / "pd-split.csv"

RUNS_DIR = P2_DIR / "runs"                     # checkpoints, logits, histories (git-ignored)
FIG_DIR = P2_DIR / "figures"

PLANTVILLAGE_URL = "https://github.com/spMohanty/PlantVillage-Dataset.git"
PLANTDOC_URL = "https://raw.githubusercontent.com/pratikkayal/PlantDoc-Object-Detection-Dataset/master"



# (short name, PlantVillage folder)
CLASSES: list[tuple[str, str]] = [
    ("bacterial-spot", "Tomato___Bacterial_spot"),
    ("early-blight", "Tomato___Early_blight"),
    ("late-blight", "Tomato___Late_blight"),
    ("leaf-mold", "Tomato___Leaf_Mold"),
    ("septoria", "Tomato___Septoria_leaf_spot"),
    ("mosaic-virus", "Tomato___Tomato_mosaic_virus"),
    ("yellow-leaf-curl", "Tomato___Tomato_Yellow_Leaf_Curl_Virus"),
    ("healthy", "Tomato___healthy"),
]

CLASS_NAMES = [c for c, _ in CLASSES]
NUM_CLASSES = len(CLASSES)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

STORE_SIZE = 256   # scale to 256x256
CROP_SIZE = 224    # then crop to 224x224

ARMS = ("baseline", "change")
SEEDS = (0, 1, 2)

@dataclass
class TrainConfig:
    arm: str = "baseline"
    seed: int = 0

    real_fraction: float = 0.0       # portion of each batch drawn from PlantDoc train crops
    batch_size: int = 64
    
    head_steps: int = 200            # stage 1
    head_lr: float = 1e-3
    full_steps: int = 600            # stage 2
    full_lr: float = 2e-4
    
    weight_decay: float = 0.01
    warmup_steps: int = 30           # linear warmup at the start of stage 2
    
    eval_every: int = 50
    eval_batch_size: int = 256

    # augmentation (identical across arms)
    hflip: bool = True
    vflip: bool = True
    color_jitter: float = 0.2  

    channels_last: bool = True
    amp: bool = True

    @property
    def total_steps(self) -> int:
        return self.head_steps + self.full_steps

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def for_arm(cls, arm: str, **kw) -> "TrainConfig":
        # data mix should be the only difference
        if arm not in ARMS:
            raise ValueError(f"arm has to be in {ARMS} - got {arm!r}")
        
        return cls(arm=arm, real_fraction=0.5 if arm == "change" else 0.0, **kw)
