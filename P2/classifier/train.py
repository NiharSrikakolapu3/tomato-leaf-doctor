from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score

from .config import ARMS, CROP_SIZE, IMAGENET_MEAN, IMAGENET_STD, NUM_CLASSES, RUNS_DIR, SEEDS, TrainConfig
from .model import build, center_crop, count_params, freeze_backbone, save, to_model_input


# HELPERS

def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def augment(batch_u8: torch.Tensor, config: TrainConfig, gen: torch.Generator) -> torch.Tensor:
    """random 224x224 crop, flips, mild color jitter, ImageNet normalization"""
    n, h, w, _ = batch_u8.shape
    device = batch_u8.device

    tops = torch.randint(0, h - CROP_SIZE + 1, (n,), generator=gen).tolist()
    lefts = torch.randint(0, w - CROP_SIZE + 1, (n,), generator=gen).tolist()
    crops = torch.stack([batch_u8[i, top : top + CROP_SIZE, left : left + CROP_SIZE] for i, (top, left) in enumerate(zip(tops, lefts))])
    x = crops.permute(0, 3, 1, 2).float().div_(255)   # N,3,H,W in [0,1]

    def coin() -> torch.Tensor:
        return (torch.rand(n, generator=gen) < 0.5).to(device).view(n, 1, 1, 1)

    if config.hflip:
        x = torch.where(coin(), x.flip(3), x)

    if config.vflip:
        x = torch.where(coin(), x.flip(2), x)

    jitter = config.color_jitter
    if jitter > 0:
        def factor() -> torch.Tensor:
            return (1 + (torch.rand(n, generator=gen) * 2 - 1) * jitter).to(device).view(n, 1, 1, 1)

        x = x * factor()                     # brightness
        gray = (0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3])
        mean_gray = gray.mean(dim=(2, 3), keepdim=True)
        x = (x - mean_gray) * factor() + mean_gray   # contrast
        gray = (0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3])
        x = (x - gray) * factor() + gray     # saturation
        x = x.clamp_(0, 1)

    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    return (x - mean) / std


@dataclass
class Subset:
    """rows `index` of a shared uint8 image array (N, 256, 256, 3) - nothing is copied"""
    images: np.ndarray
    labels: np.ndarray
    index: np.ndarray
    ids: np.ndarray | None = None      # file path / box id per row of `images`
    groups: np.ndarray | None = None   # photo id per row of `images` (bootstrap resampling unit)

    def __len__(self) -> int:
        return len(self.index)

    @property
    def y(self) -> np.ndarray:
        return self.labels[self.index]

    def take(self, positions: np.ndarray) -> np.ndarray:
        return np.asarray(self.images[self.index[positions]])


@torch.no_grad()
def predict(model, subset: Subset, device, batch_size: int, amp_dtype) -> np.ndarray:
    model.eval()
    out = []

    for start in range(0, len(subset), batch_size):
        positions = np.arange(start, min(start + batch_size, len(subset)))
        batch = torch.from_numpy(subset.take(positions)).to(device, non_blocking=True)
        inputs = to_model_input(center_crop(batch)).contiguous(memory_format=torch.channels_last)
        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            out.append(model(inputs).float().cpu())

    return torch.cat(out).numpy()


def metrics_from_logits(logits: np.ndarray, labels: np.ndarray) -> dict:
    pred = logits.argmax(1)
    return {
        "loss": F.cross_entropy(torch.from_numpy(logits), torch.from_numpy(labels).long()).item(),
        "acc": accuracy_score(labels, pred),
        "macro_f1": f1_score(labels, pred, average="macro", labels=list(range(NUM_CLASSES)), zero_division=0),
    }


def amp_dtype_for(device: torch.device, enabled: bool):
    if not enabled or device.type != "cuda":
        return None
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def balanced_weights(y: np.ndarray) -> torch.Tensor:
    # every class in the pool is drawn equally often
    counts = np.bincount(y, minlength=NUM_CLASSES)
    return torch.as_tensor(1.0 / counts[y], dtype=torch.double)


# TRAINING

@dataclass
class RunResult:
    config: dict
    train_history: pd.DataFrame
    eval_history: pd.DataFrame
    logits: dict[str, np.ndarray]
    evals: dict[str, Subset]
    final: dict[str, dict]
    seconds: float
    run_dir: Path


def train_one(
    config: TrainConfig,
    train: dict[str, Subset],
    evals: dict[str, Subset],
    device: torch.device | str = "cuda",
    run_dir: Path | None = None,
    verbose: bool = True,
) -> RunResult:
    """`train` has pool "pv" (PlantVillage) and, for the change arm, "pd" (PlantDoc crops).
    each batch element is from "pd" with probability config.real_fraction, class-balanced within its pool"""
    device = torch.device(device)
    seed_all(config.seed)
    torch.backends.cudnn.benchmark = True
    gen = torch.Generator().manual_seed(config.seed)
    amp_dtype = amp_dtype_for(device, config.amp)
    scaler = torch.amp.GradScaler(enabled=amp_dtype == torch.float16)

    pools = {"pv": train["pv"]}
    if config.real_fraction > 0:
        if "pd" not in train:
            raise ValueError(f"arm {config.arm!r} needs a 'pd' pool")
        pools["pd"] = train["pd"]
    weights = {name: balanced_weights(pool.y) for name, pool in pools.items()}

    model = build().to(device)
    if config.channels_last:
        model = model.to(memory_format=torch.channels_last)

    def make_stage(stage: int):
        if stage == 1:
            model.train()
            freeze_backbone(model, True)
            optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=config.head_lr, weight_decay=config.weight_decay)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
        else:
            freeze_backbone(model, False)
            model.train()
            optimizer = torch.optim.AdamW(model.parameters(), lr=config.full_lr, weight_decay=config.weight_decay)

            def lr_lambda(step: int) -> float:
                if step < config.warmup_steps:
                    return (step + 1) / config.warmup_steps
                progress = (step - config.warmup_steps) / max(1, config.full_steps - config.warmup_steps)
                return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        if verbose:
            print(f"  stage {stage}: {count_params(model, trainable_only=True):,} trainable params")
        return optimizer, scheduler

    def sample_batch():
        n_pd = 0
        if "pd" in pools:
            n_pd = int((torch.rand(config.batch_size, generator=gen) < config.real_fraction).sum())

        images, labels = [], []
        for name, n in (("pv", config.batch_size - n_pd), ("pd", n_pd)):
            if n == 0:
                continue
            positions = np.sort(torch.multinomial(weights[name], n, replacement=True, generator=gen).numpy())
            images.append(pools[name].take(positions))
            labels.append(pools[name].y[positions])

        return np.concatenate(images), np.concatenate(labels), n_pd

    stage = 1
    optimizer, scheduler = make_stage(1)
    train_rows, eval_rows = [], []
    logits: dict[str, np.ndarray] = {}
    start_time = time.time()

    for step in range(config.total_steps):
        if step == config.head_steps:
            stage = 2
            optimizer, scheduler = make_stage(2)

        batch_images, batch_labels, n_pd = sample_batch()
        images = torch.from_numpy(batch_images).to(device, non_blocking=True)
        labels = torch.from_numpy(batch_labels).long().to(device)
        inputs = augment(images, config, gen).contiguous(memory_format=torch.channels_last)

        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            loss = F.cross_entropy(model(inputs), labels)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        train_rows.append(
            {
                "step": step + 1,
                "stage": stage,
                "lr": lr,
                "train_loss": loss.item(),
                "pd_share": n_pd / config.batch_size,
            }
        )

        if (step + 1) % config.eval_every == 0 or step + 1 == config.total_steps:
            row = {
                "step": step + 1,
                "stage": stage,
                "train_loss_avg": float(np.mean([r["train_loss"] for r in train_rows[-config.eval_every :]])),
            }
            for name, subset in evals.items():
                logits[name] = predict(model, subset, device, config.eval_batch_size, amp_dtype)
                row.update({f"{name}_{key}": value for key, value in metrics_from_logits(logits[name], subset.y).items()})
            eval_rows.append(row)

            if verbose:
                scores = " | ".join(f"{name} F1 {row[f'{name}_macro_f1']:.3f}" for name in evals)
                print(f"  step {step + 1:4d} | stage {stage} | train {row['train_loss_avg']:.4f} | {scores} | {time.time() - start_time:5.0f}s")

            model.train()
            if stage == 1:
                model.features.eval()

    result = RunResult(
        config=config.to_dict(),
        train_history=pd.DataFrame(train_rows),
        eval_history=pd.DataFrame(eval_rows),
        logits=logits,
        evals=evals,
        final={name: metrics_from_logits(logits[name], subset.y) for name, subset in evals.items()},
        seconds=time.time() - start_time,
        run_dir=run_dir or RUNS_DIR / config.arm / f"seed{config.seed}",
    )
    save_run(result, model)
    return result


# SAVE / LOAD

def save_run(result: RunResult, model) -> None:
    run_dir = result.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    save(run_dir / "model.pt", model, result.config, result.final)
    result.train_history.to_csv(run_dir / "train_history.csv", index=False)
    result.eval_history.to_csv(run_dir / "eval_history.csv", index=False)

    empty = np.array([], dtype=object)
    for name, subset in result.evals.items():
        np.savez_compressed(
            run_dir / f"logits_{name}.npz",
            logits=result.logits[name],
            labels=subset.y,
            ids=np.asarray(subset.ids[subset.index], dtype=object) if subset.ids is not None else empty,
            groups=np.asarray(subset.groups[subset.index], dtype=object) if subset.groups is not None else empty,
        )

    metrics = {"final": result.final, "seconds": result.seconds, "config": result.config}
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))


def load_run(run_dir: Path) -> dict:
    """{'logits'|'labels'|'ids'|'groups': {evalset: array}, 'train_history', 'eval_history', 'metrics'}"""
    out = {
        "train_history": pd.read_csv(run_dir / "train_history.csv"),
        "eval_history": pd.read_csv(run_dir / "eval_history.csv"),
        "metrics": json.loads((run_dir / "metrics.json").read_text()),
        "logits": {},
        "labels": {},
        "ids": {},
        "groups": {},
    }

    for path in sorted(run_dir.glob("logits_*.npz")):
        name = path.stem.removeprefix("logits_")
        saved = np.load(path, allow_pickle=True)
        for key in ("logits", "labels", "ids", "groups"):
            out[key][name] = saved[key]

    if not out["logits"]:
        raise FileNotFoundError(f"no logits_*.npz in {run_dir} - old run format, retrain it")
    return out


# DATA FOR BOTH ARMS

def load_all_data() -> dict:
    """PlantVillage + PlantDoc train/dev crops + dev whole photos, as Subsets.
    PlantDoc final photos are dropped before any of their pixels are cached"""
    from . import data, plantdoc

    data.download()
    df = data.load_or_make_split(data.index())
    pv_images = np.array(data.load_images(df))   # copy into RAM (writable)
    pv_labels = df["label"].to_numpy()
    pv_ids = df["relpath"].to_numpy()
    pv_split = df["split"].to_numpy()

    boxes, photos = plantdoc.load_or_make_split(plantdoc.download())
    boxes = boxes[boxes["split"].isin(["train", "dev"])].reset_index(drop=True)
    crops = plantdoc.load_crops(boxes)
    crop_labels = boxes["label"].to_numpy()
    crop_ids = boxes["box_id"].to_numpy()
    crop_photos = (boxes["official"] + "/" + boxes["filename"]).to_numpy()
    crop_split = boxes["split"].to_numpy()

    # whole-photo eval only scores single-label dev photos
    dev_photos = photos[(photos["split"] == "dev") & ~photos["mixed"]].reset_index(drop=True)
    photo_images = plantdoc.load_photos(dev_photos, cache_name="pd_dev_photos_256")
    photo_ids = (dev_photos["official"] + "/" + dev_photos["filename"]).to_numpy()

    return {
        "train": {
            "pv": Subset(pv_images, pv_labels, np.flatnonzero(pv_split == "train"), pv_ids),
            "pd": Subset(crops, crop_labels, np.flatnonzero(crop_split == "train"), crop_ids, crop_photos),
        },
        "evals": {
            "pv_dev": Subset(pv_images, pv_labels, np.flatnonzero(pv_split == "dev"), pv_ids),
            "pd_dev_crop": Subset(crops, crop_labels, np.flatnonzero(crop_split == "dev"), crop_ids, crop_photos),
            "pd_dev_photo": Subset(photo_images, dev_photos["label"].to_numpy(), np.arange(len(dev_photos)), photo_ids, photo_ids),
        },
        "pv_df": df,
        "pd_boxes": boxes,
        "pd_photos": photos,
    }


def run_arms(data: dict, arms=ARMS, seeds=SEEDS, device=None, retrain: bool = True, **config_kw) -> dict[str, dict[int, dict]]:
    """train (or reload) every arm at every seed -> {arm: {seed: load_run(...)}}"""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    out: dict[str, dict[int, dict]] = {}

    for arm in arms:
        out[arm] = {}
        for seed in seeds:
            config = TrainConfig.for_arm(arm, seed=seed, **config_kw)
            run_dir = RUNS_DIR / arm / f"seed{seed}"

            if retrain or not (run_dir / "logits_pd_dev_photo.npz").exists():
                print(f"=== arm {arm} | seed {seed} ===")
                result = train_one(config, data["train"], data["evals"], device, run_dir=run_dir)
                scores = ", ".join(f"{name} F1 {metrics['macro_f1']:.4f}" for name, metrics in result.final.items())
                print(f"  done in {result.seconds:.0f}s: {scores}")

            out[arm][seed] = load_run(run_dir)

    return out


# CLI

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Fine-tune EfficientNet-B0 for both arms of the controlled comparison.")
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--head-steps", type=int, default=200)
    parser.add_argument("--full-steps", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--reuse", action="store_true", help="reload finished runs instead of retraining")
    args = parser.parse_args(argv)

    runs = run_arms(
        load_all_data(), args.arms, args.seeds, retrain=not args.reuse,
        head_steps=args.head_steps, full_steps=args.full_steps, batch_size=args.batch_size,
    )

    if set(runs) == set(ARMS):
        from .diagnostics import compare_arms

        for evalset in ("pd_dev_photo", "pd_dev_crop", "pv_dev"):
            print(f"\n{evalset}\n" + compare_arms(runs, evalset).to_string())


if __name__ == "__main__":
    main()
