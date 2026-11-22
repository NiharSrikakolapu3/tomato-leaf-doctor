"""Train a stand-in classifier for one arm and seed, and write P4's logits file for it.

Stands in for P2 until P2's runs exist. Only the network (and its usual fine-tuning rate)
differs from the pilot's pilot/train_classifier.py: same training photos, balanced sampler,
200 head steps then 600 full steps at batch 64, same augmentation, same development photos.
The final partition is never read. P4/README.md has the comparison of the backbones below.

Writes to OUT/BACKBONE/: {arm}_seed{seed}.pt (weights), .npz (P4's input format, over the
development photos), .json (scores), and classifier.json, which pipeline.py reads to
rebuild the network.

Usage (pilot repo is ksha23/leaf-doctor, data and finder results as in its README):
    python train_stand_in.py --backbone dinov3_vitb --arm lab_field --seed 0 \
        --pilot REPO/pilot --splits REPO/splits --plantvillage DATA/pv --plantdoc DATA/pd \
        --runs PILOT_RUNS --cache CACHE --out OUT
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).parents[1]))
from classifier import BACKBONES, build_classifier
from labels import CLASSES

WORKERS = 8


def make_transforms(mean, std):
    train = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return train, test


def run_steps(model, batches, optimizer, scheduler, steps, device, phase):
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    for step in range(steps):
        images, labels = next(batches)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = loss_fn(model(images.to(device)).float(), labels.to(device))
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        if (step + 1) % 200 == 0:
            print(f"{phase} step {step + 1}/{steps} loss {loss.item():.3f}", flush=True)


@torch.no_grad()
def predict(model, dataset, device):
    loader = DataLoader(dataset, batch_size=128, num_workers=WORKERS, shuffle=False)
    model.eval()
    outputs = []
    for images, _ in loader:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs.append(model(images.to(device)).float().cpu().numpy())
    return np.concatenate(outputs)


def main():
    parser = argparse.ArgumentParser()
    for name in ("pilot", "splits", "plantvillage", "plantdoc", "runs", "cache", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--backbone", choices=sorted(BACKBONES), required=True)
    parser.add_argument("--arm", choices=["lab", "lab_field"], required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # The pilot's data code: the same training photos, sampler and development photos.
    sys.path.insert(0, str(args.pilot))
    from common import CLASSES as PILOT_CLASSES, accuracy, macro_f1
    from eval_system import finder_items, whole_items
    from images import cache_plantdoc, plantdoc_rows, plantvillage_items
    from train_classifier import (BATCH_SIZE, FINE_TUNE_STEPS, HEAD_LR, HEAD_STEPS, ImageList,
                                  balanced_weights, next_batches)
    assert list(PILOT_CLASSES) == CLASSES, "class order must match labels.py"
    out = args.out / args.backbone
    out.mkdir(parents=True, exist_ok=True)
    splits, cache, runs = args.splits, args.cache, args.runs
    plantvillage, plantdoc = args.plantvillage, args.plantdoc

    torch.manual_seed(args.seed)
    device = "cuda"
    model, head, mean, std = build_classifier(args.backbone, pretrained=True)
    model.to(device)
    train_tf, test_tf = make_transforms(mean, std)

    lab_train = plantvillage_items(splits, plantvillage, "pv_train")
    field_train, _, _ = cache_plantdoc(splits, plantdoc, cache, "pd_train")
    extra = field_train if args.arm == "lab_field" else []
    sampler = WeightedRandomSampler(balanced_weights(lab_train, extra),
                                    num_samples=(HEAD_STEPS + FINE_TUNE_STEPS) * BATCH_SIZE,
                                    replacement=True, generator=torch.Generator().manual_seed(args.seed))
    loader = DataLoader(ImageList(lab_train + extra, train_tf), batch_size=BATCH_SIZE, sampler=sampler,
                        num_workers=WORKERS, persistent_workers=True, drop_last=True)
    batches = next_batches(loader)

    start = time.time()
    head_ids = {id(p) for p in head}
    for parameter in model.parameters():
        parameter.requires_grad = id(parameter) in head_ids
    run_steps(model, batches, torch.optim.AdamW(head, lr=HEAD_LR), None, HEAD_STEPS, device, "head")

    spec = BACKBONES[args.backbone]
    for parameter in model.parameters():
        parameter.requires_grad = True
    optimizer = torch.optim.AdamW(model.parameters(), lr=spec["lr"], weight_decay=spec["wd"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=FINE_TUNE_STEPS)
    run_steps(model, batches, optimizer, scheduler, FINE_TUNE_STEPS, device, "fine-tune")
    train_seconds = time.time() - start

    name = f"{args.arm}_seed{args.seed}"
    torch.save(model.state_dict(), out / f"{name}.pt")
    (out / "classifier.json").write_text(json.dumps({"backbone": args.backbone, "mean": mean, "std": std}, indent=2) + "\n")

    # P4's input format: development photos, whole photo and the finder's best box.
    rows = [r for r in plantdoc_rows(splits, "pd_dev") if r["mixed_labels"] != "1"]
    with open(runs / "finder_dev.csv", newline="") as fh:
        finder = {r["photo"]: r for r in csv.DictReader(fh)}
    whole = whole_items(splits, cache)
    best_box = [item for item in finder_items(splits, plantdoc, runs, cache) if item[4] == 0]
    assert len(whole) == len(best_box) == len(rows)
    labels = np.array([CLASSES.index(r["photo_label"]) for r in rows])
    whole_logits = predict(model, ImageList(whole, test_tf), device)
    box_logits = predict(model, ImageList(best_box, test_tf), device)
    np.savez(out / f"{name}.npz", photo=np.array([r["photo"] for r in rows]),
             group=np.array([r["group"] for r in rows]), label=labels,
             whole=whole_logits, finder_box=box_logits,
             finder_score=np.array([float(finder[r["photo"]]["score"]) for r in rows]),
             classes=np.array(CLASSES))

    result = {"backbone": args.backbone, "arm": args.arm, "seed": args.seed,
              "parameters_millions": round(sum(p.numel() for p in model.parameters()) / 1e6, 1),
              "train_seconds": round(train_seconds, 1), "fine_tune_lr": spec["lr"],
              "dev_whole_macro_f1": macro_f1(labels, whole_logits.argmax(axis=1)),
              "dev_finder_box_macro_f1": macro_f1(labels, box_logits.argmax(axis=1)),
              "dev_whole_accuracy": accuracy(labels, whole_logits.argmax(axis=1)),
              "dev_finder_box_accuracy": accuracy(labels, box_logits.argmax(axis=1))}
    (out / f"{name}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
