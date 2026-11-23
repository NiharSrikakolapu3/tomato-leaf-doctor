from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import CLASS_NAMES, NUM_CLASSES

# plot palette: categorical slots in fixed order, quiet ink / grid
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUE_RAMP = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

EVALSETS = ("pv_dev", "pd_dev_crop", "pd_dev_photo")
EVAL_TITLES = {
    "pv_dev": "PlantVillage dev (lab photos)",
    "pd_dev_crop": "PlantDoc dev, labeled crops",
    "pd_dev_photo": "PlantDoc dev, whole photos",
}

ALL_LABELS = list(range(NUM_CLASSES))


# SPLIT CHECKS

def split_table(df: pd.DataFrame) -> pd.DataFrame:
    table = pd.crosstab(df["class_name"], df["split"]).reindex(CLASS_NAMES)[["train", "dev"]]
    table["groups"] = df.groupby("class_name")["group"].nunique().reindex(CLASS_NAMES)
    table["dev_share"] = (table["dev"] / (table["train"] + table["dev"])).round(3)

    train, dev = table["train"].sum(), table["dev"].sum()
    table.loc["total"] = [train, dev, table["groups"].sum(), round(dev / (train + dev), 3)]
    return table


def dhash(images: np.ndarray, chunk: int = 1024) -> np.ndarray:
    """64-bit difference hash per image (grayscale 9x8, compare neighbours)"""
    out = []
    for start in range(0, len(images), chunk):
        x = torch.from_numpy(np.asarray(images[start : start + chunk])).float()
        gray = (0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]).unsqueeze(1)
        small = F.interpolate(gray, size=(8, 9), mode="area").squeeze(1)

        bits = (small[:, :, 1:] > small[:, :, :-1]).reshape(len(x), 64).numpy()
        out.append(np.packbits(bits, axis=1).view(">u8").ravel())

    return np.concatenate(out)


def leakage_check(hashes: np.ndarray, split: np.ndarray, max_dist: int = 4) -> dict:
    """share of dev images with a near-identical train image (hamming distance <= max_dist)"""
    def to_bits(h):
        return torch.from_numpy(np.unpackbits(h.view(np.uint8).reshape(-1, 8), axis=1).astype(np.uint8)).float()

    train_bits = to_bits(hashes[split == "train"])
    dev_bits = to_bits(hashes[split == "dev"])

    near = exact = 0
    for start in range(0, len(dev_bits), 512):
        dist = torch.cdist(dev_bits[start : start + 512], train_bits, p=1).min(dim=1).values
        near += int((dist <= max_dist).sum())
        exact += int((dist == 0).sum())

    return {
        "dev_images": len(dev_bits),
        "exact_dup_in_train": exact,
        f"near_dup_in_train(<= {max_dist} bits)": near,
        "near_dup_share": round(near / len(dev_bits), 4),
    }


# BACKGROUND SHORTCUT (Noyan 2022)

def background_pixels(images: np.ndarray, inset: int = 2) -> np.ndarray:
    """8 border pixels (4 corners + 4 edge midpoints) x RGB = 24 features"""
    h, w = images.shape[1], images.shape[2]
    top, bottom = inset, h - 1 - inset
    left, right = inset, w - 1 - inset

    points = [(top, left), (top, right), (bottom, left), (bottom, right),
              (top, w // 2), (bottom, w // 2), (h // 2, left), (h // 2, right)]
    features = np.stack([np.asarray(images[:, y, x, :]) for y, x in points], axis=1)
    return features.reshape(len(images), -1).astype(np.float32) / 255.0


def background_shortcut(images: np.ndarray, labels: np.ndarray, split: np.ndarray, seed: int = 0) -> pd.DataFrame:
    X = background_pixels(images)
    train, dev = split == "train", split == "dev"

    def score(name, pred):
        return {
            "model": name,
            "dev_acc": accuracy_score(labels[dev], pred),
            "dev_macro_f1": f1_score(labels[dev], pred, average="macro", labels=ALL_LABELS, zero_division=0),
        }

    rows = [score("majority class", np.full(dev.sum(), np.bincount(labels[train]).argmax()))]

    models = {
        "logistic regression (8 px)": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")),
        "random forest (8 px)": RandomForestClassifier(n_estimators=300, class_weight="balanced", n_jobs=-1, random_state=seed),
    }
    for name, model in models.items():
        model.fit(X[train], labels[train])
        rows.append(score(name, model.predict(X[dev])))

    return pd.DataFrame(rows).round(4)


# RESULTS

def summarize_runs(runs: dict[int, dict], evalset: str = "pd_dev_photo") -> pd.DataFrame:
    """one arm: per-seed acc / macro-F1 / loss on one eval set, plus mean and std"""
    rows = []
    for seed, run in runs.items():
        metrics = run["metrics"]["final"][evalset]
        rows.append(
            {
                "seed": seed,
                "acc": metrics["acc"],
                "macro_f1": metrics["macro_f1"],
                "loss": metrics["loss"],
                "minutes": run["metrics"]["seconds"] / 60,
            }
        )

    df = pd.DataFrame(rows).set_index("seed")
    df.loc["mean"] = df.mean()
    df.loc["std"] = df.drop(index="mean").std()
    return df.round(4)


def summarize_arms(runs_by_arm: dict[str, dict[int, dict]], evalsets=EVALSETS) -> pd.DataFrame:
    """mean ± std over seeds; rows = arm, columns = (eval set, metric)"""
    out = {}
    for arm, runs in runs_by_arm.items():
        row = {}
        for evalset in evalsets:
            table = summarize_runs(runs, evalset)
            for metric in ("acc", "macro_f1"):
                row[(evalset, metric)] = f"{table.loc['mean', metric]:.3f} ± {table.loc['std', metric]:.3f}"
        out[arm] = row

    return pd.DataFrame(out).T


def per_class_f1(runs: dict[int, dict], evalset: str = "pd_dev_photo") -> pd.DataFrame:
    columns = {}
    for seed, run in runs.items():
        pred = run["logits"][evalset].argmax(1)
        columns[f"seed{seed}"] = f1_score(run["labels"][evalset], pred, average=None, labels=ALL_LABELS, zero_division=0)

    df = pd.DataFrame(columns, index=CLASS_NAMES)
    df["mean"] = df.mean(axis=1)
    df["std"] = df.drop(columns="mean").std(axis=1)
    df["support"] = np.bincount(next(iter(runs.values()))["labels"][evalset], minlength=NUM_CLASSES)
    return df.round(4)


# ARM COMPARISON

def macro_f1_batch(y: np.ndarray, pred: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """macro-F1 for B weighted resamples at once - y, pred: (N,), weights: (B, N) counts.
    matches sklearn f1_score(average="macro", labels=range(8), zero_division=0)"""
    tp = np.stack([(weights * ((y == c) & (pred == c))).sum(1) for c in ALL_LABELS], 1)
    n_true = np.stack([(weights * (y == c)).sum(1) for c in ALL_LABELS], 1)
    n_pred = np.stack([(weights * (pred == c)).sum(1) for c in ALL_LABELS], 1)

    denom = n_true + n_pred
    f1 = np.divide(2 * tp, denom, out=np.zeros_like(denom, dtype=float), where=denom > 0)
    return f1.mean(1)


def compare_arms(runs_by_arm: dict[str, dict[int, dict]], evalset: str = "pd_dev_photo",
                 base: str = "baseline", change: str = "change", n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    """paired comparison on one eval set (the proposal's decision rule).

    macro-F1 is averaged over seeds per arm. the 95% interval resamples photos (all crops of a
    drawn photo come along) with the same draw for both arms, so the gain interval is paired"""
    arms = (base, change)
    reference = next(iter(runs_by_arm[base].values()))
    y = reference["labels"][evalset]

    groups = reference["groups"][evalset]
    if len(groups) != len(y):   # no photo ids (PlantVillage) -> each image is its own unit
        groups = np.arange(len(y))

    for arm in arms:
        for run in runs_by_arm[arm].values():
            if not np.array_equal(run["labels"][evalset], y):
                raise ValueError(f"{evalset}: runs don't share the same rows - retrain on the same split")

    # resample weights per item: (B, N)
    unique, inverse = np.unique(groups, return_inverse=True)
    draws = np.random.default_rng(seed).integers(0, len(unique), size=(n_boot, len(unique)))
    group_counts = np.stack([np.bincount(draw, minlength=len(unique)) for draw in draws])
    weights = group_counts[:, inverse].astype(float)
    ones = np.ones((1, len(y)))

    point, boot, acc = {}, {}, {}
    for arm in arms:
        preds = [run["logits"][evalset].argmax(1) for run in runs_by_arm[arm].values()]
        point[arm] = float(np.mean([macro_f1_batch(y, pred, ones)[0] for pred in preds]))
        boot[arm] = np.mean([macro_f1_batch(y, pred, weights) for pred in preds], axis=0)
        acc[arm] = float(np.mean([(pred == y).mean() for pred in preds]))

    rows = []
    for arm in arms:
        low, high = np.percentile(boot[arm], [2.5, 97.5])
        rows.append({"": arm, "macro_f1": point[arm], "ci_low": low, "ci_high": high, "acc": acc[arm], "seeds": len(runs_by_arm[arm])})

    low, high = np.percentile(boot[change] - boot[base], [2.5, 97.5])
    rows.append(
        {
            "": f"gain ({change} - {base})",
            "macro_f1": point[change] - point[base],
            "ci_low": low,
            "ci_high": high,
            "acc": acc[change] - acc[base],
            "seeds": np.nan,
        }
    )

    majority = np.full_like(y, np.bincount(y, minlength=NUM_CLASSES).argmax())
    rows.append(
        {
            "": "majority-class floor",
            "macro_f1": macro_f1_batch(y, majority, ones)[0],
            "ci_low": np.nan,
            "ci_high": np.nan,
            "acc": (majority == y).mean(),
            "seeds": np.nan,
        }
    )

    df = pd.DataFrame(rows).set_index("")
    df.attrs.update(evalset=evalset, n_units=len(unique), n_items=len(y), gain_ci_above_zero=bool(low > 0))
    return df.round(4)


# PLOTS

def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)

    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def plot_curves(runs_by_arm: dict[str, dict[int, dict]], head_steps: int, evalsets=EVALSETS, smooth: int = 25):
    """train loss + dev macro-F1 per eval set; color = arm, line style = seed"""
    import matplotlib.pyplot as plt

    line_styles = ["-", "--", ":", "-."]
    panels = [(f"train loss (rolling mean, {smooth} steps)", None)]
    panels += [(f"macro-F1: {EVAL_TITLES.get(evalset, evalset)}", f"{evalset}_macro_f1") for evalset in evalsets]

    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 4.2), facecolor=SURFACE)
    for ax, (title, column) in zip(axes, panels):
        style(ax)
        for arm_index, (arm, runs) in enumerate(runs_by_arm.items()):
            for seed_index, (seed, run) in enumerate(sorted(runs.items())):
                line = dict(color=SERIES[arm_index], lw=1.8, ls=line_styles[seed_index % len(line_styles)], label=f"{arm} seed {seed}")
                if column is None:
                    history = run["train_history"]
                    ax.plot(history["step"], history["train_loss"].rolling(smooth, min_periods=1).mean(), **line)
                else:
                    history = run["eval_history"]
                    ax.plot(history["step"], history[column], marker="o", ms=3, **line)

        ax.axvline(head_steps, color=INK_2, lw=1, ls="--")
        ax.text(head_steps, 1.0, " unfreeze backbone", transform=ax.get_xaxis_transform(), color=INK_2, fontsize=8, va="top")
        ax.set_title(title, color=INK, fontsize=11, loc="left")
        ax.set_xlabel("step", color=INK_2, fontsize=9)

    axes[0].legend(frameon=False, fontsize=8, labelcolor=INK)
    fig.tight_layout()
    return fig


def plot_confusion(labels: np.ndarray, logits_or_preds: np.ndarray, title: str = "dev confusion (row-normalized)"):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    pred = logits_or_preds.argmax(1) if logits_or_preds.ndim == 2 else logits_or_preds
    cm = confusion_matrix(labels, pred, labels=ALL_LABELS)
    norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, ax = plt.subplots(figsize=(7.5, 6.5), facecolor=SURFACE)
    image = ax.imshow(norm, cmap=LinearSegmentedColormap.from_list("blue_ramp", BLUE_RAMP), vmin=0, vmax=1)
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            if cm[i, j]:
                ax.text(j, i, f"{norm[i, j]:.2f}\n({cm[i, j]})", ha="center", va="center", fontsize=7,
                        color="white" if norm[i, j] > 0.55 else INK)

    ax.set_xticks(range(NUM_CLASSES), CLASS_NAMES, rotation=40, ha="right", fontsize=8, color=INK_2)
    ax.set_yticks(range(NUM_CLASSES), CLASS_NAMES, fontsize=8, color=INK_2)
    ax.set_xlabel("predicted", color=INK_2)
    ax.set_ylabel("true", color=INK_2)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig, cm
