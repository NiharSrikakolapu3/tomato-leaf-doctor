"""Fit the triage gate for both arms on development photos, and run the Nov 15 checks.

Reads one logits file per arm and seed (format in P4/README.md) and writes:
    gate.json            the fitted gate for every run, which the demo and final run load
    results/gate_fit.json every number behind the decisions below

The served classifier of an arm is its three seeds together, each run on the finder's best
box and on the whole photo, with the six log-probabilities averaged (gate.average_views).
Each seed alone, on both views, is also fitted to show the spread between seeds.

Decisions, in the proposal's order:
  1. top-two rule     questions become referrals in both arms if the change arm has the true
                      class in its top two on under 70% of development photos
  2. per run          temperature on development photos, retake cutoff, then the thresholds
                      with the most right advice at under 5% wrong advice and under 25% questions
  3. risk test        served change arm: under 50% referral, 25% questions, 5% wrong advice
  4. fallback         if the risk test fails, merge up to two most-confused pairs whose care
                      overlaps, in both arms, and test again

Usage:
    python fit_gate.py --logits DIR
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from gate import (OUTCOMES, TOP_TWO_RULE, average_views, calibration_error, cross_fitted,
                  evaluate, fit_gate, merge_labels, shares, softmax, top_two_share)
from labels import CLASSES
from passages import load_passages
from textmatch import content_stems, jaccard

HERE = Path(__file__).parent
ARMS = {"baseline": "lab", "change": "lab_field"}
SEEDS = (0, 1, 2)
VIEWS = ("finder_box", "whole")
SERVED = "change_ensemble"
RISK_LIMITS = {"referral": 0.50, "question": 0.25, "wrong_advice": 0.05}
CARE_OVERLAP_MIN = 0.5
MAX_MERGES = 2


def load_seeds(folder):
    seeds = {}
    for arm, prefix in ARMS.items():
        for seed in SEEDS:
            data = np.load(folder / f"{prefix}_seed{seed}.npz")
            assert list(data["classes"]) == CLASSES, "class order must match labels.py"
            seeds[(arm, seed)] = {key: data[key] for key in data.files}
    return seeds


def served_runs(seeds):
    """Gate inputs: each seed alone on both views, and each arm's three seeds together."""
    first = seeds[("change", 0)]
    for data in seeds.values():
        for key in ("photo", "label", "group", "finder_score"):
            assert np.array_equal(data[key], first[key]), f"{key} differs between runs"
    shared = {key: first[key] for key in ("photo", "label", "group", "finder_score")}

    runs = {}
    for arm in ARMS:
        for seed in SEEDS:
            views = [seeds[(arm, seed)][v] for v in VIEWS]
            runs[f"{arm}_seed{seed}"] = dict(shared, logits=average_views(views))
        views = [seeds[(arm, seed)][v] for seed in SEEDS for v in VIEWS]
        runs[f"{arm}_ensemble"] = dict(shared, logits=average_views(views))
    return runs


def macro_f1(labels, predictions):
    scores = []
    for c in np.unique(labels):
        tp = ((labels == c) & (predictions == c)).sum()
        fp = ((labels != c) & (predictions == c)).sum()
        fn = ((labels == c) & (predictions != c)).sum()
        scores.append(2 * tp / (2 * tp + fp + fn))
    return float(np.mean(scores))


def compare_inputs(seeds, runs, ask_allowed, groups):
    """The change arm through the same gate on each classifier input, means over seeds."""
    def one_view(view):
        return [dict(runs[SERVED], logits=seeds[("change", s)][view]) for s in SEEDS]
    options = {
        "finder box, one seed": one_view("finder_box"),
        "whole photo, one seed": one_view("whole"),
        "both views, one seed": [runs[f"change_seed{s}"] for s in SEEDS],
        "both views, three seeds (served)": [runs[SERVED]],
    }
    compared = {}
    for name, option in options.items():
        fits = [fit_run(run, ask_allowed, groups) for run in option]
        compared[name] = {"macro_f1": float(np.mean([f["macro_f1"] for f in fits])),
                          "in_sample": mean_shares(fits, "in_sample"),
                          "held_out_4_fold": mean_shares(fits, "held_out_4_fold")}
    return compared


def mean_shares(fits, key):
    return {o: float(np.mean([f[key]["shares"][o] for f in fits])) for o in OUTCOMES}


def top_two_check(run):
    share = top_two_share(softmax(run["logits"]), run["label"])
    return {"change_arm_top_two_share": share, "rule": TOP_TWO_RULE,
            "questions_allowed": share >= TOP_TWO_RULE}


def fit_run(run, ask_allowed, groups=None):
    logits, labels, scores = run["logits"], run["label"], run["finder_score"]
    gate = fit_gate(logits, labels, scores, ask_allowed, groups)
    names, probs = evaluate(gate, logits, labels, scores)
    merged = merge_labels(labels, groups)
    kept = scores >= gate.retake_below

    held_out = cross_fitted(logits, labels, scores, run["group"], ask_allowed=ask_allowed, groups=groups)
    raw = softmax(logits)
    if groups is not None:
        raw = np.stack([raw[:, m].sum(axis=1) for m in groups], axis=1)
    return {
        "gate": gate.to_dict(),
        "accuracy": float((probs.argmax(axis=1) == merged).mean()),
        "macro_f1": macro_f1(merged, probs.argmax(axis=1)),
        "calibration_error_before": calibration_error(raw[kept], merged[kept]),
        "calibration_error_after": calibration_error(probs[kept], merged[kept]),
        "in_sample": shares(names),
        "held_out_4_fold": shares(held_out),
        "per_photo_outcome": list(names),
    }


def risk_test(result):
    in_sample = result["in_sample"]["shares"]
    held_out = result["held_out_4_fold"]["shares"]
    return {"limits": RISK_LIMITS, "run": SERVED,
            "in_sample": {o: in_sample[o] for o in OUTCOMES},
            "held_out": {o: held_out[o] for o in OUTCOMES},
            "passed": all(in_sample[o] < limit for o, limit in RISK_LIMITS.items()),
            "held_out_would_pass": all(held_out[o] < limit for o, limit in RISK_LIMITS.items())}


def confused_pairs(runs):
    """Off-diagonal errors of each single seed, both arms, by unordered class pair."""
    counts = {}
    for name, run in runs.items():
        if name.endswith("_ensemble"):
            continue
        predictions = run["logits"].argmax(axis=1)
        for truth, guess in zip(run["label"], predictions):
            if truth != guess:
                pair = tuple(sorted((int(truth), int(guess))))
                counts[pair] = counts.get(pair, 0) + 1
    return sorted(counts.items(), key=lambda item: -item[1])


def care_overlap(passages, a, b):
    stems_a = content_stems(" ".join(passages[CLASSES[a]]["care"]))
    stems_b = content_stems(" ".join(passages[CLASSES[b]]["care"]))
    return jaccard(stems_a, stems_b)


def merge_plan(pairs, passages):
    """Most-confused pairs with overlapping care, up to MAX_MERGES, no class used twice."""
    chosen, used, considered = [], set(), []
    for (a, b), errors in pairs[:10]:
        overlap = care_overlap(passages, a, b)
        considered.append({"pair": [CLASSES[a], CLASSES[b]], "errors": errors, "care_overlap": round(overlap, 3)})
        if overlap >= CARE_OVERLAP_MIN and a not in used and b not in used and len(chosen) < MAX_MERGES:
            chosen.append((a, b))
            used.update((a, b))
    groups = [list(pair) for pair in chosen]
    groups += [[c] for c in range(len(CLASSES)) if c not in used]
    return chosen, groups, considered


def fit_all(runs, ask_allowed, groups=None):
    return {name: fit_run(run, ask_allowed, groups) for name, run in runs.items()}


def print_table(title, fitted):
    print(f"\n{title}")
    print(f"{'run':<19}{'T':>6}{'advise':>8}{'ask':>7}  " + "".join(f"{o:>14}" for o in OUTCOMES))
    for name, r in fitted.items():
        g = r["gate"]
        ask = "off" if g["ask_at"] > 1 else f"{g['ask_at']:.2f}"
        cells = "".join(f"{r['in_sample']['shares'][o]:>14.1%}" for o in OUTCOMES)
        held = "".join(f"{r['held_out_4_fold']['shares'][o]:>14.1%}" for o in OUTCOMES)
        print(f"{name:<19}{g['temperature']:>6.2f}{g['advise_at']:>8.2f}{ask:>7}  {cells}")
        print(f"{'  held out':<40}{held}")


def fit_everything(seeds, show=True):
    """Every decision above for one set of runs. Returns (report, gate_file)."""
    runs = served_runs(seeds)
    top_two = top_two_check(runs[SERVED])
    ask_allowed = top_two["questions_allowed"]

    fitted = fit_all(runs, ask_allowed)
    risk = risk_test(fitted[SERVED])
    if show:
        print_table(f"Questions allowed: {ask_allowed}", fitted)

    report = {
        "photos": int(len(runs[SERVED]["label"])),
        "served": SERVED,
        "views": list(VIEWS),
        "top_two_check": top_two,
        "runs": fitted,
        "risk_test": risk,
    }
    gate_file = {"served": SERVED, "views": list(VIEWS), "questions_allowed": ask_allowed,
                 "merged": [], "runs": {name: r["gate"] for name, r in fitted.items()}}

    if not risk["passed"]:
        passages = load_passages()
        chosen, groups, considered = merge_plan(confused_pairs(runs), passages)
        merged_fit = fit_all(runs, ask_allowed, groups)
        merged_risk = risk_test(merged_fit[SERVED])
        names = [[CLASSES[a], CLASSES[b]] for a, b in chosen]
        if show:
            print_table(f"Fallback, merged {names}", merged_fit)
        report["fallback"] = {"pairs_considered": considered, "merged": names,
                              "runs": merged_fit, "risk_test": merged_risk}
        if chosen:
            gate_file.update(merged=names, runs={n: r["gate"] for n, r in merged_fit.items()})

    groups = gate_file["runs"][SERVED]["groups"]
    report["inputs_compared"] = compare_inputs(seeds, runs, ask_allowed, groups)
    if show:
        print(f"\nChange arm by classifier input ({'merged' if groups else '8 classes'}): in sample, then held out")
        for name, c in report["inputs_compared"].items():
            cells = "  ".join(f"{c[k][o]:>6.1%}" for k in ("in_sample", "held_out_4_fold")
                              for o in ("right_advice", "wrong_advice", "question", "referral"))
            print(f"  {name:<34} F1 {c['macro_f1']:.2f}  {cells}")
        print("\nRisk test (served change arm, in sample):", json.dumps(risk["in_sample"]), "passed:", risk["passed"])
        if "fallback" in report:
            print("After merging:", json.dumps(report["fallback"]["risk_test"]["in_sample"]),
                  "passed:", report["fallback"]["risk_test"]["passed"])
    return report, gate_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logits", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=HERE)
    args = parser.parse_args()

    report, gate_file = fit_everything(load_seeds(args.logits))
    results = args.out / "results"
    results.mkdir(parents=True, exist_ok=True)
    (results / "gate_fit.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.out / "gate.json").write_text(json.dumps(gate_file, indent=2) + "\n")


if __name__ == "__main__":
    main()
