"""Put each stand-in backbone through the same gate and write results/backbone_comparison.json.

ROOT has one folder per backbone, as tools/train_stand_in.py writes them. A backbone with
all six runs (both arms, three seeds) gets every decision in fit_gate.py, including the
risk test and the fallback. A backbone with only the change arm's seed 0 was screened
out, and gets that one run on both views with all 8 classes kept.

Usage:
    python tools/compare_backbones.py --root RUNS
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))
from fit_gate import HERE, SEEDS, fit_everything, fit_run, load_seeds, macro_f1
from gate import average_views

SHARES = ("right_advice", "wrong_advice", "question", "referral")


def rounded(shares):
    return {o: round(100 * shares[o], 1) for o in SHARES}


def full_fit(folder):
    report, gate_file = fit_everything(load_seeds(folder), show=False)
    final = report.get("fallback", report)
    return {
        "runs": "both arms, three seeds",
        "change_macro_f1_8_classes": round(report["runs"]["change_ensemble"]["macro_f1"], 3),
        "change_macro_f1_per_seed": [round(report["runs"][f"change_seed{s}"]["macro_f1"], 3) for s in SEEDS],
        "lab_only_macro_f1_8_classes": round(report["runs"]["baseline_ensemble"]["macro_f1"], 3),
        "merged": gate_file["merged"],
        "risk_test_passed": final["risk_test"]["passed"],
        "per_100_photos_in_sample": rounded(final["risk_test"]["in_sample"]),
        "per_100_photos_held_out": rounded(final["risk_test"]["held_out"]),
    }


def screen_fit(folder):
    data = np.load(folder / "lab_field_seed0.npz")
    run = {key: data[key] for key in ("photo", "label", "group", "finder_score")}
    run["logits"] = average_views([data["finder_box"], data["whole"]])
    result = fit_run(run, ask_allowed=True)
    return {
        "runs": "change arm, seed 0 only",
        "change_macro_f1_8_classes": round(macro_f1(run["label"], run["logits"].argmax(axis=1)), 3),
        "per_100_photos_in_sample": rounded(result["in_sample"]["shares"]),
        "per_100_photos_held_out": rounded(result["held_out_4_fold"]["shares"]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()

    comparison = {}
    for folder in sorted(p for p in args.root.iterdir() if p.is_dir()):
        complete = all((folder / f"{arm}_seed{s}.npz").exists() for arm in ("lab", "lab_field") for s in SEEDS)
        row = full_fit(folder) if complete else screen_fit(folder)
        row["parameters_millions"] = json.loads((folder / "lab_field_seed0.json").read_text())["parameters_millions"]
        comparison[folder.name] = row
        print(folder.name, json.dumps(row))
    (HERE / "results" / "backbone_comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")


if __name__ == "__main__":
    main()
