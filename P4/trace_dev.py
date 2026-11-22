"""Trace development photos through all four stages and check the pipeline matches the gate fit.

The pipeline reruns the finder and classifier on this machine, so its routes are compared
with the routes the fitted gate gives on the saved logits. Small differences can come from
running the finder on a different GPU.

Usage:
    python trace_dev.py --weights DIR --logits DIR --photos PLANTDOC_DIR [--run change_ensemble]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from fit_gate import load_seeds, served_runs
from gate import merge_labels
from labels import CLASSES
from pipeline import HERE, LeafDoctor, open_photo
from urllib.parse import unquote


def photo_path(folder, key):
    path = folder / key
    return path if path.exists() else folder / unquote(key)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--logits", type=Path, required=True)
    parser.add_argument("--photos", type=Path, required=True)
    parser.add_argument("--run", default="change_ensemble")
    args = parser.parse_args()

    saved = served_runs(load_seeds(args.logits))[args.run]
    doctor = LeafDoctor(args.weights, args.run)
    gate = doctor.gate
    expected, _, _, _ = gate.route(saved["logits"], saved["finder_score"])
    labels = merge_labels(saved["label"], gate.groups)

    traces = []
    for index, key in enumerate(saved["photo"]):
        trace = doctor.diagnose(open_photo(photo_path(args.photos, str(key))))
        trace["photo"] = str(key)
        trace["true_class"] = CLASSES[saved["label"][index]]
        trace["gate_route_on_saved_logits"] = str(expected[index])
        trace["logit_difference"] = float(np.abs(np.array(trace["stage2_classifier"]["log_probabilities"])
                                                 - saved["logits"][index]).max())
        traces.append(trace)
        if (index + 1) % 20 == 0:
            print(f"{index + 1} photos traced", flush=True)

    gate_routes = [t["stage3_gate"]["route"] for t in traces]
    agree = float(np.mean([a == b for a, b in zip(gate_routes, expected)]))
    summary = {"run": args.run, "photos": len(traces), "route_agreement_with_saved_logits": agree,
               "median_logit_difference": float(np.median([t["logit_difference"] for t in traces])),
               "final_routes": {r: sum(t["route"] == r for t in traces)
                                for r in ("retake", "advise", "ask", "refer")}}
    print(json.dumps(summary, indent=2))
    out = HERE / "results" / "dev_traces.json"
    out.write_text(json.dumps({"summary": summary, "traces": traces}, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
