# P4: Triage gate and advice writer

P4 turns the classifier's output into what the gardener sees: a request for a clearer photo, care steps, one question between two diseases, or a referral to the UW-Madison Plant Disease Diagnostics Clinic. It also writes and checks the advice text and runs the demo.

| Stage | File | Neural | What it does |
| --- | --- | --- | --- |
| Triage gate | `gate.py`, `fit_gate.py` | No (one fitted scalar and two thresholds) | Averages the classifier's views, rescales its confidence on development photos, then retake, advise, ask or refer |
| Passages | `sources.py`, `passages.py` | No | Fetches the 8 fact sheets and cuts their symptom and care sections into numbered sentences |
| Advice writer | `writer.py`, `build_advice.py` | Yes, Qwen2.5-1.5B-Instruct, frozen | Shortens care sentences into steps and each disease's key leaf sign into a question choice |
| Step-to-source check | `source_check.py`, `textmatch.py` | No | Passes a step only if it says nothing its cited sentence does not. Anything else is quoted from the sheet |
| Demo | `pipeline.py`, `demo.py`, `trace_dev.py` | Runs all four stages | Gradio app and a trace of every development photo |

## Inputs

P2's classifier and P3's finder results are not in the repository yet. Until they are, the classifier is a stand-in, DINOv3 ViT-B/16 fine-tuned by `tools/train_stand_in.py` with the pilot's data and recipe, and the finder boxes are the pilot's OWLv2 boxes. When P2's runs exist, save them in the format below and rerun `fit_gate.py` and `build_advice.py`.

Each run is one `.npz` file named `lab_seed{0,1,2}.npz` (baseline) or `lab_field_seed{0,1,2}.npz` (change), over the 112 development photos:

| Key | Shape | Meaning |
| --- | --- | --- |
| `photo`, `group` | (112,) | PlantDoc photo key and its near-duplicate group |
| `label` | (112,) | True class index in `labels.CLASSES` order |
| `whole`, `finder_box` | (112, 8) | Logits on the whole photo and on the finder's best box (10% margin) |
| `finder_score` | (112,) | OWLv2 score of that box, 0 when none was found |
| `classes` | (8,) | Class names, checked against `labels.CLASSES` |

The weights folder holds `{lab,lab_field}_seed{0,1,2}.pt` and a `classifier.json` naming the network. The final partition is never read here. Every number below is on development photos.

DINOv3 was picked from six backbones on these same 112 photos, so its numbers are optimistic. Its change-arm macro-F1 is 0.72 against 0.59 for the proposal's EfficientNet-B0, and it is the only backbone tried that passes the risk test with all 8 classes and keeps held-out wrong advice under 5%. `results/backbone_comparison.json` has all six.

## Gate

The gate averages the classifier's log-probabilities over the three seeds, each run on the finder's best box and on the whole photo. It fits a temperature on development photos, sets the finder cutoff so under 5% of photos get a retake request, and picks the thresholds with the most right advice at under 5% wrong advice and under 25% questions.

Two rules differ from the proposal: it averages both views of all three seeds instead of one chosen view, and the 25% question cap is part of the threshold choice.

Outcome shares, 8 classes. "Held out" fits the gate on 3 of 4 folds of photo groups and scores the fourth.

| Arm | Fit | Retake | Right advice | Wrong advice | Question | Referral |
| --- | --- | --- | --- | --- | --- | --- |
| Baseline (lab only) | In sample | 4.5% | 7.1% | 4.5% | 1.8% | 82.1% |
| Baseline (lab only) | Held out | 5.4% | 5.4% | 8.0% | 4.5% | 76.8% |
| Change (lab and real) | In sample | 4.5% | 35.7% | 4.5% | 7.1% | 48.2% |
| Change (lab and real) | Held out | 5.4% | 35.7% | 4.5% | 2.7% | 51.8% |

The risk test passes in sample with all 8 classes, so no classes are merged. Held-out referral is just over the 50% limit. `results/gate_fit.json` has every run and the per-photo outcomes.

## Advice

`build_advice.py` builds one care card per class and one question per pair of classes.

- **Care steps.** The writer turns a class's numbered care sentences into 3 to 5 steps, each citing a sentence. 12 of 25 steps pass the check, and the rest are shown as quotes of the sentence they cite. Every card also quotes the sheet's own prohibitions, such as "should not be composted".
- **Questions.** Each disease's key leaf sign is shortened to one choice. 3 of 7 signs pass, and the rest are quoted. All 28 pairs get a question. "Not sure" always refers.
- **Review sheet.** `results/review_sheet.md` puts every shown step beside its source sentence for the two-teammate check.

The step-to-source check passes a step only if every content word and number is in its cited sentence, it has as many negations as that sentence, and it keeps wording such as "where allowed by local ordinance". On altered copies of all 148 care sentences (`validate_source_check.py`), it passes all 167 faithful versions and flags every removed or added negation, changed number and appended instruction, plus 16 of 21 swapped products or disposal actions.

## Running it

Requires PyTorch, torchvision, timm, transformers, nltk, Pillow and NumPy, plus gradio for the demo. The models download from Hugging Face on first use.

```bash
cd P4
python tools/train_stand_in.py --backbone dinov3_vitb --arm lab_field --seed 0 \
    --pilot REPO/pilot --splits REPO/splits --plantvillage DATA/pv --plantdoc DATA/pd \
    --runs PILOT_RUNS --cache CACHE --out RUNS          # once per arm and seed, on a GPU
python tools/compare_backbones.py --root RUNS          # writes results/backbone_comparison.json
python passages.py                                     # fetch the 8 sheets into .cache/ (git-ignored)
python fit_gate.py --logits RUNS/dinov3_vitb           # writes gate.json and results/gate_fit.json
python build_advice.py                                 # writes results/advice_table.json and review_sheet.md
python validate_source_check.py                        # writes results/source_check_validation.json
python trace_dev.py --weights RUNS/dinov3_vitb --logits RUNS/dinov3_vitb --photos PLANTDOC_DIR
python demo.py --weights RUNS/dinov3_vitb --share      # Gradio app, public link in Colab
python -m pytest tests                                 # gate and source-check unit tests
```

`fit_gate.py` takes 3 seconds, `build_advice.py` about 2 minutes on an Apple M4 Pro, and the trace about 1 minute. The trace agrees with the gate on the saved logits for 111 of 112 photos.

## Files

| File | Contents |
| --- | --- |
| `gate.json` | Fitted gate for every run. The demo and the final run load it |
| `results/gate_fit.json` | Every gate number above, per run, with per-photo outcomes |
| `results/backbone_comparison.json` | The six backbones through the same gate |
| `results/advice_table.json` | Every card, sign and question, with the writer's raw output and each check result |
| `results/review_sheet.md` | The two-teammate check |
| `results/source_check_validation.json` | The check's test on altered sentences |
| `results/dev_traces.json` | All 112 development photos through all four stages |
