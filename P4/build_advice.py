"""Build the advice table: care steps for every class, and one question for every pair.

Care steps come from the writer (Qwen2.5-1.5B) and go through the step-to-source check.
For questions, each disease's key leaf sign (the symptom sentence its sheet's photo points
at) is shortened by the writer into one choice. A pair is only asked about when both
choices use a word the other sheet lacks, otherwise the gate refers it to the clinic.
Anything that fails a check is replaced by its source sentence, quoted.
A merged pair of classes (gate.json "merged") gets its own entry built from both sheets.

Writes:
    results/advice_table.json   what the demo shows, with every check result
    results/review_sheet.md     the two-teammate check: each step beside its sentence

Usage:
    python build_advice.py [--device mps]
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from labels import CLASSES, DISPLAY
from passages import load_passages
from source_check import check_step, checked_steps, is_distinctive, quote
from sources import CLINIC
from textmatch import content_stems, jaccard, negation_count, stem, tokens
from writer import (MODEL_ID, QUESTION_STEM, SIGN_STEM, Writer, care_prompt, healthy_prompt,
                    parse_sign, parse_steps, sign_prompt)

HERE = Path(__file__).parent
NEAR_DUPLICATE = 0.8
NO_SIGN = "No, I do not see this on the leaf."
# A care sentence that forbids one of these is always shown, quoted, whatever the writer picked.
WARNING_WORDS = {stem(w) for w in "compost composted eat eaten preserve preserved handle buy".split()}
MAX_WARNINGS = 3


def sentence_records(passages, label, part):
    record = passages[label]
    return [{"text": s, "class": label, "url": record["url"], "title": record["title"]}
            for s in record[part]]


def topic(passages, labels):
    """One class, or a merged pair whose sentences are pooled without near-duplicates."""
    care, symptoms = [], []
    for label in labels:
        for part, pooled in (("care", care), ("symptoms", symptoms)):
            for record in sentence_records(passages, label, part):
                stems = content_stems(record["text"])
                if all(jaccard(stems, content_stems(r["text"])) < NEAR_DUPLICATE for r in pooled):
                    pooled.append(record)
    return {
        "key": "+".join(labels),
        "classes": list(labels),
        "name": " or ".join(DISPLAY[label] for label in labels),
        "sources": [{"title": passages[l]["title"], "url": passages[l]["url"],
                     "care_sha": passages[l]["care_sha"], "symptoms_sha": passages[l]["symptoms_sha"]}
                    for l in labels],
        "care": care,
        "symptoms": symptoms,
    }


def write_care(writer, item):
    texts = [r["text"] for r in item["care"]]
    if item["classes"] == ["healthy"]:
        prompt = healthy_prompt(texts)
    else:
        # A merged entry reads as its first class; its pooled sentences cover both sheets.
        prompt = care_prompt(item["classes"][0], texts)
    raw = writer.generate(prompt)
    parsed = parse_steps(raw)
    if not parsed:
        # Nothing usable came back: quote the first three sentences instead.
        parsed = [("", [i]) for i in range(min(3, len(texts)))]
    steps = checked_steps(parsed, texts)
    for step in steps:
        step["source"] = item["care"][step["check"]["cited"][0]]["url"]
    return {"raw": raw, "steps": steps, "warnings": warnings(item, steps)}


def warnings(item, steps):
    """The sheet's own prohibitions ("should not be composted"), quoted word for word."""
    shown = {i for step in steps for i in step["check"]["cited"]}
    found = []
    for index, record in enumerate(item["care"]):
        words = tokens(record["text"])
        forbids = negation_count(words) > 0 and any(stem(w) in WARNING_WORDS for w in words)
        if forbids and index not in shown and len(found) < MAX_WARNINGS:
            found.append({"shown": f"“{record['text']}”", "index": index, "source": record["url"]})
    return found


def key_sign_sentences(passages, label):
    """The key sign sentence, with the one before it when it starts by pointing back ("these spots")."""
    record = passages[label]
    index = record["key_sign"]
    start = index - 1 if index > 0 else index
    return record["symptoms"][start : index + 1]


def write_sign(writer, passages, label):
    sentences = key_sign_sentences(passages, label)
    raw = writer.generate(sign_prompt(sentences))
    text, cited = parse_sign(raw)
    if text:
        check = check_step(text, cited, sentences).to_dict()
    else:
        check = {"text": None, "cited": [len(sentences) - 1], "ok": False, "problems": ["no sign written"]}
    shown = text if check["ok"] else quote(sentences, [len(sentences) - 1])
    return {"class": label, "raw": raw, "sentences": sentences, "shown": shown,
            "quoted": not check["ok"], "check": check}


def topic_sign(signs, item):
    """A merged entry shows each member's sign, joined by 'or'."""
    parts = [signs[label] for label in item["classes"]]
    return {"shown": " or ".join(p["shown"] for p in parts), "quoted": any(p["quoted"] for p in parts),
            "members": [p["class"] for p in parts]}


def pair_question(signs, item_a, item_b):
    own_a = [r["text"] for r in item_a["symptoms"]]
    own_b = [r["text"] for r in item_b["symptoms"]]
    choice_a = dict(topic_sign(signs, item_a), leads_to=item_a["key"])
    choice_a["distinctive"] = is_distinctive(choice_a["shown"], own_a, own_b)
    if item_b["classes"] == ["healthy"]:
        stem = SIGN_STEM
        choice_b = {"shown": NO_SIGN, "fixed_text": True, "leads_to": item_b["key"], "distinctive": ["(no sign)"]}
    else:
        stem = QUESTION_STEM
        choice_b = dict(topic_sign(signs, item_b), leads_to=item_b["key"])
        choice_b["distinctive"] = is_distinctive(choice_b["shown"], own_b, own_a)
    asked = bool(choice_a["distinctive"]) and bool(choice_b["distinctive"])
    return {"pair": [item_a["key"], item_b["key"]], "asked": asked, "stem": stem,
            "choices": {"A": choice_a, "B": choice_b}, "not_sure": "refer"}


def review_sheet(table):
    lines = ["# Advice review sheet", "",
             "Two teammates each tick every step that says only what its quoted sentence says.",
             "Quoted steps are the fallback and need no tick. For each question, answer it from the",
             "named class's sheet and write A, B or not sure.", ""]
    for key, entry in table["care"].items():
        lines += [f"## Care: {entry['name']}", ""]
        lines += [f"Sources: " + ", ".join(f"[{s['title']}]({s['url']})" for s in entry["sources"]), ""]
        lines += ["| # | Shown to the gardener | Source sentence | Check | Reviewer 1 | Reviewer 2 |",
                  "| --- | --- | --- | --- | --- | --- |"]
        for n, step in enumerate(entry["steps"], 1):
            cited = step["check"]["cited"][0]
            source = entry["care_sentences"][cited].replace("|", "/")
            status = "quoted" if step["quoted"] else "passed"
            lines.append(f"| {n} | {step['shown']} | [{cited + 1}] {source} | {status} | [ ] | [ ] |")
        lines.append("")
    lines += ["## Leaf signs used in questions", "",
              "| Class | Shown | Source sentences | Check | Reviewer 1 | Reviewer 2 |",
              "| --- | --- | --- | --- | --- | --- |"]
    for label, sign in table["signs"].items():
        source = " ".join(sign["sentences"]).replace("|", "/")
        status = "quoted" if sign["quoted"] else "passed"
        lines.append(f"| {label} | {sign['shown']} | {source} | {status} | [ ] | [ ] |")
    lines += ["", "## Questions", "",
              "Answer each from the named class's sheet: A, B or not sure.", "",
              "| Pair | Asked | Choice A | Choice B | Answer from A's sheet | Answer from B's sheet |",
              "| --- | --- | --- | --- | --- | --- |"]
    for q in table["questions"]:
        a, b = q["choices"]["A"]["shown"], q["choices"]["B"]["shown"]
        asked = "yes" if q["asked"] else "no, refer"
        lines.append(f"| {q['pair'][0]} vs {q['pair'][1]} | {asked} | {a} | {b} | | |")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", type=Path, default=HERE / "results")
    args = parser.parse_args()

    passages = load_passages()
    gate = json.loads((HERE / "gate.json").read_text())
    groups = [[label] for label in CLASSES] + [list(pair) for pair in gate["merged"]]
    items = {"+".join(g): topic(passages, g) for g in groups}

    writer = Writer(args.device)
    care = {}
    for key, item in items.items():
        written = write_care(writer, item)
        care[key] = {"name": item["name"], "classes": item["classes"], "sources": item["sources"],
                     "care_sentences": [r["text"] for r in item["care"]], **written}
        print(f"care {key}: {sum(not s['quoted'] for s in written['steps'])} of {len(written['steps'])} steps passed")

    signs = {label: write_sign(writer, passages, label) for label in CLASSES if label != "healthy"}
    print(f"leaf signs: {sum(not s['quoted'] for s in signs.values())} of {len(signs)} written signs passed")

    questions = []
    keys = list(items)
    for key_a, key_b in itertools.combinations(keys, 2):
        if set(items[key_a]["classes"]) & set(items[key_b]["classes"]):
            continue  # a merged entry is never asked against one of its own classes
        if "healthy" in items[key_a]["classes"]:
            key_a, key_b = key_b, key_a  # the healthy side is always choice B
        questions.append(pair_question(signs, items[key_a], items[key_b]))
    print(f"questions: {sum(q['asked'] for q in questions)} of {len(questions)} pairs asked, the rest referred")

    table = {"writer": MODEL_ID, "decoding": "greedy", "clinic": CLINIC, "care": care,
             "signs": signs, "questions": questions}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "advice_table.json").write_text(json.dumps(table, indent=2, ensure_ascii=False) + "\n")
    (args.out / "review_sheet.md").write_text(review_sheet(table))


if __name__ == "__main__":
    main()
