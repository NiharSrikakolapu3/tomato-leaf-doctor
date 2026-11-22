"""How well does the step-to-source check catch altered steps? Tested on the real care sentences.

For every care sentence of every class, the check sees faithful versions, which should
pass, and altered versions, which should be flagged:
    faithful   the sentence itself, and its first clause alone
    altered    a negation removed, a negation added, a number changed, a product swapped,
               a disposal action swapped, an extra instruction appended
An altered version is only made when the sentence has the thing to alter.

Writes results/source_check_validation.json and prints the catch rate for each kind.

Usage:
    python validate_source_check.py
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from passages import load_passages
from source_check import check_step

HERE = Path(__file__).parent
NUMBER_SWAPS = {"one": "three", "two": "five", "10%": "50%", "70%": "30%", "8": "2", "10": "4",
                "14": "40", "2.75": "8", "2.5": "6", "4-5": "1-2", "seven": "two", "few": "many"}
PRODUCT_SWAPS = {"copper": "neem oil", "chlorothalonil": "sulfur", "bleach": "vinegar",
                 "alcohol": "kerosene", "pyrethrin": "malathion", "nitrogen": "potassium"}
ACTION_SWAPS = {"burning": "composting", "burned": "composted", "burn": "compost",
                "burying": "composting", "bury": "compost", "trash": "compost", "landfill": "compost pile"}
EXTRA = ", then spray the plant with a copper fungicide every week."
NEGATION_PATTERNS = [r"\bdo not\b ", r"\bshould not\b", r"\bnot\b ", r"\bavoid\b ", r"\bNOT\b "]


def first_clause(sentence):
    parts = re.split(r",\s+(?:but|and|as|which)\s+|;\s+", sentence, maxsplit=1)
    clause = parts[0].strip()
    return clause if len(parts) > 1 and len(clause.split()) >= 5 else None


def swap_first(sentence, swaps):
    for old, new in swaps.items():
        pattern = rf"\b{re.escape(old)}\b"
        if re.search(pattern, sentence):
            return re.sub(pattern, new, sentence, count=1)
    return None


def drop_negation(sentence):
    for pattern in NEGATION_PATTERNS:
        if re.search(pattern, sentence):
            dropped = re.sub(pattern, "should " if "should" in pattern else "", sentence, count=1)
            return dropped.replace("nor should", "and should")
    return None


def add_negation(sentence):
    if re.search(r"\b(not|no|avoid|never|cannot|without)\b", sentence, re.I):
        return None
    return "Do not " + sentence[0].lower() + sentence[1:]


def variants(sentence):
    altered = {
        "negation removed": drop_negation(sentence),
        "negation added": add_negation(sentence),
        "number changed": swap_first(sentence, NUMBER_SWAPS),
        "product swapped": swap_first(sentence, PRODUCT_SWAPS),
        "disposal action swapped": swap_first(sentence, ACTION_SWAPS),
        "instruction appended": sentence.rstrip(".") + EXTRA,
    }
    faithful = {"same sentence": sentence, "first clause only": first_clause(sentence)}
    return ({k: v for k, v in faithful.items() if v}, {k: v for k, v in altered.items() if v})


def main():
    passages = load_passages()
    tally = {}
    misses = []
    for label, record in passages.items():
        care = record["care"]
        for index, sentence in enumerate(care):
            faithful, altered = variants(sentence)
            for kind, text in faithful.items():
                result = check_step(text, [index], care)
                row = tally.setdefault(("faithful", kind), [0, 0])
                row[0] += result.ok
                row[1] += 1
                if not result.ok:
                    misses.append({"kind": kind, "class": label, "text": text, "problems": result.problems})
            for kind, text in altered.items():
                result = check_step(text, [index], care)
                row = tally.setdefault(("altered", kind), [0, 0])
                row[0] += not result.ok
                row[1] += 1
                if result.ok:
                    misses.append({"kind": kind, "class": label, "text": text, "problems": []})

    report = {"rows": [], "misses": misses}
    print(f"{'version':<10}{'kind':<26}{'right call':>12}")
    for (group, kind), (right, total) in tally.items():
        verdict = "passed" if group == "faithful" else "flagged"
        report["rows"].append({"version": group, "kind": kind, "right": right, "total": total,
                               "right_call": verdict})
        print(f"{group:<10}{kind:<26}{right:>5} of {total:<4} {verdict}")
    out = HERE / "results" / "source_check_validation.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"{len(misses)} wrong calls listed in {out.name}")


if __name__ == "__main__":
    main()
