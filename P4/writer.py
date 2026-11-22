"""Advice writer: Qwen2.5-1.5B rewrites retrieved fact-sheet sentences for a gardener.

Two jobs, both with greedy decoding so the output is repeatable:
  care steps   3 to 5 short steps from one class's care sentences, each citing its sentence
  question     one look-at-the-leaf question with choice A (disease A) and choice B (disease B),
               each choice citing a symptom sentence of its own disease
"""
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from labels import DISPLAY

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
SYSTEM = (
    "You help home gardeners. Use only the numbered fact sheet sentences you are given. "
    "Keep every warning, every 'do not', every number and every product name as written. "
    "Never add advice that is not in the sentences."
)
STEP_LINE = re.compile(r"^\s*\d+[.)]\s*(.+?)\s*$")
CITATION = re.compile(r"\[\s*([\d,\s]+)\]")
CHOICE_LINE = re.compile(r"^\s*(?:Choice\s+)?([AB])\s*[:.)]\s*(.+?)\s*(?:\[\s*[AB]?\s*(\d+)\s*\])?\s*\.?\s*$")


class Writer:
    def __init__(self, device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = device
        dtype = torch.float32 if device == "cpu" else torch.float16
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        self.model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=dtype).to(device).eval()

    def generate(self, prompt, max_new_tokens=300):
        messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        new_tokens = output[0][inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def numbered(sentences, prefix=""):
    return "\n".join(f"[{prefix}{i + 1}] {s}" for i, s in enumerate(sentences))


STEP_RULES = (
    "Each step shortens one sentence above into a plain instruction of at most 20 words. "
    "Reuse that sentence's own words, keep any 'do not', number, product name or 'where allowed' "
    "wording, and add nothing else. End each step with that sentence's number in brackets.\n"
    "Format:\n1. <step> [<number>]\n2. <step> [<number>]"
)


def care_prompt(label, care):
    name = DISPLAY[label]
    return (
        f"Fact sheet sentences about {name} on tomato:\n{numbered(care)}\n\n"
        f"Write 3 to 5 care steps that a home gardener can follow. {STEP_RULES}"
    )


def healthy_prompt(care):
    return (
        f"Fact sheet sentences about caring for tomato plants:\n{numbered(care)}\n\n"
        f"The plant looks healthy. Write 3 to 5 general care steps that keep it healthy. {STEP_RULES}"
    )


def parse_steps(text):
    """Return [(step text, [0-based sentence indices])] from numbered lines.

    The model puts the citation at the end, at the start, or restates the sentence in
    brackets instead. Bracketed numbers are citations, other bracketed text is dropped,
    and a step with no number is re-matched by the source check.
    """
    steps = []
    for line in text.splitlines():
        match = STEP_LINE.match(line)
        if not match:
            continue
        body = match.group(1)
        cited = [int(n) - 1 for group in CITATION.findall(body) for n in re.findall(r"\d+", group)]
        body = re.sub(r"\[[^\]]*\]", " ", body)
        body = re.sub(r"\s+", " ", body).strip(" .;")
        if body:
            steps.append((body + ".", cited))
    return steps


QUESTION_STEM = "Look closely at the leaf. Which of these fits it better?"
SIGN_STEM = "Look closely at the leaf. Do you see this?"


def sign_prompt(sentences):
    """Shorten a disease's key leaf-sign sentences into one choice a gardener can check."""
    return (
        f"Fact sheet sentences:\n{numbered(sentences)}\n\n"
        f"Describe the leaf sign in these sentences in at most 12 plain words, so a gardener can check "
        f"a leaf for it. Reuse the sentences' own words, describe only the leaf, and do not name the "
        f"disease. End with the number of the sentence you used most.\n"
        f"Format:\nA: <leaf sign> [<number>]"
    )


def parse_choices(text):
    """Return {"A": (choice, index or None), "B": ...}; a missing citation is None."""
    choices = {}
    for line in text.splitlines():
        match = CHOICE_LINE.match(line)
        if match:
            index = int(match.group(3)) - 1 if match.group(3) else None
            choices.setdefault(match.group(1), (match.group(2).strip(), index))
    return choices


def parse_sign(text):
    """The first non-empty line, with any "A:" or "B:" label and bracketed number removed."""
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        cited = [int(n) - 1 for n in re.findall(r"\[\s*<?(\d+)>?\s*\]", line)]
        line = re.sub(r"^\s*(?:Choice\s+)?[AB]\s*[:.)]\s*", "", line)
        line = re.sub(r"[\[(]\s*<?\d+>?\s*[\])]", "", line).strip(" .")
        return line + ".", cited
    return None, []
