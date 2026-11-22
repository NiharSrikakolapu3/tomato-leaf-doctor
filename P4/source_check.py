"""Step-to-source check: does each written step say only what its fact-sheet sentence says?

A step passes when all of these hold against the sentence(s) it cites:
  support   every content word of the step appears in the source (after stemming and a
            short list of same-meaning words), so a swapped product or quantity fails
  numbers   every number in the step appears in the source
  negation  the step has as many negations ("do not", "avoid") as the part of the source it
            covers, so "do not compost" cannot become "compost"
  risk      named separately when a missing word is a treatment or disposal word
  qualifier a step that burns keeps "where allowed by local ordinance", and a step that
            applies a pesticide keeps its label wording, when the source has them
A step with a wrong or missing citation is re-matched to the best sentence, or the best
two neighbouring sentences, first.
A step that still fails is replaced by its source sentence, quoted word for word.
Known blind spot: swapping two words that are both in the sentence ("bury or hot compost"
becoming "compost or hot compost") keeps the same words and passes.
"""
import re
from collections import Counter
from dataclasses import asdict, dataclass, field

from textmatch import STOPWORDS, content_stems, negation_count, numbers, stem, tokens

SUPPORT_MIN = 1.0
CLAUSE_END = re.compile(r"[,;:()]")
# Words whose addition changes what the gardener does to the plant or the fruit.
RISK_WORDS = {
    stem(w)
    for w in """fungicide fungicides copper chlorothalonil neem sulfur sulphur bleach alcohol
    insecticide insecticides pesticide pesticides herbicide herbicides spray sprays spraying
    compost composting burn burning bury burying eat eating edible soap oil oils chemical
    chemicals cure cured kill treat treatment apply application fertilizer nitrogen water
    watering prune pruning cut remove removal destroy dispose disposal landfill trash""".split()
}
# (words in the step, phrase the source sentence must not lose)
QUALIFIERS = [
    ({stem(w) for w in "burn burning burned".split()}, "local ordinance"),
    ({stem(w) for w in "fungicide fungicides insecticide insecticides pesticide copper chlorothalonil".split()},
     "label"),
]
# Words a rewrite may add without changing the instruction ("make sure your tomatoes ...").
FREE_WORDS = {stem(w) for w in "tomato tomatoes garden gardener help sure make keep".split()}


@dataclass
class StepCheck:
    text: str
    cited: list
    ok: bool
    support: float
    problems: list = field(default_factory=list)
    recited: bool = False

    def to_dict(self):
        return asdict(self)


def source_negations(step_stems, sentence):
    """Negations in the source sentence that negate something the step says.

    A negation counts when the step uses the first content word after it ("not be
    composted", "avoid overhead watering"), or a later word of its clause that the sentence
    uses nowhere else ("nor canned or otherwise preserved"). So "do not eat symptomatic
    fruit" does not count against a step about burning plants.
    """
    clauses = [tokens(clause) for clause in CLAUSE_END.split(sentence)]
    words = [w for clause in clauses for w in clause]
    in_sentence = Counter(content_stems(sentence))
    count = 0
    start = 0
    for clause in clauses:
        for offset, word in enumerate(clause):
            if negation_count([word]) == 0:
                continue
            position = start + offset
            following = [w for w in words[position + 1 : position + 5] if w not in STOPWORDS]
            negated = {stem(following[0])} if following else set()
            rest = Counter(content_stems(" ".join(clause[offset + 1 :])))
            negated |= {s for s, n in rest.items() if n == in_sentence[s]}
            if negated & step_stems:
                count += 1
        start += len(clause)
    return count


def judge(step, sentences):
    """Problems with `step` against the given source sentences, and its support."""
    source_text = " ".join(sentences)
    source_stems = set(content_stems(source_text))
    step_stems = [s for s in content_stems(step) if s not in FREE_WORDS]
    if not step_stems:
        return ["no content words"], 0.0

    missing = [s for s in step_stems if s not in source_stems]
    support = 1 - len(missing) / len(step_stems)
    problems = []
    if support < SUPPORT_MIN:
        problems.append(f"words not in source: {' '.join(sorted(set(missing)))}")

    extra_numbers = numbers(step) - numbers(source_text)
    if extra_numbers:
        problems.append(f"numbers not in source: {' '.join(sorted(extra_numbers))}")

    step_negations = negation_count(tokens(step))
    covered_negations = sum(source_negations(set(step_stems), s) for s in sentences)
    if step_negations != covered_negations:
        problems.append(f"{step_negations} negations, source part has {covered_negations}")

    risky = sorted({s for s in missing if s in RISK_WORDS})
    if risky:
        problems.append(f"adds action words: {' '.join(risky)}")

    for triggers, phrase in QUALIFIERS:
        in_source = phrase in source_text.lower()
        if in_source and triggers & set(step_stems) and phrase not in step.lower():
            problems.append(f"drops the source's '{phrase}' wording")
    return problems, support


def check_step(step, cited, passage):
    """Check one step against the passage sentences it cites (0-based indices).

    If the cited sentences fail, every single sentence is tried, then every neighbouring
    pair whose two sentences are both needed to cover the step's words (so an altered copy
    of one sentence cannot borrow a negation from its neighbour). The first span that passes
    is used. If none passes, the quoted fallback is the cited sentence, which is what the
    writer meant to rewrite, or the best-supported span when the step cited nothing usable.
    """
    valid = [i for i in cited if 0 <= i < len(passage)]
    if valid:
        problems, support = judge(step, [passage[i] for i in valid])
        if not problems:
            return StepCheck(step, valid, True, round(support, 3), [], False)

    singles = [judge(step, [sentence]) for sentence in passage]
    spans = [([i], *singles[i]) for i in range(len(passage))]
    for i in range(len(passage) - 1):
        either_covers = singles[i][1] == 1.0 or singles[i + 1][1] == 1.0
        if not either_covers:
            spans.append(([i, i + 1], *judge(step, passage[i : i + 2])))

    for span, span_problems, span_support in spans:
        if not span_problems:
            return StepCheck(step, span, True, round(span_support, 3), [], span != valid)
    if valid:
        return StepCheck(step, valid, False, round(support, 3), problems, False)
    span, span_problems, span_support = max(spans, key=lambda item: item[2])
    return StepCheck(step, span, False, round(span_support, 3), span_problems, True)


def quote(passage, indices):
    return "“" + " ".join(passage[i] for i in indices) + "”"


def checked_steps(steps, passage):
    """Run the check on (text, cited) steps, quoting the source for any that fail.

    A quote that repeats an earlier quote is dropped. A quote of a sentence that an earlier
    step only shortened replaces that step, so the rest of the sentence is not lost.
    """
    results = []
    seen_texts = set()
    for text, cited in steps:
        if text and text.lower() in seen_texts:
            continue  # the writer sometimes repeats its list
        seen_texts.add(text.lower())
        check = check_step(text, cited, passage) if text else None
        if check is None:
            check = StepCheck("", cited, False, 0.0, ["no step written"], False)
        if not check.ok:
            same = [r for r in results if set(check.cited) <= set(r["check"]["cited"])]
            if any(r["quoted"] for r in same):
                continue
            results = [r for r in results if r not in same]
        shown = text if check.ok else quote(passage, check.cited)
        results.append({"shown": shown, "quoted": not check.ok, "check": check.to_dict()})
    return results


def is_distinctive(option, own, other):
    """The option uses at least one content word that only its own disease's sheet has."""
    own_stems = set(content_stems(" ".join(own)))
    other_stems = set(content_stems(" ".join(other)))
    option_stems = set(content_stems(option))
    return sorted(option_stems & (own_stems - other_stems))
