"""Triage gate: calibrate the classifier's 8 logits, then retake, advise, ask or refer.

The logits are the classifier's log-probabilities averaged over its models and views
(average_views). A photo whose finder score is below the cutoff gets a retake request. Otherwise,
with calibrated probabilities p1 >= p2 >= ...:
    advise  if p1 >= advise_at                 (care steps for the top class)
    ask     if p1 + p2 >= ask_at               (one A-or-B question between the top two)
    refer   otherwise                          (UW Plant Disease Diagnostics Clinic)

Outcomes, as in the proposal: right advice, wrong advice (the advised class is wrong,
or a question leaves out the true class), question (the true class is one of the two),
referral and retake. They sum to 100% of photos.
"""
import hashlib
from dataclasses import asdict, dataclass

import numpy as np

OUTCOMES = ("retake", "right_advice", "wrong_advice", "question", "referral")
WRONG_LIMIT = 0.05
QUESTION_LIMIT = 0.25
RETAKE_SHARE = 0.05
TOP_TWO_RULE = 0.70
# A threshold above every probability switches that route off.
OFF = 1.01


def softmax(logits, temperature=1.0):
    scaled = np.asarray(logits, dtype=np.float64) / temperature
    scaled -= scaled.max(axis=1, keepdims=True)
    exp = np.exp(scaled)
    return exp / exp.sum(axis=1, keepdims=True)


def log_softmax(logits):
    shifted = np.asarray(logits, dtype=np.float64)
    shifted = shifted - shifted.max(axis=1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))


def average_views(views):
    """Average log-probabilities over every model and view, used as the gate's logits.

    The served classifier is an arm's three seeds, each run on the finder's box and on the
    whole photo. P4/README.md has the comparison with a single model on the box.
    """
    return np.mean([log_softmax(v) for v in views], axis=0)


def merge_probs(probs, groups):
    """Sum probabilities within each group of merged classes."""
    if groups is None:
        return probs
    return np.stack([probs[:, members].sum(axis=1) for members in groups], axis=1)


def merge_labels(labels, groups):
    if groups is None:
        return np.asarray(labels)
    lookup = {c: g for g, members in enumerate(groups) for c in members}
    return np.array([lookup[c] for c in labels])


def nll(probs, labels):
    picked = probs[np.arange(len(labels)), labels]
    return float(-np.log(picked + 1e-12).mean())


def fit_temperature(logits, labels, groups=None):
    """The one scalar T that minimizes negative log-likelihood (Guo et al. 2017)."""
    grid = np.exp(np.linspace(np.log(0.25), np.log(20.0), 600))
    losses = [nll(merge_probs(softmax(logits, t), groups), labels) for t in grid]
    return float(grid[int(np.argmin(losses))])


def calibration_error(probs, labels, bins=10):
    confidence = probs.max(axis=1)
    correct = probs.argmax(axis=1) == labels
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        chosen = (confidence > low) & (confidence <= high)
        if chosen.any():
            gap = abs(correct[chosen].mean() - confidence[chosen].mean())
            total += chosen.mean() * gap
    return float(total)


def retake_cutoff(scores, share=RETAKE_SHARE):
    """Largest finder-score cutoff that sends under `share` of photos to retake."""
    ordered = np.sort(np.asarray(scores, dtype=np.float64))
    allowed = int(np.ceil(share * len(ordered))) - 1
    if allowed <= 0:
        return 0.0
    # Midway between the last rejected and the first kept photo.
    return float((ordered[allowed - 1] + ordered[allowed]) / 2)


def top_two(probs):
    order = np.argsort(-probs, axis=1)
    rows = np.arange(len(probs))
    first, second = order[:, 0], order[:, 1]
    return first, second, probs[rows, first], probs[rows, second]


def top_two_share(probs, labels):
    first, second, _, _ = top_two(probs)
    return float(((first == labels) | (second == labels)).mean())


@dataclass
class Gate:
    temperature: float
    advise_at: float
    ask_at: float
    retake_below: float
    groups: list = None  # merged classes as lists of class indices, None for all 8

    def probabilities(self, logits):
        return merge_probs(softmax(logits, self.temperature), self.groups)

    def route(self, logits, finder_scores):
        """Return (routes, first, second, probs) for a batch of photos."""
        probs = self.probabilities(logits)
        first, second, p1, p2 = top_two(probs)
        routes = np.full(len(probs), "refer", dtype=object)
        routes[p1 + p2 >= self.ask_at] = "ask"
        routes[p1 >= self.advise_at] = "advise"
        routes[np.asarray(finder_scores) < self.retake_below] = "retake"
        return routes, first, second, probs

    def to_dict(self):
        return asdict(self)


def outcomes(routes, first, second, labels):
    """Name each photo's outcome for the gardener."""
    labels = np.asarray(labels)
    has_truth = (first == labels) | (second == labels)
    result = np.full(len(labels), "referral", dtype=object)
    result[routes == "retake"] = "retake"
    advise = routes == "advise"
    result[advise & (first == labels)] = "right_advice"
    result[advise & (first != labels)] = "wrong_advice"
    ask = routes == "ask"
    result[ask & has_truth] = "question"
    result[ask & ~has_truth] = "wrong_advice"
    return result


def shares(outcome_names):
    n = len(outcome_names)
    counts = {name: int((outcome_names == name).sum()) for name in OUTCOMES}
    return {
        "photos": n,
        "counts": counts,
        "shares": {name: counts[name] / n for name in OUTCOMES},
    }


def midpoints(values):
    """Candidate thresholds halfway between neighbouring observed values."""
    ordered = np.unique(values)
    middle = (ordered[:-1] + ordered[1:]) / 2
    return np.concatenate([[0.0], middle, [OFF]])


def choose_thresholds(probs, labels, kept, ask_allowed=True, wrong_limit=WRONG_LIMIT,
                      question_limit=QUESTION_LIMIT):
    """Thresholds with the most right advice while wrong advice and questions stay under their limits.

    The limits are the risk test's: under 5% wrong advice and under 25% of photos asked a
    question (every question counts, with or without the true class). Shares count every
    photo, including retakes (`kept` is False for those). Ties go to more questions that
    hold the true class, then to higher thresholds.
    """
    n_all = len(labels)
    probs = probs[kept]
    labels = np.asarray(labels)[kept]
    first, second, p1, p2 = top_two(probs)
    pair = p1 + p2
    correct = first == labels
    has_truth = correct | (second == labels)

    advise_grid = midpoints(p1)
    ask_grid = midpoints(pair) if ask_allowed else np.array([OFF])
    best_key, best = None, None
    for advise_at in advise_grid:
        advise = p1 >= advise_at
        right = int((advise & correct).sum())
        wrong_advised = int((advise & ~correct).sum())
        for ask_at in ask_grid:
            ask = ~advise & (pair >= ask_at)
            good_questions = int((ask & has_truth).sum())
            wrong = wrong_advised + int((ask & ~has_truth).sum())
            if wrong / n_all >= wrong_limit or ask.sum() / n_all >= question_limit:
                continue
            key = (right, good_questions, advise_at, ask_at)
            if best_key is None or key > best_key:
                best_key, best = key, (float(advise_at), float(ask_at))
    return best


def fold_of(group, folds):
    digest = hashlib.sha1(str(group).encode()).hexdigest()
    return int(digest, 16) % folds


def fit_gate(logits, labels, finder_scores, ask_allowed=True, groups=None):
    """Fit temperature, retake cutoff and thresholds on one set of development photos."""
    labels = merge_labels(labels, groups)
    retake_below = retake_cutoff(finder_scores)
    kept = np.asarray(finder_scores) >= retake_below

    # Calibrate on the photos the gate will actually see.
    temperature = fit_temperature(logits[kept], labels[kept], groups)
    probs = merge_probs(softmax(logits, temperature), groups)
    advise_at, ask_at = choose_thresholds(probs, labels, kept, ask_allowed)
    return Gate(temperature, advise_at, ask_at, retake_below, groups)


def evaluate(gate, logits, labels, finder_scores):
    labels = merge_labels(labels, gate.groups)
    routes, first, second, probs = gate.route(logits, finder_scores)
    names = outcomes(routes, first, second, labels)
    return names, probs


def cross_fitted(logits, labels, finder_scores, groups_of_photos, folds=4, **fit_args):
    """Held-out outcome shares: fit on three folds of photo groups, score the fourth."""
    folds_of = np.array([fold_of(g, folds) for g in groups_of_photos])
    names = np.empty(len(labels), dtype=object)
    for fold in range(folds):
        test = folds_of == fold
        gate = fit_gate(logits[~test], np.asarray(labels)[~test],
                        np.asarray(finder_scores)[~test], **fit_args)
        fold_names, _ = evaluate(gate, logits[test], np.asarray(labels)[test],
                                 np.asarray(finder_scores)[test])
        names[test] = fold_names
    return names
