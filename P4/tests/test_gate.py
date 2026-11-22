"""Gate checks on small made-up batches. Run: python -m pytest P4/tests"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))
from gate import OUTCOMES, Gate, average_views, choose_thresholds, fit_temperature, outcomes, retake_cutoff, shares, softmax


def test_routes_and_outcomes_sum_to_all_photos():
    gate = Gate(temperature=1.0, advise_at=0.7, ask_at=0.9, retake_below=0.2)
    logits = np.log(np.array([
        [0.80, 0.10, 0.10],  # advise class 0
        [0.50, 0.45, 0.05],  # ask 0 or 1
        [0.40, 0.30, 0.30],  # refer
        [0.90, 0.05, 0.05],  # retake, finder score too low
    ]))
    routes, first, second, _ = gate.route(logits, [0.9, 0.9, 0.9, 0.1])
    assert list(routes) == ["advise", "ask", "refer", "retake"]

    names = outcomes(routes, first, second, labels=[1, 1, 0, 0])
    assert list(names) == ["wrong_advice", "question", "referral", "retake"]
    counts = shares(names)["counts"]
    assert sum(counts[o] for o in OUTCOMES) == 4


def test_question_without_the_true_class_is_wrong_advice():
    gate = Gate(temperature=1.0, advise_at=0.9, ask_at=0.8, retake_below=0.0)
    logits = np.log(np.array([[0.5, 0.4, 0.1]]))
    routes, first, second, _ = gate.route(logits, [1.0])
    assert outcomes(routes, first, second, labels=[2])[0] == "wrong_advice"


def test_thresholds_keep_wrong_advice_under_the_limit():
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 4, 200)
    logits = rng.normal(size=(200, 4))
    logits[np.arange(200), labels] += rng.normal(1.5, 1.0, 200)
    probs = softmax(logits, fit_temperature(logits, labels))
    kept = np.ones(200, dtype=bool)
    advise_at, ask_at = choose_thresholds(probs, labels, kept)
    gate = Gate(1.0, advise_at, ask_at, 0.0)
    routes, first, second, _ = gate.route(np.log(probs), np.ones(200))
    names = outcomes(routes, first, second, labels)
    assert (names == "wrong_advice").mean() < 0.05


def test_retake_cutoff_rejects_under_five_percent():
    scores = np.linspace(0.0, 1.0, 112)
    cutoff = retake_cutoff(scores)
    assert (scores < cutoff).mean() < 0.05
    assert (scores < cutoff).sum() == 5


def test_average_views_of_one_view_is_its_log_probabilities():
    logits = np.array([[2.0, 1.0, 0.0], [0.0, 0.0, 3.0]])
    averaged = average_views([logits, logits + 5.0])
    assert np.allclose(np.exp(averaged), softmax(logits))
