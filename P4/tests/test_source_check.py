"""Source-check cases on made-up sentences. Run: python -m pytest P4/tests"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))
from source_check import check_step, checked_steps

PASSAGE = [
    "Pull the affected plants, and place them in plastic bags.",
    "Spotted fruit should not be eaten, nor should it be canned or otherwise preserved.",
    "Rotate with non-host crops every year.",
]


def test_shortened_step_passes():
    assert check_step("Pull the affected plants.", [0], PASSAGE).ok


def test_dropped_negation_is_flagged_for_every_negated_action():
    for step in ("Eat spotted fruit.", "Preserve spotted fruit."):
        assert not check_step(step, [1], PASSAGE).ok, step


def test_kept_negation_passes():
    assert check_step("Spotted fruit should not be eaten.", [1], PASSAGE).ok


def test_non_prefix_cannot_be_dropped():
    assert not check_step("Rotate with host crops every year.", [2], PASSAGE).ok
    assert check_step("Rotate with non-host crops.", [2], PASSAGE).ok


def test_quote_replaces_a_shortened_step_from_the_same_sentence():
    steps = [("Pull the affected plants.", [0]), ("Place roots and leaves in plastic bags.", [0])]
    shown = checked_steps(steps, PASSAGE)
    assert [s["quoted"] for s in shown] == [True]
    assert "plastic bags" in shown[0]["shown"]


def test_same_sentence_is_quoted_once():
    steps = [("Burn the plants.", [0]), ("Compost the plants.", [0])]
    assert len(checked_steps(steps, PASSAGE)) == 1
