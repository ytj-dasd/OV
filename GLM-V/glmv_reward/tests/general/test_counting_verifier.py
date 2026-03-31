# python3 -m pytest tests/verifiers/test_counting_verifier.py
import pytest

from glmv_reward.verifiers import CountingVerifier


def test_extract_correct_format_strict(counting_verifier):
    response = "<think>The answer is 1+x.</think><|begin_of_box|>1+x<|end_of_box|>"
    assert counting_verifier.extract_answer(response) == "1+x"


def test_extract_no_box_strict(counting_verifier):
    response = "<think>The answer is 1+x.</think>1+x"
    assert counting_verifier.extract_answer(response) is None


def test_extract_multiple_boxes_strict(counting_verifier):
    response = "<think>...</think><|begin_of_box|>1<|end_of_box|><|begin_of_box|>2<|end_of_box|>"
    assert counting_verifier.extract_answer(response) is None


def test_extract_no_think_tag(counting_verifier):  # Behavior is same for non-strict
    response = "<|begin_of_box|>1+x<|end_of_box|>"
    assert counting_verifier.extract_answer(response) is None


def test_extract_malformed_think_tag_strict(counting_verifier):
    response_nested = "<think><answer>Nested <think>...</think></think><|begin_of_box|>1+x<|end_of_box|>"
    assert counting_verifier.extract_answer(response_nested) is None


def test_judge_badcase(counting_verifier):
    assert counting_verifier.judge("two.", "22") == 0.0
    assert counting_verifier.judge("2", "22") == 0.0
    assert counting_verifier.judge("one", "11") == 0.0
    assert counting_verifier.judge("one", "11.") == 0.0
    assert counting_verifier.judge("one", "11.0") == 0.0
    assert counting_verifier.judge("one", "11.0") == 0.0
