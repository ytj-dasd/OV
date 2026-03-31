# tests/verifiers/test_mmsi_verifier.py
import pytest

from glmv_reward.verifiers import MmsiVerifier


# --- extract_answer tests ---
def test_extract_correct_format_strict(mmsi_verifier):
    response = "<think>The answer is 1+x.</think><|begin_of_box|>1+x<|end_of_box|>"
    assert mmsi_verifier.extract_answer(response) == "1+x"


def test_extract_no_box_strict(mmsi_verifier):
    response = "<think>The answer is 1+x.</think>1+x"
    assert mmsi_verifier.extract_answer(response) is None


def test_extract_multiple_boxes_strict(mmsi_verifier):
    response = "<think>...</think><|begin_of_box|>1}<|begin_of_box|>2<|end_of_box|>"
    assert mmsi_verifier.extract_answer(response) is None


def test_extract_no_think_answer_tag_strict(mmsi_verifier):
    response = "<|begin_of_box|>1+x}"
    assert mmsi_verifier.extract_answer(response) is None


# def test_extract_malformed_think_answer_tag_strict(mmsi_verifier):
#     response = "<think>Missing answer tag</think>"
#     assert mmsi_verifier.extract_answer(response) is None

#     response_nested = (
#         "<think><answer>Nested <think>...</think></answer></think>"
#         "<answer><|begin_of_box|>1+x<|end_of_box|></answer>"
#     )
#     assert mmsi_verifier.extract_answer(response_nested) is None


# --- judge tests ---
def test_judge_exact_match(mmsi_verifier):
    assert mmsi_verifier.judge("x+1", "1+x") == 1.0
    assert mmsi_verifier.judge("2", "2.0") == 1.0


def test_judge_sympy_equivalence(mmsi_verifier):
    assert mmsi_verifier.judge("(x+1)**2", "x**2 + 2*x + 1") == 1.0
    assert mmsi_verifier.judge("3/2", "1.5") == 1.0


def test_judge_none_inputs(mmsi_verifier):
    assert mmsi_verifier.judge(None, "1") == 0.0
    assert mmsi_verifier.judge("1", None) == 0.0
    assert mmsi_verifier.judge(None, None) == 0.0


def test_judge_percentage_and_degree(mmsi_verifier):
    assert mmsi_verifier.judge("50%", "0.5") == 1.0
    assert mmsi_verifier.judge("0.5", "50%") == 1.0
    assert mmsi_verifier.judge("72°", "72") == 1.0
    assert mmsi_verifier.judge("72", "72°") == 1.0
    assert mmsi_verifier.judge("50 %", "0.5") == 1.0
