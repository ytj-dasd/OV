# tests/vqa/test_vqa_verifier.py
import pytest


def test_extract_no_box_strict(vqa_verifier):
    response = "<think>The answer is 1+x.</think>1+x"
    assert vqa_verifier.extract_answer(response) is None


def test_extract_multiple_boxes_strict(vqa_verifier):
    response = "<think>...</think><|begin_of_box|>1<|end_of_box|><|begin_of_box|>2<|end_of_box|>"
    assert vqa_verifier.extract_answer(response) is None


def test_extract_no_think_tag(vqa_verifier):  # Behavior is same for non-strict
    response = "<|begin_of_box|>1+x<|end_of_box|>"
    assert vqa_verifier.extract_answer(response) is None


def test_extract_malformed_think_tag_strict(vqa_verifier):
    response = "<think>Missing answer tag</think>"
    assert vqa_verifier.extract_answer(response) is None
    response_nested = "<think>Nested <think>...</think></think>\\boxed{1+x}"
    assert vqa_verifier.extract_answer(response_nested) is None


def test_judge_exact_match(vqa_verifier):
    assert vqa_verifier.judge("x+1", "1+x") == 1.0
    assert vqa_verifier.judge("2", "2.0") == 1.0


def test_no_answer(vqa_verifier):
    assert (
        vqa_verifier.judge(
            "The question is based on an image that does not match its description; no truck or window logo is visible in the provided image.",
            "doesnotapply",
        )
        == 1.0
    )
    assert vqa_verifier.judge("Cannot be determined from the provided image", "<no answer>") == 1.0
    assert vqa_verifier.judge("The image does not display the application's version.", "<no answer>") == 1.0
    assert vqa_verifier.judge("Wii", "doesnotapply") == 0.0


def test_OCR_question(vqa_verifier):
    assert vqa_verifier.judge("airplane", "Plane", question="What has alaska airlines?") == 1.0
    assert vqa_verifier.judge("airplane", "Plane", question="What words are wrote in the image?") == 0.0
    assert vqa_verifier.judge("Bantong-2", "BENTONG-2", question="What is written in the image?") == 0.0


def test_special_characters_handling(vqa_verifier):
    assert vqa_verifier.judge("C++", "C ++") == 1.0
    assert vqa_verifier.judge("e-mail", "email") == 1.0
    assert vqa_verifier.judge("U.S.A.", "USA") == 1.0
