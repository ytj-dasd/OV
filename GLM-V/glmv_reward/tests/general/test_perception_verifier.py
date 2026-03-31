# tests/verifiers/test_perception_verifier.py
import pytest

from glmv_reward.verifiers import GeneralVerifier

# 测试列举


def test_insufficient_information_and_general(math_verifier):
    question1 = "What is the measure of angle GIY as annotated?"
    assert math_verifier.judge("Insufficient information", "Angle GIY is annotated as 77", question1) == 0.0

    question2 = "What is the length of line QP as annotated?"
    assert math_verifier.judge("Insufficient information", "Line QP is annotated as a-4", question2) == 0.0

    question3 = "What is the angle in the diagram that is equal to angle BNH?"
    assert math_verifier.judge("Angle BNH is equal to angle NHB", "angle BKH", question3) == 0.0

    question4 = "What is the length of line KO as annotated?"
    assert math_verifier.judge("Line KO is annotated as x", "Insufficient information", question4) == 0.0

    question5 = "What is the measure of angle WYV as annotated?"
    assert math_verifier.judge("Insufficient information", "Angle WYV is annotated as 1", question5) == 0.0

    question6 = "What is the angle in the diagram that is equal to angle PYF?"
    assert math_verifier.judge("Insufficient information", "Angle PYF is equal to angle YFD", question6) == 0.0

    question7 = "请说明思路：与图中UK连线平行的线为？请说明思路。"
    assert math_verifier.judge("XY连线", "图中所有与UK连线方向相同的竖直线段}", question7) == 0.0
