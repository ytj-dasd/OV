# tests/verifiers/test_math_verifier.py
import pytest

from glmv_reward.verifiers import MathVerifier


def test_extract_no_box_strict(math_verifier):
    response = "<think>The answer is 1+x.</think>1+x"
    assert math_verifier.extract_answer(response) is None


def test_extract_multiple_boxes_strict(math_verifier):
    response = "<think>...</think><|begin_of_box|>1<|end_of_box|><|begin_of_box|>2<|end_of_box|>"
    assert math_verifier.extract_answer(response) is None


def test_extract_no_think_tag(math_verifier):  # Behavior is same for non-strict
    response = "\\boxed{1+x}"
    assert math_verifier.extract_answer(response) is None


def test_extract_malformed_think_tag_strict(math_verifier):
    response = "<think>Missing answer tag</think>"
    assert math_verifier.extract_answer(response) is None
    response_nested = "<think>Nested <think>...</think></think><|begin_of_box|>1+x<|end_of_box|>"
    assert math_verifier.extract_answer(response_nested) is None


def test_judge_exact_match(math_verifier):
    assert math_verifier.judge("x+1", "1+x") == 1.0
    assert math_verifier.judge("2", "2.0") == 1.0


def test_judge_sympy_equivalence(math_verifier):
    assert math_verifier.judge("(x+1)**2", "x**2 + 2*x + 1") == 1.0
    assert math_verifier.judge("3/2", "1.5") == 1.0


def test_judge_none_inputs(math_verifier):
    assert math_verifier.judge(None, "1") == 0.0
    assert math_verifier.judge("1", None) == 0.0
    assert math_verifier.judge(None, None) == 0.0


def test_pi_in_answer(math_verifier):
    assert math_verifier.judge("pi + 1", "π + 1") == 1.0
    assert math_verifier.judge("\\pi + 1", "π + 1") == 1.0
    assert math_verifier.judge("2\\pi + 1", "1 + 2π") == 1.0


def test_fraction_and_decimal(math_verifier):
    assert math_verifier.judge("1/2", "0.5") == 1.0
    assert math_verifier.judge("0.25", "1/4") == 1.0
    assert math_verifier.judge("2/4", "0.5") == 1.0


def test_sqrt_forms(math_verifier):
    assert math_verifier.judge("sqrt(4)", "2") == 1.0
    assert math_verifier.judge("\\sqrt{9}", "3") == 1.0
    assert math_verifier.judge("√9", "3") == 1.0


def test_multiple_format_spaces(math_verifier):
    assert math_verifier.judge("72 °", "72") == 1.0
    assert math_verifier.judge("  0.5 ", "50%") == 1.0


def test_exp_notation(math_verifier):
    assert math_verifier.judge("1e3", "1000") == 1.0
    assert math_verifier.judge("2.5e-2", "0.025") == 1.0


def test_symbolic_vs_numeric(math_verifier):
    assert math_verifier.judge("sin(pi/2)", "1") == 1.0
    assert math_verifier.judge("exp(0)", "1") == 1.0


def test_negative_and_parens(math_verifier):
    assert math_verifier.judge("-3", "(-3)") == 1.0
    assert math_verifier.judge("-(2+3)", "-5") == 1.0


def test_latex_and_unicode_equiv(math_verifier):
    assert math_verifier.judge("\\frac{1}{2}", "½") == 1.0
    assert math_verifier.judge("π/2", "\\frac{\\pi}{2}") == 1.0


def test_string_float_equivalence(math_verifier):
    assert math_verifier.judge("2", "2.0") == 1.0
    assert math_verifier.judge("5.00", "5") == 1.0


def test_complex_expression_equiv(math_verifier):
    assert math_verifier.judge("2*(x+1)", "2x+2") == 1.0
    assert math_verifier.judge("a**2 + b**2 + 2*a*b", "a**2 + 2*a*b + b**2") == 1.0


def test_math_verifier_judge_equivalence(math_verifier):
    # Test decimal and fraction equivalence
    assert math_verifier.judge("0.5", "1/2") == 1.0
    assert math_verifier.judge("0.333...", "1/3") == 1.0

    # Test scientific notation
    assert math_verifier.judge("1.23e4", "12300") == 1.0
    assert math_verifier.judge("1.23E-4", "0.000123") == 1.0

    # Test square root equivalence
    assert math_verifier.judge("√4", "2") == 1.0
    assert math_verifier.judge("\\sqrt{9}", "3") == 1.0


def test_math_verifier_judge_inequality(math_verifier):
    # Test inequality expressions
    assert math_verifier.judge("x > 5", "x > 5") == 1.0
    assert math_verifier.judge("y ≤ 10", "y <= 10") == 1.0
    assert math_verifier.judge("z ≥ 0", "z >= 0") == 1.0


def test_math_verifier_judge_units(math_verifier):
    # Test unit conversions
    assert math_verifier.judge("1 kg", "1000 g", question="1 kg 等于多少 g？") == 0.0
    assert math_verifier.judge("1 hour", "3600 seconds", question="1 hour 等于多少秒？") == 0.0
    assert math_verifier.judge("1 m/s", "3.6 km/h", question="1 m/s 等于多少 km/h？") == 0.0


def test_math_verifier_judge_multiple_choice(math_verifier):
    # Test multiple choice questions with letter options
    question = "What is the color of the this?\n(A) Blue\n(B) Red\n(C) Green\n(D) Yellow"
    assert math_verifier.judge(" A", "A", question=question) == 1.0
    assert math_verifier.judge("B ", "B", question=question) == 1.0
    assert math_verifier.judge("  C", "C", question=question) == 1.0
    assert math_verifier.judge("D", "D ", question=question) == 1.0

    # Test case insensitivity for letter options
    assert math_verifier.judge("\\boxed{A}", "A", question=question) == 1.0
    assert math_verifier.judge("\\text{B}", "B", question=question) == 1.0
    assert math_verifier.judge("\\boxed{C}", "\\text{C}", question=question) == 1.0
    assert math_verifier.judge("\\boxed{D}", "D", question=question) == 1.0

    assert math_verifier.judge("A. Blue", "A", question=question) == 1.0
    assert math_verifier.judge("B. Red", "B", question=question) == 1.0
    assert math_verifier.judge("C. Green", "C", question=question) == 1.0
    assert math_verifier.judge("D. Yellow", "D", question=question) == 1.0

    # Test multiple choice with number options
    question_num = "What is the color of the this?\n(1) Blue\n(2) Red\n(3) Green\n(4) Yellow"
    assert math_verifier.judge("1", "1", question=question_num) == 1.0
    assert math_verifier.judge("2", "2", question=question_num) == 1.0
    assert math_verifier.judge("3", "3", question=question_num) == 1.0
    assert math_verifier.judge("4", "4", question=question_num) == 1.0

    assert math_verifier.judge("Red", "3", question=question) == 0.0

    # Test multiple choice with spaces
    assert math_verifier.judge(" A ", "A", question=question) == 1.0
    assert math_verifier.judge("\\boxed{B} ", "B", question=question) == 1.0


def test_unit(math_verifier):
    # Case 1: 题目中已指定单位，答案可以有单位或无单位，但若有则单位必须正确
    question1 = "一个圆柱体的体积是145立方厘米，若底面积为25平方厘米，求高。"
    assert math_verifier.judge("5.8厘米", "5.8", question=question1) == 1.0  # 有单位，数值正确
    assert math_verifier.judge("5.8", "5.8", question=question1) == 1.0  # 无单位，数值正确
    assert math_verifier.judge("5.8米", "5.8", question=question1) == 0.0  # 单位错误
    # Case 2: 题目中未指定单位，答案应带上单位且合理
    question2 = "根据图中的信息，求长方体水箱的高是多少。"
    assert math_verifier.judge("40厘米", "40厘米", question=question2) == 1.0  # 正确单位
    assert math_verifier.judge("40", "40厘米", question=question2) == 1.0
    assert math_verifier.judge("0.4米", "40厘米", question=question2) == 1.0  # 合理单位换算正确
    # Case 3: 标准答案带单位或不带单位都可能，需看数值是否一致即可
    question3 = "如果一个长方形的面积是120平方米，长为10米，求宽。"
    assert math_verifier.judge("12", "12米", question=question3) == 1.0  # 数值一致，单位略有不同
    assert math_verifier.judge("12米", "12", question=question3) == 1.0  # 数值一致，单位存在也可以
    assert math_verifier.judge("12", "12", question=question3) == 1.0  # 数值错误
