# tests/verifiers/test_general_verifier.py
import pytest

from glmv_reward.verifiers import MathVerifier

# def test_extract_correct_format_strict(general_verifier):
#     response = "<think>The answer is 1+x.</think><answer>\\boxed{1+x}</answer>"
#     assert general_verifier.extract_answer(response) == "1+x"

# def test_extract_no_box_strict(general_verifier):
#     response = "<think>The answer is 1+x.</think><answer>1+x</answer>"
#     assert general_verifier.extract_answer(response) is None

# def test_extract_multiple_boxes_strict(general_verifier):
#     response = "<think>...</think><answer>\\boxed{1}\\boxed{2}</answer>"
#     assert general_verifier.extract_answer(response) is None

# def test_extract_no_think_answer_tag(general_verifier): # Behavior is same for non-strict
#     response = "\\boxed{1+x}"
#     assert general_verifier.extract_answer(response) is None

# def test_extract_malformed_think_answer_tag_strict(general_verifier):
#     response = "<think>Missing answer tag</think>"
#     assert general_verifier.extract_answer(response) is None
#     response_nested = "<think><answer>Nested <think>...</think></answer></think><answer>\\boxed{1+x}</answer>"
#     assert general_verifier.extract_answer(response_nested) is None

# def test_judge_exact_match(general_verifier):
#     assert general_verifier.judge("x+1", "1+x") == 1.0
#     assert general_verifier.judge("2", "2.0") == 1.0

# def test_judge_sympy_equivalence(general_verifier):
#     assert general_verifier.judge("(x+1)**2", "x**2 + 2*x + 1") == 1.0
#     assert general_verifier.judge("3/2", "1.5") == 1.0

# def test_judge_tolerance(general_verifier):
#     # general_verifier uses sympy_tolerance from its config (1e-6)
#     assert general_verifier.judge("2.0000001", "2") == 1.0 # Should be 1.0 if 1e-7 < 1e-6 * (2+1e-6)
#     assert general_verifier.judge("2.001", "2") == 0.0 # 1e-3 is too large

# def test_judge_incorrect(general_verifier):
#     assert general_verifier.judge("x+1", "x+2") == 0.0
#     assert general_verifier.judge("3.1", "3.0") == 0.0

# def test_judge_none_inputs(general_verifier):
#     assert general_verifier.judge(None, "1") == 0.0
#     assert general_verifier.judge("1", None) == 0.0
#     assert general_verifier.judge(None, None) == 0.0

# def test_judge_percentage_and_degree(general_verifier):
#     assert general_verifier.judge("50%", "0.5") == 1.0
#     assert general_verifier.judge("0.5", "50%") == 1.0
#     assert general_verifier.judge("72°", "72") == 1.0
#     assert general_verifier.judge("72", "72°") == 1.0
#     assert general_verifier.judge("50 %", "0.5") == 1.0 # With space

# def test_general_verifier_extraction_correct(general_verifier):
#     response = "<think>The answer is 1+x.</think><answer>\\boxed{1+x}</answer>"
#     assert general_verifier.extract_answer(response) == "1+x"

# def test_general_verifier_judge_correct(general_verifier):
#     assert general_verifier.judge("x+1", "1+x") == 1.0
#     assert general_verifier.judge("2.000000001", "2") == 1.0 # Test tolerance

# def test_general_verifier_judge_incorrect(general_verifier):
#     assert general_verifier.judge("x+1", "x+2") == 0.0
#     assert general_verifier.judge("3.1", "3.0") == 0.0


# def test_pi_in_answer(general_verifier):
#     assert general_verifier.judge("pi + 1", "π + 1") == 1.0
#     assert general_verifier.judge("\\pi + 1", "π + 1") == 1.0
#     assert general_verifier.judge("2\\pi + 1", "1 + 2π") == 1.0

# def test_fraction_and_decimal(general_verifier):
#     assert general_verifier.judge("1/2", "0.5") == 1.0
#     assert general_verifier.judge("0.25", "1/4") == 1.0
#     assert general_verifier.judge("2/4", "0.5") == 1.0

# def test_sqrt_forms(general_verifier):
#     assert general_verifier.judge("sqrt(4)", "2") == 1.0
#     assert general_verifier.judge("\\sqrt{9}", "3") == 1.0
#     assert general_verifier.judge("√9", "3") == 1.0

# def test_multiple_format_spaces(general_verifier):
#     assert general_verifier.judge("72 °", "72") == 1.0
#     assert general_verifier.judge("  0.5 ", "50%") == 1.0

# def test_exp_notation(general_verifier):
#     assert general_verifier.judge("1e3", "1000") == 1.0
#     assert general_verifier.judge("2.5e-2", "0.025") == 1.0

# def test_symbolic_vs_numeric(general_verifier):
#     assert general_verifier.judge("sin(pi/2)", "1") == 1.0
#     assert general_verifier.judge("exp(0)", "1") == 1.0

# def test_negative_and_parens(general_verifier):
#     assert general_verifier.judge("-3", "(-3)") == 1.0
#     assert general_verifier.judge("-(2+3)", "-5") == 1.0

# def test_unit_variation(general_verifier):
#     assert general_verifier.judge("100 cm", "1 m") == 1.0   # 假如支持单位换算
#     assert general_verifier.judge("1km", "1000 m") == 1.0

# def test_bracket_and_whitespace(general_verifier):
#     assert general_verifier.judge("( 1 + x )", "x + 1") == 1.0
#     assert general_verifier.judge("  \\boxed{7}  ", "7") == 1.0

# def test_latex_and_unicode_equiv(general_verifier):
#     assert general_verifier.judge("\\frac{1}{2}", "½") == 1.0
#     assert general_verifier.judge("π/2", "\\frac{\\pi}{2}") == 1.0

# def test_large_and_small_numbers(general_verifier):
#     assert general_verifier.judge("0.00000001", "1e-8") == 1.0
#     assert general_verifier.judge("12345678", "1.2345678e7") == 1.0

# def test_string_float_equivalence(general_verifier):
#     assert general_verifier.judge("2", "2.0") == 1.0
#     assert general_verifier.judge("5.00", "5") == 1.0

# def test_complex_expression_equiv(general_verifier):
#     assert general_verifier.judge("2*(x+1)", "2x+2") == 1.0
#     assert general_verifier.judge("a**2 + b**2 + 2*a*b", "a**2 + 2*a*b + b**2") == 1.0

# def test_frag_latex_equivalence(general_verifier):
#     assert general_verifier.judge("\\frac{1}{2}", "\\frac{2}{4}") == 1.0
#     assert general_verifier.judge("\\frac{x+1}{2}", "\\frac{2x+2}{4}") == 1.0
#     assert general_verifier.judge("\\frac{\\sqrt{4}}{2}", "1") == 1.0
#     assert general_verifier.judge("\\frac{\\pi}{2}", "\\frac{3.14159}{2}") == 1.0
#     assert general_verifier.judge("\\frac{1}{\\sqrt{2}}", "\\frac{\\sqrt{2}}{2}") == 1.0
#     assert general_verifier.judge("\\frac{1}{1+\\frac{1}{x}}", "\\frac{x}{x+1}") == 1.0

# def test_general_verifier_extraction(general_verifier):
#     # Test extraction of boxed answers
#     response = "<think>Thinking...</think><answer>\\boxed{42}</answer>"
#     assert general_verifier.extract_answer(response) == "42"


# def test_general_verifier_judge_equivalence(general_verifier):
#     # Test decimal and fraction equivalence
#     assert general_verifier.judge("0.5", "1/2") == 1.0
#     assert general_verifier.judge("0.333...", "1/3") == 1.0

#     # Test scientific notation
#     assert general_verifier.judge("1.23e4", "12300") == 1.0
#     assert general_verifier.judge("1.23E-4", "0.000123") == 1.0

#     # Test square root equivalence
#     assert general_verifier.judge("√4", "2") == 1.0
#     assert general_verifier.judge("\\sqrt{9}", "3") == 1.0

# def test_general_verifier_judge_inequality(general_verifier):
#     # Test inequality expressions
#     assert general_verifier.judge("x > 5", "x > 5") == 1.0
#     assert general_verifier.judge("y ≤ 10", "y <= 10") == 1.0
#     assert general_verifier.judge("z ≥ 0", "z >= 0") == 1.0

# def test_general_verifier_judge_units(general_verifier):
#     # Test unit conversions
#     assert general_verifier.judge("1 kg", "1000 g") == 1.0
#     assert general_verifier.judge("1 hour", "3600 seconds") == 1.0
#     assert general_verifier.judge("1 m/s", "3.6 km/h") == 1.0

# def test_general_verifier_judge_geometry(general_verifier):
#     # Test geometric expressions
#     assert general_verifier.judge("πr²", "πr^2") == 1.0
#     assert general_verifier.judge("2πr", "2πr") == 1.0
#     assert general_verifier.judge("sin²θ + cos²θ", "1") == 1.0

#     # Test more complex mathematical expressions
#     assert general_verifier.judge("e^(iπ)", "-1") == 1.0
#     assert general_verifier.judge("log(e)", "1") == 1.0
#     assert general_verifier.judge("sin(π/2)", "1") == 1.0
#     assert general_verifier.judge("cos(π)", "-1") == 1.0
#     assert general_verifier.judge("tan(π/4)", "1") == 1.0
#     assert general_verifier.judge("cot(π/4)", "1") == 1.0
#     assert general_verifier.judge("sec(0)", "1") == 1.0
#     assert general_verifier.judge("csc(π/2)", "1") == 1.0
#     assert general_verifier.judge("arcsin(1)", "π/2") == 1.0
#     assert general_verifier.judge("arccos(0)", "π/2") == 1.0
#     assert general_verifier.judge("arctan(1)", "π/4") == 1.0

#     # Test more complex mathematical expressions
#     assert general_verifier.judge("sinh(0)", "0") == 1.0
#     assert general_verifier.judge("cosh(0)", "1") == 1.0
#     assert general_verifier.judge("tanh(0)", "0") == 1.0

#     # Test matrix operations
#     assert general_verifier.judge("[[1,2],[3,4]]", "[[1,2],[3,4]]") == 1.0
#     assert general_verifier.judge("det([[1,2],[3,4]])", "-2") == 1.0


# def test_general_verifier_judge_multiple_choice(general_verifier):
#     # Test multiple choice questions with letter options
#     question = "What is the color of the this?\n(A) Blue\n(B) Red\n(C) Green\n(D) Yellow"
#     assert general_verifier.judge("A", "A", question=question) == 1.0
#     assert general_verifier.judge("B", "B", question=question) == 1.0
#     assert general_verifier.judge("C", "C", question=question) == 1.0
#     assert general_verifier.judge("D", "D", question=question) == 1.0

#     # Test case insensitivity for letter options
#     assert general_verifier.judge("A", "A", question=question) == 1.0
#     assert general_verifier.judge("B", "B", question=question) == 1.0
#     assert general_verifier.judge("C", "C", question=question) == 1.0
#     assert general_verifier.judge("D", "D", question=question) == 1.0


#     assert general_verifier.judge("A. Blue", "A", question=question) == 1.0
#     assert general_verifier.judge("B. Red", "B", question=question) == 1.0
#     assert general_verifier.judge("C. Green", "C", question=question) == 1.0
#     assert general_verifier.judge("D. Yellow", "D", question=question) == 1.0

#     # Test multiple choice with number options
#     question_num = "What is the color of the this?\n(1) Blue\n(2) Red\n(3) Green\n(4) Yellow"
#     assert general_verifier.judge("1", "1", question=question_num) == 1.0
#     assert general_verifier.judge("2", "2", question=question_num) == 1.0
#     assert general_verifier.judge("3", "3", question=question_num) == 1.0
#     assert general_verifier.judge("4", "4", question=question_num) == 1.0

#     assert general_verifier.judge("Red", "3", question=question) == 0.0

#     # Test multiple choice with spaces
#     assert general_verifier.judge(" A ", "A", question=question) == 1.0
#     assert general_verifier.judge("\\boxed{B} ", "B", question=question) == 1.0
