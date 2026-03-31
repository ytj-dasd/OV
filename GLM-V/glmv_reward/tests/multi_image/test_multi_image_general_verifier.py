# tests/verifiers/test_multi_image_general_verifier.py
import pytest

from glmv_reward.verifiers import MultiImageVerifier


def test_extract_correct_format_strict(multi_image_general_verifier):
    response = "<think>The answer is 1+x.</think><|begin_of_box|>1+x<|end_of_box|>"
    assert multi_image_general_verifier.extract_answer(response) == "1+x"


def test_extract_no_box_strict(multi_image_general_verifier):
    response = "<think>The answer is 1+x.</think>1+x"
    assert multi_image_general_verifier.extract_answer(response) is None


def test_extract_multiple_boxes_strict(multi_image_general_verifier):
    response = "<think>...</think><|begin_of_box|>1<|end_of_box|><|begin_of_box|>2<|end_of_box|>"
    assert multi_image_general_verifier.extract_answer(response) is None


def test_extract_no_think_tag(multi_image_general_verifier):  # Behavior is same for non-strict
    response = "<|begin_of_box|>1+x<|end_of_box|>"
    assert multi_image_general_verifier.extract_answer(response) is None


def test_extract_malformed_think_tag_strict(multi_image_general_verifier):
    response_nested = "<think>Nested <think>...</think></think><|begin_of_box|>1+x<|end_of_box|>"
    assert multi_image_general_verifier.extract_answer(response_nested) is None


def test_extract_answer_boxed_with_special_characters(multi_image_general_verifier):
    """测试包含特殊字符的boxed内容"""
    # 包含标点符号的答案
    response = "<think>Punctuation</think><|begin_of_box|>Hello, World!<|end_of_box|>"
    assert multi_image_general_verifier.extract_answer(response) == "Hello, World!"

    # 包含空格的答案
    response = "<think>Spaces</think><|begin_of_box|>New York City<|end_of_box|>"
    assert multi_image_general_verifier.extract_answer(response) == "New York City"

    # 包含特殊符号的答案
    response = "<think>Symbols</think><|begin_of_box|>@#$%^&*()<|end_of_box|>"
    assert multi_image_general_verifier.extract_answer(response) == "@#$%^&*()"


def test_judge_exact_match(multi_image_general_verifier):
    assert multi_image_general_verifier.judge("x+1", "1+x") == 1.0
    assert multi_image_general_verifier.judge("2", "2.0") == 1.0


def test_judge_sympy_equivalence(multi_image_general_verifier):
    assert multi_image_general_verifier.judge("(x+1)**2", "x**2 + 2*x + 1") == 1.0
    assert multi_image_general_verifier.judge("3/2", "1.5") == 1.0


def test_judge_incorrect(multi_image_general_verifier):
    assert multi_image_general_verifier.judge("x+1", "x+2") == 0.0
    assert multi_image_general_verifier.judge("3.1", "3.0") == 0.0


def test_judge_none_inputs(multi_image_general_verifier):
    assert multi_image_general_verifier.judge(None, "1") == 0.0
    assert multi_image_general_verifier.judge("1", None) == 0.0
    assert multi_image_general_verifier.judge(None, None) == 0.0


def test_multi_image_general_verifier_extraction_correct(multi_image_general_verifier):
    response = "<think>The answer is 1+x.</think><|begin_of_box|>1+x<|end_of_box|>"
    assert multi_image_general_verifier.extract_answer(response) == "1+x"


def test_multi_image_general_verifier_judge_correct(multi_image_general_verifier):
    assert multi_image_general_verifier.judge("x+1", "1+x") == 1.0
    assert multi_image_general_verifier.judge("2.000000001", "2") == 1.0  # Test tolerance


def test_multi_image_general_verifier_judge_incorrect(multi_image_general_verifier):
    assert multi_image_general_verifier.judge("x+1", "x+2") == 0.0
    assert multi_image_general_verifier.judge("3.1", "3.0") == 0.0


def test_pi_in_answer(multi_image_general_verifier):
    assert multi_image_general_verifier.judge("pi + 1", "π + 1") == 1.0
    assert multi_image_general_verifier.judge("\\pi + 1", "π + 1") == 1.0
    assert multi_image_general_verifier.judge("2\\pi + 1", "1 + 2π") == 1.0


def test_sqrt_forms(multi_image_general_verifier):
    assert multi_image_general_verifier.judge("sqrt(4)", "2") == 1.0
    assert multi_image_general_verifier.judge("\\sqrt{9}", "3") == 1.0
    assert multi_image_general_verifier.judge("√9", "3") == 1.0


def test_exp_notation(multi_image_general_verifier):
    assert multi_image_general_verifier.judge("1e3", "1000") == 1.0
    assert multi_image_general_verifier.judge("2.5e-2", "0.025") == 1.0


def test_symbolic_vs_numeric(multi_image_general_verifier):
    assert multi_image_general_verifier.judge("sin(pi/2)", "1") == 1.0
    assert multi_image_general_verifier.judge("exp(0)", "1") == 1.0


def test_negative_and_parens(multi_image_general_verifier):
    assert multi_image_general_verifier.judge("-3", "(-3)") == 1.0
    assert multi_image_general_verifier.judge("-(2+3)", "-5") == 1.0


def test_bracket_and_whitespace(multi_image_general_verifier):
    assert multi_image_general_verifier.judge("( 1 + x )", "x + 1") == 1.0
    assert multi_image_general_verifier.judge("  \\boxed{7}  ", "7") == 1.0


def test_latex_and_unicode_equiv(multi_image_general_verifier):
    assert multi_image_general_verifier.judge("\\frac{1}{2}", "½") == 1.0
    assert multi_image_general_verifier.judge("π/2", "\\frac{\\pi}{2}") == 1.0


def test_string_float_equivalence(multi_image_general_verifier):
    assert multi_image_general_verifier.judge("2", "2.0") == 1.0
    assert multi_image_general_verifier.judge("5.00", "5") == 1.0


def test_complex_expression_equiv(multi_image_general_verifier):
    assert multi_image_general_verifier.judge("2*(x+1)", "2x+2") == 1.0
    assert multi_image_general_verifier.judge("a**2 + b**2 + 2*a*b", "a**2 + 2*a*b + b**2") == 1.0


def test_multi_image_general_verifier_extraction(multi_image_general_verifier):
    # Test extraction of boxed answers
    response = "<think>Thinking...</think><|begin_of_box|>42<|end_of_box|>"
    assert multi_image_general_verifier.extract_answer(response) == "42"


def test_multi_image_general_verifier_judge_inequality(multi_image_general_verifier):
    # Test inequality expressions
    assert multi_image_general_verifier.judge("x > 5", "x > 5") == 1.0
    assert multi_image_general_verifier.judge("y ≤ 10", "y <= 10") == 1.0
    assert multi_image_general_verifier.judge("z ≥ 0", "z >= 0") == 1.0


def test_multi_image_general_verifier_judge_geometry(multi_image_general_verifier):
    # Test geometric expressions
    assert multi_image_general_verifier.judge("πr²", "πr^2") == 1.0
    assert multi_image_general_verifier.judge("2πr", "2πr") == 1.0
    assert multi_image_general_verifier.judge("sin²θ + cos²θ", "1") == 1.0

    # Test more complex mathematical expressions
    assert multi_image_general_verifier.judge("e^(iπ)", "-1") == 1.0
    assert multi_image_general_verifier.judge("log(e)", "1") == 1.0
    assert multi_image_general_verifier.judge("sin(π/2)", "1") == 1.0
    assert multi_image_general_verifier.judge("cos(π)", "-1") == 1.0
    assert multi_image_general_verifier.judge("tan(π/4)", "1") == 1.0
    assert multi_image_general_verifier.judge("cot(π/4)", "1") == 1.0
    assert multi_image_general_verifier.judge("sec(0)", "1") == 1.0
    assert multi_image_general_verifier.judge("csc(π/2)", "1") == 1.0
    assert multi_image_general_verifier.judge("arcsin(1)", "π/2") == 1.0
    assert multi_image_general_verifier.judge("arccos(0)", "π/2") == 1.0
    assert multi_image_general_verifier.judge("arctan(1)", "π/4") == 1.0

    # Test more complex mathematical expressions
    assert multi_image_general_verifier.judge("sinh(0)", "0") == 1.0
    assert multi_image_general_verifier.judge("cosh(0)", "1") == 1.0
    assert multi_image_general_verifier.judge("tanh(0)", "0") == 1.0

    # Test matrix operations
    assert multi_image_general_verifier.judge("[[1,2],[3,4]]", "[[1,2],[3,4]]") == 1.0
    assert multi_image_general_verifier.judge("det([[1,2],[3,4]])", "-2") == 1.0


def test_multi_image_general_verifier_judge_matching(chart_verifier):
    # Test matching questions
    question = "Match the countries with their capitals:\nA) France\nB) Germany\nC) Italy"
    answer = "A-Paris, B-Berlin, C-Rome"
    assert chart_verifier.judge(answer, answer, question=question) == 1.0
    assert chart_verifier.judge("A:Paris B:Berlin C:Rome", answer, question=question) == 1.0  # Different format
    assert chart_verifier.judge("A-Berlin, B-Paris, C-Rome", answer, question=question) == 0.0  # Wrong matches


def test_multi_image_general_verifier_judge_fill_in_blank(chart_verifier):
    # Test fill in the blank questions
    question = "The capital of France is _____."
    assert chart_verifier.judge("Paris", "Paris", question=question) == 1.0
    assert chart_verifier.judge("paris", "Paris", question=question) == 1.0  # Case insensitive
    assert chart_verifier.judge("Paris, France", "Paris", question=question) == 1.0  # Extra info
    assert chart_verifier.judge("London", "Paris", question=question) == 0.0  # Wrong answer


def test_extract_answer_complex_boxed_content(multi_image_general_verifier):
    """测试复杂的boxed内容"""
    # 包含分数的boxed
    response = "<think>Fraction</think><|begin_of_box|>\\frac{1}{2}<|end_of_box|>"
    assert multi_image_general_verifier.extract_answer(response) == "\\frac{1}{2}"

    # 包含多个变量的表达式
    response = "<think>Complex expr</think><|begin_of_box|>x^2 + 2xy + y^2<|end_of_box|>"
    assert multi_image_general_verifier.extract_answer(response) == "x^2 + 2xy + y^2"

    # 包含特殊字符
    response = "<think>Special chars</think><|begin_of_box|>π + √2 + ∞<|end_of_box|>"
    assert multi_image_general_verifier.extract_answer(response) == "π + √2 + ∞"


def test_extract_answer_whitespace_handling(multi_image_general_verifier):
    """测试空白字符的处理"""
    # 标签间有额外空白
    response = "<think>  Some thinking  </think>    <|begin_of_box|>42<|end_of_box|>  "
    assert multi_image_general_verifier.extract_answer(response) == "42"

    # 换行符
    response = "<think>\nThinking\n</think>\n\n<|begin_of_box|>42<|end_of_box|>\n"
    assert multi_image_general_verifier.extract_answer(response) == "42"


def test_judge_sympy_parsing_errors(multi_image_general_verifier):
    """测试sympy解析错误的情况"""
    # 无效的数学表达式
    assert multi_image_general_verifier.judge("invalid_expr", "42") == 0.0
    assert multi_image_general_verifier.judge("42", "invalid_expr") == 0.0

    # 包含未定义函数的表达式
    assert multi_image_general_verifier.judge("undefined_func(x)", "42") == 0.0

    # 语法错误的表达式
    assert multi_image_general_verifier.judge("2 +", "2") == 0.0
    assert multi_image_general_verifier.judge("(2 + 3", "5") == 0.0


# 非学科场景测试用例


def test_extract_answer_edge_cases(multi_image_general_verifier):
    """测试答案提取的边界情况"""
    # 空字符串
    assert multi_image_general_verifier.extract_answer("") is None

    # 只有空白字符
    assert multi_image_general_verifier.extract_answer("   \n\t   ") is None

    # 不完整的标签
    assert multi_image_general_verifier.extract_answer("<think>incomplete") is None
    assert multi_image_general_verifier.extract_answer("incomplete") is None

    # 空的think和answer标签
    assert multi_image_general_verifier.extract_answer("<think></think>") is None
    assert multi_image_general_verifier.extract_answer("<think>   </think>   ") is None

    # 只有think标签，没有answer标签
    assert multi_image_general_verifier.extract_answer("<think>Some thinking</think>") is None

    # 只有answer标签，没有think标签
    assert multi_image_general_verifier.extract_answer("<|begin_of_box|>42<|end_of_box|>") is None


def test_extract_answer_case_insensitive(multi_image_general_verifier):
    """测试标签的大小写不敏感性"""
    # 大写标签
    response = "<THINK>The answer is 42.</THINK><|begin_of_box|>42<|end_of_box|>"
    assert multi_image_general_verifier.extract_answer(response) == "42"

    # 混合大小写
    response = "<Think>The answer is 42.</Think><|begin_of_box|>42<|end_of_box|>"
    assert multi_image_general_verifier.extract_answer(response) == "42"


def test_general_knowledge_questions(multi_image_general_verifier):
    """测试常识问答"""
    # 地理常识
    question = "What is the capital of France?"
    assert multi_image_general_verifier.judge("Paris", "Paris", question=question) == 1.0
    assert multi_image_general_verifier.judge("London", "Paris", question=question) == 0.0

    # 历史常识
    question = "Who painted the Mona Lisa?"
    assert multi_image_general_verifier.judge("Leonardo da Vinci", "Leonardo da Vinci", question=question) == 1.0
    assert multi_image_general_verifier.judge("Michelangelo", "Leonardo da Vinci", question=question) == 0.0


def test_image_description_scenarios(multi_image_general_verifier):
    """测试图片描述场景"""
    # 颜色识别
    question = "What color is the car in the image?"
    assert multi_image_general_verifier.judge("red", "red", question=question) == 1.0
    assert multi_image_general_verifier.judge("blue", "red", question=question) == 0.0

    # 物体识别
    question = "What animal is shown in the picture?"
    assert multi_image_general_verifier.judge("dog", "dog", question=question) == 1.0
    assert multi_image_general_verifier.judge("cat", "dog", question=question) == 0.0

    # 数量计数
    question = "How many people are in the image?"
    assert multi_image_general_verifier.judge("3", "3", question=question) == 1.0
    assert multi_image_general_verifier.judge("2", "3", question=question) == 0.0


def test_yes_no_questions(multi_image_general_verifier):
    """测试是非题"""
    question = "Is there a cat in the image?"
    assert multi_image_general_verifier.judge("yes", "yes", question=question) == 1.0
    assert multi_image_general_verifier.judge("no", "yes", question=question) == 0.0

    # 布尔值形式
    question = "Is the statement true?"
    assert multi_image_general_verifier.judge("true", "true", question=question) == 1.0
    assert multi_image_general_verifier.judge("false", "true", question=question) == 0.0


def test_text_content_extraction(multi_image_general_verifier):
    """测试文本内容提取"""
    # OCR文本识别
    question = "What text is displayed on the sign?"
    assert multi_image_general_verifier.judge("STOP", "STOP", question=question) == 1.0
    assert multi_image_general_verifier.judge("Stop", "STOP", question=question) == 1.0
    assert multi_image_general_verifier.judge("GO", "STOP", question=question) == 0.0


def test_location_and_scene_recognition(multi_image_general_verifier):
    """测试地点和场景识别"""
    # 场所识别
    question = "What type of building is this?"
    assert multi_image_general_verifier.judge("hospital", "hospital", question=question) == 1.0
    assert multi_image_general_verifier.judge("school", "hospital", question=question) == 0.0

    # 室内外判断
    question = "Is this an indoor or outdoor scene?"
    assert multi_image_general_verifier.judge("indoor", "indoor", question=question) == 1.0
    assert multi_image_general_verifier.judge("outdoor", "indoor", question=question) == 0.0


def test_action_and_activity_recognition(multi_image_general_verifier):
    """测试动作和活动识别"""
    # 动作识别
    question = "What is the person doing?"
    assert multi_image_general_verifier.judge("running", "running", question=question) == 1.0
    assert multi_image_general_verifier.judge("walking", "running", question=question) == 0.0

    # 活动场景
    question = "What sport is being played?"
    assert multi_image_general_verifier.judge("basketball", "basketball", question=question) == 1.0
    assert multi_image_general_verifier.judge("football", "basketball", question=question) == 0.0


def test_emotion_and_expression_recognition(multi_image_general_verifier):
    """测试情绪和表情识别"""
    question = "What emotion is the person showing?"
    assert multi_image_general_verifier.judge("happy", "happy", question=question) == 1.0
    assert multi_image_general_verifier.judge("sad", "happy", question=question) == 0.0


def test_weather_and_time_recognition(multi_image_general_verifier):
    """测试天气和时间识别"""
    # 天气识别
    question = "What is the weather like?"
    assert multi_image_general_verifier.judge("sunny", "sunny", question=question) == 1.0
    assert multi_image_general_verifier.judge("rainy", "sunny", question=question) == 0.0

    # 时间段识别
    question = "What time of day is it?"
    assert multi_image_general_verifier.judge("morning", "morning", question=question) == 1.0
    assert multi_image_general_verifier.judge("evening", "morning", question=question) == 0.0


def test_food_and_cuisine_recognition(multi_image_general_verifier):
    """测试食物和菜系识别"""
    question = "What type of food is this?"
    assert multi_image_general_verifier.judge("pizza", "pizza", question=question) == 1.0
    assert multi_image_general_verifier.judge("burger", "pizza", question=question) == 0.0

    # 菜系识别
    question = "What cuisine is this?"
    assert multi_image_general_verifier.judge("Italian", "Italian", question=question) == 1.0
    assert multi_image_general_verifier.judge("Chinese", "Italian", question=question) == 0.0


def test_brand_and_logo_recognition(multi_image_general_verifier):
    """测试品牌和标志识别"""
    question = "What brand logo is shown?"
    assert multi_image_general_verifier.judge("Nike", "Nike", question=question) == 1.0
    assert multi_image_general_verifier.judge("Adidas", "Nike", question=question) == 0.0


def test_transportation_recognition(multi_image_general_verifier):
    """测试交通工具识别"""
    question = "What type of vehicle is this?"
    assert multi_image_general_verifier.judge("car", "car", question=question) == 1.0
    assert multi_image_general_verifier.judge("truck", "car", question=question) == 0.0


def test_age_and_gender_recognition(multi_image_general_verifier):
    """测试年龄和性别识别"""
    # 性别识别
    question = "What is the gender of the person?"
    assert multi_image_general_verifier.judge("male", "male", question=question) == 1.0
    assert multi_image_general_verifier.judge("female", "male", question=question) == 0.0

    # 年龄段识别
    question = "What age group does this person belong to?"
    assert multi_image_general_verifier.judge("adult", "adult", question=question) == 1.0
    assert multi_image_general_verifier.judge("child", "adult", question=question) == 0.0


def test_descriptive_questions(multi_image_general_verifier):
    """测试描述性问题"""
    # 外观描述
    question = "Describe the main object's color"
    assert multi_image_general_verifier.judge("blue", "blue", question=question) == 1.0
    assert multi_image_general_verifier.judge("red", "blue", question=question) == 0.0

    # 材质描述
    question = "What material is the object made of?"
    assert multi_image_general_verifier.judge("wood", "wood", question=question) == 1.0
    assert multi_image_general_verifier.judge("metal", "wood", question=question) == 0.0


def test_spatial_relationship_questions(multi_image_general_verifier):
    """测试空间关系问题"""
    # 位置关系
    question = "Where is the cat relative to the dog?"
    assert multi_image_general_verifier.judge("above", "above", question=question) == 1.0
    assert multi_image_general_verifier.judge("below", "above", question=question) == 0.0

    # 方向关系
    question = "Which direction is the person facing?"
    assert multi_image_general_verifier.judge("left", "left", question=question) == 1.0
    assert multi_image_general_verifier.judge("right", "left", question=question) == 0.0


def test_comparative_questions(multi_image_general_verifier):
    """测试比较性问题"""
    # 大小比较
    question = "Which object is larger?"
    assert multi_image_general_verifier.judge("elephant", "elephant", question=question) == 1.0
    assert multi_image_general_verifier.judge("mouse", "elephant", question=question) == 0.0

    # 数量比较
    question = "Are there more cats or dogs?"
    assert multi_image_general_verifier.judge("cats", "cats", question=question) == 1.0
    assert multi_image_general_verifier.judge("dogs", "cats", question=question) == 0.0


def test_temporal_questions(multi_image_general_verifier):
    """测试时间相关问题"""
    # 季节识别
    question = "What season is depicted?"
    assert multi_image_general_verifier.judge("winter", "winter", question=question) == 1.0
    assert multi_image_general_verifier.judge("summer", "winter", question=question) == 0.0

    # 历史时期
    question = "From which era is this photograph?"
    assert multi_image_general_verifier.judge("1950s", "1950s", question=question) == 1.0
    assert multi_image_general_verifier.judge("2000s", "1950s", question=question) == 0.0


def test_cultural_and_social_context(multi_image_general_verifier):
    """测试文化和社会背景"""
    # 文化识别
    question = "What cultural style is this architecture?"
    assert multi_image_general_verifier.judge("Gothic", "Gothic", question=question) == 1.0
    assert multi_image_general_verifier.judge("Modern", "Gothic", question=question) == 0.0

    # 社会场景
    question = "What type of social gathering is this?"
    assert multi_image_general_verifier.judge("wedding", "wedding", question=question) == 1.0
    assert multi_image_general_verifier.judge("birthday party", "wedding", question=question) == 0.0


def test_multimodal_content_integration(multi_image_general_verifier):
    """测试多模态内容整合"""
    # 图文结合问题
    question = "Based on the image and text, what is the main topic?"
    assert multi_image_general_verifier.judge("climate change", "climate change", question=question) == 1.0
    assert multi_image_general_verifier.judge("technology", "climate change", question=question) == 0.0


def test_judge_non_string_inputs(multi_image_general_verifier):
    """测试非字符串输入的处理"""
    # 数字输入
    assert multi_image_general_verifier.judge(42, "42") == 0.0
    assert multi_image_general_verifier.judge("42", 42) == 0.0
    assert multi_image_general_verifier.judge(42, 42) == 0.0

    # 列表输入
    assert multi_image_general_verifier.judge(["42"], "42") == 0.0
    assert multi_image_general_verifier.judge("42", ["42"]) == 0.0

    # 布尔输入
    assert multi_image_general_verifier.judge(True, "True") == 0.0
    assert multi_image_general_verifier.judge("True", False) == 0.0


def test_judge_with_question_parameter(multi_image_general_verifier):
    """测试带有question参数的judge方法"""
    question = "What is the color of the car?"

    # 正确答案
    assert multi_image_general_verifier.judge("red", "red", question=question) == 1.0

    # 错误答案
    assert multi_image_general_verifier.judge("blue", "red", question=question) == 0.0


def test_extract_answer_unicode_characters(multi_image_general_verifier):
    """测试包含Unicode字符的答案提取"""
    # 包含中文字符
    response = "<think>Chinese</think><|begin_of_box|>北京<|end_of_box|>"
    assert multi_image_general_verifier.extract_answer(response) == "北京"

    # 包含emoji
    response = "<think>Emoji</think><|begin_of_box|>😀🎉🌟<|end_of_box|>"
    assert multi_image_general_verifier.extract_answer(response) == "😀🎉🌟"

    # 包含其他语言
    response = "<think>Japanese</think><|begin_of_box|>こんにちは<|end_of_box|>"
    assert multi_image_general_verifier.extract_answer(response) == "こんにちは"


def test_multi_image_comparison_scenarios(multi_image_general_verifier):
    """测试多图像对比场景"""
    # 图像差异识别
    question = "What is different between the two images?"
    assert multi_image_general_verifier.judge("color of the car", "color of the car", question=question) == 1.0
    assert multi_image_general_verifier.judge("size of the building", "color of the car", question=question) == 0.0

    # 图像相似性判断
    question = "Are these two images similar?"
    assert multi_image_general_verifier.judge("yes", "yes", question=question) == 1.0
    assert multi_image_general_verifier.judge("no", "yes", question=question) == 0.0


def test_sequential_image_analysis(multi_image_general_verifier):
    """测试序列图像分析"""
    # 时间序列分析
    question = "What happens from the first image to the second image?"
    assert multi_image_general_verifier.judge("the flower blooms", "the flower blooms", question=question) == 1.0
    assert multi_image_general_verifier.judge("the flower wilts", "the flower blooms", question=question) == 0.0


def test_complex_scene_understanding(multi_image_general_verifier):
    """测试复杂场景理解"""
    # 场景上下文推理
    question = "Based on the context, what is likely to happen next?"
    assert multi_image_general_verifier.judge("it will rain", "it will rain", question=question) == 1.0
    assert multi_image_general_verifier.judge("it will snow", "it will rain", question=question) == 0.0


def test_abstract_concept_recognition(multi_image_general_verifier):
    """测试抽象概念识别"""
    # 情感概念
    question = "What emotion is conveyed by the image sequence?"
    assert multi_image_general_verifier.judge("melancholy", "melancholy", question=question) == 1.0
    assert multi_image_general_verifier.judge("joy", "melancholy", question=question) == 0.0


def test_technical_visual_analysis(multi_image_general_verifier):
    """测试技术性视觉分析"""
    # 图像质量评估
    question = "Which image has better quality?"
    assert multi_image_general_verifier.judge("image A", "image A", question=question) == 1.0
    assert multi_image_general_verifier.judge("image B", "image A", question=question) == 0.0

    # 技术参数识别
    question = "What camera angle was used?"
    assert multi_image_general_verifier.judge("bird's eye view", "bird's eye view", question=question) == 1.0
    assert multi_image_general_verifier.judge("ground level", "bird's eye view", question=question) == 0.0


def test_cultural_context_analysis(multi_image_general_verifier):
    """测试文化背景分析"""
    # 文化标识
    question = "What cultural elements are visible?"
    assert multi_image_general_verifier.judge("traditional clothing", "traditional clothing", question=question) == 1.0
    assert multi_image_general_verifier.judge("modern architecture", "traditional clothing", question=question) == 0.0

    # 地域特色
    question = "Which region does this represent?"
    assert multi_image_general_verifier.judge("East Asia", "East Asia", question=question) == 1.0
    assert multi_image_general_verifier.judge("Europe", "East Asia", question=question) == 0.0


def test_educational_content_analysis(multi_image_general_verifier):
    """测试教育内容分析"""
    # 学习材料理解
    question = "What educational concept is being demonstrated?"
    assert multi_image_general_verifier.judge("photosynthesis", "photosynthesis", question=question) == 1.0
    assert multi_image_general_verifier.judge("respiration", "photosynthesis", question=question) == 0.0

    # 步骤说明理解
    question = "What is the next step in the procedure?"
    assert multi_image_general_verifier.judge("add water", "add water", question=question) == 1.0
    assert multi_image_general_verifier.judge("heat mixture", "add water", question=question) == 0.0


def test_creative_content_evaluation(multi_image_general_verifier):
    """测试创意内容评价"""
    # 艺术风格识别
    question = "What art style is this?"
    assert multi_image_general_verifier.judge("impressionism", "impressionism", question=question) == 1.0
    assert multi_image_general_verifier.judge("cubism", "impressionism", question=question) == 0.0


def test_edge_case_responses(multi_image_general_verifier):
    """测试边界情况响应"""
    # 空响应处理
    response = "<think>Cannot determine</think><|begin_of_box|>N/A<|end_of_box|>"
    assert multi_image_general_verifier.extract_answer(response) == "N/A"

    # 特殊字符响应
    response = "<think>Special case</think><|begin_of_box|>N/A<|end_of_box|>"
    assert multi_image_general_verifier.extract_answer(response) == "N/A"

    # 数字范围响应
    response = "<think>Range answer</think><|begin_of_box|>5-10<|end_of_box|>"
    assert multi_image_general_verifier.extract_answer(response) == "5-10"


def test_structured_data_responses(multi_image_general_verifier):
    """测试结构化数据响应"""
    # JSON格式响应（在boxed内）
    response = '<think>Structured</think><|begin_of_box|>"color": "red", "size": "large"<|end_of_box|>'
    assert multi_image_general_verifier.extract_answer(response) == '"color": "red", "size": "large"'

    # 列表格式响应
    response = "<think>List format</think><|begin_of_box|>[apple, banana, orange]<|end_of_box|>"
    assert multi_image_general_verifier.extract_answer(response) == "[apple, banana, orange]"


# 新增测试用例 - 内容理解类_动作识别


def test_spelling(math_verifier):
    # 相似词语
    question1 = "What does the lady hold in her hands at the end of the video?"
    assert math_verifier.judge("a pen", "paper", question1) == 0.0

    # 拼写错误
    question2 = "Which object was closed by the person?"
    assert math_verifier.judge("aptop", "The laptop.", question1) == 0.0

    question3 = "What brand is shown?"
    assert math_verifier.judge("Aple", "Apple", question3) == 0.0
    assert math_verifier.judge("Gogle", "Google", question3) == 0.0

    # 单复数差异
    question5 = "How many animals are in the picture?"
    assert math_verifier.judge("a cat", "cats", question5) == 0.0
    assert math_verifier.judge("dogs", "a dog", question5) == 0.0

    # 部分匹配但意思不同
    question6 = "What action is being performed?"
    assert math_verifier.judge("reading", "reaching", question6) == 0.0
