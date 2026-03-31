def test_judge_none_inputs(counting_verifier):
    assert counting_verifier.judge(None, "1") == 0.0
    assert counting_verifier.judge("1", None) == 0.0
    assert counting_verifier.judge(None, None) == 0.0


def test_judge_exact_number_matches(counting_verifier):
    question = "<|begin_of_image|><|end_of_image|>\nHow many people are in the photo?"
    assert counting_verifier.judge("1", "one", question=question) == 1.0
    assert counting_verifier.judge("The correct number of photo", "2", question=question) == 0.0


def test_judge_decimal_format_variations(counting_verifier):
    question = "<|begin_of_image|><|end_of_image|>\nHow many people are in the photo?"
    assert counting_verifier.judge("1", "1.0", question=question) == 1.0
    assert counting_verifier.judge("1", "1.00", question=question) == 1.0
    assert counting_verifier.judge("1", "1.000", question=question) == 1.0
    assert counting_verifier.judge("1", "1.0000", question=question) == 1.0


def test_judge_incorrect_matches(counting_verifier):
    question = "<|begin_of_image|><|end_of_image|>\nHow many people are in the photo?"
    assert counting_verifier.judge("4", "five", question=question) == 0.0
    assert counting_verifier.judge("6", "five", question=question) == 0.0
    assert counting_verifier.judge("7", "five", question=question) == 0.0


def test_judge_number_word_incorrect(counting_verifier):
    question = "<|begin_of_image|><|end_of_image|>\nHow many people are in the photo?"
    # Test incorrect number word variations
    assert counting_verifier.judge("zero", "1", question=question) == 0.0  # Wrong number
    assert counting_verifier.judge("one", "2", question=question) == 0.0  # Wrong number
    assert counting_verifier.judge("two", "3", question=question) == 0.0  # Wrong number
    assert counting_verifier.judge("three", "4", question=question) == 0.0  # Wrong number
    assert counting_verifier.judge("four", "5", question=question) == 0.0  # Wrong number
    assert counting_verifier.judge("five", "6", question=question) == 0.0  # Wrong number
    assert counting_verifier.judge("six", "7", question=question) == 0.0  # Wrong number
    assert counting_verifier.judge("seven", "8", question=question) == 0.0  # Wrong number
    assert counting_verifier.judge("eight", "9", question=question) == 0.0  # Wrong number
    assert counting_verifier.judge("nine", "10", question=question) == 0.0  # Wrong number
    assert counting_verifier.judge("ten", "11", question=question) == 0.0  # Wrong number


def test_judge_chinese_number_words_incorrect(counting_verifier):
    # Test different questions for incorrect Chinese number words
    questions = [
        "<|begin_of_image|><|end_of_image|>\nHow many cats are in the photo?",
        "<|begin_of_image|><|end_of_image|>\nHow many flowers are in the photo?",
        "<|begin_of_image|><|end_of_image|>\nHow many trees are in the photo?",
        "<|begin_of_image|><|end_of_image|>\nHow many books are in the photo?",
    ]

    # Test incorrect Chinese number word variations with different questions
    assert counting_verifier.judge("三只猫", "4", question=questions[0]) == 0.0  # Wrong number
    assert counting_verifier.judge("五朵花", "6", question=questions[1]) == 0.0  # Wrong number
    assert counting_verifier.judge("七棵树", "8", question=questions[2]) == 0.0  # Wrong number
    assert counting_verifier.judge("十本书", "11", question=questions[3]) == 0.0  # Wrong number
