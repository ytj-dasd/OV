import pytest

from glmv_reward.verifiers import ChartVerifier


def test_extract_answer(chart_verifier):
    response = "<think>The answer is 1+x.</think><|begin_of_box|>1+x<|end_of_box|>"
    assert chart_verifier.extract_answer(response) == "1+x"
    response = (
        "<think>The answer is 1+x.</think><|begin_of_box|>1+x<|end_of_box|> and <|begin_of_box|>2+y<|end_of_box|>"
    )
    assert chart_verifier.extract_answer(response) == None


def test_judge_none_inputs(chart_verifier):
    assert chart_verifier.judge(None, "1") == 0.0
    assert chart_verifier.judge("1", None) == 0.0
    assert chart_verifier.judge(None, None) == 0.0
    assert chart_verifier.judge("", "1") == 0.0
    assert chart_verifier.judge("   ", "1") == 0.0


def test_multiple_format_spaces(chart_verifier):
    assert chart_verifier.judge("72 °", "72") == 1.0
    assert chart_verifier.judge("  0.5 ", "50%") == 1.0
    assert chart_verifier.judge("  50  %  ", "0.5") == 1.0
    assert chart_verifier.judge("90 ° ", "90") == 1.0
    assert chart_verifier.judge(" 180 ° ", "180") == 1.0
    assert chart_verifier.judge("72\t°", "72") == 1.0
    assert chart_verifier.judge("50%\n", "0.5") == 1.0
    assert chart_verifier.judge("  3.14  ", "3.14") == 1.0
    assert chart_verifier.judge(" 42 ", "42") == 1.0
    assert chart_verifier.judge(" x + 1 ", "x+1") == 1.0


def test_chart_verifier_judge_true_false(chart_verifier):
    # Test true/false questions
    question = "Is the sky blue?"
    assert chart_verifier.judge("True", "True", question=question) == 1.0
    assert chart_verifier.judge("False", "False", question=question) == 1.0
    assert chart_verifier.judge("true", "True", question=question) == 1.0  # Case insensitive
    assert chart_verifier.judge("TRUE", "True", question=question) == 1.0
    assert chart_verifier.judge("T", "True", question=question) == 1.0
    assert chart_verifier.judge("Yes", "True", question=question) == 1.0
    assert chart_verifier.judge("Y", "True", question=question) == 1.0
    assert chart_verifier.judge("1", "True", question=question) == 1.0
    assert chart_verifier.judge("FALSE", "False", question=question) == 1.0
    assert chart_verifier.judge("F", "False", question=question) == 1.0
    assert chart_verifier.judge("No", "False", question=question) == 1.0
    assert chart_verifier.judge("0", "False", question=question) == 1.0
    assert chart_verifier.judge(" True ", "True", question=question) == 1.0
    assert chart_verifier.judge(" False ", "False", question=question) == 1.0
    assert chart_verifier.judge("True", "False", question=question) == 0.0
    assert chart_verifier.judge("Maybe", "True", question=question) == 0.0


def test_chart_verifier_judge_short_answer(chart_verifier):
    # Test short answer questions
    question = "What is the capital of France?"
    assert chart_verifier.judge("Paris", "Paris", question=question) == 1.0
    assert chart_verifier.judge("paris", "Paris", question=question) == 1.0  # Case insensitive
    assert chart_verifier.judge("Paris, France", "Paris", question=question) == 1.0  # Extra info
    assert chart_verifier.judge("London", "Paris", question=question) == 0.0  # Wrong answer

    math_question = "What is 2+2?"
    assert chart_verifier.judge("4", "4", question=math_question) == 1.0
    assert chart_verifier.judge("four", "4", question=math_question) == 1.0
    assert chart_verifier.judge("Four", "4", question=math_question) == 1.0

    geo_question = "What is the largest ocean?"
    assert chart_verifier.judge("Pacific", "Pacific", question=geo_question) == 1.0
    assert chart_verifier.judge("Pacific Ocean", "Pacific", question=geo_question) == 1.0
    assert chart_verifier.judge("The Pacific", "Pacific", question=geo_question) == 1.0

    science_question = "What is H2O?"
    assert chart_verifier.judge("water", "water", question=science_question) == 1.0
    assert chart_verifier.judge("Water", "water", question=science_question) == 1.0
    assert chart_verifier.judge("dihydrogen monoxide", "water", question=science_question) == 1.0

    history_question = "Who was the first president of the USA?"
    assert chart_verifier.judge("Washington", "Washington", question=history_question) == 1.0
    assert chart_verifier.judge("George Washington", "Washington", question=history_question) == 1.0
    assert chart_verifier.judge("President Washington", "Washington", question=history_question) == 1.0


def test_chart_verifier_judge_essay(chart_verifier):
    # Test essay questions
    question = "Explain the process of photosynthesis."
    answer = "Photosynthesis is the process by which plants convert light energy into chemical energy."
    assert chart_verifier.judge(answer, answer, question=question) == 1.0
    assert chart_verifier.judge(answer.lower(), answer, question=question) == 1.0  # Case insensitive


def test_chart_verifier_judge_matching(chart_verifier):
    # Test matching questions
    question = "Match the countries with their capitals:\nA) France\nB) Germany\nC) Italy"
    answer = "A-Paris, B-Berlin, C-Rome"
    assert chart_verifier.judge(answer, answer, question=question) == 1.0
    assert chart_verifier.judge("A:Paris B:Berlin C:Rome", answer, question=question) == 1.0  # Different format
    assert chart_verifier.judge("A-Berlin, B-Paris, C-Rome", answer, question=question) == 0.0  # Wrong matches
    assert chart_verifier.judge("A=Paris, B=Berlin, C=Rome", answer, question=question) == 1.0
    assert chart_verifier.judge("A->Paris, B->Berlin, C->Rome", answer, question=question) == 1.0
    assert chart_verifier.judge("A Paris B Berlin C Rome", answer, question=question) == 1.0
    assert chart_verifier.judge("B-Berlin, A-Paris, C-Rome", answer, question=question) == 1.0
    assert chart_verifier.judge("C-Rome, B-Berlin, A-Paris", answer, question=question) == 1.0
    assert chart_verifier.judge("(A)Paris (B)Berlin (C)Rome", answer, question=question) == 1.0
    assert chart_verifier.judge("A)Paris B)Berlin C)Rome", answer, question=question) == 1.0

    number_question = "Match numbers to words:\n1) One\n2) Two\n3) Three"
    number_answer = "1-One, 2-Two, 3-Three"
    assert chart_verifier.judge(number_answer, number_answer, question=number_question) == 1.0
    assert chart_verifier.judge("1:One 2:Two 3:Three", number_answer, question=number_question) == 1.0


def test_chart_verifier_judge_fill_in_blank(chart_verifier):
    # Test fill in the blank questions
    question = "The capital of France is _____."
    assert chart_verifier.judge("Paris", "Paris", question=question) == 1.0
    assert chart_verifier.judge("paris", "Paris", question=question) == 1.0  # Case insensitive
    assert chart_verifier.judge("Paris, France", "Paris", question=question) == 1.0  # Extra info
    assert chart_verifier.judge("London", "Paris", question=question) == 0.0  # Wrong answer


def test_chart_verifier_judge_number_word_incorrect(chart_verifier):
    question = "How many objects are liked by more than 80 percent of people?"

    # Test incorrect number word variations
    assert chart_verifier.judge("zero", "1", question=question) == 0.0  # Wrong number
    assert chart_verifier.judge("one", "2", question=question) == 0.0  # Wrong number
    assert chart_verifier.judge("two", "3", question=question) == 0.0  # Wrong number
    assert chart_verifier.judge("three", "4", question=question) == 0.0  # Wrong number
    assert chart_verifier.judge("four", "5", question=question) == 0.0  # Wrong number
    assert chart_verifier.judge("five", "6", question=question) == 0.0  # Wrong number
    assert chart_verifier.judge("six", "7", question=question) == 0.0  # Wrong number
    assert chart_verifier.judge("seven", "8", question=question) == 0.0  # Wrong number
    assert chart_verifier.judge("eight", "9", question=question) == 0.0  # Wrong number
    assert chart_verifier.judge("nine", "10", question=question) == 0.0  # Wrong number
    assert chart_verifier.judge("ten", "11", question=question) == 0.0  # Wrong number
    assert chart_verifier.judge("eleven", "12", question=question) == 0.0  # Wrong number
    assert chart_verifier.judge("twelve", "13", question=question) == 0.0  # Wrong number
    assert chart_verifier.judge("thirteen", "14", question=question) == 0.0  # Wrong number
    assert chart_verifier.judge("fourteen", "15", question=question) == 0.0  # Wrong number
    assert chart_verifier.judge("fifteen", "16", question=question) == 0.0  # Wrong number
    assert chart_verifier.judge("sixteen", "17", question=question) == 0.0  # Wrong number
    assert chart_verifier.judge("seventeen", "18", question=question) == 0.0  # Wrong number
    assert chart_verifier.judge("eighteen", "19", question=question) == 0.0  # Wrong number
    assert chart_verifier.judge("nineteen", "20", question=question) == 0.0  # Wrong number
    assert chart_verifier.judge("twenty", "21", question=question) == 0.0  # Wrong number


def test_chart_verifier_judge_essay_with_multiple_elements(chart_verifier):
    # Test fill in the blank questions
    question = "Which two x-axis labels of 2 sums up to 20.3 ?"
    assert chart_verifier.judge("Beijing and Tokyo", "Tokyo, Beijing", question=question) == 1.0


def test_chart_verifier_judge_Ambiguous_answers(chart_verifier):
    question = "What is the sum of the average of 1963.0  and the average of 1961.0 ?"
    assert (
        chart_verifier.judge(
            "\\text{label from the graph (identified by visual inspection or analysis of bar heights)}",
            "1.0",
            question=question,
        )
        == 0.0
    )
    question = "What is the average of the highest and lowest value of AVG ?"
    assert (
        chart_verifier.judge(
            "\\text{labels from the bar graph where 1971 values have the specified difference (specific labels based on graph data)}",
            "0.28",
            question=question,
        )
        == 0.0
    )
    question = "What is the sum of the average of 1963.0  and the average of 1961.0 ?"
    assert (
        chart_verifier.judge(
            "\\text{l\\text{value from graph (based on visual data)}", "-5207080204.63", question=question
        )
        == 0.0
    )


def test_chart_verifier_judge_year_question(chart_verifier):
    question = (
        "Which x-axis label has the minimum difference between Dominica and Belize"  # question中不出现year等关键词
    )
    assert chart_verifier.judge("2010", "2010", question=question) == 1.0
    assert chart_verifier.judge("2010", "2011", question=question) == 0.0
    question_with_year = (
        "Which year has the minimum difference between Dominica and Belize"  # question中出现year等关键词
    )
    assert chart_verifier.judge("2010", "2010", question=question_with_year) == 1.0
    assert chart_verifier.judge("2010", "2011", question=question_with_year) == 0.0
