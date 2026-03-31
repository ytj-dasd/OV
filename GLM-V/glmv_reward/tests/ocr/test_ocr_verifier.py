import pytest


def test_ocr_text_recognition(ocr_verifier):
    """Test OCR text recognition with various text formats"""
    # Test Chinese text recognition
    question = "What is the text on the sign?"
    exact_match = ocr_verifier.judge("社区服", "社区服", question=question)
    wrong_char = ocr_verifier.judge("社区服务", "社区服", question=question)
    missing_chars = ocr_verifier.judge("西城区委", "中共西城区委", question=question)
    full_match = ocr_verifier.judge("社区服务", "社区服务", question=question)
    partial_match = ocr_verifier.judge("社区", "社区服", question=question)

    assert exact_match == 1.0
    assert full_match == 1.0
    assert exact_match > wrong_char
    assert exact_match > missing_chars
    assert exact_match > partial_match
    assert full_match > wrong_char
    assert full_match > missing_chars
    assert full_match > partial_match

    # Test English text recognition
    question = "What is the name of the establishment?"
    exact_match = ocr_verifier.judge("yeoldeUNIONOYSTER HOUSE", "yeoldeUNIONOYSTER HOUSE", question=question)
    missing_word = ocr_verifier.judge("yeoldeUNIONOYSTER", "yeoldeUNIONOYSTER HOUSE", question=question)
    wrong_word = ocr_verifier.judge("THE UNION PUB", "THE UNION BAR", question=question)
    diff_capital = ocr_verifier.judge("OLD UNION OYSTER HOUSE", "yeoldeUNIONOYSTER HOUSE", question=question)
    partial_name = ocr_verifier.judge("THE UNION", "THE UNION BAR", question=question)

    assert exact_match == 1.0
    assert exact_match > missing_word
    assert exact_match > wrong_word
    assert exact_match > diff_capital
    assert exact_match > partial_name

    # Test mixed language text
    question = "What is the address?"
    exact_match = ocr_verifier.judge("复兴路32号院社区居委会", "复兴路32号院社区居委会", question=question)
    wrong_num = ocr_verifier.judge("复兴路33号院社区居委会", "复兴路32号院社区居委会", question=question)
    missing_num = ocr_verifier.judge("育才路", "育才路53号", question=question)
    incomplete = ocr_verifier.judge("复兴路32号", "复兴路32号院社区居委会", question=question)

    assert exact_match == 1.0
    assert exact_match > wrong_num
    assert exact_match > missing_num
    assert exact_match > incomplete

    # Test mathematical expressions
    question = "What is the mathematical expression?"
    exact_match = ocr_verifier.judge(
        "x=α₁y₁+...+αᵢ₋₁yᵢ₋₁+αᵢ₊₁yᵢ₊₁+...αₓₙ₊₁yₙ₊₁", "x=α₁y₁+...+αᵢ₋₁yᵢ₋₁+αᵢ₊₁yᵢ₊₁+...αₓₙ₊₁yₙ₊₁", question=question
    )
    missing_sub = ocr_verifier.judge(
        "x=α₁y₁+...+αᵢ₋₁yᵢ₋₁+αᵢ₊₁yᵢ₊₁+...αₓₙ₊₁yₙ", "x=α₁y₁+...+αᵢ₋₁yᵢ₋₁+αᵢ₊₁yᵢ₊₁+...αₓₙ₊₁yₙ₊₁", question=question
    )
    wrong_sub = ocr_verifier.judge(
        "x=α₁y₁+...+αᵢ₋₁yᵢ₋₁+αᵢ₊₁yᵢ₊₁+...αₓₙ₊₁yₙ₊₂", "x=α₁y₁+...+αᵢ₋₁yᵢ₋₁+αᵢ₊₁yᵢ₊₁+...αₓₙ₊₁yₙ₊₁", question=question
    )

    assert exact_match == 1.0
    assert exact_match > missing_sub
    assert exact_match > wrong_sub


def test_judge_similarity(ocr_verifier):
    exact_match = ocr_verifier.judge("x+1", "x+1")
    similar = ocr_verifier.judge("x+1", "x+2")
    different = ocr_verifier.judge("x+1", "y+1")

    assert exact_match == 1.0
    assert exact_match > similar
    assert exact_match > different


def test_judge_badcase(ocr_verifier):
    chinese_diff = ocr_verifier.judge("这是小苹果", "我不知道")
    english_diff = ocr_verifier.judge("hello world", "thanks.")

    assert chinese_diff == english_diff

    # Test similar but different expressions
    similar_expr1 = ocr_verifier.judge("2x + 3y = 5", "2x + 3y = 6")
    similar_expr2 = ocr_verifier.judge("2x + 3y = 5", "2x + 4y = 5")
    similar_expr3 = ocr_verifier.judge("2x + 3y = 5", "3x + 3y = 5")

    assert similar_expr1 > 0.0
    assert similar_expr2 > 0.0
    assert similar_expr3 > 0.0
    assert similar_expr1 == similar_expr2
    assert similar_expr2 == similar_expr3

    # Test partial matches
    partial1 = ocr_verifier.judge("The quick brown fox", "The quick brown fox jumps")
    partial2 = ocr_verifier.judge("The quick brown fox", "The quick brown fox jumps over")
    partial3 = ocr_verifier.judge("The quick brown fox", "The quick brown fox jumps over the lazy dog")

    assert partial1 > 0.0
    assert partial2 > 0.0
    assert partial3 > 0.0
    assert partial1 > partial2
    assert partial2 > partial3

    # Test mixed content
    mixed1 = ocr_verifier.judge("Price: $99.99", "Price: $99.99")
    mixed2 = ocr_verifier.judge("Price: $99.99", "Price: $100.00")
    mixed3 = ocr_verifier.judge("Price: $99.99", "Price: $199.99")
    mixed4 = ocr_verifier.judge("Price: $99.99", "Pr1ce: S188.88")

    assert mixed1 > 0.0
    assert mixed2 > 0.0
    assert mixed3 > 0.0
    assert mixed1 > mixed2
    assert mixed1 > mixed3
    assert mixed1 > mixed4
    assert mixed2 > mixed4
    assert mixed3 > mixed4

    # Test various text formats and edge cases
    # Test Chinese movie theater names
    theater1 = ocr_verifier.judge("峨影1958", "峨影1958")
    theater2 = ocr_verifier.judge("峨影", "峨影1958")
    theater3 = ocr_verifier.judge("峨影1958", "峨影1958影城")

    assert theater1 == 1.0
    assert theater2 > 0.0
    assert theater3 > 0.0

    # Test mixed content with numbers and text
    mixed_num1 = ocr_verifier.judge("182692", "182692")
    mixed_num2 = ocr_verifier.judge("182692", "182693")
    mixed_num3 = ocr_verifier.judge("182692", "182692-00001")

    assert mixed_num1 == 1.0
    assert mixed_num2 > 0.0
    assert mixed_num3 > 0.0
    assert mixed_num1 > mixed_num2
    assert mixed_num1 > mixed_num3

    # Test mathematical expressions
    math1 = ocr_verifier.judge("X^{Y}", "X^{Y}")
    math2 = ocr_verifier.judge("c=\sqrt{2}", "c=\sqrt{2}")
    math3 = ocr_verifier.judge("X^{Y}", "X^{Z}")

    assert math1 == 1.0
    assert math2 == 1.0
    assert math3 > 0.0
    assert math1 > math3

    # Test mixed language content
    mixed_lang1 = ocr_verifier.judge("三楼：影院、KTV、轻", "三楼：影院、KTV、轻")
    mixed_lang2 = ocr_verifier.judge("三楼：影院、KTV", "三楼：影院、KTV、轻")
    mixed_lang3 = ocr_verifier.judge("三楼：影院", "三楼：影院、KTV、轻")

    assert mixed_lang1 == 1.0
    assert mixed_lang1 > mixed_lang2
    assert mixed_lang2 > mixed_lang3

    # Test scientific content
    science1 = ocr_verifier.judge("C.氢、氮等轻核元素是宇宙中天然元素之母", "C.氢、氮等轻核元素是宇宙中天然元素之母")
    science2 = ocr_verifier.judge("C.氢、氮等轻核元素", "C.氢、氮等轻核元素是宇宙中天然元素之母")
    science3 = ocr_verifier.judge("C.氢、氮等", "C.氢、氮等轻核元素是宇宙中天然元素之母")

    assert science1 > science2
    assert science2 > science3


def test_ocr_verifier_additional_cases(ocr_verifier):
    # Test special characters and symbols
    special1 = ocr_verifier.judge("© 2023", "© 2023")
    special2 = ocr_verifier.judge("© 2023", "© 2024")
    special3 = ocr_verifier.judge("© 2023", "© 2023 All Rights Reserved")

    assert special1 == 1.0
    assert special1 > special2
    assert special1 > special3

    # Test dates and times
    date1 = ocr_verifier.judge("2023-12-31", "2023-12-31")
    date2 = ocr_verifier.judge("2023-12-31", "2023-12-30")
    date3 = ocr_verifier.judge("2023-12-31", "2023-12-31 23:59")

    assert date1 == 1.0
    assert date1 > date2
    assert date1 > date3

    # Test addresses and locations
    address1 = ocr_verifier.judge("123 Main St, City", "123 Main St, City")
    address2 = ocr_verifier.judge("123 Main St", "123 Main St, City")
    address3 = ocr_verifier.judge("123 Main", "123 Main St, City")

    assert address1 == 1.0
    assert address1 > address2
    assert address2 > address3

    # Test product codes and serial numbers
    code1 = ocr_verifier.judge("ABC-123-XYZ", "ABC-123-XYZ")
    code2 = ocr_verifier.judge("ABC-123", "ABC-123-XYZ")
    code3 = ocr_verifier.judge("ABC", "ABC-123-XYZ")

    assert code1 == 1.0
    assert code1 > code2
    assert code2 > code3

    # Test mixed format content
    mixed1 = ocr_verifier.judge("Order #12345: $99.99", "Order #12345: $99.99")
    mixed2 = ocr_verifier.judge("Order #12345", "Order #12345: $99.99")
    mixed3 = ocr_verifier.judge("Order", "Order #12345: $99.99")

    assert mixed1 == 1.0
    assert mixed1 > mixed2
    assert mixed2 > mixed3

    # Test OCR text with various formats and languages
    ocr1 = ocr_verifier.judge("Robert P. Langlais", "Robert P. Langlais")
    ocr2 = ocr_verifier.judge("Robert Langlais", "Robert P. Langlais")
    ocr3 = ocr_verifier.judge("R. Langlais", "Robert P. Langlais")

    assert ocr1 == 1.0
    assert ocr1 > ocr2
    assert ocr2 > ocr3

    # Test mixed language content
    mixed_lang1 = ocr_verifier.judge("西班牙风情街", "西班牙风情街")
    mixed_lang2 = ocr_verifier.judge("西班牙街", "西班牙风情街")
    mixed_lang3 = ocr_verifier.judge("西班牙", "西班牙风情街")

    assert mixed_lang1 == 1.0
    assert mixed_lang1 > mixed_lang2
    assert mixed_lang2 > mixed_lang3

    # Test mathematical expressions
    math1 = ocr_verifier.judge("3.045dm^{3}=(6)cm^{3}(700)cm^{3}", "3.045dm^{3}=(6)cm^{3}(700)cm^{3}")
    math2 = ocr_verifier.judge("3.045dm^3=6cm^3+700cm^3", "3.045dm^{3}=(6)cm^{3}(700)cm^{3}")
    math3 = ocr_verifier.judge("3.045dm^3", "3.045dm^{3}=(6)cm^{3}(700)cm^{3}")

    assert math1 == 1.0
    assert math1 > math2
    assert math2 > math3

    # Test phone numbers and codes
    phone1 = ocr_verifier.judge("13026141688", "13026141688")
    phone2 = ocr_verifier.judge("1302614168", "13026141688")
    phone3 = ocr_verifier.judge("1302614", "13026141688")

    assert phone1 == 1.0
    assert phone1 > phone2
    assert phone2 > phone3

    # Test dates and times with different formats
    date_time1 = ocr_verifier.judge("08/30/21-09/14/21", "08/30/21-09/14/21")
    date_time2 = ocr_verifier.judge("08/30/21", "08/30/21-09/14/21")
    date_time3 = ocr_verifier.judge("08/30", "08/30/21-09/14/21")

    assert date_time1 == 1.0
    assert date_time1 > date_time2
    assert date_time2 > date_time3


def test_upper_lower_case(ocr_verifier):
    ocr_verifier.ignore_case = True
    case1 = ocr_verifier.judge("AFSDF", "AFSDF")
    case2 = ocr_verifier.judge("AFSDF", "afsdf")
    case3 = ocr_verifier.judge("afsdf", "AFSDF")
    case4 = ocr_verifier.judge("AFSDF", "AFSDF.")

    assert case1 == case2 == case3 == 1.0
    assert case1 > case4
