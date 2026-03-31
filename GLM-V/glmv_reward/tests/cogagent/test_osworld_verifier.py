import pytest

TYPE_WEIGHT, PARAM_WEIGHT = 0.01, 0.99


def test_basic_click_actions_valid(osworld_verifier):
    basic_clicks = {
        "left_click": "[100, 200]",
        "right_click": "[300, 400]",
        "middle_click": "[250, 350]",
        "hover": "[150, 250]",
        "left_double_click": "[200, 300]",
    }

    for action, coords in basic_clicks.items():
        gt = f"""<think>test</think>Action: <|begin_of_box|>{action}(start_box='{coords}')<|end_of_box|>"""
        gt_dict = osworld_verifier.extract_answer(gt)
        assert osworld_verifier.judge(gt_dict, gt_dict) >= 0.8
        coords_list = eval(coords)
        slight_off = f"[{coords_list[0] + 10}, {coords_list[1] - 10}]"
        ans = f"""<think>test</think>Action: <|begin_of_box|>{action}(start_box='{slight_off}')<|end_of_box|>"""
        ans_dict = osworld_verifier.extract_answer(ans)
        assert osworld_verifier.judge(ans_dict, gt_dict) >= 0.5
        ans = f"""<think>test</think>Action: <|begin_of_box|>{action}(start_box='{coords}', element_info='123456')<|end_of_box|>"""
        ans_dict = osworld_verifier.extract_answer(ans)
        assert osworld_verifier.judge(ans_dict, gt_dict) >= 0.8


def test_complex_actions(osworld_verifier):
    gt = """<think>test</think>Action: <|begin_of_box|>left_drag(start_box='[100, 100]', end_box='[300, 300]')<|end_of_box|>"""
    ans = """<think>test</think>Action: <|begin_of_box|>left_drag(start_box='[105, 95]', end_box='[295, 305]')<|end_of_box|>"""
    gt_dict = osworld_verifier.extract_answer(gt)
    ans_dict = osworld_verifier.extract_answer(ans)
    assert osworld_verifier.judge(ans_dict, gt_dict) >= 0.8


def test_text_input(osworld_verifier):
    text_tests = [
        ("Hello World!", "Hello World!", 1),
        ("Hello World!", "HeLloworld!", 0.6),
        ("Hello World!", "Goodbye World!", 0.2),
    ]

    for gt_text, ans_text, expected_score in text_tests:
        gt = f"""<think>test</think>Action: <|begin_of_box|>type(content='{gt_text}')<|end_of_box|>"""
        ans = f"""<think>test</think>Action: <|begin_of_box|>type(content='{ans_text}')<|end_of_box|>"""
        gt_dict = osworld_verifier.extract_answer(gt)
        ans_dict = osworld_verifier.extract_answer(ans)
        assert osworld_verifier.judge(ans_dict, gt_dict) >= expected_score


def test_scroll_actions(osworld_verifier):
    scroll_tests = [
        ("down", 3, "down", 3, 0.8),
        ("up", 2, "up", 2, 0.8),
        ("down", 3, "up", 3, 0.0),
        ("down", 3, "down", 10, 0.0),
    ]

    for gt_dir, gt_step, ans_dir, ans_step, expected_score in scroll_tests:
        gt = f"""<think>test</think>Action: <|begin_of_box|>scroll(start_box='[200, 200]', direction='{gt_dir}', step={gt_step})<|end_of_box|>"""
        ans = f"""<think>test</think>Action: <|begin_of_box|>scroll(start_box='[198, 202]', direction='{ans_dir}', step={ans_step})<|end_of_box|>"""
        gt_dict = osworld_verifier.extract_answer(gt)
        ans_dict = osworld_verifier.extract_answer(ans)
        assert osworld_verifier.judge(ans_dict, gt_dict) >= expected_score


def test_special_actions(osworld_verifier):
    special_actions = ["WAIT()", "DONE()", "FAIL()"]

    for action in special_actions:
        gt = f"""<think>test</think>Action: <|begin_of_box|>{action}<|end_of_box|>"""
        gt_dict = osworld_verifier.extract_answer(gt)
        assert osworld_verifier.judge(gt_dict, gt_dict) == 1
        if action.lower() != action:
            ans = f"""<think>test</think>Action: <|begin_of_box|>{action.lower()}<|end_of_box|>"""
            ans_dict = osworld_verifier.extract_answer(ans)
            assert osworld_verifier.judge(ans_dict, gt_dict) == 0.0
        else:
            ans = f"""<think>test</think>Action: <|begin_of_box|>{action.upper()}<|end_of_box|>"""
            ans_dict = osworld_verifier.extract_answer(ans)
            assert osworld_verifier.judge(ans_dict, gt_dict) == 0.0
