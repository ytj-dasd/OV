import pytest


def test_android_world_verifier_judge(android_world_verifier):
    # Test basic number comparison
    # assert android_world_verifier.judge("<|begin_of_box|>", "\\boxed{2.0}") == 1.0
    gt = """Memory: None
Reason: I have selected the image of the Camellia within the PictureThis app. The current screen is asking to "Choose related organ". The image shows a flower, so I need to select 'flower' to help the app identify the plant. Clicking on the 'flower' option with the associated image will provide the necessary information for the app to proceed with the identification.
Action: <|begin_of_box|>{"action_type": "click", "box_2d": [[503,580,718,702]]}<|end_of_box|>"""
    ans_swap_pos = """Memory: None
Reason: I have selected the image of the Camellia within the PictureThis app. The current screen is asking to "Choose related organ". The image shows a flower, so I need to select 'flower' to help the app identify the plant. Clicking on the 'flower' option with the associated image will provide the necessary information for the app to proceed with the identification.
Action: <|begin_of_box|>{"box_2d": [[503,580,718,702]], "action_type": "click"}<|end_of_box|>"""

    # Extract dictionaries from strings first
    gt_dict = android_world_verifier.extract_answer(gt)
    ans_dict = android_world_verifier.extract_answer(ans_swap_pos)

    assert android_world_verifier.judge(ground_truth=gt_dict, extracted_answer=ans_dict) == 1.0

    ans_more_param = """Memory: None
Reason: I have selected the image of the Camellia within the PictureThis app. The current screen is asking to "Choose related organ". The image shows a flower, so I need to select 'flower' to help the app identify the plant. Clicking on the 'flower' option with the associated image will provide the necessary information for the app to proceed with the identification.
Action: <|begin_of_box|>{"action_type": "click", "box_2d": [[503,580,718,702]], "element_type": "button"}<|end_of_box|>"""
    ans_more_dict = android_world_verifier.extract_answer(ans_more_param)
    assert android_world_verifier.judge(ground_truth=gt_dict, extracted_answer=ans_more_dict) == 0.0

    ans_less_param = """Memory: None
Reason: I have selected the image of the Camellia within the PictureThis app. The current screen is asking to "Choose related organ". The image shows a flower, so I need to select 'flower' to help the app identify the plant. Clicking on the 'flower' option with the associated image will provide the necessary information for the app to proceed with the identification.
Action: <|begin_of_box|>{"action_type": "click"}<|end_of_box|>"""
    ans_less_dict = android_world_verifier.extract_answer(ans_less_param)
    assert android_world_verifier.judge(ground_truth=gt_dict, extracted_answer=ans_less_dict) == 0.0

    ans_wrong_type = """Memory: None
Reason: I have selected the image of the Camellia within the PictureThis app. The current screen is asking to "Choose related organ". The image shows a flower, so I need to select 'flower' to help the app identify the plant. Clicking on the 'flower' option with the associated image will provide the necessary information for the app to proceed with the identification.
Action: <|begin_of_box|>{"action_type": "wait", "box_2d": [[503,580,718,702]]}<|end_of_box|>"""
    ans_wrong_dict = android_world_verifier.extract_answer(ans_wrong_type)
    assert android_world_verifier.judge(ground_truth=gt_dict, extracted_answer=ans_wrong_dict) == 0.0

    ans_invalid_box = """Memory: None
Reason: I have selected the image of the Camellia within the PictureThis app. The current screen is asking to "Choose related organ". The image shows a flower, so I need to select 'flower' to help the app identify the plant. Clicking on the 'flower' option with the associated image will provide the necessary information for the app to proceed with the identification.
Action: <|begin_of_box|>{"box_2d": [[550,580,550,702]], "action_type": "click"}<|end_of_box|>"""
    ans_invalid_dict = android_world_verifier.extract_answer(ans_invalid_box)
    assert android_world_verifier.judge(ground_truth=gt_dict, extracted_answer=ans_invalid_dict) == 0.0

    ans_outside_box = """Memory: None
Reason: I have selected the image of the Camellia within the PictureThis app. The current screen is asking to "Choose related organ". The image shows a flower, so I need to select 'flower' to help the app identify the plant. Clicking on the 'flower' option with the associated image will provide the necessary information for the app to proceed with the identification.
Action: <|begin_of_box|>{"box_2d": [[203,580,418,702]], "action_type": "click"}<|end_of_box|>"""
    ans_outside_dict = android_world_verifier.extract_answer(ans_outside_box)
    assert android_world_verifier.judge(ground_truth=gt_dict, extracted_answer=ans_outside_dict) == 0.0

    ans_box_miner_diff = """Memory: None
Reason: I have selected the image of the Camellia within the PictureThis app. The current screen is asking to "Choose related organ". The image shows a flower, so I need to select 'flower' to help the app identify the plant. Clicking on the 'flower' option with the associated image will provide the necessary information for the app to proceed with the identification.
Action: <|begin_of_box|>{"box_2d": [[503,480,718,602]], "action_type": "click"}<|end_of_box|>"""
    ans_box_miner_diff_dict = android_world_verifier.extract_answer(ans_box_miner_diff)
    assert 0.0 < android_world_verifier.judge(ground_truth=gt_dict, extracted_answer=ans_box_miner_diff_dict) < 1.0

    ans_box_format_error = """Memory: None
Reason: I have selected the image of the Camellia within the PictureThis app. The current screen is asking to "Choose related organ". The image shows a flower, so I need to select 'flower' to help the app identify the plant. Clicking on the 'flower' option with the associated image will provide the necessary information for the app to proceed with the identification.
Action: <|begin_of_box|>{{"box_2d": [[503,480,718,602]], "action_type": "click"}}<|end_of_box|>"""
    ans_box_format_error_dict = android_world_verifier.extract_answer(ans_box_format_error)
    assert android_world_verifier.judge(ground_truth=gt_dict, extracted_answer=ans_box_format_error_dict) == 0.0

    gt_no_override = """Memory: None
    Reason: I have selected the image of the Camellia within the PictureThis app. The current screen is asking to "Choose related organ"
Action: <|begin_of_box|>{'action_type': 'input_text', 'text': 'Personal Finance Tracker', 'box_2d': [[38, 115, 961, 140]]}<|end_of_box|>
"""

    gt_override = """Memory: None
Reason: I have selected the image of the Camellia within the PictureThis app. The current screen is asking to "Choose related organ"
Action: <|begin_of_box|>{'action_type': 'input_text', 'text': 'Personal Finance Tracker', 'override': True, 'box_2d': [[38, 115, 961, 140]]}<|end_of_box|>
"""

    ans_no_override = """Memory: None
    Reason: I have selected the image of the Camellia within the PictureThis app. The current screen is asking to "Choose related organ"
Action: <|begin_of_box|>{'action_type': 'input_text', 'text': 'Personal Finance Tracker', 'box_2d': [[38, 115, 961, 140]]}<|end_of_box|>
"""

    ans_override = """Memory: None
Reason: I have selected the image of the Camellia within the PictureThis app. The current screen is asking to "Choose related organ"
Action: <|begin_of_box|>{'action_type': 'input_text', 'text': 'Personal Finance Tracker', 'override': True, 'box_2d': [[38, 115, 961, 140]]}<|end_of_box|>
"""
    ans_override2 = """Memory: None
Reason: I have selected the image of the Camellia within the PictureThis app. The current screen is asking to "Choose related organ"
Action: <|begin_of_box|>{'action_type': 'input_text', 'text': 'Personal Finance Tracker', 'override': true, 'box_2d': [[38, 115, 961, 140]]}<|end_of_box|>
"""

    gt_no_override_dict = android_world_verifier.extract_answer(gt_no_override)
    ans_override_dict = android_world_verifier.extract_answer(ans_override)
    # Different keys should return 0.0
    assert android_world_verifier.judge(ground_truth=gt_no_override_dict, extracted_answer=ans_override_dict) == 0.0

    gt_override_dict = android_world_verifier.extract_answer(gt_override)
    ans_no_override_dict = android_world_verifier.extract_answer(ans_no_override)
    # Different keys should return 0.0
    assert android_world_verifier.judge(ground_truth=gt_override_dict, extracted_answer=ans_no_override_dict) == 0.0

    ans_override2_dict = android_world_verifier.extract_answer(ans_override2)
    # Same keys should return 1.0
    assert android_world_verifier.judge(ground_truth=gt_no_override_dict, extracted_answer=ans_no_override_dict) == 1.0
    assert android_world_verifier.judge(ground_truth=gt_override_dict, extracted_answer=ans_override_dict) == 1.0
    assert android_world_verifier.judge(ground_truth=gt_override_dict, extracted_answer=ans_override2_dict) == 1.0
