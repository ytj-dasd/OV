import pytest


def test_webvoyager_verifier_judge_number(webvoyager_verifier):
    # Test basic number comparison

    # assert webvoyager_verifier.judge("\\boxed{2}", "\\boxed{2.0}") == 1.0
    # assert webvoyager_verifier.judge("\\boxed{2.0}","\\boxed{2}") == 1.0
    # assert webvoyager_verifier.judge("\\boxed{2}", "\\boxed{TWO}") == 1.0
    # assert webvoyager_verifier.judge("\\boxed{TWO}", "\\boxed{2}") == 1.0
    examples = [
        [
            "<think></think><answer><|begin_of_box|>CLICK(point=(214, 100), element_info='Dinners')<|end_of_box|></answer>",
            "<think></think><answer><|begin_of_box|>CLICK(point=(230, 100), element_info='Dinners')<|end_of_box|></answer>",
        ],
        [
            "<think></think><answer><|begin_of_box|>CLICK(point=(214, 100), box=[[198,92,229,107]], element_info='Dinners')<|end_of_box|></answer>",
            "<think></think><answer><|begin_of_box|>CLICK(point=(230, 100), box=[[198,92,229,107]], element_info='Dinners')<|end_of_box|></answer>",
        ],
        [
            "<think></think><answer><|begin_of_box|>CLICK(point=(214, 100), box=[[198,92,229,107]], element_info='Dinners')<|end_of_box|></answer>",
            "<think></think><answer><|begin_of_box|>CLICK(point=(200, 100), box=[[198,92,229,107]], element_info='Dinners')<|end_of_box|></answer>",
        ],
        [
            "<think></think><answer><|begin_of_box|>CLICK(point=(214, 100), box=[[198,92,229,107]], element_info='Dinners')<|end_of_box|></answer>",
            "<think></think><answer><|begin_of_box|>CLICK(point=(214, 100), box=[[198,92,229,100]], element_info='Dinners')<|end_of_box|></answer>",
        ],
        [
            "<think></think><answer><|begin_of_box|>TYPE(point=(214, 100), box=[[198,92,229,107]], element_info='Dinners')<|end_of_box|></answer>",
            "<think></think><answer><|begin_of_box|>CLICK(point=(200, 100), box=[[198,92,229,100]], element_info='Dinners')<|end_of_box|></answer>",
        ],
        [
            "<think></think><answer><|begin_of_box|>TYPE(point=(214, 100), text='bcd', box=[[198,92,229,107]], element_info='Dinners')<|end_of_box|></answer>",
            "<think></think><answer><|begin_of_box|>TYPE(point=(200, 100), text='abcd', box=[[198,92,229,107]], element_info='Dinners')<|end_of_box|></answer>",
        ],
        [
            "<think></think><answer><|begin_of_box|>TYPE(point=(214, 100), text='bcd', box=[[0,0,1,1]], element_info='Dinners')<|end_of_box|></answer>",
            "<think></think><answer><|begin_of_box|>TYPE(point=(200, 100), text='abcd', box=[[198,92,229,107]], element_info='Dinners')<|end_of_box|></answer>",
        ],
        [
            "<think></think><answer><|begin_of_box|>KEY_PRESS(key='Return')<|end_of_box|></answer>",
            "<think></think><answer><|begin_of_box|>KEY_PRESS(key='Enter')<|end_of_box|></answer>",
        ],
        [
            "<think></think><answer><|begin_of_box|>KEY_PRESS(key='Enter')<|end_of_box|></answer>",
            "<think></think><answer><|begin_of_box|>KEY_PRESS(key='Enter')<|end_of_box|></answer>",
        ],
        [
            "<think></think><answer><|begin_of_box|>SCROLL_DOWN(point=(214, 100), distance=50, element_info='Dinners')<|end_of_box|></answer>",
            "<think></think><answer><|begin_of_box|>SCROLL_DOWN(point=(214, 100), distance=0, element_info='Dinners')<|end_of_box|></answer>",
        ],
        [
            "<think></think><answer><|begin_of_box|>SCROLL_DOWN(point=(214, 100), box=[[198,92,229,107]], distance=0.5, element_info='Dinners')<|end_of_box|></answer>",
            "<think></think><answer><|begin_of_box|>SCROLL_DOWN(point=(214, 100), box=[[198,92,229,107]], distance=0.75, element_info='Dinners')<|end_of_box|></answer>",
        ],
        [
            "<think></think><answer><|begin_of_box|>SCROLL_DOWN(point=(214, 100), box=[[198,92,229,107]], distance=0.75, element_info='Dinners')<|end_of_box|></answer>",
            "<think></think><answer><|begin_of_box|>SCROLL_DOWN(point=(214, 100), box=[[198,92,229,107]], distance=0.5, element_info='Dinners')<|end_of_box|></answer>",
        ],
        [
            "<think></think><answer><|begin_of_box|>ANSWER(content='Dinners')<|end_of_box|></answer>",
            "<think></think><answer><|begin_of_box|>ANSWER(content='Dinners')<|end_of_box|></answer>",
        ],
    ]
    for example_id, example in enumerate(examples):
        if example_id == 0:
            assert webvoyager_verifier.judge(example[0], example[1]) == 0.05, (
                f"Example {example_id} failed: {example[0]} vs {example[1]}"
            )
