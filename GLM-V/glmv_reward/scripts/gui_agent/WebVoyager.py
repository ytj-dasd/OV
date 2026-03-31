# verifiers/webvoyager.py
import re
from typing import Any, Optional


def extract_answer(response: str, question: Optional[str] = None) -> Optional[str]:
    return re.findall(r"<\|begin_of_box\|>(.*?)<\|end_of_box\|>", response, re.DOTALL)[0]


def lcs(x, y):
    m = len(x)
    n = len(y)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


def judge(
    extracted_answer: Any,
    ground_truth: Any,
    question: Optional[str] = None,
    image_path=None,
) -> float:
    def extract_information(text):
        # New format patterns
        new_patterns = {
            "click": r"CLICK\(point=\((\d+),\s*(\d+)\)(?:,\s*box=\[\[([^\]]+)\]\])?(?:,\s*element_info='([^']*)')?\)",
            "type": r"TYPE\(point=\((\d+),\s*(\d+)\)(?:,\s*text='([^']*)')?(?:,\s*box=\[\[([^\]]+)\]\])?(?:,\s*element_info='([^']*)')?\)",
            "key": r"KEY_PRESS\(key='([^']*)'\)",
            "scroll_down": r"SCROLL_DOWN\(point=\((\d+),\s*(\d+)\)(?:,\s*box=\[\[([^\]]+)\]\])?(?:,\s*distance=([^,\)]+))?(?:,\s*element_info='([^']*)')?\)",
            "scroll_up": r"SCROLL_UP\(point=\((\d+),\s*(\d+)\)(?:,\s*box=\[\[([^\]]+)\]\])?(?:,\s*distance=([^,\)]+))?(?:,\s*element_info='([^']*)')?\)",
            "answer": r"ANSWER\(content='([^']*)'\)",
        }

        # Try new format first
        for key, pattern in new_patterns.items():
            match = re.search(pattern, text, re.DOTALL)
            if match:
                if key == "click":
                    x, y, box, element_info = match.groups()
                    return "click", {"x": int(x), "y": int(y), "box": box, "element_info": element_info}
                elif key == "type":
                    x, y, text_content, box, element_info = match.groups()
                    return "type", {
                        "x": int(x),
                        "y": int(y),
                        "text": text_content or "",
                        "box": box,
                        "element_info": element_info,
                    }
                elif key == "key":
                    key_name = match.group(1)
                    return "key", {"key": key_name}
                elif key in ["scroll_down", "scroll_up"]:
                    x, y, box, distance, element_info = match.groups()
                    return "scroll", {
                        "x": int(x),
                        "y": int(y),
                        "box": box,
                        "distance": distance,
                        "element_info": element_info,
                        "direction": key.split("_")[1],
                    }
                elif key == "answer":
                    content = match.group(1)
                    return "answer", {"content": content}

        # Fall back to old format patterns
        old_patterns = {
            "click": r"Click \[?(\d+)\]?",
            "type": r"Type \[?(\d+)\]?[; ]+\[?(.[^\]]*)\]?",
            "key": r"Key[; ]+\[?(.[^\]]*)\]?",
            "scroll": r"Scroll \[?(\d+|WINDOW)\]?[; ]+\[?(up|down)\]?",
            "wait": r"^Wait",
            "goback": r"^GoBack",
            "google": r"^Google",
            "bing": r"^Bing",
            "answer": r"ANSWER[; ]+<content>(.*?)</content>",
        }

        for key, pattern in old_patterns.items():
            match = re.search(pattern, text, re.DOTALL)
            if match:
                if key in ["click", "wait", "goback", "google", "bing"]:
                    # no content
                    return key, match.groups()
                else:
                    return key, {"number": match.group(1), "content": match.group(2)} if key in [
                        "type",
                        "scroll",
                    ] else {"content": match.group(1)}
        return None, None

    action, info = extract_information(extracted_answer)
    gt_action, gt_info = extract_information(ground_truth)
    if action != gt_action:
        return 0.0

    if action == "click":
        # Handle new format
        if isinstance(info, dict) and isinstance(gt_info, dict):
            # New format: check point distance and element_info
            if info.get("x") == gt_info.get("x") and info.get("y") == gt_info.get("y"):
                return 1.0
            elif info.get("element_info") == gt_info.get("element_info"):
                # Same element but different position
                return 0.05
            else:
                return 0.0
        # Handle old format
        elif isinstance(info, tuple) and isinstance(gt_info, tuple):
            click_ele_number = int(info[0])
            gt_click_ele_number = int(gt_info[0])
            if click_ele_number == gt_click_ele_number:
                return 1.0
            else:
                return 0.0
        else:
            return 0.0

    elif action == "type":
        # Handle new format
        if isinstance(info, dict) and isinstance(gt_info, dict) and "text" in info:
            input_text = info.get("text", "")
            gt_input_text = gt_info.get("text", "")

            # Calculate text similarity
            if input_text == gt_input_text:
                text_similarity = 1.0
            elif len(input_text) > 0 and len(gt_input_text) > 0:
                text_similarity = lcs(input_text, gt_input_text) / max(len(input_text), len(gt_input_text))
            else:
                text_similarity = 0.0

            # Check position match
            if info.get("x") == gt_info.get("x") and info.get("y") == gt_info.get("y"):
                # Same position, return text similarity
                return text_similarity
            else:
                # Different position, return text similarity with small penalty
                return text_similarity * 0.95 if text_similarity > 0.05 else 0.05
        # Handle old format
        elif isinstance(info, dict) and isinstance(gt_info, dict) and "number" in info:
            type_ele_number = int(info["number"])
            gt_type_ele_number = int(gt_info["number"])
            input_text = info["content"]
            gt_input_text = gt_info["content"]
            if type_ele_number == gt_type_ele_number:
                return lcs(input_text, gt_input_text) / max(len(input_text), len(gt_input_text))
            else:
                return 0.0
        else:
            return 0.0

    elif action == "key":
        # Handle new format
        if isinstance(info, dict) and isinstance(gt_info, dict) and "key" in info:
            key_name = info["key"].lower()
            gt_key_name = gt_info["key"].lower()
            if key_name == gt_key_name:
                return 1.0
            # Handle equivalent keys
            elif (key_name == "return" and gt_key_name == "enter") or (key_name == "enter" and gt_key_name == "return"):
                return 0.05
            else:
                return 0.0
        # Handle old format
        elif isinstance(info, dict) and isinstance(gt_info, dict) and "content" in info:
            key_name = info["content"].lower()
            gt_key_name = gt_info["content"].lower()
            if key_name == gt_key_name:
                return 1.0
            else:
                return 0.0
        else:
            return 0.0

    elif action == "scroll":
        # Handle new format
        if isinstance(info, dict) and isinstance(gt_info, dict) and "distance" in info:
            if (
                info.get("x") == gt_info.get("x")
                and info.get("y") == gt_info.get("y")
                and info.get("direction") == gt_info.get("direction")
            ):
                # Same position and direction, compare distance
                try:
                    distance = float(info.get("distance", 0))
                    gt_distance = float(gt_info.get("distance", 0))
                    if distance == gt_distance:
                        return 1.0
                    else:
                        # Calculate similarity based on distance difference
                        max_distance = max(abs(distance), abs(gt_distance))
                        if max_distance == 0:
                            return 1.0
                        diff = abs(distance - gt_distance)
                        return max(0.0, 1.0 - diff / max_distance)
                except (ValueError, TypeError):
                    return 0.0
            else:
                return 0.0
        # Handle old format
        elif isinstance(info, dict) and isinstance(gt_info, dict) and "number" in info:
            scroll_ele_number = info["number"]
            scroll_content = info["content"]
            gt_scroll_ele_number = gt_info["number"]
            gt_scroll_content = gt_info["content"]
            if scroll_ele_number == gt_scroll_ele_number:
                if scroll_content == gt_scroll_content:
                    return 1.0
                else:
                    return 0.0
            else:
                return 0.0
        else:
            return 0.0

    elif action == "wait":
        return 1.0
    elif action == "goback":
        return 1.0
    elif action == "google":
        return 1.0
    elif action == "bing":
        return 1.0
    elif action == "answer":
        if isinstance(info, dict) and isinstance(gt_info, dict):
            answer_content = info.get("content", "")
            gt_answer_content = gt_info.get("content", "")
            if answer_content == gt_answer_content:
                return 1.0
            elif len(answer_content) > 0 and len(gt_answer_content) > 0:
                return lcs(answer_content, gt_answer_content) / max(len(answer_content), len(gt_answer_content))
            else:
                return 0.0
        else:
            return 0.0
    else:
        return 0.0
