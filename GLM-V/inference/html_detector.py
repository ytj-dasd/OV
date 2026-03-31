import re
import time
from dataclasses import dataclass


@dataclass
class StreamingResult:
    output: str
    html_detected: bool


class HTMLDetector:
    MAX_BUFFER_CHUNKS = 5

    def __init__(self, streaming_mode: bool):
        self.streaming_mode = streaming_mode
        self.html_detected = False
        self.buffer = ""
        self.html_prefix_added = False
        self.detection_threshold = 50
        self.chunk_count = 0
        self.html_start_patterns = [
            re.compile(r"^<!DOCTYPE\s+html", re.IGNORECASE),
            re.compile(r"^<html[\s>]", re.IGNORECASE),
            re.compile(r"^<head[\s>]", re.IGNORECASE),
            re.compile(
                r"^<bod" r"y[\s>]",
                re.IGNORECASE,
            ),
            re.compile(r"^<div[\s>]", re.IGNORECASE),
            re.compile(r"^<section[\s>]", re.IGNORECASE),
            re.compile(r"^<article[\s>]", re.IGNORECASE),
            re.compile(r"^<main[\s>]", re.IGNORECASE),
            re.compile(r"^<header[\s>]", re.IGNORECASE),
            re.compile(r"^<footer[\s>]", re.IGNORECASE),
            re.compile(r"^<nav[\s>]", re.IGNORECASE),
        ]

        self.html_structure_pattern = re.compile(
            r"<[a-zA-Z][^>]*>.*</[a-zA-Z]+>", re.DOTALL
        )
        self.html_features_pattern = re.compile(
            r"(?:<!DOCTYPE|<html|<head|<body|<div|class=|id=|href=|src=)", re.IGNORECASE
        )

    def count_occurrences(self, text: str, substring: str) -> int:
        return text.count(substring)

    def find_html_start_position(self, text: str) -> int:
        positions = []
        html_patterns = [
            r"<!DOCTYPE\s+html",
            r"<html\b",
            r"<head\b",
            r"<body\b",
            r"<div\b",
            r"<section\b",
            r"<article\b",
            r"<main\b",
            r"<header\b",
            r"<footer\b",
            r"<nav\b",
            r"<[a-zA-Z]+",
        ]
        for pat in html_patterns:
            for match in re.finditer(pat, text, re.IGNORECASE):
                positions.append(match.start())
        return min(positions) if positions else 0

    def is_html_content(self, text: str) -> bool:
        trimmed = text.strip()
        if "```" in trimmed:
            return False

        if any(p.search(trimmed) for p in self.html_start_patterns):
            return True

        has_html_features = bool(self.html_features_pattern.search(trimmed))

        has_tag_like_structure = bool(re.search(r"<[a-zA-Z][^>]*>", trimmed))

        has_html_structure = bool(self.html_structure_pattern.search(trimmed))

        escaped_newlines = self.count_occurrences(text, "\\n")
        actual_newlines = self.count_occurrences(text, "\n")

        if self.streaming_mode:
            if has_html_features or has_tag_like_structure:
                return True

            return has_html_structure and (escaped_newlines > 0 or actual_newlines > 0)
        else:
            has_multiple_lines = escaped_newlines > 2 or actual_newlines > 2
            return has_html_structure and has_html_features and has_multiple_lines

    def process_streaming_chunk(self, chunk: str) -> StreamingResult:
        if not self.html_detected and self.chunk_count >= self.MAX_BUFFER_CHUNKS:
            output = self.buffer + chunk
            self.buffer = ""
            return StreamingResult(output, False)

        if (
            not self.html_detected
            and len(self.buffer) + len(chunk) > self.detection_threshold
        ):
            output = self.buffer + chunk
            self.buffer = ""
            return StreamingResult(output, False)

        self.chunk_count += 1

        if not self.html_detected:
            self.buffer += chunk
            if self.is_html_content(self.buffer):
                self.html_detected = True
                self.html_prefix_added = True

                html_start_pos = self.find_html_start_position(self.buffer)
                if html_start_pos > 0:
                    before_html = self.buffer[:html_start_pos]
                    html_part = self.buffer[html_start_pos:]
                    output = before_html + "```html\n" + html_part
                else:
                    output = "```html\n" + self.buffer

                self.buffer = ""
                return StreamingResult(output, True)
            else:
                # 未检测到HTML，提取可以安全输出的部分
                self.buffer = (
                    self.buffer.replace("\\n", "\n")
                    .replace('\\"', '"')
                    .replace("\\'", "'")
                )
                if self.buffer.endswith("\\"):
                    output = self.buffer[:-1]
                    self.buffer = "\\"
                    return StreamingResult(output, False)
                else:
                    output = self.buffer
                    self.buffer = ""
                    return StreamingResult(output, False)
        else:
            self.buffer += chunk
            self.buffer = (
                self.buffer.replace("\\n", "\n").replace('\\"', '"').replace("\\'", "'")
            )
            if self.buffer.endswith("\\"):
                output = self.buffer[:-1]
                self.buffer = "\\"
                return StreamingResult(output, False)
            else:
                output = self.buffer
                self.buffer = ""
                return StreamingResult(output, False)

    def finalize_stream(self) -> str:
        if not self.streaming_mode:
            return ""

        result = ""
        if self.buffer:
            if self.html_detected and not self.html_prefix_added:
                result = "```html\n" + self.buffer + "\n```"
            else:
                result = self.buffer
            self.buffer = ""

        if self.html_detected and self.html_prefix_added:
            result += "\n```"

        self.html_detected = False
        self.html_prefix_added = False
        self.chunk_count = 0
        return result


if __name__ == "__main__":
    print("=== Streaming HTML Detection Demo ===")
    print("Simulating AI model streaming HTML content output...")

    detector = HTMLDetector(streaming_mode=True)
    html_stream = [
        "Let me create a webpage for you:\n\n<",
        "html",
        ">",
        "\n<head>",
        "\n  <title>",
        "Example Page",
        "</title>",
        "\n  <meta charset='utf-8'>",
        "\n</head>",
        "\n<body>",
        "\n  <div class='container'>",
        "\n    <h1>",
        "Welcome to my website",
        "</h1>",
        "\n    <p class='intro'>",
        "This is an introductory paragraph.",
        "</p>",
        "\n    <ul>",
        "\n      <li>Feature One</li>",
        "\n      <li>Feature Two</li>",
        "\n    </ul>",
        "\n  </div>",
        "\n</body>",
        "\n</html>",
    ]

    print("🚀 Starting streaming output...")
    print("-" * 50)

    for i, chunk in enumerate(html_stream):
        result = detector.process_streaming_chunk(chunk)

        # If there is output, display it immediately
        if result.output:
            print(result.output, end="", flush=True)

        # If HTML is detected, show a prompt
        if result.html_detected:
            print(
                f"\nBlock {i + 1} detected HTML, markdown code block prefix added",
                flush=True,
            )

        time.sleep(0.15)

    final_output = detector.finalize_stream()
    if final_output:
        print(final_output, end="", flush=True)

    print("\n" + "-" * 50)
    print("Streaming processing complete!")
    print("=== Non-streaming HTML Detection Test ===")
    print("Testing HTML detection results for different types of content...\n")

    non_stream_detector = HTMLDetector(streaming_mode=False)
    test_cases = [
        {
            "name": "Complete HTML Document",
            "content": """<!DOCTYPE html>
                <html>
                <head>
                    <title>Test Page</title>
                </head>
                <body>
                    <h1>Title</h1>
                    <p class="content">This is a paragraph</p>
                </body>
                </html>""",
            "expected": True,
        },
        {
            "name": "Plain Text Content",
            "content": "This is a piece of normal text, without any HTML tags. Here are some special characters: < > &",
            "expected": False,
        },
        {
            "name": "Fragment with HTML-like Features",
            "content": 'This is not complete HTML, but has a <div class="test"> tag and attribute',
            "expected": False,
        },
        {
            "name": "HTML Inside Code Block",
            "content": "```html\n<div>This HTML is inside a code block</div>\n```",
            "expected": False,
        },
        {
            "name": "HTML Fragment with Multiple Line Breaks",
            "content": """<div>
                    <p>First line</p>
                    <p>Second line</p>
                    <p>Third line</p>
                </div>""",
            "expected": True,
        },
        {
            "name": "HTML with Only Opening Tags",
            "content": "<html><head><body><div>",
            "expected": False,
        },
    ]

    for i, test in enumerate(test_cases, 1):
        result = non_stream_detector.is_html_content(test["content"])
        status = "✓" if result == test["expected"] else "✗"
        print(f"Test {i}: {test['name']}")
        print(f"  Expected: {test['expected']}")
        print(f"  Actual: {result}")
        print(f"  Test Result: {status}\n")

    print("Non-streaming tests complete!")
