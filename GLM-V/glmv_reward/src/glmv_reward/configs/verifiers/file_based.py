# -*- coding: utf-8 -*-


from typing import Optional

import msgspec


class FileBasedVerifierConfig(msgspec.Struct, frozen=True, tag_field="verifier_type", tag="file_based"):
    extract_answer_file_path: str
    extract_answer_func_name: str
    judge_func_path: str
    judge_func_name: str
    load_once: bool = True
