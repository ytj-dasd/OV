# -*- coding: utf-8 -*-


import msgspec


class AndroidworldVerifierConfig(msgspec.Struct, frozen=True, tag_field="verifier_type", tag="AndroidWorld"):
    extract_answer_file_path: str
    extract_answer_func_name: str
    judge_func_path: str
    judge_func_name: str
    load_once: bool = True
