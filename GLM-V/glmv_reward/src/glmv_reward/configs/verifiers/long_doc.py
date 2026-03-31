# -*- coding: utf-8 -*-


from typing import List, Optional

import msgspec


class LongDocVerifierConfig(msgspec.Struct, frozen=True, tag_field="verifier_type", tag="long_doc"):
    llm_api_key: List[str]
    llm_judge_url: List[str]
    llm_model: List[str]
    answer_extraction_regex: Optional[str] = None
    llm_max_tokens: int = 10
    llm_temperature: float = 0.1
    llm_top_p: float = 1.0
