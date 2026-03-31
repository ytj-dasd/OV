# # tests/verifiers/test_general_verifier.py
# import pytest
# from glmv_reward.verifiers import  GeneralVerifier

# @pytest.mark.skip(reason="跳过所有测试")

# def test_general_verifier_judge_multiple_choice(general_verifier):
#     # Test multiple choice questions with letter options
#     question = "What is the color of the sky?\n(A) Blue\n(B) Red\n(C) Green\n(D) Yellow"
#     assert general_verifier.judge("A", "A", question=question) == 1.0
#     assert general_verifier.judge("B", "B", question=question) == 1.0
#     assert general_verifier.judge("C", "C", question=question) == 1.0
#     assert general_verifier.judge("D", "D", question=question) == 1.0

#     # Test case insensitivity for letter options
#     assert general_verifier.judge("a", "A", question=question) == 1.0
#     assert general_verifier.judge("b", "B", question=question) == 1.0

#     # Test multiple choice with number options
#     question_num = "What is 2+2?\n(1) 3\n(2) 4\n(3) 5\n(4) 6"
#     assert general_verifier.judge("2", "2", question=question_num) == 1.0
#     assert general_verifier.judge("4", "4", question=question_num) == 0.0  # Wrong answer

#     # Test multiple choice with spaces and formatting
#     assert general_verifier.judge(" A ", "A", question=question) == 1.0
#     assert general_verifier.judge("(A)", "A", question=question) == 1.0
#     assert general_verifier.judge("A.", "A", question=question) == 1.0

# def test_general_verifier_judge_true_false(general_verifier):
#     # Test true/false questions
#     question = "Is the sky blue?"
#     assert general_verifier.judge("True", "True", question=question) == 1.0
#     assert general_verifier.judge("False", "False", question=question) == 1.0
#     assert general_verifier.judge("true", "True", question=question) == 1.0  # Case insensitive
#     assert general_verifier.judge("T", "True", question=question) == 1.0  # Abbreviated
#     assert general_verifier.judge("F", "False", question=question) == 1.0  # Abbreviated

# def test_general_verifier_judge_short_answer(general_verifier):
#     # Test short answer questions
#     question = "What is the capital of France?"
#     assert general_verifier.judge("Paris", "Paris", question=question) == 1.0
#     assert general_verifier.judge("paris", "Paris", question=question) == 1.0  # Case insensitive
#     assert general_verifier.judge("Paris, France", "Paris", question=question) == 0.0  # Extra info
#     assert general_verifier.judge("London", "Paris", question=question) == 0.0  # Wrong answer

# def test_general_verifier_judge_essay(general_verifier):
#     # Test essay questions
#     question = "Explain the process of photosynthesis."
#     answer = "Photosynthesis is the process by which plants convert light energy into chemical energy."
#     assert general_verifier.judge(answer, answer, question=question) == 1.0
#     assert general_verifier.judge(answer.lower(), answer, question=question) == 1.0  # Case insensitive
#     assert general_verifier.judge("Different answer", answer, question=question) == 0.0  # Wrong answer

# def test_general_verifier_judge_matching(general_verifier):
#     # Test matching questions
#     question = "Match the countries with their capitals:\nA) France\nB) Germany\nC) Italy"
#     answer = "A-Paris, B-Berlin, C-Rome"
#     assert general_verifier.judge(answer, answer, question=question) == 1.0
#     assert general_verifier.judge("A:Paris B:Berlin C:Rome", answer, question=question) == 1.0  # Different format
#     assert general_verifier.judge("A-Berlin, B-Paris, C-Rome", answer, question=question) == 0.0  # Wrong matches

# def test_general_verifier_judge_fill_in_blank(general_verifier):
#     # Test fill in the blank questions
#     question = "The capital of France is _____."
#     assert general_verifier.judge("Paris", "Paris", question=question) == 1.0
#     assert general_verifier.judge("paris", "Paris", question=question) == 1.0  # Case insensitive
#     assert general_verifier.judge("Paris, France", "Paris", question=question) == 1.0  # Extra info
#     assert general_verifier.judge("London", "Paris", question=question) == 0.0  # Wrong answer


# def test_general_verifier_judge_chinese_literature(general_verifier):
#     # Test Chinese literature questions
#     question = "《红楼梦》的作者是谁？"
#     assert general_verifier.judge("高鹗和曹雪芹", "曹雪芹和高鹗", question=question) == 1.0
#     assert general_verifier.judge("曹雪芹、高鹗", "曹雪芹和高鹗", question=question) == 1.0  # Extra info
#     assert general_verifier.judge("施耐庵", "曹雪芹和高鹗", question=question) == 0.0  # Wrong answer

#     question = "《静夜思》的作者是谁？"
#     assert general_verifier.judge("李白", "李白", question=question) == 1.0
#     assert general_verifier.judge("李太白", "李白", question=question) == 1.0  # Alternative name
#     assert general_verifier.judge("杜甫", "李白", question=question) == 0.0  # Wrong answer

# def test_general_verifier_judge_poetry(general_verifier):
#     # Test poetry questions
#     question = "请默写《静夜思》的前两句"
#     answer = "床前明月光，疑是地上霜"
#     assert general_verifier.judge(answer, answer, question=question) == 1.0
#     assert general_verifier.judge("床前明月光，疑是地上霜。", answer, question=question) == 1.0  # Extra punctuation
#     assert general_verifier.judge("窗前明月光，不是地上霜", answer, question=question) == 0.0  # Wrong character

# def test_general_verifier_judge_politics(general_verifier):
#     # Test politics questions
#     question = "中国共产党的最高理想和最终目标是什么？"
#     answer = "实现共产主义"
#     assert general_verifier.judge(answer, answer, question=question) == 1.0
#     assert general_verifier.judge("实现共产主义社会", answer, question=question) == 1.0  # Similar meaning
#     assert general_verifier.judge("实现共同富裕", answer, question=question) == 0.0  # Wrong answer

# def test_general_verifier_judge_economics(general_verifier):
#     # Test economics questions
#     question = "什么是通货膨胀？"
#     answer = "货币供应量增加导致物价普遍上涨的现象"
#     assert general_verifier.judge(answer, answer, question=question) == 1.0
#     assert general_verifier.judge("物价普遍上涨", answer, question=question) == 0.0  # Simplified answer
#     assert general_verifier.judge("货币贬值", answer, question=question) == 0.0  # Incomplete answer

# def test_general_verifier_judge_social_science(general_verifier):
#     # Test social science questions
#     question = "什么是社会契约论？"
#     answer = "人民通过契约建立政府，让渡部分权利以换取保护"
#     assert general_verifier.judge(answer, answer, question=question) == 1.0
#     assert general_verifier.judge("人民通过契约让渡部分权利给政府以获得保护", answer, question=question) == 1.0  # Equivalent expression
#     assert general_verifier.judge("社会契约论是卢梭提出的理论", answer, question=question) == 0.0  # Wrong focus

# def test_general_verifier_judge_geography(general_verifier):
#     # Test geography questions
#     question = "长江发源于哪个省份？"
#     assert general_verifier.judge("青海省附近", "青海省", question=question) == 1.0
#     assert general_verifier.judge("甘肃青海地带", "青海省", question=question) == 0.0  # Simplified answer
#     assert general_verifier.judge("西藏", "青海省", question=question) == 0.0  # Wrong answer

#     question = "中国最大的淡水湖是什么？"
#     assert general_verifier.judge("鄱阳湖", "鄱阳湖", question=question) == 1.0
#     assert general_verifier.judge("江西鄱阳湖", "鄱阳湖", question=question) == 1.0  # Extra info
#     assert general_verifier.judge("洞庭湖", "鄱阳湖", question=question) == 0.0  # Wrong answer


# def test_general_verifier_judge_common_sense(general_verifier):
#     # Test common sense questions
#     question = "水在什么温度下会结冰？"
#     assert general_verifier.judge("0摄氏度", "0℃", question=question) == 1.0
#     assert general_verifier.judge("零度", "0℃", question=question) == 1.0
#     assert general_verifier.judge("1度", "0℃", question=question) == 0.0

#     question = "一天有多少小时？"
#     assert general_verifier.judge("24小时", "24小时", question=question) == 1.0
#     assert general_verifier.judge("24h", "24小时", question=question) == 1.0
#     assert general_verifier.judge("1440 minutes", "24小时", question=question) == 0.0
#     assert general_verifier.judge("24.0", "24小时", question=question) == 1.0

# def test_general_verifier_judge_life_skills(general_verifier):
#     # Test life skills questions
#     question = "如何正确洗手？"
#     answer = "用肥皂和流动水洗手至少20秒"
#     assert general_verifier.judge(answer, answer, question=question) == 1.0
#     assert general_verifier.judge("用肥皂洗手10秒", answer, question=question) == 0.0
#     assert general_verifier.judge("用水冲一下", answer, question=question) == 0.0

#     question = "如何正确刷牙？"
#     answer = "每天早晚各刷一次，每次至少2分钟"
#     assert general_verifier.judge(answer, answer, question=question) == 1.0
#     assert general_verifier.judge("每天刷两次牙", answer, question=question) == 0.0
#     assert general_verifier.judge("随便刷一下", answer, question=question) == 0.0

# def test_general_verifier_judge_science(general_verifier):
#     # Test science questions
#     question = "光合作用的基本过程是什么？"
#     answer = "植物利用光能将二氧化碳和水转化为有机物和氧气"
#     assert general_verifier.judge(answer, answer, question=question) == 1.0
#     assert general_verifier.judge("植物利用阳光制造食物", answer, question=question) == 0.0
#     assert general_verifier.judge("植物吸收水分", answer, question=question) == 0.0

#     question = "地球的自转方向是什么？"
#     assert general_verifier.judge("自西向东", "自西向东", question=question) == 1.0
#     assert general_verifier.judge("从西向东", "自西向东", question=question) == 1.0
#     assert general_verifier.judge("自东向西", "自西向东", question=question) == 0.0

# def test_general_verifier_judge_english_grammar(general_verifier):
#     # Test English grammar questions
#     question = "What is the past tense of 'go'?"
#     assert general_verifier.judge("went", "went", question=question) == 1.0
#     assert general_verifier.judge("WENT", "went", question=question) == 1.0
#     assert general_verifier.judge("goed", "went", question=question) == 0.0

#     question = "Which is correct: 'He is taller than I' or 'He is taller than me'?"
#     assert general_verifier.judge("He is taller than I", "He is taller than I", question=question) == 1.0
#     assert general_verifier.judge("He is taller than I am", "He is taller than I", question=question) == 1.0
#     assert general_verifier.judge("He is taller than me", "He is taller than I", question=question) == 0.0

# def test_general_verifier_judge_subjects(general_verifier):
#     # Test various subject questions
#     question = "什么是质数？"
#     answer = "只能被1和自身整除的大于1的自然数"
#     assert general_verifier.judge(answer, answer, question=question) == 1.0
#     assert general_verifier.judge("只能被1和自身整除的数", answer, question=question) == 1.0
#     assert general_verifier.judge("能被2整除的数", answer, question=question) == 0.0

#     question = "什么是化学反应？"
#     answer = "物质之间发生化学变化，产生新物质的过程"
#     assert general_verifier.judge(answer, answer, question=question) == 1.0
#     assert general_verifier.judge("产生新物质的反应", answer, question=question) == 1.0
#     assert general_verifier.judge("物质混合", answer, question=question) == 0.0

#     question = "什么是光合作用？"
#     answer = "植物利用光能将二氧化碳和水转化为有机物和氧气的过程"
#     assert general_verifier.judge(answer, answer, question=question) == 1.0
#     assert general_verifier.judge("植物制造食物的过程", answer, question=question) == 1.0
#     assert general_verifier.judge("植物呼吸", answer, question=question) == 0.0

# def test_general_verifier_judge_history(general_verifier):
#     # Test history questions
#     question = "秦始皇统一六国是在哪一年？"
#     assert general_verifier.judge("公元前221年", "公元前221年", question=question) == 1.0
#     assert general_verifier.judge("公元前222年，秦始皇统一六国", "公元前221年", question=question) == 0.0
#     assert general_verifier.judge("公元前220年", "公元前221年", question=question) == 0.0

#     question = "第一次世界大战爆发于哪一年？"
#     assert general_verifier.judge("1914年", "1914年", question=question) == 1.0
#     assert general_verifier.judge("1914", "1914年", question=question) == 1.0
#     assert general_verifier.judge("1915年", "1914年", question=question) == 0.0

# def test_general_verifier_judge_art(general_verifier):
#     # Test art questions
#     question = "《蒙娜丽莎》的作者是谁？"
#     assert general_verifier.judge("达芬奇", "达芬奇", question=question) == 1.0
#     assert general_verifier.judge("列奥纳多·达·芬奇", "达芬奇", question=question) == 1.0
#     assert general_verifier.judge("毕加索", "达芬奇", question=question) == 0.0

#     question = "《清明上河图》描绘的是哪个朝代？"
#     assert general_verifier.judge("北宋", "北宋", question=question) == 1.0
#     assert general_verifier.judge("南宋", "北宋", question=question) == 0.0
#     assert general_verifier.judge("唐朝", "北宋", question=question) == 0.0

# def test_general_verifier_judge_sports(general_verifier):
#     # Test sports questions
#     question = "足球比赛中，一个队有多少名球员？"
#     assert general_verifier.judge("11名", "11名", question=question) == 1.0
#     assert general_verifier.judge("11", "11名", question=question) == 1.0
#     assert general_verifier.judge("10名", "11名", question=question) == 0.0

#     question = "奥运会每几年举办一次？"
#     assert general_verifier.judge("4年", "4年", question=question) == 1.0
#     assert general_verifier.judge("四年", "4年", question=question) == 1.0
#     assert general_verifier.judge("2年", "4年", question=question) == 0.0

# def test_general_verifier_judge_technology(general_verifier):
#     # Test technology questions
#     question = "什么是人工智能？"
#     answer = "让计算机模拟人类智能的技术"
#     assert general_verifier.judge(answer, answer, question=question) == 1.0
#     assert general_verifier.judge("为了让计算机模拟人类智能的技术", answer, question=question) == 1.0
#     assert general_verifier.judge("机器人技术", answer, question=question) == 0.0

#     question = "什么是区块链？"
#     answer = "分布式账本技术，具有去中心化、不可篡改的特点"
#     assert general_verifier.judge(answer, answer, question=question) == 1.0
#     assert general_verifier.judge("一种铁链条，跟区块相关", answer, question=question) == 0.0

# def test_general_verifier_judge_family_relations(general_verifier):
#     # Test family relations questions
#     question = "妈妈的妈妈叫什么？"
#     assert general_verifier.judge("外婆", "外婆", question=question) == 1.0
#     assert general_verifier.judge("姥姥", "外婆", question=question) == 1.0
#     assert general_verifier.judge("奶奶", "外婆", question=question) == 0.0

#     question = "爸爸的哥哥叫什么？"
#     assert general_verifier.judge("大伯", "大伯", question=question) == 1.0
#     assert general_verifier.judge("伯父", "大伯", question=question) == 1.0
#     assert general_verifier.judge("叔叔", "大伯", question=question) == 0.0

# def test_general_verifier_judge_age(general_verifier):
#     # Test age-related questions
#     question = "法定成年年龄是多少岁？"
#     assert general_verifier.judge("18岁", "18岁", question=question) == 1.0
#     assert general_verifier.judge("18", "18岁", question=question) == 1.0
#     assert general_verifier.judge("20岁", "18岁", question=question) == 0.0

#     question = "退休年龄是多少岁？"
#     assert general_verifier.judge("60岁", "60岁", question=question) == 1.0
#     assert general_verifier.judge("60", "60岁", question=question) == 1.0
#     assert general_verifier.judge("65岁", "60岁", question=question) == 0.0

# def test_general_verifier_judge_marriage(general_verifier):
#     # Test marriage-related questions
#     question = "法定结婚年龄是多少岁？"
#     assert general_verifier.judge("22岁", "22岁", question=question) == 1.0
#     assert general_verifier.judge("22", "22岁", question=question) == 1.0
#     assert general_verifier.judge("20岁", "22岁", question=question) == 0.0

#     question = "结婚纪念日叫什么？"
#     assert general_verifier.judge("结婚纪念日", "结婚纪念日", question=question) == 1.0
#     assert general_verifier.judge("结婚日", "结婚纪念日", question=question) == 1.0
#     assert general_verifier.judge("婚礼日", "结婚纪念日", question=question) == 0.0
