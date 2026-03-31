def test_geoquest_extract_answer(geoquest_verifier):
    question = "这张照片是哪里拍的？"
    response = "<think>123</think>从图片中的场景分析，首先可见大型恐龙骨架与非洲象标本同处一个空间，这类展品组合常见于自然历史博物馆。建筑风格具有显著特征：高大的拱形天花板带有规则排列的天窗，提供充足自然光；墙壁装饰有圆形浮雕和古典柱式，二层回廊结构搭配拱形门洞，整体呈现古典复兴式建筑风格，符合大型博物馆的庄重感与历史感。展品布局上，恐龙骨架（蜥脚类）占据中央空间，大象标本立于基座之上，周围有少量参观者，显示出公共教育场所的开放性。结合自然历史博物馆典型的展品类型（古生物化石与动物标本）和建筑设计特点，可推断此处为专注于自然科学与历史展示的博物馆。\n\n结论：这里可能是一座<|begin_of_box|>自然历史博物馆<|end_of_box|>，其古典建筑风格与大型生物标本的陈列方式，符合此类博物馆用于展示地球生命演化与自然遗产的功能定位。"
    gt_answer = "自然历史博物馆"
    extracted_answer = geoquest_verifier.extract_answer(response, question)
    assert extracted_answer == gt_answer


def test_geoquest_multiple_guesses(geoquest_verifier):
    """测试多次猜测的情况"""
    question = "图片：拍摄人在哪里？"
    gt_answer = '{"place_name": "苍岩山", "address": "中国石家庄市井陉县苍岩山 邮政编码: 050304"}'
    response = "这可能是中国华北地区的某个山区，具体来说可能是河北省、河南省或山西省的某个风景名胜区，从建筑风格看应该是佛教寺庙建筑群"
    assert geoquest_verifier.judge(response, gt_answer, question) == 0.0
