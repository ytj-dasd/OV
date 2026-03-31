# tests/verifiers/test_logic_verifier.py
import pytest


def test_logic_verifier_judge_negations(logic_verifier):
    # Test negations and opposite statements
    question = "描述图片中的情况"
    assert logic_verifier.judge("不是红色", "颜色不是红色", question) == 1.0
    assert logic_verifier.judge("没有苹果", "不存在苹果", question) == 1.0
    assert logic_verifier.judge("不是圆形", "不是圆形的", question) == 1.0
    assert logic_verifier.judge("没有学生", "不存在学生", question) == 1.0


def test_logic_verifier_judge_conditions(logic_verifier):
    # Test conditional statements
    question = "描述图片中的条件关系"
    assert logic_verifier.judge("如果是红色，那么是圆形", "如果是圆形，那么是红色", question) == 0.0
    assert logic_verifier.judge("如果下雨，那么带伞", "如果带伞，那么下雨", question) == 0.0
    assert logic_verifier.judge("如果x>0，那么x²>0", "如果x²>0，那么x>0", question) == 0.0
    assert logic_verifier.judge("如果a=b，那么a²=b²", "如果a²=b²，那么a=b", question) == 0.0


def test_logic_verifier_judge_set_operations(logic_verifier):
    # Test set operations and relationships
    question = "描述图片中的集合关系"
    assert logic_verifier.judge("A是B的子集", "B包含A", question) == 1.0
    assert logic_verifier.judge("A和B的交集为空", "A和B不相交", question) == 1.0
    assert logic_verifier.judge("A是B的真子集", "B包含A且A不等于B", question) == 1.0
