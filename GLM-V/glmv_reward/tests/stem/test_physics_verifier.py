import re

import pytest

from glmv_reward.verifiers import PhysicsVerifier


def test_physics_verifier_judge_unit_conversions(physics_verifier):
    # Test unit conversions
    assert physics_verifier.judge("1 km", "1000 m") == 1.0
    assert physics_verifier.judge("1 hour", "3600 seconds") == 1.0
    assert physics_verifier.judge("1 eV", "1.60218 × 10^-19 J") == 1.0
    # assert physics_verifier.judge("10 m/s", "36 km/h") == 1.0


def test_physics_verifier_judge_equations(physics_verifier):
    # Test physics equations
    assert physics_verifier.judge("F = ma", "F = ma") == 1.0
    assert physics_verifier.judge("E = mc²", "E = mc^2") == 1.0
    assert physics_verifier.judge("v = d/t", "v = d/t") == 1.0
    assert physics_verifier.judge("P = F/A", "P = F/A") == 1.0
    assert physics_verifier.judge("W = Fd", "W = Fd") == 1.0
    assert physics_verifier.judge("v = \\frac{s}{t}", "v = \\dfrac{s}{t}") == 1.0


def test_physics_verifier_judge_vectors(physics_verifier):
    # Test vector notation
    assert physics_verifier.judge("F⃗ = ma⃗", "F = ma") == 1.0
    # assert physics_verifier.judge("v⃗ = d⃗/t", "v = d/t") == 1.0
    assert physics_verifier.judge("F⃗ = -kx⃗", "F = -kx") == 1.0


def test_physics_verifier_judge_uncertainty(physics_verifier):
    # Test measurements with uncertainty
    assert physics_verifier.judge("(9.81 ± 0.01) m/s²", "9.81 ± 0.01 m/s²") == 1.0
    assert physics_verifier.judge("(3.00 ± 0.01) × 10^8 m/s", "3.00e8 ± 0.01e8 m/s") == 1.0


def test_physics_verifier_judge_kinematics(physics_verifier):
    # Test kinematic equations
    assert physics_verifier.judge("v = v₀ + at", "v = v0 + at") == 1.0
    assert physics_verifier.judge("d = v₀t + ½at²", "d = v0t + 0.5at^2") == 1.0
    assert physics_verifier.judge("v² = v₀² + 2ad", "v^2 = v0^2 + 2ad") == 1.0


def test_physics_verifier_judge_energy(physics_verifier):
    # Test energy equations
    assert physics_verifier.judge("KE = ½mv²", "KE = 0.5mv^2") == 1.0
    assert physics_verifier.judge("PE = mgh", "PE = mgh") == 1.0
    assert physics_verifier.judge("E = KE + PE", "E = KE + PE") == 1.0


def test_physics_verifier_judge_waves(physics_verifier):
    # Test wave equations
    assert physics_verifier.judge("v = fλ", "v = fλ") == 1.0
    assert physics_verifier.judge("f = 1/T", "f = 1/T") == 1.0
    assert physics_verifier.judge("E = hf", "E = hf") == 1.0


def test_physics_verifier_judge_circuits(physics_verifier):
    # Test circuit equations
    assert physics_verifier.judge("V = IR", "V = IR") == 1.0
    assert physics_verifier.judge("P = IV", "P = IV") == 1.0
    assert physics_verifier.judge("R = ρL/A", "R = ρL/A") == 1.0


def test_physics_verifier_judge_thermodynamics(physics_verifier):
    # Test thermodynamics equations
    assert physics_verifier.judge("PV = nRT", "PV = nRT") == 1.0
    assert physics_verifier.judge("Q = mcΔT", "Q = mcΔT") == 1.0
    assert physics_verifier.judge("W = PΔV", "W = PΔV") == 1.0


def test_physics_verifier_judge_quantum(physics_verifier):
    # Test quantum mechanics equations
    assert physics_verifier.judge("E = hf", "E = hf") == 1.0
    assert physics_verifier.judge("λ = h/p", "λ = h/p") == 1.0
    assert physics_verifier.judge("ΔxΔp ≥ ℏ/2", "ΔxΔp ≥ ℏ/2") == 1.0


def test_physics_verifier_judge_relativity(physics_verifier):
    # Test relativity equations
    assert physics_verifier.judge("E = mc²", "E = mc^2") == 1.0
    assert physics_verifier.judge("γ = 1/√(1-v²/c²)", "γ = 1/sqrt(1-v^2/c^2)") == 1.0
    assert physics_verifier.judge("t' = γ(t-vx/c²)", "t' = γ(t-vx/c^2)") == 1.0


def test_physics_verifier_judge_physical_equation_equivalence(physics_verifier):
    # 代数上等价的公式
    assert physics_verifier.judge("F=ma", "a=F/m") == 1.0
    # assert physics_verifier.judge("v=s/t", "s=vt") == 1.0

    # 非等价公式
    assert physics_verifier.judge("F=ma", "F=mg") == 0.0
    assert physics_verifier.judge("v=s/t", "v=s*t") == 0.0

    # 易错补充
    assert physics_verifier.judge("P=IV", "I=P/V") == 1.0
    assert physics_verifier.judge("Q=mcΔT", "ΔT=Q/(mc)") == 1.0
    assert physics_verifier.judge("P=IV", "P=I/V") == 0.0
    assert physics_verifier.judge("Q=mcΔT", "Q=mv^2") == 0.0


def test_physics_verifier_judge_physical_extra_info_with_question(physics_verifier):
    question = "电磁炉的工作原理是什么？"
    assert (
        physics_verifier.judge(
            "电磁炉通过电磁感应原理加热金属锅底，同时锅内的水因受热而沸腾",
            "电磁炉利用电磁感应加热金属锅底",
            question=question,
        )
        == 1.0
    )

    # 核心信息匹配，额外信息不影响
    question = "牛顿第一定律的内容是什么？"
    assert (
        physics_verifier.judge(
            "物体在不受外力或合外力为零时保持静止或匀速直线运动状态，这个定律也被称为惯性定律，是伽利略研究的基础",
            "物体在合外力为零时保持静止或匀速直线运动",
            question=question,
        )
        == 1.0
    )

    # 光的干涉现象，包含额外背景信息
    question = "什么是光的干涉现象？"
    assert (
        physics_verifier.judge(
            "光的干涉是指两束或多束相干光波相遇时产生明暗相间的条纹，这种现象最早由托马斯·杨在双缝实验中观察到",
            "光的干涉是相干光波相遇产生明暗条纹的现象",
            question=question,
        )
        == 1.0
    )

    # 电阻定律，含有计算示例
    question = "欧姆定律的表达式是什么？"
    assert (
        physics_verifier.judge(
            "欧姆定律表示为U=IR，其中U是电压，I是电流，R是电阻，例如当电阻为10Ω，电流为2A时，电压为20V",
            "欧姆定律的表达式是U=IR",
            question=question,
        )
        == 1.0
    )

    # 自由落体运动，包含历史背景
    question = "自由落体运动的特点是什么？"
    assert (
        physics_verifier.judge(
            "自由落体运动是物体只在重力作用下从静止开始的运动，加速度恒定为g≈9.8m/s²，伽利略通过比萨斜塔实验证明了这一点",
            "自由落体运动是初速度为零，加速度为g的匀加速运动",
            question=question,
        )
        == 1.0
    )

    # 热力学第一定律，含有应用举例
    question = "热力学第一定律的内容是什么？"
    assert (
        physics_verifier.judge(
            "热力学第一定律表明能量守恒，即ΔU=Q-W，其中ΔU是内能变化，Q是吸收的热量，W是对外做功，这个定律在蒸汽机和内燃机中都有重要应用",
            "热力学第一定律是能量守恒定律，表达式为ΔU=Q-W",
            question=question,
        )
        == 1.0
    )


def test_physics_verifier_judge_unit_variations(physics_verifier):
    # Test different unit representations
    assert physics_verifier.judge("5 N", "5 newtons") == 1.0
    assert physics_verifier.judge("10 kPa", "10 kilopascals") == 1.0
    assert physics_verifier.judge("2.5 μF", "2.5 microfarads") == 1.0
    assert physics_verifier.judge("15°C", "15 degrees Celsius") == 1.0


def test_physics_verifier_judge_symbol_variations(physics_verifier):
    # Test different symbol usage
    assert physics_verifier.judge("F = ma", "a = F/m") == 1.0
    assert physics_verifier.judge("v = u + at", "v = at + u") == 1.0
    assert physics_verifier.judge("E = mc²", "E = m c^2") == 1.0


def test_physics_verifier_judge_numeric_equivalence(physics_verifier):
    # Test numeric equivalence
    assert physics_verifier.judge("1/2", "0.5") == 1.0
    assert physics_verifier.judge("根号2", "sqrt(2)") == 1.0
    assert physics_verifier.judge("π", "3.14159265359") == 1.0


def test_physics_verifier_judge_algebraic_equivalence(physics_verifier):
    # 乘法符号省略
    assert physics_verifier.judge("F = k*(q1*q2)/r²", "F = kq1q2/r²") == 1.0

    # 因式分解等效
    assert physics_verifier.judge("E = ½mv² + mgh", "E = m(½v² + gh)") == 1.0

    # 合并同类项
    assert physics_verifier.judge("3R + 2R - 5T", "5R - 5T") == 1.0

    # 三角恒等式
    assert physics_verifier.judge("sinθ/cosθ", "tanθ") == 1.0

    # 展开多项式
    assert physics_verifier.judge("(a+b)²", "a² + 2ab + b²") == 1.0

    # 分数简化
    assert physics_verifier.judge("(2π/4π)", "0.5") == 1.0

    # 不等效案例（应返回0.0）
    # assert physics_verifier.judge("F = ma", "F = mv²/r") == 0.0 needs question

    # 矢量表达式等效
    assert physics_verifier.judge("A·B + A·C", "A·(B + C)") == 1.0

    # 指数等效性
    assert physics_verifier.judge("e^(x+y)", "e^x * e^y") == 1.0

    # 对数性质
    assert physics_verifier.judge("ln(a) + ln(b)", "ln(ab)") == 1.0


def test_physics_verifier_judge_physical_direction_equivalence(physics_verifier):
    # 方向表述不同，但物理结果等效
    question = "如图所示，两个相同电荷量的小球分别放置在距点P 0.2m 处，求点P的电场强度。"
    assert (
        physics_verifier.judge(
            "两个带正电的小球分别放在点P的左侧和右侧0.2m处",
            "两个小球放在点P右侧和左侧0.2m处，均带正电",
            question=question,
        )
        == 1.0
    )

    # 新增测试样例1：力的平衡条件，位置描述等效
    question = "一物体在三个力作用下保持平衡，其中两个力的大小和方向已知。"
    assert (
        physics_verifier.judge(
            "第一个力向上10N，第二个力向右5N", "一个力竖直向上10N，另一个力水平向右5N", question=question
        )
        == 1.0
    )

    # 新增测试样例2：对称电场中的等效描述
    question = "两个等量异种点电荷分别位于x轴上的±a位置，求原点处的电场。"
    assert (
        physics_verifier.judge("正电荷在右侧+a处，负电荷在左侧-a处", "负电荷位于-a，正电荷位于+a", question=question)
        == 1.0
    )

    # 新增测试样例3：圆周运动中的等效方向描述
    question = "粒子在匀强磁场中做圆周运动，磁场垂直纸面，求粒子运动方向。"
    assert physics_verifier.judge("粒子顺时针运动", "粒子沿顺时针方向做圆周运动", question=question) == 1.0

    # 新增测试样例4：光路的等效描述
    question = "平行光经过凸透镜后的光路如何？"
    assert physics_verifier.judge("平行光会聚于焦点", "光线在焦点处汇聚", question=question) == 1.0

    # 新增测试样例5：弹性碰撞中的等效方向表述
    question = "两球发生弹性正碰，碰撞前后的运动状态如何？"
    assert (
        physics_verifier.judge("球A向右运动，球B向左运动", "球B朝左方向运动，球A朝右方向运动", question=question) == 1.0
    )

    # 易错补充
    question = "物体受重力方向？"
    assert physics_verifier.judge("竖直向下", "沿地心方向", question=question) == 1.0
    assert physics_verifier.judge("竖直向下", "竖直向上", question=question) == 0.0
    question = "电流方向？"
    assert physics_verifier.judge("从正极流向负极", "由高电位流向低电位", question=question) == 1.0
    assert physics_verifier.judge("从正极流向负极", "从负极流向正极", question=question) == 0.0


def test_physics_verifier_judge_physical_direction_difference_invalid(physics_verifier):
    # 方向关键，错误描述应判0分
    question = "一辆小车受到两个相反方向的力作用，哪个方向的合力更大？"
    assert (
        physics_verifier.judge(
            "左侧的合力为5N，右侧为3N，所以合力向右", "左侧合力为5N，右侧为3N，因此合力向左", question=question
        )
        == 0.0
    )

    # 题干明确指出磁场方向，错误判断方向应为0
    question = "在垂直向上的磁场中，一电子以垂直磁场方向射入，洛伦兹力方向如何？"
    assert physics_verifier.judge("洛伦兹力方向向左", "洛伦兹力方向向右", question=question) == 0.0

    assert (
        physics_verifier.judge(
            "相互作用力方向相反、大小相等",
            "作用力与反作用力大小相等、方向相反",
        )
        == 1.0
    )

    # 错误理解，方向反了，判0
    question = "一导体棒在竖直方向上匀速下落，处于水平磁场中，感应电流方向如何？"
    assert physics_verifier.judge("电流从左端流向右端", "电流从右端流向左端", question=question) == 0.0

    # 重力方向错误判断
    question = "在地球表面，物体受到的重力方向如何？"
    assert physics_verifier.judge("重力方向竖直向下", "重力方向竖直向上", question=question) == 0.0

    # 光的折射方向错误
    question = "光从空气射入水中，入射角30°，折射角约22°，折射光线相对法线的位置？"
    assert physics_verifier.judge("折射光线向法线偏折", "折射光线远离法线偏折", question=question) == 0.0


def test_physics_verifier_judge_physical_direction_with_extra_context(physics_verifier):
    # 方向不同但满足镜像对称条件，判等效
    question = "在两相反带电的平行板中间放置一个中性小球，球从距离正极0.05m处释放。"
    assert (
        physics_verifier.judge(
            "球向负极方向运动，加速度为2m/s²", "球沿与正极相反方向加速，大小为2m/s²", question=question
        )
        == 1.0
    )

    # 力的方向表述等效
    question = "质量为2kg的物体受到水平向右的10N拉力，求加速度方向。"
    assert physics_verifier.judge("加速度方向水平向右", "加速度沿正x轴方向", question=question) == 1.0

    # 磁场方向表述等效
    question = "载流导线在磁场中受力，已知电流向上，磁场垂直纸面向外。"
    assert physics_verifier.judge("导线受力方向向右", "导线受安培力，方向指向右侧", question=question) == 1.0

    # 圆周运动向心力方向等效表述
    question = "物体做匀速圆周运动时，向心力的方向特点？"
    assert physics_verifier.judge("向心力方向指向圆心", "向心力沿半径方向向内", question=question) == 1.0


def test_physics_verifier_judge_physical_ordered_list_should_match(physics_verifier):
    # 顺序应保留的情况（有序）
    question = "自由落体运动的过程依次经历哪些阶段？"
    assert physics_verifier.judge("静止→加速下落→到达地面", "加速下落→静止→到达地面", question=question) == 0.0

    # 电磁感应过程顺序错误
    question = "通过改变磁通量产生感应电流的过程是？"
    assert (
        physics_verifier.judge(
            "磁通量变化→产生感应电动势→形成感应电流", "产生感应电动势→磁通量变化→形成感应电流", question=question
        )
        == 0.0
    )

    # 光电效应过程顺序错误
    question = "光电效应的基本过程依次是？"
    assert (
        physics_verifier.judge("光子入射→电子吸收能量→电子逸出", "电子逸出→光子入射→电子吸收能量", question=question)
        == 0.0
    )

    # 简谐振动一个周期的顺序错误
    question = "弹簧振子从平衡位置开始一个完整周期的运动顺序？"
    assert (
        physics_verifier.judge(
            "平衡位置→最大位移→平衡位置→反向最大位移→平衡位置",
            "最大位移→平衡位置→反向最大位移→平衡位置→平衡位置",
            question=question,
        )
        == 0.0
    )

    # 机械波传播过程顺序错误
    question = "机械波的产生和传播过程依次是？"
    assert (
        physics_verifier.judge("振源振动→介质质点振动→波动传播", "波动传播→振源振动→介质质点振动", question=question)
        == 0.0
    )

    # α衰变过程顺序错误
    question = "原子核α衰变的过程步骤是？"
    assert (
        physics_verifier.judge("不稳定核→发射α粒子→形成新核", "形成新核→不稳定核→发射α粒子", question=question) == 0.0
    )

    # 易错补充
    question = "实验步骤？"
    assert physics_verifier.judge("准备→操作→记录→分析", "准备 操作 记录 分析", question=question) == 1.0
    question = "电磁波的传播顺序？"
    assert (
        physics_verifier.judge("电场变化→磁场变化→电磁波传播", "电场变化 磁场变化 电磁波传播", question=question) == 1.0
    )
    assert (
        physics_verifier.judge("电场变化→磁场变化→电磁波传播", "磁场变化 电场变化 电磁波传播", question=question) == 0.0
    )


def test_physics_verifier_judge_physical_ordered_list_match(physics_verifier):
    # 顺序正确
    question = "声音传播的基本过程？"
    assert (
        physics_verifier.judge("振动→产生声波→传播→人耳接收", "振动 产生声波 传播 人耳接收", question=question) == 1.0
    )

    # 牛顿运动定律应用步骤正确
    question = "用牛顿第二定律解题的基本步骤？"
    assert (
        physics_verifier.judge("受力分析→建立坐标系→列方程→求解", "受力分析 建立坐标系 列方程 求解", question=question)
        == 1.0
    )

    # 电路分析步骤正确
    question = "分析复杂电路的基本步骤？"
    assert (
        physics_verifier.judge(
            "简化电路→确定等效电阻→计算总电流→分析各支路",
            "简化电路 确定等效电阻 计算总电流 分析各支路",
            question=question,
        )
        == 1.0
    )

    # 能量转化过程正确
    question = "单摆运动中能量转化的过程？"
    assert physics_verifier.judge("重力势能→动能→重力势能", "重力势能 动能 重力势能", question=question) == 1.0

    # 物态变化过程正确
    question = "水从固态到气态的完整过程？"
    assert physics_verifier.judge("冰→水→水蒸气", "冰 水 水蒸气", question=question) == 1.0

    # question = "电流方向是_____．"
    # assert physics_verifier.judge(
    #     "abcd",
    #     "a→b→c→d→a",
    #     question=question
    # ) == 1.0

    # 光的双缝干涉实验步骤正确
    question = "双缝干涉实验的基本步骤？"
    assert (
        physics_verifier.judge("单色光源→单缝→双缝→屏幕观察", "单色光源 单缝 双缝 屏幕观察", question=question) == 1.0
    )
