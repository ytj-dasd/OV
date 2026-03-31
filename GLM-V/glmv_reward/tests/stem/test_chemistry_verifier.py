import pytest

from glmv_reward.verifiers import ChemistryVerifier


def test_chemistry_verifier_judge_chemical_formulas(chemistry_verifier):
    """测试基础化学式的等价性判断，包括常见写法、下标、中文描述等。"""
    # Test basic chemical formulas
    assert chemistry_verifier.judge("H2O", "H2O") == 1.0
    assert chemistry_verifier.judge("CO2", "CO2") == 1.0
    assert chemistry_verifier.judge("CH4", "CH4") == 1.0
    assert chemistry_verifier.judge("NaCl", "NaCl") == 1.0
    assert chemistry_verifier.judge("H2SO4", "H2SO4") == 1.0
    # 挑战性用例
    assert chemistry_verifier.judge("H₂O", "H2O") == 1.0  # Unicode下标
    assert chemistry_verifier.judge("H2O2", "H2O") == 0.0  # 不是同一物质
    assert chemistry_verifier.judge("氯化钠", "NaCl") == 1.0  # 中文与化学式
    assert chemistry_verifier.judge("NaCl", "KCl") == 0.0  # 不同物质
    assert chemistry_verifier.judge("C6H12O6", "葡萄糖") == 1.0  # 中文名与化学式
    assert chemistry_verifier.judge("3N_A", "1.806×10²⁴") == 1.0  # 数值不同

    assert chemistry_verifier.judge("CH₄", "CH_{4}") == 1.0
    assert chemistry_verifier.judge("H_2", "H₂") == 1.0
    assert chemistry_verifier.judge("H₂SO₄", "H2SO4") == 1.0
    assert chemistry_verifier.judge("加热碳酸钠", "加热Na2CO3") == 1.0
    assert chemistry_verifier.judge("加热氢氧化钠", "加热NaOH") == 1.0
    # assert chemistry_verifier.judge("氢气和氧气反应生成水", "2H2 + O2 → 2H2O") == 1.0


def test_chemistry_verifier_judge_chemical_states(chemistry_verifier):
    """测试物质状态（气、液、固、溶液）标注的等价性。"""
    # Test chemical states
    assert chemistry_verifier.judge("H2O(l)", "H2O(l)") == 1.0
    assert chemistry_verifier.judge("CO2(g)", "CO2(g)") == 1.0
    assert chemistry_verifier.judge("NaCl(s)", "NaCl(s)") == 1.0
    assert chemistry_verifier.judge("H2SO4(aq)", "H2SO4(aq)") == 1.0
    # 挑战性用例
    assert chemistry_verifier.judge("H2O(l)", "H2O") == 1.0  # 缺省状态
    assert chemistry_verifier.judge("CO2(g)", "CO2(s)") == 0.0  # 状态不同
    assert chemistry_verifier.judge("NaCl(s)", "NaCl(aq)") == 0.0  # 固体与溶液
    assert chemistry_verifier.judge("H2SO4(aq)", "H2SO4(l)") == 0.0  # 溶液与液体
    assert chemistry_verifier.judge("NaCl", "NaCl(s)") == 1.0  # 缺省为固体

    # Test chemical states
    assert chemistry_verifier.judge("H2O(l)", "H2O") == 1.0
    assert chemistry_verifier.judge("CO2", "CO2(g)") == 1.0
    assert chemistry_verifier.judge("NaCl", "NaCl(s)") == 1.0
    assert chemistry_verifier.judge("H2SO4(aq)", "H2SO4") == 1.0


def test_chemistry_verifier_judge_chemical_charges(chemistry_verifier):
    """测试带电粒子的等价性，包括离子、价态等。"""
    # Test chemical charges
    assert chemistry_verifier.judge("Na+", "Na+") == 1.0
    assert chemistry_verifier.judge("Cl-", "Cl-") == 1.0
    assert chemistry_verifier.judge("SO4^2-", "SO4^2-") == 1.0
    assert chemistry_verifier.judge("NH4+", "NH4+") == 1.0
    # 挑战性用例
    assert chemistry_verifier.judge("Fe3+", "Fe^{3+}") == 1.0  # 不同写法
    assert chemistry_verifier.judge("Cl-", "Cl+") == 0.0  # 电荷相反
    assert chemistry_verifier.judge("SO4^{2-}", "SO4^2-") == 1.0  # 不同写法
    assert chemistry_verifier.judge("NH4+", "NH3") == 0.0  # 离子与分子
    assert chemistry_verifier.judge("Na+", "钠离子") == 1.0  # 中文名


def test_chemistry_verifier_judge_chemical_reactions(chemistry_verifier):
    """测试化学反应描述的等价性，包括能量、热、光等。"""
    # Test chemical reactions
    assert chemistry_verifier.judge("2H2 + O2 → 2H2O + energy", "2H2 + O2 = 2H2O + energy") == 1.0
    assert chemistry_verifier.judge("CH4 + 2O2 → CO2 + 2H2O", "CH4 + 2O2 = CO2 + 2H2O") == 1.0
    assert chemistry_verifier.judge("2Na + Cl2 → 2NaCl", "2Na + Cl2 = 2NaCl") == 1.0


def test_chemistry_verifier_judge_chemical_equilibrium(chemistry_verifier):
    """测试化学平衡方程式的等价性，包括箭头、顺序等。"""
    # Test chemical equilibrium
    assert chemistry_verifier.judge("H2 + I2 ⇌ 2HI", "H2 + I2 ⇌ 2HI") == 1.0
    assert chemistry_verifier.judge("N2 + 3H2 ⇌ 2NH3", "N2 + 3H2 ⇌ 2NH3") == 1.0
    assert chemistry_verifier.judge("H2O ⇌ H+ + OH-", "H2O ⇌ H+ + OH-") == 1.0
    # 挑战性用例
    assert chemistry_verifier.judge("H2 + I2 ⇌ 2HI", "I2 + H2 ⇌ 2HI") == 1.0  # 顺序不同
    assert chemistry_verifier.judge("H2O ⇌ H+ + OH-", "H2O → H+ + OH-") == 1.0  # 箭头类型
    assert chemistry_verifier.judge("H2O ⇌ H+ + OH-", "H2O ⇌ H+ + O2-") == 0.0  # 产物不同
    assert chemistry_verifier.judge("N2 + 3H2 ⇌ 2NH3", "N2 + 3H2 ⇌ 2NH2") == 0.0  # 产物不同
    assert chemistry_verifier.judge("N2 + 3H2 ⇌ 2NH3", "N2 + 3H2 = 2NH3") == 1.0


def test_chemistry_verifier_judge_chemical_ph(chemistry_verifier):
    # Test pH values
    assert chemistry_verifier.judge("pH = 7", "pH = 7") == 1.0
    assert chemistry_verifier.judge("pH = 14", "pH = 14") == 1.0
    assert chemistry_verifier.judge("pH = 0", "pH = 0") == 1.0
    assert chemistry_verifier.judge("pH = 7.0", "pH = 7") == 1.0


def test_chemistry_verifier_judge_chemical_units(chemistry_verifier):
    # Test chemical units and concentrations
    assert chemistry_verifier.judge("1 M", "1 mol/L") == 1.0
    assert chemistry_verifier.judge("0.1 M", "0.1 mol/L") == 1.0
    assert chemistry_verifier.judge("1 mM", "0.001 mol/L") == 1.0
    assert chemistry_verifier.judge("1 μM", "1e-6 mol/L") == 1.0
    assert chemistry_verifier.judge("1 nM", "1e-9 mol/L") == 1.0


def test_chemistry_verifier_judge_chemical_quantities(chemistry_verifier):
    # Test chemical quantities with units
    assert chemistry_verifier.judge("1 mol", "1 mol") == 1.0
    assert chemistry_verifier.judge("2.5 mol", "2.5 mol") == 1.0
    assert chemistry_verifier.judge("1 g", "1 g") == 1.0
    assert chemistry_verifier.judge("1 kg", "1000 g") == 1.0
    assert chemistry_verifier.judge("1 L", "1 L") == 1.0
    assert chemistry_verifier.judge("1 mL", "0.001 L") == 1.0


def test_chemistry_verifier_judge_chemical_energies(chemistry_verifier):
    # Test chemical energies with units
    assert chemistry_verifier.judge("1 kJ/mol", "1 kJ/mol") == 1.0
    assert chemistry_verifier.judge("1 kcal/mol", "4.184 kJ/mol") == 1.0
    assert chemistry_verifier.judge("1 J/mol", "0.001 kJ/mol") == 1.0


def test_chemistry_verifier_judge_chemical_pressures(chemistry_verifier):
    # Test chemical pressures with units
    assert chemistry_verifier.judge("1 atm", "101.325 kPa") == 1.0
    assert chemistry_verifier.judge("1 bar", "100 kPa") == 1.0
    assert chemistry_verifier.judge("1 mmHg", "0.133322 kPa") == 1.0
    assert chemistry_verifier.judge("1 torr", "0.133322 kPa") == 1.0


def test_chemistry_verifier_judge_chemical_temperatures(chemistry_verifier):
    # Test chemical temperatures with units
    assert chemistry_verifier.judge("273.15 K", "0 °C") == 1.0
    assert chemistry_verifier.judge("373.15 K", "100 °C") == 1.0
    assert chemistry_verifier.judge("298.15 K", "25 °C") == 1.0
    assert chemistry_verifier.judge("0 K", "-273.15 °C") == 1.0


def test_chemistry_verifier_judge_units_equivalence(chemistry_verifier):
    # 数值相同，单位略有差异，应该判定为等价
    assert chemistry_verifier.judge("1.5N_{A}", "1.5N_A") == 1.0
    assert chemistry_verifier.judge("200 kJ/mol", "200 kJ mol^{-1}") == 1.0
    assert chemistry_verifier.judge("8.314 J mol^{-1} K^{-1}", "8.314 J/(mol·K)") == 1.0
    assert chemistry_verifier.judge("0.0821 L·atm/mol·K", "0.0821 L atm mol^{-1} K^{-1}") == 1.0


def test_chemistry_verifier_judge_units_nonequivalence(chemistry_verifier):
    # 数值不同，不等价
    assert chemistry_verifier.judge("1.0N_A", "2.0N_A") == 0.0
    assert chemistry_verifier.judge("100 kJ/mol", "200 kJ/mol") == 0.0


def test_chemistry_verifier_judge_chemical_equations_equivalence(chemistry_verifier):
    # 同种反应，不同写法，符合计量关系，应等价
    assert chemistry_verifier.judge("2H2 + O2 → 2H2O", "2H2 + O2 = 2H2O") == 1.0
    assert chemistry_verifier.judge("H2 + 0.5 O2 → H2O", "2H2 + O2 = 2H2O") == 1.0
    assert chemistry_verifier.judge("NaCl(aq) + AgNO3 → AgCl↓ + NaNO3", "NaCl + AgNO3 = AgCl + NaNO3") == 1.0
    assert chemistry_verifier.judge("CaCO3(s) → CaO(s) + CO2(g)", "CaCO3 = CaO + CO2") == 1.0
    assert chemistry_verifier.judge("\\mathrm{S} + \\mathrm{O_2} → \\mathrm{SO_2}", "S + O₂ → SO₂") == 1.0


# def test_chemistry_verifier_judge_chemical_equilibrium(chemistry_verifier):
#     assert chemistry_verifier.judge("N2 + 3H2 ⇌ 2NH3", "N2 + 3H2 = 2NH3") == 1.0

# #


def test_notation_format_differences(chemistry_verifier):
    assert chemistry_verifier.judge("6.022e23", "6.022×10²³") == 1.0
    assert chemistry_verifier.judge("1.0 mol", "1 mol") == 1.0
    assert chemistry_verifier.judge("1.0 mol", "1.00 mol") == 1.0
    assert chemistry_verifier.judge("1 mol", "one mole") == 1.0
    assert chemistry_verifier.judge("0.5 mol", "1/2 mol") == 1.0


def test_contextual_inference_extended(chemistry_verifier):
    assert chemistry_verifier.judge("盐酸", "HCl 溶液") == 1.0
    assert chemistry_verifier.judge("盐酸", "氯化氢的水溶液") == 1.0

    assert chemistry_verifier.judge("NaOH", "氢氧化钠") == 1.0
    assert chemistry_verifier.judge("NaOH", "烧碱") == 1.0

    assert chemistry_verifier.judge("醋酸", "乙酸") == 1.0
    assert chemistry_verifier.judge("CH₃COOH", "乙酸") == 1.0

    assert chemistry_verifier.judge("酒精", "乙醇") == 1.0
    assert chemistry_verifier.judge("C₂H₅OH", "酒精") == 1.0

    assert chemistry_verifier.judge("石灰水", "Ca(OH)₂ 饱和溶液") == 1.0
    assert chemistry_verifier.judge("石灰水", "氢氧化钙溶液") == 1.0

    assert chemistry_verifier.judge("生石灰", "氧化钙") == 1.0
    assert chemistry_verifier.judge("CaO", "生石灰") == 1.0

    assert chemistry_verifier.judge("熟石灰", "氢氧化钙") == 1.0
    assert chemistry_verifier.judge("Ca(OH)₂", "熟石灰") == 1.0

    assert chemistry_verifier.judge("加热氯酸钾和二氧化锰混合物", "加热KClO3和MnO2的混合物") == 1.0

    assert (
        chemistry_verifier.judge("能通过控制滴加液体的快慢从而控制反应速率", "反应速率可通过控制液体滴加的速度来调节")
        == 1.0
    )

    assert (
        chemistry_verifier.judge(
            "通过控制滴加稀盐酸的速度，可以调节锌与酸反应产生氢气的速率",
            "调节稀盐酸滴加的快慢能控制锌与稀盐酸反应时氢气生成的速度",
        )
        == 1.0
    )

    assert (
        chemistry_verifier.judge(
            "向锌中滴加稀盐酸，通过控制滴加速度可以控制反应速率",
            "锌与稀盐酸反应时，调节酸的滴加速度能调控氢气产生的快慢",
        )
        == 1.0
    )

    assert (
        chemistry_verifier.judge(
            "锌与稀盐酸反应过程中，可以通过滴加酸液的快慢来控制氢气生成的速度",
            "反应中控制酸的滴加速度可以调节氢气的产生速率",
        )
        == 1.0
    )

    assert (
        chemistry_verifier.judge(
            "调节酸液滴入锌粒的速度，可以影响氢气的生成速率", "通过控制酸液加入速度，能调节锌和酸反应的快慢"
        )
        == 1.0
    )
