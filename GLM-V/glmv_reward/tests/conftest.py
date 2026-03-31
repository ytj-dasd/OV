# tests/conftest.py
import pytest

from glmv_reward.reward_system import RewardSystem


@pytest.fixture(scope="session")
def reward_system_instance():
    """
    Provides a RewardSystem instance initialized with the default YAML config.
    Scope is session as it's unlikely to change state between tests.
    """
    return RewardSystem.from_yaml("configs/full_config.yaml")


@pytest.fixture
def geoquest_verifier(reward_system_instance):
    return reward_system_instance.get_verifier_from_datasource("geoguess")


@pytest.fixture
def math_verifier(reward_system_instance):
    return reward_system_instance.get_verifier_from_datasource("math")


@pytest.fixture
def general_verifier(reward_system_instance):
    return reward_system_instance.get_verifier_from_datasource("general")


@pytest.fixture
def physics_verifier(reward_system_instance):
    return reward_system_instance.get_verifier_from_datasource("physics")


@pytest.fixture
def chemistry_verifier(reward_system_instance):
    return reward_system_instance.get_verifier_from_datasource("chemistry")


@pytest.fixture
def android_world_verifier(reward_system_instance):
    return reward_system_instance.get_verifier_from_datasource("AndroidWorld")


@pytest.fixture
def webvoyager_verifier(reward_system_instance):
    return reward_system_instance.get_verifier_from_datasource("WebVoyager")


@pytest.fixture
def osworld_verifier(reward_system_instance):
    return reward_system_instance.get_verifier_from_datasource("OSWorld")


@pytest.fixture
def chart_verifier(reward_system_instance):
    return reward_system_instance.get_verifier_from_datasource("chart")


@pytest.fixture
def logic_verifier(reward_system_instance):
    # temp use math as logic verifier
    return reward_system_instance.get_verifier_from_datasource("math")


@pytest.fixture
def multi_image_general_verifier(reward_system_instance):
    return reward_system_instance.get_verifier_from_datasource("multi_image")


@pytest.fixture
def ocr_verifier(reward_system_instance):
    return reward_system_instance.get_verifier_from_datasource("ocr")


@pytest.fixture
def counting_verifier(reward_system_instance):
    return reward_system_instance.get_verifier_from_datasource("counting")


@pytest.fixture
def mmsi_verifier(reward_system_instance):
    return reward_system_instance.get_verifier_from_datasource("mmsi")


@pytest.fixture
def vqa_verifier(reward_system_instance):
    return reward_system_instance.get_verifier_from_datasource("vqa")


@pytest.fixture
def long_doc_verifier(reward_system_instance):
    return reward_system_instance.get_verifier_from_datasource("long_doc")


@pytest.fixture
def language_mix_verifier(reward_system_instance):
    return reward_system_instance.get_verifier_from_datasource("language_mix")
