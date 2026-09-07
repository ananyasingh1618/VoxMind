from voxmind.core.config import Settings
from voxmind.services.llm.factory import build_llm_provider
from voxmind.services.llm.mock_provider import MockLlmProvider


def _settings(**overrides) -> Settings:
    base = dict(JWT_SECRET_KEY="x" * 32, ENV="test")
    base.update(overrides)
    return Settings(**base)


def test_no_provider_configured_returns_none():
    settings = _settings(LLM_PROVIDER="local_dev")
    assert build_llm_provider(settings) is None


def test_anthropic_without_api_key_returns_none_not_a_fake_provider():
    settings = _settings(LLM_PROVIDER="anthropic", ANTHROPIC_API_KEY=None)
    assert build_llm_provider(settings) is None


def test_openai_without_api_key_returns_none():
    settings = _settings(LLM_PROVIDER="openai", OPENAI_API_KEY=None)
    assert build_llm_provider(settings) is None


def test_anthropic_with_api_key_returns_real_provider_instance():
    settings = _settings(LLM_PROVIDER="anthropic", ANTHROPIC_API_KEY="sk-test-not-real")
    provider = build_llm_provider(settings)
    assert provider is not None
    assert provider.provider_name == "anthropic"


def test_mock_llm_flag_returns_mock_provider_only_in_test_env():
    settings = _settings(ENV="test", MOCK_LLM=True, LLM_PROVIDER="local_dev")
    provider = build_llm_provider(settings)
    assert isinstance(provider, MockLlmProvider)


def test_mock_llm_flag_outside_test_env_is_rejected_at_settings_construction():
    import pytest

    with pytest.raises(ValueError):
        Settings(JWT_SECRET_KEY="x" * 32, ENV="dev", MOCK_LLM=True)
