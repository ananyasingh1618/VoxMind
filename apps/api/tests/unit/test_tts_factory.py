from voxmind.core.config import Settings
from voxmind.services.tts.factory import build_tts_provider


def _settings(**overrides) -> Settings:
    base = dict(JWT_SECRET_KEY="x" * 32, ENV="test")
    base.update(overrides)
    return Settings(**base)


def test_local_hf_is_the_default_and_always_available_without_credentials():
    settings = _settings()
    assert settings.TTS_PROVIDER == "local_hf"
    provider = build_tts_provider(settings)
    assert provider is not None
    assert provider.provider_name == "local_hf"


def test_disabled_returns_none():
    settings = _settings(TTS_PROVIDER="disabled")
    assert build_tts_provider(settings) is None


def test_openai_without_api_key_returns_none_not_a_fake_provider():
    settings = _settings(TTS_PROVIDER="openai", OPENAI_API_KEY=None)
    assert build_tts_provider(settings) is None


def test_openai_with_api_key_returns_real_provider_instance():
    settings = _settings(TTS_PROVIDER="openai", OPENAI_API_KEY="sk-test-not-real")
    provider = build_tts_provider(settings)
    assert provider is not None
    assert provider.provider_name == "openai"
