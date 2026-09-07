import pytest

from voxmind.services.incongruence.analyzer import DeterministicIncongruenceAnalyzer


@pytest.mark.asyncio
async def test_aligned_signals_yield_low_incongruence():
    analyzer = DeterministicIncongruenceAnalyzer()
    semantic = {"sentiment_label": "positive", "sentiment_score": 0.9}
    vocal = {"predicted_label": "happy", "confidence": 0.9}
    signal = await analyzer.analyze(semantic, vocal)
    assert signal.incongruence_score < 0.2
    assert signal.signal_category == "analytical_not_diagnostic"
    assert "not indicate deception" in signal.explanation.lower()


@pytest.mark.asyncio
async def test_divergent_signals_yield_high_incongruence():
    analyzer = DeterministicIncongruenceAnalyzer()
    semantic = {"sentiment_label": "positive", "sentiment_score": 0.95}
    vocal = {"predicted_label": "sad", "confidence": 0.9}
    signal = await analyzer.analyze(semantic, vocal)
    assert signal.incongruence_score > 0.6


@pytest.mark.asyncio
async def test_never_frames_as_deception_or_lie_detection():
    analyzer = DeterministicIncongruenceAnalyzer()
    semantic = {"sentiment_label": "negative", "sentiment_score": 0.8}
    vocal = {"predicted_label": "happy", "confidence": 0.8}
    signal = await analyzer.analyze(semantic, vocal)
    explanation_lower = signal.explanation.lower()
    assert "lie" not in explanation_lower
    assert "dishonest" in explanation_lower or "deception" in explanation_lower
    assert "does not indicate deception" in explanation_lower or "not indicate deception" in explanation_lower


@pytest.mark.asyncio
async def test_confidence_limited_by_weaker_upstream_signal():
    analyzer = DeterministicIncongruenceAnalyzer()
    semantic = {"sentiment_label": "positive", "sentiment_score": 0.99}
    vocal = {"predicted_label": "sad", "confidence": 0.3}
    signal = await analyzer.analyze(semantic, vocal)
    assert signal.confidence == pytest.approx(0.3)
