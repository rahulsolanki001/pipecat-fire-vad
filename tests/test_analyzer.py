import sys
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

# Stub out fireredvad before importing our package,
# so the module-level import in fire_vad.py doesn't fail.
sys.modules["fireredvad"] = MagicMock()

# NOW it's safe to import our package
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat_firered_vad import FireVadAnalyzer, VadMode


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_pcm(n_samples: int, value: int = 0) -> bytes:
    """Return n_samples of silent int16 PCM bytes."""
    return np.full(n_samples, value, dtype=np.int16).tobytes()

def make_frame_result(smoothed_prob: float = 0.0, raw_prob: float = 0.0):
    r = MagicMock()
    r.smoothed_prob = smoothed_prob
    r.raw_prob = raw_prob
    r.is_speech = int(smoothed_prob > 0.4)
    return r


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_vad():
    """Patch FireRedStreamVad so no model weights are needed."""
    instance = MagicMock()
    instance.config.speech_threshold = 0.4
    instance.detect_frame.return_value = make_frame_result(0.0)

    with patch("pipecat_firered_vad.fire_vad.FireRedStreamVad") as MockClass:
        MockClass.from_pretrained.return_value = instance
        yield instance

@pytest.fixture
def analyzer(mock_vad):
    return FireVadAnalyzer(
        model_dir="/fake/model/dir",
        sample_rate=16000,
        params=VADParams(confidence=0.5, start_secs=0.2, stop_secs=0.3),
    )


# ── Sample rate validation ────────────────────────────────────────────────────

def test_rejects_wrong_sample_rate(mock_vad):
    with pytest.raises(ValueError, match="16000"):
        FireVadAnalyzer(model_dir="/fake", sample_rate=8000)

def test_accepts_16k(analyzer):
    analyzer.set_sample_rate(16000)  # should not raise

def test_set_sample_rate_rejects_wrong(analyzer):
    with pytest.raises(ValueError):
        analyzer.set_sample_rate(44100)


# ── num_frames_required ───────────────────────────────────────────────────────

def test_num_frames_required(analyzer):
    assert analyzer.num_frames_required() == 160


# ── voice_confidence: basic ───────────────────────────────────────────────────

def test_returns_zero_on_silence(analyzer, mock_vad):
    mock_vad.detect_frame.return_value = make_frame_result(smoothed_prob=0.0)
    result = analyzer.voice_confidence(make_pcm(160))
    assert result == 0.0

def test_returns_smoothed_prob_on_speech(analyzer, mock_vad):
    mock_vad.detect_frame.return_value = make_frame_result(smoothed_prob=0.85)
    result = analyzer.voice_confidence(make_pcm(160))
    assert abs(result - 0.85) < 1e-6

def test_confidence_clipped_to_1(analyzer, mock_vad):
    mock_vad.detect_frame.return_value = make_frame_result(smoothed_prob=1.5)
    result = analyzer.voice_confidence(make_pcm(160))
    assert result <= 1.0

def test_confidence_clipped_to_0(analyzer, mock_vad):
    mock_vad.detect_frame.return_value = make_frame_result(smoothed_prob=-0.3)
    result = analyzer.voice_confidence(make_pcm(160))
    assert result >= 0.0


# ── voice_confidence: remainder buffer ───────────────────────────────────────

def test_remainder_carried_forward(analyzer, mock_vad):
    """80 samples then 80 samples should produce exactly one detect_frame call."""
    mock_vad.detect_frame.return_value = make_frame_result(smoothed_prob=0.9)

    analyzer.voice_confidence(make_pcm(80))   # not enough for a frame
    assert mock_vad.detect_frame.call_count == 0

    analyzer.voice_confidence(make_pcm(80))   # completes the frame
    assert mock_vad.detect_frame.call_count == 1

def test_multiple_frames_in_one_call(analyzer, mock_vad):
    """320 bytes = 2 frames → detect_frame called twice, peak returned."""
    mock_vad.detect_frame.side_effect = [
        make_frame_result(smoothed_prob=0.3),
        make_frame_result(smoothed_prob=0.9),
    ]
    result = analyzer.voice_confidence(make_pcm(320))
    assert mock_vad.detect_frame.call_count == 2
    assert abs(result - 0.9) < 1e-6  # peak is returned

def test_empty_buffer_returns_zero(analyzer, mock_vad):
    result = analyzer.voice_confidence(b"")
    assert result == 0.0
    mock_vad.detect_frame.assert_not_called()

def test_sub_frame_buffer_returns_zero(analyzer, mock_vad):
    """Fewer than 160 samples → no frame processed, returns 0.0."""
    result = analyzer.voice_confidence(make_pcm(100))
    assert result == 0.0
    mock_vad.detect_frame.assert_not_called()


# ── reset ─────────────────────────────────────────────────────────────────────

def test_reset_clears_remainder(analyzer, mock_vad):
    analyzer.voice_confidence(make_pcm(80))   # load remainder
    assert analyzer._remainder.size == 80
    analyzer.reset()
    assert analyzer._remainder.size == 0

def test_reset_calls_vad_reset(analyzer, mock_vad):
    analyzer.reset()
    mock_vad.reset.assert_called_once()


# ── set_mode ──────────────────────────────────────────────────────────────────

def test_set_mode_delegates_to_vad(analyzer, mock_vad):
    analyzer.set_mode(VadMode.AGGRESSIVE)
    mock_vad.set_mode.assert_called_once_with(VadMode.AGGRESSIVE)

def test_mode_applied_at_init(mock_vad):
    FireVadAnalyzer(model_dir="/fake", mode=VadMode.VERY_AGGRESSIVE)
    mock_vad.set_mode.assert_called_once_with(VadMode.VERY_AGGRESSIVE)


# ── error handling ────────────────────────────────────────────────────────────

def test_detect_frame_exception_returns_zero(analyzer, mock_vad):
    """Any exception in detect_frame must not propagate — return 0.0."""
    mock_vad.detect_frame.side_effect = RuntimeError("model exploded")
    result = analyzer.voice_confidence(make_pcm(160))
    assert result == 0.0

def test_missing_smoothed_prob_falls_back_to_raw(analyzer, mock_vad):
    """If smoothed_prob is missing, fall back to raw_prob gracefully."""
    result_obj = MagicMock(spec=["raw_prob"])  # no smoothed_prob attribute
    result_obj.raw_prob = 0.77
    mock_vad.detect_frame.return_value = result_obj
    result = analyzer.voice_confidence(make_pcm(160))
    assert abs(result - 0.77) < 1e-6