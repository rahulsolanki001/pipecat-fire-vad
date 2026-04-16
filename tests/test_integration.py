# Run with: pytest tests/test_integration.py -v -m integration
#
# Requires:
#   FIREREDVAD_MODEL_DIR env var pointing to Stream-VAD weights
#   fireredvad installed and on PYTHONPATH

import os
import numpy as np
import pytest
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat_firered_vad import FireVadAnalyzer, VadMode

MODEL_DIR = os.getenv("FIREREDVAD_MODEL_DIR")

@pytest.mark.integration
@pytest.mark.skipif(not MODEL_DIR, reason="FIREREDVAD_MODEL_DIR not set")
class TestRealModel:

    @pytest.fixture(scope="class")
    def analyzer(self):
        return FireVadAnalyzer(
            model_dir=MODEL_DIR,
            sample_rate=16000,
            params=VADParams(confidence=0.5, start_secs=0.2, stop_secs=0.3),
            mode=VadMode.PERMISSIVE,
        )

    def test_loads_without_error(self, analyzer):
        assert analyzer is not None

    def test_silence_gives_low_confidence(self, analyzer):
        silence = np.zeros(160, dtype=np.int16).tobytes()
        score = analyzer.voice_confidence(silence)
        assert score < 0.5, f"Expected low confidence for silence, got {score}"

    def test_num_frames_required(self, analyzer):
        assert analyzer.num_frames_required() == 160

    def test_reset_does_not_crash(self, analyzer):
        analyzer.voice_confidence(np.zeros(160, dtype=np.int16).tobytes())
        analyzer.reset()  # should not raise

    def test_set_mode_does_not_crash(self, analyzer):
        for mode in [VadMode.VERY_PERMISSIVE, VadMode.PERMISSIVE,
                     VadMode.AGGRESSIVE, VadMode.VERY_AGGRESSIVE]:
            analyzer.set_mode(mode)

    def test_confidence_in_valid_range(self, analyzer):
        rng = np.random.default_rng(42)
        noise = (rng.integers(-1000, 1000, 160, dtype=np.int16)).tobytes()
        score = analyzer.voice_confidence(noise)
        assert 0.0 <= score <= 1.0