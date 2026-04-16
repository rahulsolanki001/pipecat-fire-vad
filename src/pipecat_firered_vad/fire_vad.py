"""FireRedVAD analyzer integration for Pipecat.

This module provides a VAD analyzer implementation using FireRedVAD's
streaming backend. Audio is processed one 10 ms frame at a time via
`FireRedStreamVad.detect_frame()`, which returns a `StreamVadFrameResult`
carrying both raw and smoothed speech probabilities.

FireRedVAD internal pipeline (per frame):
    160 int16 PCM samples
        → log filterbank feature extraction (+ CMVN normalisation)
        → DFSMN forward pass (causal, stateful via model_caches)
        → raw_prob  (direct neural network output, 0.0–1.0)
        → sliding-window smoother  → smoothed_prob
        → post-processing rules (min_speech_frame, min_silence_frame, etc.)
        → StreamVadFrameResult

We feed `smoothed_prob` to Pipecat's VADAnalyzer smoothing layer, because
it is already denoised by the model's own window smoother and is the value
against which `speech_threshold` is applied internally.

Audio requirements: 16 kHz, 16-bit mono PCM (enforced at construction time).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from loguru import logger

from pipecat.audio.vad.vad_analyzer import VADAnalyzer, VADParams

try:
    from fireredvad import FireRedStreamVad, FireRedStreamVadConfig
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "FireRedVAD is not installed.\n"
        "  git clone https://github.com/FireRedTeam/FireRedVAD.git\n"
        "  cd FireRedVAD && pip install -r requirements.txt\n"
        "  export PYTHONPATH=$PWD:$PYTHONPATH\n"
        "Then download model weights:\n"
        "  huggingface-cli download FireRedTeam/FireRedVAD "
        "--local-dir ./pretrained_models/FireRedVAD"
    ) from exc


# FireRedVAD operates at 16 kHz with 10 ms frames → 160 samples per frame.
# FRAME_LENGTH_SAMPLE is hardcoded in the FireRedVAD source; we mirror it here.
_REQUIRED_SAMPLE_RATE: int = 16_000
_FRAME_SAMPLES: int = 160  # 10 ms × 16000 Hz


class VadMode:
    """Pre-tuned sensitivity presets mirroring FireRedStreamVad.set_mode().

    Attributes:
        VERY_PERMISSIVE: Catches soft/distant speech. May increase false alarms.
        PERMISSIVE:      Balanced — good starting point for most use cases.
        AGGRESSIVE:      Suppresses background noise well. May clip quiet speech.
        VERY_AGGRESSIVE: Maximum noise rejection. Best for loud environments.
    """
    VERY_PERMISSIVE = 0
    PERMISSIVE      = 1
    AGGRESSIVE      = 2
    VERY_AGGRESSIVE = 3


class FireVadAnalyzer(VADAnalyzer):
    """Pipecat VAD analyzer backed by FireRedStreamVad.

    Processes audio one 10 ms frame (160 samples) at a time using
    `detect_frame()`. The DFSMN model's hidden state (`model_caches`) is
    preserved across calls, giving it memory of recent audio context.

    Args:
        model_dir: Path to the downloaded Stream-VAD model directory,
            e.g. ``"pretrained_models/FireRedVAD/Stream-VAD"``.
        sample_rate: Audio sample rate in Hz. Must be 16000 if provided.
        params: Optional VADParams controlling Pipecat-level smoothing
            (confidence, start_secs, stop_secs).
        mode: Optional VadMode preset (0-3). When set, overrides the
            individual threshold/frame parameters below.
        use_gpu: Run DFSMN inference on GPU (requires CUDA).
        smooth_window_size: Frames in the model's internal sliding-window
            smoother. Larger = less jitter, slightly more onset latency.
        speech_threshold: Model-level gate. Frames with smoothed_prob above
            this are considered speech. Range 0.0-1.0 (default 0.4).
        pad_start_frame: Extra frames prepended at speech onset to avoid
            clipping the leading edge of a word.
        min_speech_frame: Minimum consecutive speech frames before a segment
            is confirmed. Prevents single-frame false positives.
        max_speech_frame: Maximum frames in one speech segment before a
            forced split.
        min_silence_frame: Silence frames required to close a speech segment.
            Higher = bot waits longer before deciding the turn ended.
    """

    def __init__(
        self,
        *,
        model_dir: str,
        sample_rate: Optional[int] = None,
        params: Optional[VADParams] = None,
        mode: Optional[int] = None,
        # FireRedStreamVadConfig knobs
        use_gpu: bool = False,
        smooth_window_size: int = 5,
        speech_threshold: float = 0.4,
        pad_start_frame: int = 5,
        min_speech_frame: int = 8,
        max_speech_frame: int = 2000,
        min_silence_frame: int = 20,
    ) -> None:
        super().__init__(sample_rate=sample_rate, params=params)

        if sample_rate is not None:
            _require_16k(sample_rate)

        self._model_dir = model_dir

        vad_config = FireRedStreamVadConfig(
            use_gpu=use_gpu,
            smooth_window_size=smooth_window_size,
            speech_threshold=speech_threshold,
            pad_start_frame=pad_start_frame,
            min_speech_frame=min_speech_frame,
            max_speech_frame=max_speech_frame,
            min_silence_frame=min_silence_frame,
            chunk_max_frame=max_speech_frame,
        )

        logger.info("FireVadAnalyzer: loading model from '{}'", model_dir)
        self._vad: FireRedStreamVad = FireRedStreamVad.from_pretrained(
            model_dir, vad_config
        )

        if mode is not None:
            self.set_mode(mode)

        # Leftover samples that didn't fill a complete 160-sample frame.
        # Carried forward to the next voice_confidence() call.
        self._remainder: np.ndarray = np.empty(0, dtype=np.int16)

        logger.info(
            "FireVadAnalyzer: ready (mode={}, threshold={:.2f})",
            mode if mode is not None else "custom",
            self._vad.config.speech_threshold,
        )

    # ------------------------------------------------------------------
    # VADAnalyzer interface
    # ------------------------------------------------------------------

    def set_sample_rate(self, sample_rate: int) -> None:
        _require_16k(sample_rate)
        super().set_sample_rate(sample_rate)

    def num_frames_required(self) -> int:
        """Return the number of PCM samples Pipecat should deliver per call.

        Returning 160 tells Pipecat to hand us exactly one 10 ms frame at a
        time, which matches detect_frame()'s hard requirement of exactly
        FRAME_LENGTH_SAMPLE samples.
        """
        return _FRAME_SAMPLES

    def reset(self) -> None:
        """Reset all internal state.

        Calls FireRedStreamVad.reset() which clears three things:
          - model_caches  : the DFSMN hidden state between frames
          - audio_feat    : the feature extractor's CMVN running buffer
          - postprocessor : the smoothing window and speech-segment state

        Also discards any leftover samples held in _remainder.
        Call this between sessions (e.g. on_participant_left) so that one
        caller's audio context does not bleed into the next.
        """
        self._vad.reset()
        self._remainder = np.empty(0, dtype=np.int16)
        logger.debug("FireVadAnalyzer: state reset.")

    def set_mode(self, mode: int) -> None:
        """Apply a pre-tuned sensitivity preset.

        Delegates to FireRedStreamVad.set_mode(), which adjusts three knobs
        simultaneously:

            Mode 0 VERY_PERMISSIVE  threshold=0.3  min_speech=8   min_silence=20
            Mode 1 PERMISSIVE       threshold=0.5  min_speech=10  min_silence=15
            Mode 2 AGGRESSIVE       threshold=0.7  min_speech=15  min_silence=10
            Mode 3 VERY_AGGRESSIVE  threshold=0.9  min_speech=20  min_silence=5

        Args:
            mode: One of the VadMode constants (0-3).
        """
        self._vad.set_mode(mode)
        logger.info("FireVadAnalyzer: mode set to {}", mode)

    def voice_confidence(self, buffer: bytes) -> float:
        """Compute voice confidence for the given PCM16 audio buffer.

        Because num_frames_required() returns 160, Pipecat will normally call
        this with exactly 320 bytes (160 samples × 2 bytes). We nonetheless
        handle arbitrary sizes robustly via a remainder buffer, so the
        integration is safe even if the upstream chunk size changes.

        Per-frame flow:
            1. Convert bytes → int16 array, prepend any leftover samples.
            2. Slice into 160-sample frames.
            3. Call detect_frame(frame) → StreamVadFrameResult.
            4. Read smoothed_prob from the result.
            5. Save incomplete trailing samples to _remainder.
            6. Return the peak smoothed_prob across all processed frames.

        We return smoothed_prob (not raw_prob) because:
          - It is already noise-reduced by the model's window smoother.
          - It is the value the model's internal threshold is applied against.
          - Feeding it to Pipecat's own smoother gives two complementary
            layers of noise rejection without double-counting raw variance.

        Returns:
            Peak smoothed_prob across all frames, clipped to [0.0, 1.0].
            Returns 0.0 if no complete frame was available or on any error.
        """
        try:
            incoming = np.frombuffer(buffer, dtype=np.int16)

            # Prepend leftover samples from the previous call.
            if self._remainder.size:
                incoming = np.concatenate([self._remainder, incoming])

            confidences: list[float] = []
            offset = 0

            while offset + _FRAME_SAMPLES <= len(incoming):
                frame = incoming[offset : offset + _FRAME_SAMPLES]
                offset += _FRAME_SAMPLES

                # detect_frame() requires exactly FRAME_LENGTH_SAMPLE (160)
                # samples and returns a StreamVadFrameResult dataclass.
                result = self._vad.detect_frame(frame)

                # smoothed_prob is the model's window-averaged probability —
                # the primary confidence signal. We fall back to raw_prob
                # defensively in case of an unexpected API change.
                prob = getattr(result, "smoothed_prob",
                               getattr(result, "raw_prob", 0.0))
                confidences.append(float(np.clip(prob, 0.0, 1.0)))

            # Hold any sub-frame remainder for the next call.
            self._remainder = incoming[offset:].copy()

            # Peak confidence surfaces a speech onset at any frame boundary.
            return max(confidences) if confidences else 0.0

        except Exception as exc:
            logger.error("FireVadAnalyzer: error in voice_confidence: {}", exc)
            return 0.0


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------

def _require_16k(sample_rate: int) -> None:
    """Raise ValueError if sample_rate is not 16000."""
    if sample_rate != _REQUIRED_SAMPLE_RATE:
        raise ValueError(
            f"FireRedVAD requires {_REQUIRED_SAMPLE_RATE} Hz audio, "
            f"got {sample_rate} Hz. "
            "Convert with: ffmpeg -i in.wav -ar 16000 -ac 1 -acodec pcm_s16le out.wav"
        )