# pipecat-ai-fireredvad

[![PyPI version](https://img.shields.io/pypi/v/pipecat-firered-vad.svg)](https://pypi.org/project/pipecat-firered-vad/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

A [Pipecat](https://github.com/pipecat-ai/pipecat) integration for **FireRedVAD** — a SOTA industrial-grade streaming Voice Activity Detection model that supports 100+ languages and outperforms Silero-VAD, TEN-VAD, FunASR-VAD, and WebRTC-VAD on the FLEURS-VAD-102 benchmark.

| Metric       | FireRedVAD | Silero-VAD | TEN-VAD | WebRTC-VAD |
|--------------|-----------|------------|---------|------------|
| F1 Score ↑   | **97.57** | 95.95      | 95.19   | 52.30      |
| AUC-ROC ↑    | **99.60** | 97.99      | 97.81   | —          |
| False Alarm ↓| **2.69**  | 9.41       | 15.47   | 2.83       |

---

## Requirements

- Python 3.10+
- `pipecat-ai >= 0.0.90`
- `fireredvad` (installed manually from GitHub — see setup below)
- Audio: **16 kHz, 16-bit mono PCM**

---

## Installation

### 1. Install this package

```bash
pip install pipecat-firered-vad
```

### 2. Install FireRedVAD

`fireredvad` is not on PyPI. Clone and install it from GitHub:

```bash
git clone https://github.com/FireRedTeam/FireRedVAD.git
cd FireRedVAD
pip install -r requirements.txt
export PYTHONPATH=$PWD:$PYTHONPATH
```

### 3. Download model weights

```bash
# via Hugging Face
pip install -U "huggingface_hub[cli]"
huggingface-cli download FireRedTeam/FireRedVAD \
    --local-dir ./pretrained_models/FireRedVAD

# or via ModelScope (recommended if you're in China)
pip install -U modelscope
modelscope download --model xukaituo/FireRedVAD \
    --local_dir ./pretrained_models/FireRedVAD
```

### 4. Configure environment

```bash
cp .env.example .env
# Edit .env and set FIREREDVAD_MODEL_DIR to your downloaded weights path
```

---

## Quick Start

```python
import asyncio
from dotenv import load_dotenv
import os

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat_ai_fireredvad import FireVadAnalyzer

load_dotenv()

async def main():
    vad = FireVadAnalyzer(
        model_dir=os.environ["FIREREDVAD_MODEL_DIR"],
        params=VADParams(
            confidence=0.7,
            start_secs=0.2,
            stop_secs=0.5,
        ),
        speech_threshold=0.4,
        smooth_window_size=5,
    )

    # Pass the analyzer to your transport, e.g. DailyTransport:
    # transport = DailyTransport(..., vad_analyzer=vad)

asyncio.run(main())
```

---

## Configuration Reference

### `FireVadAnalyzer` constructor parameters

| Parameter            | Type    | Default | Description                                              |
|----------------------|---------|---------|----------------------------------------------------------|
| `model_dir`          | `str`   | —       | **Required.** Path to the `Stream-VAD` model directory. |
| `sample_rate`        | `int`   | `None`  | Must be `16000` if provided (enforced).                  |
| `params`             | `VADParams` | `None` | Pipecat-level smoothing (confidence, start/stop secs). |
| `use_gpu`            | `bool`  | `False` | Run inference on GPU.                                   |
| `smooth_window_size` | `int`   | `5`     | Frame-level confidence smoothing window inside FireRedVAD. |
| `speech_threshold`   | `float` | `0.4`   | Raw model threshold for speech vs silence.              |
| `pad_start_frame`    | `int`   | `5`     | Frames prepended at speech onset to avoid clipping.     |
| `min_speech_frame`   | `int`   | `8`     | Minimum consecutive frames before segment is confirmed. |
| `max_speech_frame`   | `int`   | `2000`  | Maximum frames in a single speech segment.              |
| `min_silence_frame`  | `int`   | `20`    | Silence frames required before a segment ends.          |
| `max_buffer_frames`  | `int`   | `50`    | Ring buffer capacity (oldest frames evicted on overflow). |

### Audio requirements

FireRedVAD only accepts **16 kHz, 16-bit mono PCM**. Convert other formats with:

```bash
ffmpeg -i input.wav -ar 16000 -ac 1 -acodec pcm_s16le output.wav
```

---

## Environment Variables

| Variable                | Description                                          |
|-------------------------|------------------------------------------------------|
| `FIREREDVAD_MODEL_DIR`  | Path to the downloaded `Stream-VAD` model directory. |
| `FIREREDVAD_USE_GPU`    | Set to `1` to enable GPU inference (default: `0`).  |

See [`.env.example`](.env.example) for a ready-to-copy template.

---

## Related packages

- [`pipecat-firered-vad`](https://pypi.org/project/pipecat-firered-vad/) — this package
- [`pipecat-ai-ten-vad`](https://pypi.org/project/pipecat-ai-ten-vad/) — TEN-VAD integration (same author)
- [FireRedVAD on GitHub](https://github.com/FireRedTeam/FireRedVAD)
- [FireRedVAD on HuggingFace](https://huggingface.co/FireRedTeam/FireRedVAD)

---

## Contributing

Pull requests are welcome. For major changes, please open an issue first.

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-change`
3. Run linting: `ruff check . && ruff format .`
4. Run tests: `pytest`
5. Open a PR

---

## License

Apache License 2.0 — see [LICENSE](LICENSE) for details.

FireRedVAD model weights are released under their own license. See the
[FireRedVAD repository](https://github.com/FireRedTeam/FireRedVAD) for details.