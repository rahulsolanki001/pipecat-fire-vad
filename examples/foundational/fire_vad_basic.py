"""Basic voice agent example using FireRedVAD with Pipecat.

This example demonstrates a complete voice pipeline using FireRedVAD for
voice activity detection, Deepgram for speech-to-text, OpenAI for the
LLM, and Cartesia for text-to-speech.

FireRedVAD is a SOTA streaming VAD model supporting 100+ languages,
outperforming Silero-VAD, TEN-VAD, FunASR-VAD, and WebRTC-VAD.

Requirements:
    # 1. Install Pipecat with required service extras
    pip install "pipecat-ai[daily,deepgram,openai,cartesia]"

    # 2. Install FireRedVAD (not on PyPI — install from GitHub)
    git clone https://github.com/FireRedTeam/FireRedVAD.git
    cd FireRedVAD && pip install -r requirements.txt
    export PYTHONPATH=$PWD:$PYTHONPATH

    # 3. Download model weights
    huggingface-cli download FireRedTeam/FireRedVAD \\
        --local-dir ./pretrained_models/FireRedVAD

    # 4. Install this package
    pip install pipecat-firered-vad

Usage:
    Copy .env.example to .env, fill in your API keys and model path, then:
        python fireredvad_basic.py

Audio note:
    FireRedVAD requires 16 kHz 16-bit mono PCM. DailyTransport handles this
    automatically when sample_rate=16000 is set below.
"""

import asyncio
import os

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.services.daily import DailyParams, DailyTransport

from pipecat_ai_fireredvad import FireVadAnalyzer, VadMode

load_dotenv()


async def main() -> None:
    # ------------------------------------------------------------------
    # FireRedVAD analyzer
    # ------------------------------------------------------------------
    # FIREREDVAD_MODEL_DIR must point to the Stream-VAD subdirectory, e.g.:
    #   ./pretrained_models/FireRedVAD/Stream-VAD
    #
    # Sensitivity is controlled via VadMode presets (recommended) or by
    # setting individual knobs manually (advanced).
    #
    # VadMode presets adjust speech_threshold, min_speech_frame, and
    # min_silence_frame together as a matched set:
    #
    #   VadMode.VERY_PERMISSIVE  — soft/distant speech, noisy mics
    #   VadMode.PERMISSIVE       — balanced, good default (used below)
    #   VadMode.AGGRESSIVE       — clean environments, reduces false alarms
    #   VadMode.VERY_AGGRESSIVE  — maximum noise rejection
    #
    # Pipecat-level VADParams tune the turn-detection layer on top:
    #   confidence  — minimum smoothed_prob to count as speech (0.0–1.0)
    #   start_secs  — seconds of speech before the turn is confirmed
    #   stop_secs   — seconds of silence before the turn ends;
    #                 raise to 0.5–0.8 if the bot cuts off mid-sentence
    # ------------------------------------------------------------------
    vad = FireVadAnalyzer(
        model_dir=os.environ["FIREREDVAD_MODEL_DIR"],
        sample_rate=16000,
        params=VADParams(
            confidence=0.7,   # Pipecat-level gate on smoothed_prob
            start_secs=0.2,   # silence → speech: confirmed after this long
            stop_secs=0.3,    # speech → silence: end-of-turn after this long
        ),
        mode=VadMode.PERMISSIVE,   # swap to AGGRESSIVE in noisy environments
        use_gpu=os.getenv("FIREREDVAD_USE_GPU", "0") == "1",
    )

    # ------------------------------------------------------------------
    # Daily WebRTC transport
    # ------------------------------------------------------------------
    transport = DailyTransport(
        os.environ["DAILY_ROOM_URL"],
        os.getenv("DAILY_TOKEN"),
        "FireRed VAD Bot",
        DailyParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_enabled=True,
            vad_analyzer=vad,
            vad_audio_passthrough=True,
        ),
    )

    # ------------------------------------------------------------------
    # AI services
    # ------------------------------------------------------------------
    stt = DeepgramSTTService(api_key=os.environ["DEEPGRAM_API_KEY"])

    llm = OpenAILLMService(
        api_key=os.environ["OPENAI_API_KEY"],
        model="gpt-4o-mini",
    )

    tts = CartesiaTTSService(
        api_key=os.environ["CARTESIA_API_KEY"],
        voice_id=os.getenv("CARTESIA_VOICE_ID", "71a7ad14-091d-4441-9a1e-4e1c48c89f41"),
    )

    # ------------------------------------------------------------------
    # Conversation context
    # ------------------------------------------------------------------
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful voice assistant. "
                "Keep your responses concise and conversational — "
                "ideally one or two sentences. "
                "Avoid markdown, bullet points, or lists."
            ),
        }
    ]

    context = OpenAILLMContext(messages)
    context_aggregator = llm.create_context_aggregator(context)

    # ------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------
    pipeline = Pipeline(
        [
            transport.input(),               # receives audio from the user
            stt,                             # transcribes speech → text
            context_aggregator.user(),       # collects the user turn
            llm,                             # generates a response
            tts,                             # synthesises speech
            transport.output(),              # sends audio to the user
            context_aggregator.assistant(),  # stores the bot turn
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(allow_interruptions=True),
    )

    # ------------------------------------------------------------------
    # Transport event handlers
    # ------------------------------------------------------------------
    @transport.event_handler("on_first_participant_joined")
    async def on_first_participant_joined(transport, participant):
        logger.info("Participant joined: {}", participant["id"])
        await transport.capture_participant_transcription(participant["id"])
        # Seed the pipeline with an empty user context frame so the LLM
        # is primed and ready before the user speaks.
        await task.queue_frames([context_aggregator.user().get_context_frame()])

    @transport.event_handler("on_participant_left")
    async def on_participant_left(transport, participant, reason):
        logger.info("Participant left: {} (reason: {})", participant["id"], reason)
        # Reset VAD state between sessions so stale hidden state doesn't
        # bleed into the next caller.
        vad.reset()
        await task.cancel()

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------
    runner = PipelineRunner()
    await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())