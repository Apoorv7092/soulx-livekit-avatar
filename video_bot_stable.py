"""
video_bot.py — End-to-end talking-avatar bot over LiveKit WebRTC.

Flow
────
  LiveKit audio-in (participant mic)
    → VAD (silero, in-process)
    → Ori STT  (WebSocket, streaming)         — raw websockets, no pipecat
    → OpenAI LLM  (chat completions stream)   — raw openai SDK
    → Ori TTS  (WebSocket, streaming)         — raw websockets, no pipecat
    → SoulX FlashHead GPU inference           — audio → lip-synced video frames
    → LiveKit A/V publish (own rtc tracks)    — video + audio streamed to room

Architecture
────────────
  AvatarBot          — top-level orchestrator; owns the LiveKit room, all
                        WebSocket connections, and the GPU model.
  STTClient          — wraps the Ori STT WebSocket; feeds raw PCM chunks and
                        yields final transcripts.
  TTSClient          — wraps the Ori TTS WebSocket; sends text, streams back
                        PCM audio bytes.
  VideoInferencePusher
                     — background asyncio task that drains a generation_queue
                        of PCM audio slices, runs FlashHead inference, and
                        pushes (rgba, audio_bytes) pairs onto a playback_queue.
  PlaybackLoop       — background asyncio task that drains playback_queue and
                        calls video_source.capture_frame / audio_source.capture_frame
                        at the target FPS.
  VAD                — Silero VAD run per 30ms frame; emits speech-start /
                        speech-end events that gate the STT pipeline.

Key design decisions
────────────────────
  • No pipecat — every service is called directly so there are zero framework
    surprises hiding bugs.
  • The bot mutes itself while the user is speaking (drops mic audio) and
    mutes the user's track while the bot is speaking, preventing echo /
    feedback loops.
  • STT runs in streaming mode: 20ms PCM chunks → JSON over the existing WS.
    Only "recognized" (final) transcripts trigger the LLM.
  • TTS text is streamed from the LLM token-by-token; we accumulate into
    sentence-sized chunks and send each chunk to TTS as soon as it ends
    (sentence streaming) so audio starts before the full LLM response is done.
  • Silero VAD detects speech boundaries so we know when the user has finished
    speaking before sending the transcript to the LLM.

Required env vars (.env in same directory):
  LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET
  ORI_STT_API_KEY, ORI_STT_BASE_URL          (wss://…)
  ORI_TTS_API_KEY, ORI_TTS_WS_URL            (wss://…)
  ORI_TTS_VOICE_ID
  ORI_STT_LANGUAGE                            (default: en)
  ORI_TTS_LANGUAGE                            (default: hi)
  OPENAI_API_KEY
  LIVEKIT_ROOM                                (default: soulx-flashhead-room)
  AVATAR_IMAGE                                (default: examples/girl.png)
  MODEL_CKPT_DIR                              (default: models/SoulX-FlashHead-1_3B)
  WAV2VEC_DIR                                 (default: models/hindi_base_wav2vec2)
"""

import asyncio
import collections
import io
import json
import os
import re
import sys
import time
import uuid
import wave

import cv2
import numpy as np
import requests
import torch
import websockets
from dotenv import load_dotenv
from loguru import logger
from openai import AsyncOpenAI

from livekit import api, rtc
import base64
from flash_head.inference import (
    get_audio_embedding,
    get_base_data,
    get_infer_params,
    get_pipeline,
    run_pipeline,
)

load_dotenv(override=True)

# ─────────────────────────────────────────────────────────────────────────────
# Constants / tunables
# ─────────────────────────────────────────────────────────────────────────────
SAMPLE_RATE = 16_000          # Hz  — must match FlashHead model
CHANNELS    = 1
CHUNK_MS    = 20              # ms per STT audio chunk
CHUNK_SAMPLES = SAMPLE_RATE * CHUNK_MS // 1000   # 320 samples
CHUNK_BYTES   = CHUNK_SAMPLES * 2                # 640 bytes  (int16)

VAD_WINDOW_MS      = 32       # ms for Silero VAD analysis window — EXACTLY 512 samples @ 16kHz
                              # model enforces: x.shape[-1] must == 512 for sr=16000
VAD_THRESHOLD      = 0.5      # speech probability threshold
VAD_SPEECH_HOLD_MS = 400      # ms to stay "speaking" after prob drops below threshold
VAD_MIN_SPEECH_MS  = 200      # ignore speech bursts shorter than this

TTS_ENCODING    = "pcm_16000"  # must match SAMPLE_RATE above
TTS_CHUNK_BYTES = SAMPLE_RATE * 2  # 1-second chunks fed to FlashHead

ANIMATION_MULTIPLIER = 1.5     # scale audio embedding → more expressive lip-sync
PLAYBACK_QUEUE_MAX   = 999     # frames before we start dropping to catch up

SYSTEM_PROMPT = """You are a helpful voiceAI named priya"""
# Regex that splits LLM text into sentence-sized TTS chunks
_SENTENCE_END = re.compile(r'(?<=[.!?,;])\s+')


# ─────────────────────────────────────────────────────────────────────────────
# Silero VAD wrapper (no pipecat dependency)
# ─────────────────────────────────────────────────────────────────────────────
class SileroVAD:
    """Thin wrapper around the Silero VAD torch model.

    Supports both the legacy stateful API (v4, returns prob, h, c) and the
    newer stateless API (v5+, forward(x, sr) → prob scalar directly).
    """

    def __init__(self, threshold: float = VAD_THRESHOLD):
        self.threshold = threshold
        self._model, self._utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            onnx=False,
            verbose=False,
        )
        self._model.eval()

    def reset(self):
        self._model.reset_states()

    @torch.no_grad()
    def is_speech(self, pcm_bytes: bytes) -> float:
        """Return speech probability for a PCM-16 frame."""
        samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        tensor = torch.from_numpy(samples).unsqueeze(0)  # (1, N)
        prob = self._model(tensor, SAMPLE_RATE)
        return float(prob)


# ─────────────────────────────────────────────────────────────────────────────
# STT client — Ori STT REST API (OpenAI-compatible)
# ─────────────────────────────────────────────────────────────────────────────
class STTClient:
    """Transcribes utterances via the Ori STT REST API.

    Sends buffered PCM as an in-memory WAV file; no WebSocket involved.

    Usage:
        transcript = await stt.transcribe_utterance(pcm_bytes)
    """

    def __init__(self, api_key: str, base_url: str, language: str = "hi",
                 model: str = "ori-prime-v2.3"):
        self._api_key  = api_key
        # base_url may be ws:// or https://; normalise to https REST endpoint
        self._api_url  = 'https://ori-stt-test.oriserve.com/openai/v1/audio/transcriptions'
        self._language = language
        self._model    = model

    @staticmethod
    def _pcm_to_wav(pcm_bytes: bytes, sample_rate: int = SAMPLE_RATE,
                    channels: int = 1, sampwidth: int = 2) -> bytes:
        """Wrap raw PCM-16 bytes in a WAV container (in memory)."""
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(sampwidth)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm_bytes)
        return buf.getvalue()

    def _post(self, wav_bytes: bytes) -> str:
        """Blocking HTTP POST — run via asyncio.to_thread."""
        resp = requests.post(
            self._api_url,
            headers={"Authorization": f"Bearer {self._api_key}"},
            data={
                "language":    self._language,
                "stream":      "false",
                "model":       self._model,
                "temperature": "0.0",
            },
            files={"file": ("utterance.wav", wav_bytes, "audio/wav")},
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json().get("text", "").strip()

    async def transcribe_utterance(self, pcm_bytes: bytes) -> str:
        """Convert PCM to WAV and POST to Ori STT; return transcript or ''."""
        logger.info(f"STT: transcribing {len(pcm_bytes)} bytes via REST…")
        wav = self._pcm_to_wav(pcm_bytes)
        try:
            text = await asyncio.to_thread(self._post, wav)
            logger.info(f"STT transcript: {text!r}")
            return text
        except Exception as e:
            logger.error(f"STT error: {e}")
            return ""


# ─────────────────────────────────────────────────────────────────────────────
# TTS client — wraps Ori TTS WebSocket
# ─────────────────────────────────────────────────────────────────────────────
class TTSClient:
    """Sends text to Ori TTS and yields PCM bytes as they arrive.

    Usage:
        async with TTSClient(...) as tts:
            async for pcm in tts.synthesize("Hello world"):
                process(pcm)
    """

    def __init__(self, api_key: str, ws_url: str, voice_id: str,
                 language: str = "hi"):
        self._api_key  = api_key
        self._ws_url   = ws_url
        self._voice_id = voice_id
        self._language = language
        self._ws       = None

    async def connect(self):
        logger.info(f"TTS connecting to {self._ws_url}")
        self._ws = await websockets.connect(
            self._ws_url,
            additional_headers={"Authorization": f"Bearer {self._api_key}"},
            max_size=16 * 1024 * 1024,
        )
        logger.info("TTS connected.")

    async def disconnect(self):
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
        self._ws = None

    async def synthesize(self, text: str):
        """Send `text` for synthesis; async-yield decoded PCM bytes per chunk."""
        if not self._ws:
            return
        if not text.strip():
            return

        req = {
            "text":       text,
            "language":   self._language,
            "voice_id":   self._voice_id,
            "encoding":   TTS_ENCODING,
            "speechReqId": str(uuid.uuid4()),
        }
        logger.debug(f"TTS synthesize: {text!r}")
        try:
            await self._ws.send(json.dumps(req))
        except Exception as e:
            logger.error(f"TTS send error: {e}")
            return

        # Receive audio chunks until audio_streaming_complete
        try:
            async for raw in self._ws:
                msg = json.loads(raw)
                chunks   = msg.get("audio_chunks", [])
                complete = msg.get("audio_streaming_complete", False)

                for b64 in chunks:
                    audio = base64.b64decode(b64)
                    # Skip 4092-byte non-audio metadata packets (model embeddings)
                    if len(audio) == 4092:
                        continue
                    yield audio

                if complete:
                    break
        except Exception as e:
            logger.error(f"TTS recv error: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Avatar bot — ties everything together
# ─────────────────────────────────────────────────────────────────────────────
class AvatarBot:
    def __init__(self):
        # ── env ──────────────────────────────────────────────────────────────
        self.lk_url     = os.environ["LIVEKIT_URL"]
        self.lk_key     = os.environ["LIVEKIT_API_KEY"]
        self.lk_secret  = os.environ["LIVEKIT_API_SECRET"]
        self.room_name  = os.getenv("LIVEKIT_ROOM", "soulx-flashhead-room")

        self.stt_key    = os.environ["ORI_STT_API_KEY"]
        self.stt_url    = os.environ["ORI_STT_BASE_URL"]
        self.stt_lang   = os.getenv("ORI_STT_LANGUAGE", "en")

        self.tts_key    = os.environ["ORI_TTS_API_KEY"]
        self.tts_ws_url = os.environ["ORI_TTS_WS_URL"]
        self.tts_voice  = os.environ["ORI_TTS_VOICE_ID"]
        self.tts_lang   = os.getenv("ORI_TTS_LANGUAGE", "hi")

        self.openai_key     = os.environ["OPENAI_API_KEY"]
        self.avatar_img     = os.getenv("AVATAR_IMAGE",    "examples/girl2.png")
        self.model_ckpt_dir = os.getenv("MODEL_CKPT_DIR",  "models/SoulX-FlashHead-1_3B")
        self.wav2vec_dir    = os.getenv("WAV2VEC_DIR",      "models/hindi_base_wav2vec2")

        # ── state ─────────────────────────────────────────────────────────────
        self.room: rtc.Room = rtc.Room()
        self.openai = AsyncOpenAI(api_key=self.openai_key)
        self.conversation: list[dict] = []   # OpenAI messages history

        # service clients
        self.stt: STTClient  = STTClient(self.stt_key, self.stt_url, self.stt_lang)
        self.tts: TTSClient  = TTSClient(self.tts_key, self.tts_ws_url,
                                         self.tts_voice, self.tts_lang)

        # ── FlashHead inference params ────────────────────────────────────────
        ip = get_infer_params()
        self.sr           = ip["sample_rate"]
        self.tgt_fps      = ip["tgt_fps"]
        self.cad          = ip["cached_audio_duration"]   # cached audio duration
        self.frame_num    = ip["frame_num"]
        self.motion_frames_num = 2
        self.slice_len    = self.frame_num - self.motion_frames_num

        self.audio_end_idx   = self.cad * self.tgt_fps
        self.audio_start_idx = self.audio_end_idx - self.frame_num
        self.cached_len      = self.sr * self.cad
        self.slice_samples   = self.slice_len * self.sr // self.tgt_fps

        # Rolling audio window (silence-initialised)
        self.audio_dq = collections.deque(
            [0.0] * self.cached_len, maxlen=self.cached_len
        )

        # ── queues ────────────────────────────────────────────────────────────
        # (float_array, bytes) slices waiting for GPU inference
        self.generation_queue: asyncio.Queue = asyncio.Queue()
        # (rgba_np, audio_bytes) pairs waiting for LiveKit publish
        self.playback_queue: collections.deque = collections.deque()

        # ── LiveKit A/V sources / tracks ──────────────────────────────────────
        self.video_source = rtc.VideoSource(512, 512)
        self.video_track  = rtc.LocalVideoTrack.create_video_track(
            "bot-video", self.video_source
        )
        self.audio_source = rtc.AudioSource(self.sr, 1)
        self.audio_track  = rtc.LocalAudioTrack.create_audio_track(
            "bot-audio", self.audio_source
        )

        # ── Idle frame ────────────────────────────────────────────────────────
        img = cv2.imread(self.avatar_img)
        if img is None:
            logger.warning(f"Avatar image not found: {self.avatar_img!r} — using black frame")
            img = np.zeros((512, 512, 3), dtype=np.uint8)
        self.idle_rgba = cv2.cvtColor(cv2.resize(img, (512, 512)), cv2.COLOR_BGR2RGBA)

        # ── bot-speaking flag (mutes user audio processing while bot talks) ───
        self._bot_speaking = False

        # ── VAD ───────────────────────────────────────────────────────────────
        self._vad = SileroVAD(threshold=VAD_THRESHOLD)

        # model pipeline (loaded in run())
        self.model_pipeline = None

    # ──────────────────────────────────────────────────────────────────────────
    # GPU inference background task
    # ──────────────────────────────────────────────────────────────────────────
    async def _inference_loop(self):
        logger.info("Inference loop started.")
        while True:
            chunk_floats, chunk_bytes = await self.generation_queue.get()
            self.audio_dq.extend(chunk_floats.tolist())
            audio_array = np.array(self.audio_dq)

            try:
                def _infer():
                    torch.cuda.synchronize()
                    emb = get_audio_embedding(
                        self.model_pipeline, audio_array,
                        self.audio_start_idx, self.audio_end_idx,
                    )
                    emb = emb * ANIMATION_MULTIPLIER
                    video = run_pipeline(self.model_pipeline, emb)
                    torch.cuda.synchronize()
                    return video.cpu().numpy()

                video_np = await asyncio.to_thread(_infer)
                n_frames = video_np.shape[0]
                if n_frames == 0 or len(chunk_bytes) == 0:
                    continue

                bpf = len(chunk_bytes) // n_frames   # bytes per video frame

                for i in range(n_frames):
                    vf = video_np[i]
                    if vf.ndim == 3 and vf.shape[0] == 3:
                        vf = np.transpose(vf, (1, 2, 0))
                    elif vf.ndim == 2:
                        vf = np.stack([vf] * 3, axis=-1)
                    if vf.dtype != np.uint8:
                        vf = np.clip(vf, 0, 255).astype(np.uint8)
                    rgba = cv2.cvtColor(vf, cv2.COLOR_RGB2RGBA) if vf.shape[2] == 3 else vf
                    audio_slice = chunk_bytes[i * bpf : (i + 1) * bpf]
                    self.playback_queue.append((rgba, audio_slice))

            except Exception as e:
                logger.error(f"Inference error: {e}")

    # ──────────────────────────────────────────────────────────────────────────
    # Playback background task — publishes A/V to LiveKit at tgt_fps
    # ──────────────────────────────────────────────────────────────────────────
    async def _playback_loop(self):
        logger.info("Playback loop started.")
        frame_dur = 1.0 / self.tgt_fps

        while True:
            t0 = time.time()
            try:
                if len(self.playback_queue) > PLAYBACK_QUEUE_MAX:
                    logger.warning("Playback queue overflow — dropping stale frames.")
                    while len(self.playback_queue) > 25:
                        self.playback_queue.popleft()

                if self.playback_queue:
                    rgba, audio_bytes = self.playback_queue.popleft()
                    self.video_source.capture_frame(
                        rtc.VideoFrame(
                            rgba.shape[1], rgba.shape[0],
                            rtc.VideoBufferType.RGBA, rgba.tobytes(),
                        )
                    )
                    if audio_bytes:
                        await self.audio_source.capture_frame(
                            rtc.AudioFrame(
                                data=audio_bytes,
                                sample_rate=self.sr,
                                num_channels=1,
                                samples_per_channel=len(audio_bytes) // 2,
                            )
                        )
                else:
                    # Idle — stream static avatar image
                    self.video_source.capture_frame(
                        rtc.VideoFrame(
                            self.idle_rgba.shape[1], self.idle_rgba.shape[0],
                            rtc.VideoBufferType.RGBA, self.idle_rgba.tobytes(),
                        )
                    )
                    await asyncio.sleep(frame_dur)
                    continue

            except Exception as e:
                logger.error(f"Playback error: {e}")

            elapsed = time.time() - t0
            await asyncio.sleep(max(0.0, frame_dur - elapsed))

    # ──────────────────────────────────────────────────────────────────────────
    # Feed TTS PCM audio into the inference pipeline
    # ──────────────────────────────────────────────────────────────────────────
    async def _feed_tts_audio(self, tts_audio_buf: bytearray,
                               float_buf: list,
                               pcm_bytes: bytes):
        """Accumulate pcm_bytes; when we have slice_samples worth, push to GPU queue."""
        tts_audio_buf.extend(pcm_bytes)
        floats = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        float_buf.extend(floats.tolist())

        while len(float_buf) >= self.slice_samples:
            chunk_floats = np.array(float_buf[:self.slice_samples])
            float_buf[:] = float_buf[self.slice_samples:]

            byte_len = self.slice_samples * 2
            chunk_bytes = bytes(tts_audio_buf[:byte_len])
            del tts_audio_buf[:byte_len]

            await self.generation_queue.put((chunk_floats, chunk_bytes))

    async def _flush_tts_audio(self, tts_audio_buf: bytearray, float_buf: list):
        """Zero-pad and flush any remaining audio so the last animation plays."""
        if not float_buf:
            return
        # pad to a full slice
        needed = self.slice_samples - len(float_buf)
        float_buf.extend([0.0] * needed)
        tts_audio_buf.extend(bytes(needed * 2))
        await self._feed_tts_audio(tts_audio_buf, float_buf, b"")

    # ──────────────────────────────────────────────────────────────────────────
    # LLM + TTS — called once a final transcript arrives
    # ──────────────────────────────────────────────────────────────────────────
    async def _respond(self, transcript: str):
        if not transcript.strip():
            return

        logger.info(f"User: {transcript!r}")
        self._bot_speaking = True

        self.conversation.append({"role": "user", "content": transcript})

        # ── LLM streaming ─────────────────────────────────────────────────────
        full_response = ""
        sentence_buf  = ""

        tts_audio_buf: bytearray = bytearray()
        float_buf:     list      = []

        try:
            stream = await self.openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": SYSTEM_PROMPT}] + self.conversation,
                stream=True,
                temperature=0.7,
            )

            async for chunk in stream:
                delta = chunk.choices[0].delta.content or ""
                if not delta:
                    continue
                full_response += delta
                sentence_buf  += delta
                logger.debug(f"LLM token: {delta!r}")

                # Send to TTS whenever we hit a sentence boundary
                parts = _SENTENCE_END.split(sentence_buf)
                if len(parts) > 1:
                    # All but the last part are complete sentences
                    for sentence in parts[:-1]:
                        sentence = sentence.strip()
                        if sentence:
                            logger.info(f"TTS sentence: {sentence!r}")
                            async for pcm in self.tts.synthesize(sentence):
                                await self._feed_tts_audio(tts_audio_buf, float_buf, pcm)
                    sentence_buf = parts[-1]

            # Flush the remaining partial sentence
            if sentence_buf.strip():
                logger.info(f"TTS flush: {sentence_buf.strip()!r}")
                async for pcm in self.tts.synthesize(sentence_buf.strip()):
                    await self._feed_tts_audio(tts_audio_buf, float_buf, pcm)

        except Exception as e:
            logger.error(f"LLM/TTS error: {e}")
        finally:
            # Flush remaining audio to inference pipeline
            await self._flush_tts_audio(tts_audio_buf, float_buf)

            self.conversation.append({"role": "assistant", "content": full_response})
            logger.info(f"Assistant: {full_response!r}")

            # Wait for the playback queue to drain before accepting new input
            logger.debug("Waiting for playback queue to drain…")
            while self.playback_queue or not self.generation_queue.empty():
                await asyncio.sleep(0.1)

            self._bot_speaking = False
            logger.info("Bot finished speaking — ready for next user input.")

    # ──────────────────────────────────────────────────────────────────────────
    # Audio input handler — VAD gating → STT → LLM+TTS
    # ──────────────────────────────────────────────────────────────────────────
    async def _handle_audio_stream(self, audio_stream: rtc.AudioStream):
        """Consume a participant's audio stream with VAD + STT.

        Strategy: buffer all PCM during the utterance; when VAD detects
        end-of-speech open a *fresh* STT WebSocket for that utterance only.
        This avoids server-side idle-timeouts that killed persistent connections.
        """
        logger.info("Audio stream handler started.")

        # Silero v5 requires exactly 512 samples (1024 bytes) per inference call.
        VAD_FRAME_BYTES = 512 * 2

        speech_start_ms: float = 0.0
        last_speech_ms:  float = 0.0
        is_speaking:     bool  = False
        vad_buf          = bytearray()
        utterance_buf    = bytearray()   # accumulates raw PCM for the whole utterance

        async for audio_event in audio_stream:
            frame: rtc.AudioFrame = audio_event.frame
            pcm = bytes(frame.data)

            # Drop audio while the bot is talking (prevents STT echo loop)
            if self._bot_speaking:
                continue

            vad_buf.extend(pcm)

            while len(vad_buf) >= VAD_FRAME_BYTES:
                now_ms    = time.time() * 1000
                vad_chunk = bytes(vad_buf[:VAD_FRAME_BYTES])
                vad_buf   = vad_buf[VAD_FRAME_BYTES:]

                prob = self._vad.is_speech(vad_chunk)
                logger.trace(f"VAD prob={prob:.2f}")

                if prob >= VAD_THRESHOLD:
                    last_speech_ms = now_ms
                    if not is_speaking:
                        is_speaking     = True
                        speech_start_ms = now_ms
                        utterance_buf   = bytearray()   # start fresh buffer
                        logger.info("VAD: speech started")

                # Buffer VAD chunks while user is speaking
                if is_speaking:
                    utterance_buf.extend(vad_chunk)

                    silence_ms = now_ms - last_speech_ms
                    if silence_ms >= VAD_SPEECH_HOLD_MS:
                        duration_ms = now_ms - speech_start_ms
                        is_speaking = False
                        self._vad.reset()
                        logger.info(f"VAD: speech ended ({duration_ms:.0f}ms)")

                        if duration_ms < VAD_MIN_SPEECH_MS:
                            logger.info("VAD: utterance too short — ignoring")
                            utterance_buf = bytearray()
                            continue

                        # Transcribe the buffered utterance with a fresh connection
                        logger.info(f"STT: transcribing {len(utterance_buf)} bytes…")
                        captured = bytes(utterance_buf)
                        utterance_buf = bytearray()

                        transcript = await self.stt.transcribe_utterance(captured)
                        if transcript:
                            logger.info(f"Triggering LLM for: {transcript!r}")
                            asyncio.create_task(self._respond(transcript))
                        else:
                            logger.warning("STT: no final transcript received")

    # ──────────────────────────────────────────────────────────────────────────
    # Greet the first participant
    # ──────────────────────────────────────────────────────────────────────────
    async def _greet(self):
        await asyncio.sleep(0.5)   # brief pause so tracks are published
        logger.info("Sending greeting…")
        await self._respond("Greet the user warmly and briefly in English hindi(devanagri script for hindi strictly) mix, then ask sarcastically")

    # ──────────────────────────────────────────────────────────────────────────
    # Main entry point
    # ──────────────────────────────────────────────────────────────────────────
    async def run(self):
        # ── Load FlashHead model ───────────────────────────────────────────────
        logger.info("Loading SoulX FlashHead model into VRAM…")
        self.model_pipeline = get_pipeline(
            world_size=1,
            ckpt_dir=self.model_ckpt_dir,
            wav2vec_dir=self.wav2vec_dir,
            model_type="lite",
        )
        get_base_data(
            self.model_pipeline,
            cond_image_path_or_dir=self.avatar_img,
            base_seed=42,
            use_face_crop=False,
        )

        # ── GPU pre-warm ───────────────────────────────────────────────────────
        logger.info("Pre-warming GPU (~30-40s)…")
        dummy = np.zeros(self.cached_len, dtype=np.float32)
        torch.cuda.synchronize()
        dummy_emb = get_audio_embedding(
            self.model_pipeline, dummy,
            self.audio_start_idx, self.audio_end_idx,
        )
        run_pipeline(self.model_pipeline, dummy_emb)
        torch.cuda.synchronize()
        logger.info("GPU ready.")

        # ── Connect TTS (persistent; STT uses connect-per-utterance) ────────────
        await self.tts.connect()

        # ── Connect to LiveKit room ────────────────────────────────────────────
        token = (
            api.AccessToken(self.lk_key, self.lk_secret)
            .with_identity("soulx-video-bot")
            .with_name("SoulX Avatar")
            .with_grants(api.VideoGrants(
                room_join=True, room=self.room_name,
                can_publish=True, can_subscribe=True, can_publish_data=True,
            ))
            .to_jwt()
        )

        logger.info(f"Connecting to LiveKit room {self.room_name!r}…")

        # Event: first participant joins → greet + start audio pipeline
        first_participant_event = asyncio.Event()

        @self.room.on("track_subscribed")
        def on_track_subscribed(track, publication, participant):
            if track.kind == rtc.TrackKind.KIND_AUDIO:
                logger.info(f"Audio track subscribed from {participant.identity!r}")
                audio_stream = rtc.AudioStream(
                    track,
                    sample_rate=SAMPLE_RATE,
                    num_channels=CHANNELS,
                )
                asyncio.ensure_future(self._handle_audio_stream(audio_stream))
                if not first_participant_event.is_set():
                    first_participant_event.set()

        await self.room.connect(
            self.lk_url,
            token,
            options=rtc.RoomOptions(auto_subscribe=True),
        )
        logger.info("Room connected.")

        # ── Publish bot's own A/V tracks ──────────────────────────────────────
        await self.room.local_participant.publish_track(
            self.video_track,
            rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_CAMERA),
        )
        await self.room.local_participant.publish_track(
            self.audio_track,
            rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE),
        )
        logger.info("Bot A/V tracks published.")

        # ── Start background loops ─────────────────────────────────────────────
        asyncio.create_task(self._inference_loop())
        asyncio.create_task(self._playback_loop())

        # ── Wait for first participant then greet ──────────────────────────────
        logger.info("Waiting for first participant…")
        await first_participant_event.wait()
        asyncio.create_task(self._greet())

        # ── Keep alive ────────────────────────────────────────────────────────
        try:
            while True:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            pass
        finally:
            logger.info("Shutting down…")
            await self.tts.disconnect()
            await self.room.disconnect()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
async def main():
    bot = AvatarBot()
    await bot.run()


if __name__ == "__main__":
    logger.remove(0)
    logger.add(sys.stderr, level="DEBUG")
    asyncio.run(main())
