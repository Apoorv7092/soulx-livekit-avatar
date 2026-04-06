"""
video_bot_dev.py — Development version of the talking-avatar bot.

New features vs video_bot_stable.py
────────────────────────────────────
  1. Dual TTS providers — Ori TTS (default) *or* ElevenLabs (lowest-latency
     Flash model over WebSocket).  Select via TTS_PROVIDER env var.

  2. Idle head-motion video loop — instead of a freeze-frame image, the bot
     cycles through a short head-motion MP4 while not speaking, and switches
     back to FlashHead inference frames the moment speech starts.  Set
     IDLE_VIDEO to a path; falls back to static image if unset.

  3. Local LiveKit Meet client — run `serve_meet.py` (same directory) to start
     a local token server + static file server on port 8080.  Open the
     browser at http://localhost:8080 to join the room without sending any
     video data to meet.livekit.io.

Flow
────
  LiveKit audio-in (participant mic)
    → VAD (silero, in-process)
    → Ori STT  (REST, per-utterance WAV POST)
    → OpenAI LLM  (chat completions stream)
    → TTS  (Ori WebSocket  OR  ElevenLabs WebSocket, streaming)
    → SoulX FlashHead GPU inference  (audio → lip-synced video frames)
    → LiveKit A/V publish (own rtc tracks)

Required env vars (.env in same directory):
  LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET
  ORI_STT_API_KEY, ORI_STT_BASE_URL
  ORI_TTS_API_KEY, ORI_TTS_WS_URL, ORI_TTS_VOICE_ID
  ORI_STT_LANGUAGE                    (default: en)
  ORI_TTS_LANGUAGE                    (default: hi)
  OPENAI_API_KEY
  LIVEKIT_ROOM                        (default: soulx-flashhead-room)
  AVATAR_IMAGE                        (default: examples/girl.png)
  MODEL_CKPT_DIR                      (default: models/SoulX-FlashHead-1_3B)
  WAV2VEC_DIR                         (default: models/hindi_base_wav2vec2)

New / dev-only env vars:
  TTS_PROVIDER          "ori" (default) | "elevenlabs"
  ELEVENLABS_API_KEY    ElevenLabs API key (required when TTS_PROVIDER=elevenlabs)
  ELEVENLABS_VOICE_ID   ElevenLabs voice ID
  ELEVENLABS_MODEL_ID   Model to use (default: eleven_flash_v2_5)
  IDLE_VIDEO            Path to head-motion MP4 for idle state (optional)
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
SAMPLE_RATE = 16_000          # Hz — must match FlashHead model
CHANNELS    = 1
CHUNK_MS    = 20              # ms per STT audio chunk
CHUNK_SAMPLES = SAMPLE_RATE * CHUNK_MS // 1000   # 320 samples
CHUNK_BYTES   = CHUNK_SAMPLES * 2                # 640 bytes (int16)

VAD_WINDOW_MS      = 32       # ms for Silero VAD analysis window
VAD_THRESHOLD      = 0.5
VAD_SPEECH_HOLD_MS = 400
VAD_MIN_SPEECH_MS  = 200

TTS_ENCODING        = "pcm_16000"
TTS_TAIL_SILENCE_MS = 800   # ms of silence appended after every TTS response so the
                             # video animation winds down gracefully before idle kicks in
TTS_CHUNK_BYTES = SAMPLE_RATE * 2   # 1-second chunks fed to FlashHead

ANIMATION_MULTIPLIER = 1.5
PLAYBACK_QUEUE_MAX   = 999

SYSTEM_PROMPT = """You are a helpful voiceAI named ALI who speaks in eng. If the user speaks in hindi roman-scripture, you always reply in hindi devanagri script only. Strictly refrain from responding in hindi latin scripts like "kya", "aap" "mein" etc. ONLY RESPLY IN DEVANAGRI IF THE USER IS SPEAKING IN HINDI ROMANIZED ELSE IN ENGLISH NO TRANSLITERATED RESPONSES FROM YOUR SIDE OR YOU WILL BE FIRED FROM YOUR JOB. strictly no emojis as this is a voice conversation. STRICTLY REFRAIN FROM GENERATING Exclaimation marks, only generate full stop as a sentence ender, not even comma."""
_SENTENCE_END = re.compile(r'(?<=[.!?,;])\s+')

# Max idle video frames cached in RAM (~20s at 30fps = 600 frames).
_IDLE_MAX_FRAMES = 600


# ─────────────────────────────────────────────────────────────────────────────
# Silero VAD wrapper
# ─────────────────────────────────────────────────────────────────────────────
class SileroVAD:
    """Thin wrapper around the Silero VAD torch model."""

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
        samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        tensor = torch.from_numpy(samples).unsqueeze(0)
        prob = self._model(tensor, SAMPLE_RATE)
        return float(prob)


# ─────────────────────────────────────────────────────────────────────────────
# STT client — Ori STT REST API
# ─────────────────────────────────────────────────────────────────────────────
class STTClient:
    """Transcribes utterances via the Ori STT REST API (per-utterance WAV POST)."""

    def __init__(self, api_key: str, base_url: str, language: str = "hi",
                 model: str = "ori-prime-v2.3"):
        self._api_key  = api_key
        self._api_url  = "https://ori-stt-test.oriserve.com/openai/v1/audio/transcriptions"
        self._language = language
        self._model    = model

    @staticmethod
    def _pcm_to_wav(pcm_bytes: bytes, sample_rate: int = SAMPLE_RATE,
                    channels: int = 1, sampwidth: int = 2) -> bytes:
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(sampwidth)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm_bytes)
        return buf.getvalue()

    def _post(self, wav_bytes: bytes) -> str:
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
        logger.info(f"STT: transcribing {len(pcm_bytes)} bytes via REST…")
        wav = self._pcm_to_wav(pcm_bytes)
        try:
            text = await asyncio.to_thread(self._post, wav)
            if text=='nan': #noisy/unintelligible chunk transcript
                logger.info(f"STT transcript: {text!r}, skipping...")
                return ""
                
            logger.info(f"STT transcript: {text!r}")
            return text
        except Exception as e:
            logger.error(f"STT error: {e}")
            return ""


# ─────────────────────────────────────────────────────────────────────────────
# TTS client — Ori TTS WebSocket (unchanged from stable)
# ─────────────────────────────────────────────────────────────────────────────
class OriTTSClient:
    """Sends text to Ori TTS and yields PCM bytes as they arrive.

    Kept identical to stable TTSClient; renamed to OriTTSClient to coexist
    with ElevenLabsTTSClient under a unified interface.
    """

    def __init__(self, api_key: str, ws_url: str, voice_id: str,
                 language: str = "hi"):
        self._api_key  = api_key
        self._ws_url   = ws_url
        self._voice_id = voice_id
        self._language = language
        self._ws       = None

    async def connect(self):
        logger.info(f"Ori TTS connecting to {self._ws_url}")
        self._ws = await websockets.connect(
            self._ws_url,
            additional_headers={"Authorization": f"Bearer {self._api_key}"},
            max_size=16 * 1024 * 1024,
        )
        logger.info("Ori TTS connected.")

    async def disconnect(self):
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
        self._ws = None

    async def synthesize(self, text: str):
        """Send `text`; async-yield decoded PCM bytes per chunk."""
        if not self._ws or not text.strip():
            return

        req = {
            "text":        text,
            "language":    self._language,
            "voice_id":    self._voice_id,
            "encoding":    TTS_ENCODING,
            "speechReqId": str(uuid.uuid4()),
        }
        logger.debug(f"Ori TTS synthesize: {text!r}")
        try:
            await self._ws.send(json.dumps(req))
        except Exception as e:
            logger.error(f"Ori TTS send error: {e}")
            return

        try:
            async for raw in self._ws:
                msg      = json.loads(raw)
                chunks   = msg.get("audio_chunks", [])
                complete = msg.get("audio_streaming_complete", False)

                for b64 in chunks:
                    audio = base64.b64decode(b64)
                    # Skip 4092-byte non-audio metadata packets
                    if len(audio) == 4092:
                        continue
                    yield audio

                if complete:
                    break
        except Exception as e:
            logger.error(f"Ori TTS recv error: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# TTS client — ElevenLabs WebSocket  (NEW)
# ─────────────────────────────────────────────────────────────────────────────
class ElevenLabsTTSClient:
    """Streams text to ElevenLabs and yields raw PCM-16 bytes at 16 kHz.

    Uses the ElevenLabs text-to-speech WebSocket streaming API with:
      • eleven_flash_v2_5 model  (~75ms TTFB, lowest latency)
      • pcm_16000 output format  (matches FlashHead SAMPLE_RATE exactly)
      • optimize_streaming_latency=4  (maximum latency reduction)
      • flush=true per sentence   (forces generation without buffering)

    The WebSocket is kept alive for the session; sentences are streamed one
    by one with flush=true.  The connection is re-established automatically
    if it drops between turns.
    """

    _WS_TEMPLATE = (
        "wss://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream-input"
        "?model_id={model_id}&output_format=pcm_16000&optimize_streaming_latency=4"
    )

    def __init__(self, api_key: str, voice_id: str,
                 model_id: str = "eleven_flash_v2_5"):
        self._api_key  = api_key
        self._voice_id = voice_id
        self._model_id = model_id
        self._ws       = None

    # ── internal helpers ──────────────────────────────────────────────────────

    @property
    def _ws_url(self) -> str:
        return self._WS_TEMPLATE.format(
            voice_id=self._voice_id,
            model_id=self._model_id,
        )

    async def _open_ws(self):
        """Open a fresh WebSocket and send the ElevenLabs init handshake."""
        logger.info(f"ElevenLabs TTS connecting…")
        ws = await websockets.connect(
            self._ws_url,
            additional_headers={"xi-api-key": self._api_key},
            max_size=16 * 1024 * 1024,
        )
        # ElevenLabs requires an init message with a single space before any text.
        await ws.send(json.dumps({
            "text": " ",
            "voice_settings": {
                "stability":        0.5,
                "similarity_boost": 0.8,
                "style":            0,
                "use_speaker_boost": False,
            },
            "generation_config": {
                # Small schedule so audio starts quickly; flush=true bypasses it anyway.
                "chunk_length_schedule": [50, 100, 150, 200],
            },
        }))
        logger.info("ElevenLabs TTS connected.")
        return ws

    async def _ensure_connected(self):
        """Re-establish the WebSocket if it has been closed."""
        if self._ws is None or self._ws.closed:
            self._ws = await self._open_ws()

    # ── public interface (same as OriTTSClient) ───────────────────────────────

    async def connect(self):
        self._ws = await self._open_ws()

    async def disconnect(self):
        if self._ws:
            try:
                # Graceful close: empty text signals end-of-stream.
                await self._ws.send(json.dumps({"text": ""}))
                await self._ws.close()
            except Exception:
                pass
        self._ws = None

    async def synthesize(self, text: str):
        """Send one sentence; async-yield raw PCM-16 bytes as they arrive."""
        if not text.strip():
            return

        await self._ensure_connected()

        # ElevenLabs requires text to end with a trailing space.
        payload = {"text": text.rstrip() + " ", "flush": True}
        logger.debug(f"ElevenLabs TTS synthesize: {text!r}")

        try:
            await self._ws.send(json.dumps(payload))
        except Exception as e:
            logger.error(f"ElevenLabs TTS send error: {e} — reconnecting")
            # Reconnect and retry once.
            self._ws = None
            await self._ensure_connected()
            await self._ws.send(json.dumps(payload))

        try:
            async for raw in self._ws:
                data     = json.loads(raw)
                audio_b64 = data.get("audio")
                is_final  = data.get("isFinal", False)

                if audio_b64:
                    # Decoded bytes are raw S16LE PCM at 16 kHz — no further processing needed.
                    yield base64.b64decode(audio_b64)

                if is_final:
                    break
        except Exception as e:
            logger.error(f"ElevenLabs TTS recv error: {e}")
            self._ws = None   # mark for reconnect on next call


# ─────────────────────────────────────────────────────────────────────────────
# Avatar bot — ties everything together
# ─────────────────────────────────────────────────────────────────────────────
class AvatarBot:
    def __init__(self):
        # ── env ──────────────────────────────────────────────────────────────
        self.lk_url    = os.environ["LIVEKIT_URL"]
        self.lk_key    = os.environ["LIVEKIT_API_KEY"]
        self.lk_secret = os.environ["LIVEKIT_API_SECRET"]
        self.room_name = os.getenv("LIVEKIT_ROOM", "soulx-flashhead-room")

        self.stt_key  = os.environ["ORI_STT_API_KEY"]
        self.stt_url  = os.environ["ORI_STT_BASE_URL"]
        self.stt_lang = os.getenv("ORI_STT_LANGUAGE", "en")

        self.tts_provider = os.getenv("TTS_PROVIDER", "ori").lower()

        self.openai_key     = os.environ["OPENAI_API_KEY"]
        self.avatar_img     = os.getenv("AVATAR_IMAGE",   "examples/girl2.png")
        self.model_ckpt_dir = os.getenv("MODEL_CKPT_DIR", "models/SoulX-FlashHead-1_3B")
        self.wav2vec_dir    = os.getenv("WAV2VEC_DIR",    "models/hindi_base_wav2vec2")

        # ── state ─────────────────────────────────────────────────────────────
        self.room   = rtc.Room()
        self.openai = AsyncOpenAI(api_key=self.openai_key)
        self.conversation: list[dict] = []

        # ── TTS client selection ──────────────────────────────────────────────
        if self.tts_provider == "elevenlabs":
            el_key      = os.environ["ELEVENLABS_API_KEY"]
            el_voice    = os.environ["ELEVENLABS_VOICE_ID"]
            el_model    = os.getenv("ELEVENLABS_MODEL_ID", "eleven_flash_v2_5")
            self.tts    = ElevenLabsTTSClient(el_key, el_voice, el_model)
            logger.info(f"TTS provider: ElevenLabs (model={el_model}, voice={el_voice})")
        else:
            tts_key     = os.environ["ORI_TTS_API_KEY"]
            tts_ws_url  = os.environ["ORI_TTS_WS_URL"]
            tts_voice   = os.environ["ORI_TTS_VOICE_ID"]
            tts_lang    = os.getenv("ORI_TTS_LANGUAGE", "hi")
            self.tts    = OriTTSClient(tts_key, tts_ws_url, tts_voice, tts_lang)
            logger.info("TTS provider: Ori TTS")

        # ── STT ───────────────────────────────────────────────────────────────
        self.stt = STTClient(self.stt_key, self.stt_url, self.stt_lang)

        # ── FlashHead inference params ────────────────────────────────────────
        ip = get_infer_params()
        self.sr                = ip["sample_rate"]
        self.tgt_fps           = ip["tgt_fps"]
        self.cad               = ip["cached_audio_duration"]
        self.frame_num         = ip["frame_num"]
        self.motion_frames_num = 2
        self.slice_len         = self.frame_num - self.motion_frames_num

        self.audio_end_idx   = self.cad * self.tgt_fps
        self.audio_start_idx = self.audio_end_idx - self.frame_num
        self.cached_len      = self.sr * self.cad
        self.slice_samples   = self.slice_len * self.sr // self.tgt_fps

        self.audio_dq = collections.deque(
            [0.0] * self.cached_len, maxlen=self.cached_len
        )

        # ── queues ────────────────────────────────────────────────────────────
        self.generation_queue: asyncio.Queue    = asyncio.Queue()
        self.playback_queue: collections.deque  = collections.deque()

        # ── LiveKit A/V sources ───────────────────────────────────────────────
        self.video_source = rtc.VideoSource(512, 512)
        self.video_track  = rtc.LocalVideoTrack.create_video_track(
            "bot-video", self.video_source
        )
        self.audio_source = rtc.AudioSource(self.sr, 1)
        self.audio_track  = rtc.LocalAudioTrack.create_audio_track(
            "bot-audio", self.audio_source
        )

        # ── Static fallback idle frame (RGBA) ─────────────────────────────────
        img = cv2.imread(self.avatar_img)
        if img is None:
            logger.warning(f"Avatar image not found: {self.avatar_img!r} — using black frame")
            img = np.zeros((512, 512, 3), dtype=np.uint8)
        self.idle_rgba = cv2.cvtColor(cv2.resize(img, (512, 512)), cv2.COLOR_BGR2RGBA)

        # ── Idle head-motion video frames  (NEW: feature #2) ─────────────────
        # When IDLE_VIDEO is set, frames are loaded into RAM and cycled
        # in _playback_loop whenever the bot is not speaking.
        # Falls back to the static idle_rgba if the file is missing or unset.
        self._idle_frames: list[np.ndarray] = []   # list of RGBA uint8 arrays
        self._idle_frame_idx: int           = 0

        idle_video_path = os.getenv("IDLE_VIDEO", "")
        if idle_video_path:
            self._load_idle_video(idle_video_path)
        else:
            logger.info("IDLE_VIDEO not set — using static avatar image in idle state.")

        # ── bot-speaking flag ─────────────────────────────────────────────────
        self._bot_speaking = False

        # ── VAD ───────────────────────────────────────────────────────────────
        self._vad = SileroVAD(threshold=VAD_THRESHOLD)

        self.model_pipeline = None

    # ──────────────────────────────────────────────────────────────────────────
    # Idle video helpers  (NEW: feature #2)
    # ──────────────────────────────────────────────────────────────────────────
    def _load_idle_video(self, path: str):
        """Read up to _IDLE_MAX_FRAMES frames from an MP4/video file into RAM."""
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            logger.warning(f"Could not open idle video: {path!r} — using static frame.")
            return

        fps    = cap.get(cv2.CAP_PROP_FPS) or self.tgt_fps
        total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        logger.info(
            f"Loading idle video {path!r}  ({fps:.1f}fps, ~{total} frames, "
            f"capping at {_IDLE_MAX_FRAMES})…"
        )

        count = 0
        while count < _IDLE_MAX_FRAMES:
            ret, frame = cap.read()
            if not ret:
                break
            rgba = cv2.cvtColor(cv2.resize(frame, (512, 512)), cv2.COLOR_BGR2RGBA)
            self._idle_frames.append(rgba)
            count += 1

        cap.release()
        logger.info(f"Idle video ready: {len(self._idle_frames)} frames cached.")

    def _next_idle_frame(self) -> np.ndarray:
        """Return the next RGBA idle frame, cycling the video loop."""
        if not self._idle_frames:
            return self.idle_rgba
        frame = self._idle_frames[self._idle_frame_idx % len(self._idle_frames)]
        self._idle_frame_idx += 1
        return frame

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
                    emb   = emb * ANIMATION_MULTIPLIER
                    video = run_pipeline(self.model_pipeline, emb)
                    torch.cuda.synchronize()
                    return video.cpu().numpy()

                video_np = await asyncio.to_thread(_infer)
                n_frames = video_np.shape[0]
                if n_frames == 0 or len(chunk_bytes) == 0:
                    continue

                bpf = len(chunk_bytes) // n_frames

                for i in range(n_frames):
                    vf = video_np[i]
                    if vf.ndim == 3 and vf.shape[0] == 3:
                        vf = np.transpose(vf, (1, 2, 0))
                    elif vf.ndim == 2:
                        vf = np.stack([vf] * 3, axis=-1)
                    if vf.dtype != np.uint8:
                        vf = np.clip(vf, 0, 255).astype(np.uint8)
                    if vf.shape[:2] != (512, 512):
                        vf = cv2.resize(vf, (512, 512))
                    rgba        = cv2.cvtColor(vf, cv2.COLOR_RGB2RGBA) if vf.shape[2] == 3 else vf
                    audio_slice = chunk_bytes[i * bpf : (i + 1) * bpf]
                    self.playback_queue.append((rgba, audio_slice))

            except Exception as e:
                logger.error(f"Inference error: {e}")

    # ──────────────────────────────────────────────────────────────────────────
    # Playback background task — publishes A/V to LiveKit at tgt_fps
    # ──────────────────────────────────────────────────────────────────────────
    async def _playback_loop(self):
        """Drain playback_queue at tgt_fps.

        NEW (feature #2): when the queue is empty (bot not speaking), cycle
        through the idle head-motion video frames instead of freezing on a
        static image.  When frames arrive from FlashHead, they seamlessly
        replace the idle video with lip-synced content.
        """
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
                    # ── Speaking: publish FlashHead lip-sync frame ─────────────
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
                    # ── Idle: cycle head-motion video (or static frame) ────────
                    idle_frame = self._next_idle_frame()
                    self.video_source.capture_frame(
                        rtc.VideoFrame(
                            idle_frame.shape[1], idle_frame.shape[0],
                            rtc.VideoBufferType.RGBA, idle_frame.tobytes(),
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
        tts_audio_buf.extend(pcm_bytes)
        floats = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        float_buf.extend(floats.tolist())

        while len(float_buf) >= self.slice_samples:
            chunk_floats = np.array(float_buf[:self.slice_samples])
            float_buf[:] = float_buf[self.slice_samples:]

            byte_len    = self.slice_samples * 2
            chunk_bytes = bytes(tts_audio_buf[:byte_len])
            del tts_audio_buf[:byte_len]

            await self.generation_queue.put((chunk_floats, chunk_bytes))

    async def _flush_tts_audio(self, tts_audio_buf: bytearray, float_buf: list):
        if not float_buf:
            return
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

                parts = _SENTENCE_END.split(sentence_buf)
                if len(parts) > 1:
                    for sentence in parts[:-1]:
                        sentence = sentence.strip()
                        if sentence:
                            logger.info(f"TTS sentence: {sentence!r}")
                            async for pcm in self.tts.synthesize(sentence):
                                await self._feed_tts_audio(tts_audio_buf, float_buf, pcm)
                    sentence_buf = parts[-1]

            if sentence_buf.strip():
                logger.info(f"TTS flush: {sentence_buf.strip()!r}")
                async for pcm in self.tts.synthesize(sentence_buf.strip()):
                    await self._feed_tts_audio(tts_audio_buf, float_buf, pcm)

        except Exception as e:
            logger.error(f"LLM/TTS error: {e}")
        finally:
            await self._flush_tts_audio(tts_audio_buf, float_buf)

            # Tail silence — 800 ms of zeros pushed through the same inference
            # pipeline so FlashHead eases the mouth back to rest before the idle
            # video takes over, rather than snapping abruptly.
            tail_samples              = SAMPLE_RATE * TTS_TAIL_SILENCE_MS // 1000
            tail_buf: bytearray       = bytearray()
            tail_floats: list         = []
            await self._feed_tts_audio(tail_buf, tail_floats, bytes(tail_samples * 2))
            await self._flush_tts_audio(tail_buf, tail_floats)

            self.conversation.append({"role": "assistant", "content": full_response})
            logger.info(f"Assistant: {full_response!r}")

            logger.debug("Waiting for playback queue to drain…")
            while self.playback_queue or not self.generation_queue.empty():
                await asyncio.sleep(0.1)

            self._bot_speaking = False
            logger.info("Bot finished speaking — ready for next user input.")

    # ──────────────────────────────────────────────────────────────────────────
    # Audio input handler — VAD gating → STT → LLM+TTS
    # ──────────────────────────────────────────────────────────────────────────
    async def _handle_audio_stream(self, audio_stream: rtc.AudioStream):
        logger.info("Audio stream handler started.")

        VAD_FRAME_BYTES = 512 * 2

        speech_start_ms: float = 0.0
        last_speech_ms:  float = 0.0
        is_speaking:     bool  = False
        vad_buf          = bytearray()
        utterance_buf    = bytearray()

        async for audio_event in audio_stream:
            frame: rtc.AudioFrame = audio_event.frame
            pcm = bytes(frame.data)

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
                        utterance_buf   = bytearray()
                        logger.info("VAD: speech started")

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

                        logger.info(f"STT: transcribing {len(utterance_buf)} bytes…")
                        captured      = bytes(utterance_buf)
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
        await asyncio.sleep(0.5)
        logger.info("Sending greeting…")
        await self._respond(
            "Greet the user warmly and briefly in English, then ask sarcastically(strictly no emojis)"
        )

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

        # ── Connect TTS ────────────────────────────────────────────────────────
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

        await self.room.local_participant.publish_track(
            self.video_track,
            rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_CAMERA),
        )
        await self.room.local_participant.publish_track(
            self.audio_track,
            rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE),
        )
        logger.info("Bot A/V tracks published.")

        asyncio.create_task(self._inference_loop())
        asyncio.create_task(self._playback_loop())

        logger.info("Waiting for first participant…")
        await first_participant_event.wait()
        asyncio.create_task(self._greet())

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
