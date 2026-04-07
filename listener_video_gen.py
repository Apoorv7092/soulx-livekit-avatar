"""
listener_video_gen.py — Generate a 20-second "intently listening" idle video.

Takes an avatar image, feeds silence (or a reference audio file) through the
FlashHead model, and writes an MP4 suitable for use as IDLE_VIDEO in
video_bot_dev.py.

Silence audio → zero lip movement → natural listening/idle expression.
Reference audio → lip/head motion driven by that audio → no audio in output.

Usage
─────
  python listener_video_gen.py --image examples/girl.png
  python listener_video_gen.py --image examples/girl.png --out idle_listen.mp4 --duration 20
  python listener_video_gen.py --image examples/girl.png --audio ref.wav
  python listener_video_gen.py --image examples/girl.png --audio ref.mp3 --out driven.mp4

Args
────
  --image     Path to avatar image (required)
  --out       Output MP4 path (default: idle_listen.mp4)
  --duration  Video length in seconds (default: 20)
  --audio     Optional reference audio file (MP3/WAV) to drive head/lip motion
              instead of silence. Output video will have no audio track.
  --model     "lite" or "pro" (default: lite)
  --ckpt      Model checkpoint dir (default: models/SoulX-FlashHead-1_3B)
  --wav2vec   Wav2Vec model dir   (default: models/hindi_base_wav2vec2)
"""

import argparse
import collections
import os
import subprocess
import sys

import cv2
import numpy as np
import torch
from loguru import logger

from flash_head.inference import (
    get_audio_embedding,
    get_base_data,
    get_infer_params,
    get_pipeline,
    run_pipeline,
)


# ─────────────────────────────────────────────────────────────────────────────
# Args
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Generate idle listening video via FlashHead")
    p.add_argument("--image",    required=True,                          help="Avatar image path")
    p.add_argument("--out",      default="idle_listen.mp4",              help="Output MP4 path")
    p.add_argument("--duration", type=float, default=20.0,               help="Video duration in seconds")
    p.add_argument("--audio",    default=None,                           help="Reference audio file (MP3/WAV) to drive motion instead of silence")
    p.add_argument("--model",    default="lite", choices=["lite", "pro"],help="Model variant")
    p.add_argument("--ckpt",     default="models/SoulX-FlashHead-1_3B", help="Checkpoint directory")
    p.add_argument("--wav2vec",  default="models/hindi_base_wav2vec2",  help="Wav2Vec model directory")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    if not os.path.isfile(args.image):
        logger.error(f"Image not found: {args.image!r}")
        sys.exit(1)

    if args.audio is not None and not os.path.isfile(args.audio):
        logger.error(f"Audio file not found: {args.audio!r}")
        sys.exit(1)

    # ── Load model ────────────────────────────────────────────────────────────
    logger.info(f"Loading FlashHead ({args.model}) …")
    pipeline = get_pipeline(
        world_size=1,
        ckpt_dir=args.ckpt,
        wav2vec_dir=args.wav2vec,
        model_type=args.model,
    )
    get_base_data(
        pipeline,
        cond_image_path_or_dir=args.image,
        base_seed=42,
        use_face_crop=False,
    )

    # ── Inference params ──────────────────────────────────────────────────────
    ip               = get_infer_params()
    sr               = ip["sample_rate"]        # 16000
    tgt_fps          = ip["tgt_fps"]            # 25
    cad              = ip["cached_audio_duration"]  # 8  (seconds)
    frame_num        = ip["frame_num"]          # 33  (frames per inference call)
    motion_frames_num = 2                       # overlap frames between chunks
    slice_len        = frame_num - motion_frames_num   # 31 new frames per chunk
    slice_samples    = slice_len * sr // tgt_fps       # 19840 samples per chunk

    cached_len      = sr * cad                         # 128000 samples (8s window)
    audio_end_idx   = cad * tgt_fps                    # 200
    audio_start_idx = audio_end_idx - frame_num        # 167

    # ── Targets ───────────────────────────────────────────────────────────────
    total_frames_needed = int(args.duration * tgt_fps)   # e.g. 500 for 20s
    # Each chunk yields frame_num raw frames; we keep slice_len after the first.
    # First chunk keeps all frame_num frames (no previous context to overlap).
    chunks_needed = max(1, -(-total_frames_needed // slice_len))  # ceil div

    logger.info(
        f"Generating {args.duration}s × {tgt_fps}fps = {total_frames_needed} frames "
        f"across {chunks_needed} inference chunks …"
    )

    # ── Load reference audio (if provided) ───────────────────────────────────
    ref_audio: np.ndarray | None = None
    if args.audio is not None:
        try:
            import librosa
        except ImportError:
            logger.error("librosa is required for --audio support: pip install librosa")
            sys.exit(1)

        logger.info(f"Loading reference audio: {args.audio!r}")
        ref_audio, _ = librosa.load(args.audio, sr=sr, mono=True)
        logger.info(f"  {len(ref_audio)} samples @ {sr} Hz = {len(ref_audio) / sr:.1f}s")

        total_samples_needed = chunks_needed * slice_samples
        if len(ref_audio) < total_samples_needed:
            logger.info(
                f"  Audio shorter than video target — padding "
                f"{total_samples_needed - len(ref_audio)} samples with silence"
            )
            ref_audio = np.pad(ref_audio, (0, total_samples_needed - len(ref_audio)))
        else:
            ref_audio = ref_audio[:total_samples_needed]
    else:
        logger.info("No reference audio provided — using silence for idle motion")

    # ── GPU pre-warm ──────────────────────────────────────────────────────────
    logger.info("Pre-warming GPU …")
    dummy = np.zeros(cached_len, dtype=np.float32)
    torch.cuda.synchronize()
    dummy_emb = get_audio_embedding(pipeline, dummy, audio_start_idx, audio_end_idx)
    run_pipeline(pipeline, dummy_emb)
    torch.cuda.synchronize()
    logger.info("GPU ready.")

    # ── Rolling audio deque (silence or reference audio) ─────────────────────
    audio_dq = collections.deque([0.0] * cached_len, maxlen=cached_len)

    # ── Inference loop ────────────────────────────────────────────────────────
    all_frames: list[np.ndarray] = []   # list of (512, 512, 3) uint8 BGR frames

    for chunk_idx in range(chunks_needed):
        if ref_audio is not None:
            start = chunk_idx * slice_samples
            chunk = ref_audio[start : start + slice_samples]
            if len(chunk) < slice_samples:
                chunk = np.pad(chunk, (0, slice_samples - len(chunk)))
            audio_dq.extend(chunk.astype(np.float32))
        else:
            audio_dq.extend([0.0] * slice_samples)

        audio_array = np.array(audio_dq, dtype=np.float32)

        torch.cuda.synchronize()
        emb = get_audio_embedding(pipeline, audio_array, audio_start_idx, audio_end_idx)
        video_np = run_pipeline(pipeline, emb)   # (frame_num, H, W, C) uint8 tensor
        torch.cuda.synchronize()

        frames = video_np.cpu().numpy()   # shape: (frame_num, H, W, 3) or (frame_num, 3, H, W)

        for i in range(frames.shape[0]):
            vf = frames[i]

            # Handle (C, H, W) → (H, W, C)
            if vf.ndim == 3 and vf.shape[0] == 3:
                vf = np.transpose(vf, (1, 2, 0))
            elif vf.ndim == 2:
                vf = np.stack([vf] * 3, axis=-1)

            if vf.dtype != np.uint8:
                vf = np.clip(vf, 0, 255).astype(np.uint8)

            # Ensure 512×512
            if vf.shape[:2] != (512, 512):
                vf = cv2.resize(vf, (512, 512))

            # FlashHead outputs RGB → convert to BGR for cv2
            bgr = cv2.cvtColor(vf, cv2.COLOR_RGB2BGR)
            all_frames.append(bgr)

        collected = len(all_frames)
        logger.info(
            f"  Chunk {chunk_idx + 1}/{chunks_needed} → "
            f"{frames.shape[0]} frames  (total: {collected})"
        )

        if collected >= total_frames_needed:
            break

    # Trim to exact target length
    all_frames = all_frames[:total_frames_needed]
    logger.info(f"Collected {len(all_frames)} frames — writing MP4 …")

    # ── Write MP4 ─────────────────────────────────────────────────────────────
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.out, fourcc, float(tgt_fps), (512, 512))

    if not writer.isOpened():
        logger.error(f"Could not open VideoWriter for {args.out!r}")
        sys.exit(1)

    for frame in all_frames:
        writer.write(frame)

    writer.release()
    logger.info(f"Done — saved to {args.out!r}  ({len(all_frames)} frames @ {tgt_fps}fps)")

    # ── Strip audio track when reference audio was used ───────────────────────
    if args.audio is not None:
        logger.info("Stripping audio track from output video (ffmpeg) …")
        tmp = args.out + "._noaudio_tmp.mp4"
        os.rename(args.out, tmp)
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-i", tmp, "-an", "-c:v", "copy", args.out],
                check=True,
                capture_output=True,
            )
            logger.info("Audio track removed — output is video-only.")
        except subprocess.CalledProcessError as e:
            logger.warning(f"ffmpeg strip failed: {e.stderr.decode().strip()}")
            logger.warning("Keeping original file (may contain audio metadata).")
            os.rename(tmp, args.out)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

    logger.info(f"Set  IDLE_VIDEO={args.out}  in your .env to use this in video_bot_dev.py")


if __name__ == "__main__":
    logger.remove(0)
    logger.add(sys.stderr, level="INFO")
    main()
