"""
listener_video_gen.py — Generate a 20-second "intently listening" idle video.

Takes an avatar image, feeds silence through the FlashHead model, and writes
an MP4 suitable for use as IDLE_VIDEO in video_bot_dev.py.

Silence audio → zero lip movement → natural listening/idle expression.

Usage
─────
  python listener_video_gen.py --image examples/girl.png
  python listener_video_gen.py --image examples/girl.png --out idle_listen.mp4 --duration 20

Args
────
  --image     Path to avatar image (required)
  --out       Output MP4 path (default: idle_listen.mp4)
  --duration  Video length in seconds (default: 20)
  --model     "lite" or "pro" (default: lite)
  --ckpt      Model checkpoint dir (default: models/SoulX-FlashHead-1_3B)
  --wav2vec   Wav2Vec model dir   (default: models/hindi_base_wav2vec2)
"""

import argparse
import collections
import os
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
    slice_samples    = slice_len * sr // tgt_fps       # 19840 silence samples per chunk

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

    # ── GPU pre-warm ──────────────────────────────────────────────────────────
    logger.info("Pre-warming GPU …")
    dummy = np.zeros(cached_len, dtype=np.float32)
    torch.cuda.synchronize()
    dummy_emb = get_audio_embedding(pipeline, dummy, audio_start_idx, audio_end_idx)
    run_pipeline(pipeline, dummy_emb)
    torch.cuda.synchronize()
    logger.info("GPU ready.")

    # ── Rolling silence audio deque (same as AvatarBot) ──────────────────────
    audio_dq = collections.deque([0.0] * cached_len, maxlen=cached_len)

    # ── Inference loop ────────────────────────────────────────────────────────
    all_frames: list[np.ndarray] = []   # list of (512, 512, 3) uint8 BGR frames

    for chunk_idx in range(chunks_needed):
        # Feed one slice of silence into the rolling window
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
    logger.info(f"Set  IDLE_VIDEO={args.out}  in your .env to use this in video_bot_dev.py")


if __name__ == "__main__":
    logger.remove(0)
    logger.add(sys.stderr, level="INFO")
    main()
