#!/usr/bin/env python3
"""LensVoice runtime entry point.

Usage:
    venv/bin/python scripts/run_pipeline.py --mock            # mock (no GPU/webcam)
    venv/bin/python scripts/run_pipeline.py --config my.yaml  # real models
    venv/bin/python scripts/run_pipeline.py --frames 120 --mock --drain

--mock forces camera + models + TTS to their deterministic stubs regardless of
config. A bounded mock camera (--frames N) ends the run cleanly.
"""
from __future__ import annotations

import argparse

from lensvoice.camera.mock_camera import MockCamera
from lensvoice.camera.webcam import WebcamCamera
from lensvoice.config.settings import load_settings
from lensvoice.runtime.pipeline import Pipeline


def main() -> int:
    ap = argparse.ArgumentParser(description="LensVoice runtime")
    ap.add_argument("--config", default=None, help="yaml config (default configs/default.yaml)")
    ap.add_argument("--mock", action="store_true",
                    help="use mock camera/models/TTS regardless of config")
    ap.add_argument("--frames", type=int, default=None,
                    help="run N frames then exit (bounded mock camera; default: forever)")
    ap.add_argument("--drain", action="store_true",
                    help="print everything the TTS stub would have spoken")
    args = ap.parse_args()

    settings = load_settings(args.config)
    mock = args.mock
    camera = (MockCamera(settings.camera, max_frames=args.frames) if (mock or args.frames)
              else WebcamCamera(settings.camera))

    from lensvoice.audio.tts import SilentTTS
    spoken: list[str] = []
    tts = SilentTTS(persist=spoken)
    on_utterance = (lambda level, text: print(f"{level}: {text}")) if args.drain else None

    pipeline = Pipeline(settings, camera, tts=tts, on_utterance=on_utterance)
    pipeline.run()
    if args.drain:
        print("--- spoken ---")
        for s in spoken:
            print(s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())