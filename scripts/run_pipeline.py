#!/usr/bin/env python3
"""LensVoice runtime entry point.

Usage:
    venv/bin/python scripts/run_pipeline.py --mock            # mock (no GPU/webcam)
    venv/bin/python scripts/run_pipeline.py --config my.yaml  # real models
    venv/bin/python scripts/run_pipeline.py --frames 120 --mock --drain

    # Live viewer: Mac webcam, or a video file played at its own speed
    venv/bin/python scripts/run_pipeline.py --config configs/lv640_live.yaml --show
    venv/bin/python scripts/run_pipeline.py --config configs/lv640_live.yaml --show \
        --source video-testing/gemini_generated_video_dab576c6.mp4

--mock forces camera + models + TTS to their deterministic stubs regardless of
config. A bounded mock camera (--frames N) ends the run cleanly.
--source overrides camera.source (an index like 0, or a video file path).
--show opens the viewer window (q/Esc quit, d depth overlay, p pause);
--record out.mp4 also saves the annotated view; --silent disables the voice.
"""
from __future__ import annotations

import argparse
import os

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
    ap.add_argument("--source", default=None,
                    help="camera index (e.g. 0) or video file path; overrides camera.source")
    ap.add_argument("--show", action="store_true",
                    help="open the live viewer window (boxes, distance, depth, subtitles)")
    ap.add_argument("--record", default=None, help="save the annotated view to this .mp4")
    ap.add_argument("--warm-voice", action="store_true",
                    help="slowly pre-fetch the common short sentences in the background (one request at a time, "
                         "yields to live speech, stops if rate limited; ~2,000 characters billed once, cached on disk)")
    ap.add_argument("--sarvam-only", action="store_true",
                    help="never fall back to the system voice (a sentence waits for Sarvam audio however long it takes)")
    ap.add_argument("--silent", action="store_true",
                    help="no voice (no API key needed): the viewer shows each sentence as it is produced")
    ap.add_argument("--no-realtime", action="store_true",
                    help="video files: process every frame as fast as possible instead of at video speed")
    args = ap.parse_args()

    settings = load_settings(args.config)
    if args.source is not None:
        if args.source.isdigit():
            settings.camera.source = int(args.source)
        elif os.path.isfile(args.source):
            settings.camera.source = args.source
        else:
            ap.error(f"--source: not a camera index or an existing file: {args.source}")
    if args.silent:
        settings.tts.provider = "silent"
        settings.audio.async_playback = False
    if args.sarvam_only:
        settings.tts.fallback_local_ms = 0
    mock = args.mock
    camera = (MockCamera(settings.camera, max_frames=args.frames) if (mock or args.frames)
              else WebcamCamera(settings.camera, realtime=not args.no_realtime))

    from lensvoice.audio.tts import SilentTTS, create_tts
    spoken: list[str] = []
    # --mock always uses the silent stub; otherwise tts.provider (silent | sarvam) decides
    tts = SilentTTS(persist=spoken) if mock else create_tts(settings.tts, persist=spoken)

    viewer = None
    if args.show or args.record:
        from lensvoice.display import LiveViewer
        depth_cfg = settings.model("depth")
        depth_label = (depth_cfg.weight.split("/")[-1] if depth_cfg.enabled and depth_cfg.weight
                       else "mock")
        voice_label = settings.tts.provider if not mock else "silent"
        viewer = LiveViewer(settings.spatial, settings.safety, depth_label=depth_label,
                            voice_label=voice_label, show=args.show, record=args.record)

    printer = (lambda level, text: print(f"{level}: {text}")) if args.drain else None
    on_utterance = printer
    if viewer is not None:
        if not mock and settings.tts.provider != "silent":
            tts.on_play = viewer.on_spoken          # subtitle appears when the audio starts
            viewer.lag_fn = lambda: tts.lags[-1] if tts.lags else None
        else:                                       # silent: show the sentence when it is produced
            def on_utterance(level, text, _p=printer, _v=viewer):
                if _p:
                    _p(level, text)
                _v.on_spoken(text, level)

    if (args.warm_voice or settings.tts.warm_on_start) and not mock and settings.tts.provider != "silent":
        from lensvoice.narration.phrases import common_phrases, core_phrases
        pick_set = common_phrases if args.warm_voice else core_phrases
        phrases = pick_set(settings.class_names, settings.narration)
        print(f"warming {len(phrases)} sentences slowly in the background "
              f"(~{sum(len(p) for p in phrases)} characters, billed once; cached on disk afterwards)")
        tts.warm(phrases)

    pipeline = Pipeline(settings, camera, tts=tts, on_utterance=on_utterance, on_frame=viewer)
    if viewer is not None and pipeline.picture is not None:
        viewer.on_summary = pipeline.picture.request_summary       # key 's' in the window
    try:
        pipeline.run()
    finally:
        if viewer is not None:
            viewer.close()
    if pipeline.picture is not None:
        st = pipeline.picture.stats()
        ul = (f"urgent latency (confirmed -> audio start): median {st['urgent_latency_ms_median']:.0f} ms, "
              f"max {st['urgent_latency_ms_max']:.0f} ms over {st['urgent_n']}") if st["urgent_n"] else "no urgent events"
        pa = (f"voice responsiveness (picked -> audio start): median {st['pick_to_audio_ms_median']:.0f} ms, "
              f"p90 {st['pick_to_audio_ms_p90']:.0f} ms, max {st['pick_to_audio_ms_max']:.0f} ms"
              if st["pick_to_audio_ms_median"] is not None else "")
        print(f"narration (pull): candidates {st['candidates']} -> spoken {st['spoken']} "
              f"(summaries {st['summaries']}, urgent interrupts {st['interrupts']}, "
              f"camera-motion frames ignored {st['ego_motion_frames']})\n  {ul}\n  {pa}")
    lags = sorted(getattr(tts, "lags", [])) if pipeline.picture is None else []
    if lags:
        q = lambda f: lags[min(len(lags) - 1, int(f * len(lags)))]
        print(f"voice lag (sentence queued -> audio start) over {len(lags)} sentences: "
              f"median {q(0.5)/1000:.1f}s, p90 {q(0.9)/1000:.1f}s, max {lags[-1]/1000:.1f}s | "
              f"api calls {tts.api_calls}, cache hits {tts.cache_hits}, "
              f"stale dropped {pipeline.audio.stale_dropped}, repeats dropped {pipeline.audio.dupes_dropped}")
        print(f"voice: {len(lags) - tts.fallback_count} sentences in the Sarvam voice, "
              f"{tts.fallback_count} in the system voice (instant fallback) | "
              f"rate-limited (429) responses: {tts.rate_limited} | failed requests: {tts.failures}")
    if args.drain:
        print("--- spoken ---")
        for s in spoken:
            print(s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
