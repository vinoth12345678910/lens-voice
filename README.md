# LensVoice

A laptop-only AI accessibility assistant for blind users. A webcam feeds a
perception stack on the laptop; the assistant **speaks only what the evidence
supports** — no invented distance, speed, intent, or identity.

This repository contains the **runtime pipeline** (`src/lensvoice`), the
**unified 26-class dataset** tooling, and the **training experiments**. The
perception models are trained by the ML owner on the merged dataset; the
runtime is model-agnostic behind clean interfaces.

> Status: runtime skeleton (mock-first — runs with no GPU/weights/webcam).
> Real model adapters are wired; they need trained weights.

---

## 1. Overview

Camera → detect objects (person/car/…) → track them → reason spatially/temporally
→ extract text, signs, depth, scene → detect *changes* → score by priority →
generate sentences → speak (bounded, priority-ordered audio queue).

The 26-class taxonomy and the dataset are shared with the ML owner; the only
place ids live in the runtime is `configs/default.yaml`.

## 2. Quickstart (mock — nothing to download)

```bash
venv/bin/python -m pip install -e .            # editable install + pytest
venv/bin/python -m pytest tests/ -q            # 49 tests, mocks only
venv/bin/python scripts/run_pipeline.py --mock --frames 100 --drain
```

`--drain` prints every sentence the TTS stub would have spoken.
Try `--frames 0` (no `--mock`) for the webcam with still-mocked perception.

## 3. Project structure

```
src/lensvoice/        runtime package (perception + reasoning + narration + audio)
configs/default.yaml  the ONLY place weights/camera/tuning values live
scripts/run_pipeline.py
tests/                unit + integration (mock-first)
docs/                 architecture / models / pipeline runbook
tools/                dataset + training tooling (Friend 1 / data prep)
training/             training scripts + exports
experiments/          experiment configs + outputs (gitignored)
models/               weights dir (gitignored, out-of-band transfer)
datasets/             built datasets (gitignored)
```

`src/lensvoice/models/` = **perception adapters**; root `models/` = **weights**
(do not confuse).

## 4. Architecture

See [docs/architecture.md](docs/architecture.md). One-liner: models extract
evidence → pipeline interprets → narrator speaks. Language is produced in
exactly one module (`narration/narrator.py`).

## 5. Data contracts

`src/lensvoice/models/schemas.py` — plain dataclasses (NumPy only, serialisable
via `dataclasses.asdict`). `BBox(x1,y1,x2,y2)` is absolute pixels, validated.
All adapters convert framework output into these types; nothing Ultralytics /
PaddleOCR / torch leaks into the pipeline.

## 6. Taxonomy (locked)

| id | class | id | class | id | class |
|----|-------|----|-------|----|-------|
| 0 | person | 9 | couch | 18 | cell_phone |
| 1 | car | 10 | bed | 19 | book |
| 2 | motorcycle | 11 | backpack | 20 | keyboard |
| 3 | bus | 12 | bag | 21 | mouse |
| 4 | truck | 13 | suitcase | 22 | tv_monitor |
| 5 | autorickshaw | 14 | bottle | 23 | bench |
| 6 | bicycle | 15 | cup | 24 | umbrella |
| 7 | chair | 16 | bowl | 25 | dog |
| 8 | table | 17 | laptop | | |

Reference copy: `configs/default.yaml`. Never redefine ids in code.

## 7. Model integration contract

[Full contract in docs/models.md](docs/models.md#interfaces-the-integration-contract).
Five interface ABCs (`VisionModel`, `OCRModel`, `TrafficSignModel`,
`DepthModel`, `SceneModel`) in `models/interfaces.py`. Adapt to them; the
pipeline never sees frameworks.

## 8. Model status

| slot | runtime adapter | needs (Friend 1 / later) | priority |
|------|-----------------|--------------------------|----------|
| yolo (26-class) | IMPLEMENTED (YOLOAdapter) | trained unified `best.pt` | Highest |
| ocr | IMPLEMENTED (PaddleOCRAdapter) | PP-OCRv4 model | High |
| traffic_sign | SKELETON (detector wired, classifier TODO) | det. + classifier weights | High |
| depth | SKELETON (MiDaS socket) | Depth Anything / MiDaS weights | Medium |
| scene | SKELETON (CLIP socket) | fine-tuned CLIP | Medium |
| tts | STUB (SilentTTS) | piper or system voice | Medium |
| tracking | IMPLEMENTED mock IoU | ByteTrack / BoT-SORT | High |

PLANNED (design reserved, not implemented): audio interruption, tap-to-identify
query API, persistent memory of attended objects, multi-threaded model workers.

## 9. Scheduling

`FrameScheduler` (frame-count deterministic): each module gets its own FPS
budget vs the camera rate. Ids divisible by `round(camera_fps/module_fps)`
run that module. Tune `scheduler.*_fps` in YAML.

## 10. Camera

`MockCamera` (synthetic, bounded — used by tests/`--mock`) and `WebcamCamera`
(OpenCV). Swap via the `Camera` ABC; the runtime never calls cv2 directly.

## 11. Tracking

`tracking/tracker.py`: deterministic greedy-IoU `MockTracker` retains a
`Trajectory` per id (bboxes, first/last seen, missing frames). ByteTrack /
BoT-SORT are planned behind the same `.update()` API.

## 12. Spatial reasoning

Maps each object into a 3×3 zone of the frame (left/center/right ×
upper/middle/lower) and a relative-depth bucket. Depth is never real distance.

## 13. Temporal reasoning

`TemporalAnalyzer`: new/persistent/disappeared, stationary/moving/
approaching/receding — all from bbox-size ratios over the trajectory, never
absolute velocity.

## 14. Change detection

Todo list of evidence events — `new_object`, `object_moved`,
`object_closer`, `object_gone`, `text_detected`, sign gating — each with
cooldowns so the narrator isn't spammy.

## 15. Safety

Hazard events only when evidence agrees: close (`very_near`/`near`) **and** in
the collision band; vehicles additionally require observed approach. Signs
(stop/yield/crossing) are attention events with their own cooldown. The
pipeline never predicts intent ("will cross") — it reports what was measured.

## 16. Priority

`PriorityScorer`: every event → weighted score → `CRITICAL/HIGH/MEDIUM/LOW`
(weights in YAML). Repeated announcements get a repetition penalty so the user
isn't nagged.

## 17. Narration

`Narrator.generate(ctx, scored)` produces `(level, sentence)` for scored,
non-stale events only. Sentence builders use fields the event actually has —
no inventions.

## 18. Audio

`PriorityAudioQueue`: bounded (default 4), priority-ordered; a full queue drops
the lowest-priority new utterance; CRITICAL may interrupt current speech.
`TTS` ABC + `SilentTTS` stub (records everything — used by mock mode and tests).

## 19. Runtime

`runtime/pipeline.py` wires camera → scheduler → perception → tracking →
context → change → safety → priority → narration → audio → telemetry, and
exposes `process_frame` (returns `Context`) and `run` (main loop).

## 20. Configuration

Everything tunable lives in `configs/default.yaml`; a custom file can be passed
via `--config`. Typed dataclasses in `config/settings.py`. Real adapters refuse
to run with an unset weight — safe defaults.

## 21. Telemetry & metrics

`telemetry/logger.py` — structured logger that redacts images and OCR text.
`telemetry/metrics.py` — sliding-window counts/rates/percentiles per metric
(used for frame_ms, detection counts, narration counts).

## 22. Privacy

- No raw frames are ever logged.
- No OCR/recognised text is persisted or uploaded by default.
- No cloud calls; inference is local.
- Depth is relative-only; nothing claims metric distance.
- Compromise surface is small on purpose — see [docs/architecture.md](docs/architecture.md).

## 23. Testing

```bash
venv/bin/python -m pytest tests/ -q
```

49 tests across unit (schemas, config, scheduler, tracker, spatial, temporal,
change/safety/priority, narration/audio, mock models) and integration (full
mock pipeline end-to-end, adapter weight guards).

## 24. Dataset & training (ML owner)

Unified 26-class dataset lives in `datasets/lensvoice-unified/`
(train/val/test with manifests + audits), built by `tools/build_unified.py`.
Training configs: `experiments/exp_unified_diag`, `experiments/exp_train_l`.
Runbook for the GPU box: `RUNBOOK_friend.md`. The runtime prizes these
artifacts only as adapter inputs — never hardcoded.

## 25. Team ownership

| area | owner |
|------|-------|
| perception models: unified YOLO, OCR, traffic signs, depth, scene | **Friend 1 (ML)** |
| runtime: interfaces, adapters, reasoning, narration, audio, telemetry | **Friend 2 (Runtime)** |
| dataset curation + experiments + exports | Friend 1 + collaboration |
| I/O: camera, TTS provider choice, audio UX | Friend 2 |

Integration point: `configs/default.yaml` (weights/enable flags) + the
`models/interfaces.py` contract + `docs/models.md` + this table.

## 26. Repo conventions

- Python 3.10+, dataclasses only (no Pydantic), NumPy + PyYAML as runtime deps.
- Weights, datasets, experiments, logs, runs are gitignored and transferred
  out-of-band.
- Never redefine the taxonomy in code; reference `configs/default.yaml`.
- No comments unless they earn their place; interfaces carry docstrings.