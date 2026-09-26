# LensVoice architecture

LensVoice is a **laptop-only** AI assistant for blind users. The camera feeds
a laptop webcam; perception models extract structured evidence per frame; a
reasoning layer interprets it; and the assistant **speaks only what the
evidence supports**.

## Core principle

> **Models extract evidence. The pipeline interprets. The narrator speaks.**

There is exactly one arrow for language: `evidence -> structured events ->
sentences`. No model output ever lands directly in the TTS path. This prevents
hallucinated distance, speed, intent, identity or emotion.

## Package layout

```
src/lensvoice/
  camera/        camera sources (webcam + deterministic MockCamera)
  scheduler/     per-module frame-rate scheduling
  models/        perception interfaces, adapters, mocks, schemas (contracts)
  tracking/      multi-object tracker (mock IoU now, ByteTrack later)
  reasoning/     spatial, temporal, change detection, safety, priority, context
  narration/     the only sentence generator
  audio/         bounded priority speech queue + TTS
  runtime/       Pipeline (glues everything together)
  telemetry/     structured privacy-safe logger + metrics
  config/        typed settings from configs/*.yaml
  utils/         clock/path helpers
```

## Stage-by-stage flow

```
Frame
 └─ Scheduler decides which modules run this frame (ids % interval == 0)
     ├─ VisionModel.predict   -> DetectedObject[]
     ├─ OCRModel.read         -> OCRResult[]
     ├─ TrafficSignModel.detect -> TrafficSignResult[]
     ├─ DepthModel.estimate   -> DepthResult (RELATIVE, 0=far..1=near)
     └─ SceneModel.classify   -> SceneResult
 └─ Tracker.update(detections) assigns track_ids, retains Trajectory history
 └─ ContextEngine.build(evidence, tracks)
      → Context { scene, objects, object_states (zones/dist/motion), text, signs }
 └─ ChangeDetector          → new_object / object_moved / object_closer / gone / text_detected
 └─ SafetyAnalyzer          → hazard_path / vehicle_in_path / sign_attention (cooldown-gated)
 └─ PriorityScorer          → PriorityScore { score, CRITICAL|HIGH|MEDIUM|LOW }
 └─ Narrator.generate(ctx, scored) -> (level, sentence) — the ONLY valid language
 └─ PriorityAudioQueue      → bounded, priority-ordered; CRITICAL may interrupt
 └─ TTS.speak()
```

## Why dataclasses (no Pydantic)?

A framework-free contract is more robust: the schema module imports only
NumPy. This makes the pipeline trivially testable and keeps the model-loading
frameworks (Ultralytics/PaddleOCR/torch) at the adapters, where they belong.

## Mock-first design

Every perception slot has a deterministic mock. With `--mock`, the pipeline
runs without a GPU, weights, webcam, or TTS device. Real adapters are wired by
flipping `enabled: true` in `configs/default.yaml` and supplying weights.

## IoC seams

- `Clock` (lensvoice.utils.time) — inject fake clocks in tests.
- `TTS` (lensvoice.audio.tts) — SilentTTS stub now; Piper/system later.
- `Camera` (lensvoice.camera.camera) — MockCamera vs WebcamCamera.
- `MockTracker` → ByteTrack/BoT-SORT behind the same `.update()` API.