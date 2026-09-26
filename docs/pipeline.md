# LensVoice pipeline runbook

## Run the mock pipeline (no GPU, no weights, no webcam)

```bash
venv/bin/python -m pip install -e .          # first time only
venv/bin/python scripts/run_pipeline.py --mock --frames 100 --drain
```

`--frames N` runs N bounded synthetic frames then exits cleanly.
`--drain` prints every sentence the TTS stub would have spoken.

## Run against a webcam (mock perception)

```bash
venv/bin/python scripts/run_pipeline.py --frames 0           # no --mock => webcam
```

Cameras/models stay mocked unless you enable them in YAML; `--frames` alone
uses a real webcam with mock perception (safe to try).

## Wire real models

1. Put trained weights under `models/`.
2. Edit `configs/default.yaml` (or a copy):
   - `models.yolo.enabled: true`, `models.yolo.weight: models/yolo-best.pt`
3. Run: `venv/bin/python scripts/run_pipeline.py --config configs/default.yaml`

Adapters raise a clear error if `enabled: true` with an empty weight.

## Config cheat sheet

| key | meaning |
|-----|---------|
| `scheduler.<x>_fps` | per-module FrameScheduler budget vs `camera_fps` |
| `spatial.distance_bins` | relative-depth buckets (very_near/near/medium/far) |
| `safety.center_band` | fraction of frame width that is "in path" when close |
| `safety.require_approaching_for_vehicle` | vehicles must show approach before hazard |
| `priority.weights` | event category weights → score → level |
| `audio.max_queue_size` / `cooldown_ms` | speech queue bound + repetition cooldown |
| `tts.provider` | `silent` now; `piper`/`system` planned |

## Testing

```bash
venv/bin/python -m pytest tests/ -q      # 49 tests: unit + integration (mocks only)
```

## Privacy guarantees in this build

- The logger never logs raw images or OCR text by default (redacts them).
- No data leaves the machine; no cloud calls.
- Depth is relative-only; nothing claims metric distance.
- The narrator can only speak from structured evidence — no invented intent.