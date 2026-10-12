\# Abhinav Runtime Runbook



\## Branch

Abhinav-runtime



\## Role

Abhinav — Runtime / Pipeline Engineer



\## Responsibilities

\- Camera and frame scheduling

\- Model adapters and runtime interfaces

\- Object tracking

\- Spatial and temporal reasoning

\- Change detection

\- Context management

\- Safety and priority

\- Narration

\- Audio / TTS

\- Runtime orchestration

\- Telemetry

\- Integration testing



\## Current Baseline



\### Tests

Command:

python -m pytest tests/ -q



Status:

PASS



\### Mock Pipeline

Command:

python scripts/run\_pipeline.py --mock --frames 100 --drain



Status:

PASS



\## Important Rules



\- Do not modify the locked 26-class taxonomy.

\- Do not commit model weights.

\- Do not hardcode model paths.

\- Keep mock models for testing.

\- Framework-specific code must stay inside adapters.

\- Do not let Ultralytics objects leak into the runtime pipeline.

\- Preserve existing runtime interfaces unless a concrete issue requires a change.



\## Current Phase

Phase 0 — Baseline / Setup



\## Next Phase

Phase 1 — Integrate Friend 1's unified YOLO model.

