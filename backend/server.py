import asyncio
import base64
import json
import logging
import os
import time
from collections import OrderedDict

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from ultralytics import YOLO

from pipeline.tracker import Tracker
from pipeline.urgency import classify_urgency
from pipeline.change_detector import ChangeDetector
from pipeline.description import generate_description
from pipeline.sarvam import synthesize

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("lensvoice-server")

MODEL_PATH = os.path.join(os.path.dirname(__file__), "best.pt")
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "best.pt")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

log.info(f"Loading model from {MODEL_PATH}")
model = YOLO(MODEL_PATH)
class_names = model.names
log.info(f"Model loaded. Classes: {list(class_names.values())}")

FRAME_WIDTH = 512
FRAME_HEIGHT = 512
CONF_THRESHOLD = 0.25
HEARTBEAT_INTERVAL = 2.0
INFERENCE_INTERVAL = 0.5

app = FastAPI(title="LensVoice Backend")


async def send_json(websocket: WebSocket, data: dict):
    try:
        await websocket.send_json(data)
    except Exception:
        pass


async def process_session(websocket: WebSocket):
    tracker = Tracker()
    change_detector = ChangeDetector()
    language = "en-IN"
    speaker = "priya"
    session_ready = False
    frame_count = 0
    last_inference_time = 0.0
    last_heartbeat_time = time.monotonic()
    connected = True

    log.info("New client connected")

    async def heartbeat_loop():
        nonlocal last_heartbeat_time
        while connected:
            now = time.monotonic()
            if now - last_heartbeat_time >= HEARTBEAT_INTERVAL:
                last_heartbeat_time = now
                obj_count = len(tracker.objects) if tracker else 0
                await send_json(websocket, {
                    "type": "status",
                    "connected": True,
                    "objects_in_view": obj_count,
                })
            await asyncio.sleep(0.5)

    heartbeat_task = asyncio.create_task(heartbeat_loop())

    try:
        while True:
            message = await websocket.receive()

            if message.get("type") == "websocket.disconnect":
                break

            if "bytes" in message:
                raw_bytes = message["bytes"]
            elif "text" in message:
                text = message["text"]
                try:
                    msg = json.loads(text)
                except json.JSONDecodeError:
                    await send_json(websocket, {
                        "type": "error",
                        "message": "Invalid JSON",
                        "recoverable": True,
                    })
                    continue

                if msg.get("type") == "config":
                    language = msg.get("language", "en-IN")
                    speaker = msg.get("speaker", "priya")
                    session_ready = True
                    log.info(f"Session configured: language={language}, speaker={speaker}")
                    await send_json(websocket, {
                        "type": "status",
                        "connected": True,
                        "objects_in_view": 0,
                        "configured": True,
                    })
                continue
            else:
                continue

            if not session_ready:
                await send_json(websocket, {
                    "type": "error",
                    "message": "Send config message first",
                    "recoverable": True,
                })
                continue

            now = time.monotonic()
            if now - last_inference_time < INFERENCE_INTERVAL:
                continue
            last_inference_time = now

            frame_count += 1
            log.info(f"[Frame {frame_count}] Received {len(raw_bytes)} bytes")

            try:
                np_arr = np.frombuffer(raw_bytes, np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                if frame is None:
                    log.warning("Failed to decode frame")
                    continue
            except Exception as e:
                log.warning(f"Frame decode error: {e}")
                continue

            try:
                frame_resized = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
            except Exception as e:
                log.warning(f"Frame resize error: {e}")
                continue

            try:
                results = model.predict(
                    frame_resized,
                    conf=CONF_THRESHOLD,
                    device="mps",
                    verbose=False,
                )
                detections = []
                for box in results[0].boxes:
                    cls_id = int(box.cls[0])
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    detections.append({
                        "bbox": [x1, y1, x2, y2],
                        "class": class_names[cls_id],
                    })
                log.info(f"  Detected {len(detections)} objects: {[d['class'] for d in detections]}")
            except Exception as e:
                log.error(f"Inference error: {e}")
                await send_json(websocket, {
                    "type": "error",
                    "message": "Inference failed",
                    "recoverable": True,
                })
                continue

            if not detections:
                continue

            try:
                tracked = tracker.update(detections)
                urgency_results = classify_urgency(tracked)
            except Exception as e:
                log.error(f"Tracking/urgency error: {e}")
                continue

            hazards = [o for o in urgency_results if o["urgency"] == "HAZARD"]
            info_objects = [o for o in urgency_results if o["urgency"] == "INFO"]

            try:
                if hazards:
                    obj = hazards[0]
                    sentence = generate_description(obj)
                    log.info(f"  -> HAZARD: {sentence}")
                    translated, audio_bytes = synthesize(sentence, language, speaker)
                    announcement = {
                        "type": "announcement",
                        "urgency": "HAZARD",
                        "text_en": sentence,
                        "text_translated": translated,
                        "audio_base64": base64.b64encode(audio_bytes).decode("utf-8") if audio_bytes else None,
                    }
                    await send_json(websocket, announcement)
                else:
                    changes = change_detector.check(info_objects, FRAME_WIDTH, FRAME_HEIGHT)
                    if changes:
                        obj = changes[0]
                        sentence = generate_description(obj)
                        log.info(f"  -> INFO: {sentence}")
                        translated, audio_bytes = synthesize(sentence, language, speaker)
                        announcement = {
                            "type": "announcement",
                            "urgency": "INFO",
                            "text_en": sentence,
                            "text_translated": translated,
                            "audio_base64": base64.b64encode(audio_bytes).decode("utf-8") if audio_bytes else None,
                        }
                        await send_json(websocket, announcement)
            except Exception as e:
                log.error(f"Description/synthesis error: {e}")
                await send_json(websocket, {
                    "type": "error",
                    "message": f"Failed to generate announcement: {str(e)}",
                    "recoverable": True,
                })
                continue

    except WebSocketDisconnect:
        log.info("Client disconnected")
    except asyncio.CancelledError:
        pass
    except Exception as e:
        log.error(f"Session error: {e}")
    finally:
        connected = False
        heartbeat_task.cancel()
        try:
            await heartbeat_task
        except asyncio.CancelledError:
            pass
        log.info("Session cleaned up")


@app.websocket("/stream")
async def stream_endpoint(websocket: WebSocket):
    await websocket.accept()
    await process_session(websocket)


@app.get("/health")
async def health():
    return {"status": "ok"}
