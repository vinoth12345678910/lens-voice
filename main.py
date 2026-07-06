import cv2
import time
import base64
import requests
from dotenv import load_dotenv
import os
from ultralytics import YOLO
import numpy as np
from collections import OrderedDict

# ============ CONFIG ============
CAMERA_SOURCE = "http://10.175.135.234:8080/video"   # phone stream — change if IP changes
MODEL_PATH = "best.pt"                                 # confirm this matches your actual filename
TARGET_LANGUAGE = "ta-IN"                              # "en-IN", "hi-IN", or "ta-IN"
SPEAKER = "priya"
SAMPLE_EVERY_N_FRAMES = 15
CONF_THRESHOLD = 0.25   # lowered from 0.3 — weak model, give it a better chance to catch things

# ============ SETUP ============
load_dotenv()
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
if not SARVAM_API_KEY:
    print("WARNING: SARVAM_API_KEY not found in .env — speech will fail.")
else:
    SARVAM_API_KEY = SARVAM_API_KEY.strip()
HEADERS = {"api-subscription-key": SARVAM_API_KEY, "Content-Type": "application/json"}

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at '{MODEL_PATH}'. Run: ls *.pt  to see what's actually there.")

print(f"Loading model from {MODEL_PATH}...")
model = YOLO(MODEL_PATH)
class_names = model.names
print(f"Model loaded. Classes: {list(class_names.values())}")

# ============ TRACKER ============
class Tracker:
    def __init__(self, max_missed_frames=5, iou_threshold=0.3, centroid_dist_threshold=60):
        self.next_id = 0
        self.objects = OrderedDict()
        self.max_missed_frames = max_missed_frames
        self.iou_threshold = iou_threshold
        self.centroid_dist_threshold = centroid_dist_threshold

    def _centroid(self, bbox):
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def _bbox_area(self, bbox):
        x1, y1, x2, y2 = bbox
        return max(0, x2 - x1) * max(0, y2 - y1)

    def _iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
        inter = max(0, xB - xA) * max(0, yB - yA)
        union = self._bbox_area(boxA) + self._bbox_area(boxB) - inter
        return inter / union if union > 0 else 0

    def update(self, detections):
        unmatched = list(range(len(detections)))
        for obj_id, obj in list(self.objects.items()):
            best_match, best_iou = None, -1
            obj_centroid = obj["centroid"]
            for i in unmatched:
                det = detections[i]
                if det["class"] != obj["class"]:
                    continue
                iou = self._iou(obj["bbox"], det["bbox"])
                det_centroid = self._centroid(det["bbox"])
                dist = np.hypot(obj_centroid[0]-det_centroid[0], obj_centroid[1]-det_centroid[1])
                qualifies = (iou > self.iou_threshold) or (dist < self.centroid_dist_threshold)
                if qualifies and iou > best_iou:
                    best_iou, best_match = iou, i

            if best_match is not None:
                det = detections[best_match]
                area_prev = self._bbox_area(obj["bbox"])
                area_now = self._bbox_area(det["bbox"])
                obj["bbox"] = det["bbox"]
                obj["centroid"] = self._centroid(det["bbox"])
                obj["missed"] = 0
                obj["area_history"].append(area_now)
                if len(obj["area_history"]) > 5:
                    obj["area_history"].pop(0)
                growth = (area_now - area_prev) / area_prev if area_prev > 0 else 0
                obj["motion"] = "approaching" if growth > 0.08 else "receding" if growth < -0.08 else "static"
                unmatched.remove(best_match)
            else:
                obj["missed"] += 1

        for obj_id in list(self.objects.keys()):
            if self.objects[obj_id]["missed"] > self.max_missed_frames:
                del self.objects[obj_id]

        for i in unmatched:
            det = detections[i]
            self.objects[self.next_id] = {
                "bbox": det["bbox"], "class": det["class"],
                "centroid": self._centroid(det["bbox"]),
                "area_history": [self._bbox_area(det["bbox"])],
                "missed": 0, "motion": "new",
            }
            self.next_id += 1
        return self.objects

# ============ URGENCY CLASSIFIER ============
HAZARD_VEHICLE_CLASSES = {"car","bus","truck","motorcycle","autorickshaw","bicycle","trailer","caravan","train","vehicle fallback"}
HAZARD_PERSON_CLASSES = {"person","rider"}
CLOSE_AREA_THRESHOLD = 0.15 * (512*512)

def classify_urgency(tracked_objects):
    results = []
    for obj_id, obj in tracked_objects.items():
        cls, motion = obj["class"], obj["motion"]
        area = obj["area_history"][-1] if obj["area_history"] else 0
        urgency = "INFO"
        if cls in HAZARD_VEHICLE_CLASSES and motion == "approaching":
            urgency = "HAZARD"
        elif cls in HAZARD_PERSON_CLASSES and motion == "approaching" and area > CLOSE_AREA_THRESHOLD:
            urgency = "HAZARD"
        elif cls in HAZARD_VEHICLE_CLASSES and area > CLOSE_AREA_THRESHOLD * 1.5:
            urgency = "HAZARD"
        results.append({"id": obj_id, "class": cls, "motion": motion, "area": area,
                         "bbox": obj["bbox"], "urgency": urgency})
    return results

# ============ CHANGE DETECTOR ============
class ChangeDetector:
    def __init__(self, position_bucket_size=3):
        self.last_spoken_state = {}
        self.position_bucket_size = position_bucket_size

    def _bucket_position(self, bbox, frame_width):
        cx = (bbox[0] + bbox[2]) / 2
        bucket_width = frame_width / self.position_bucket_size
        idx = int(cx // bucket_width)
        labels = ["left","center","right"]
        return labels[min(idx, len(labels)-1)]

    def _bucket_distance(self, area, frame_area):
        ratio = area / frame_area
        return "near" if ratio > 0.15 else "medium" if ratio > 0.04 else "far"

    def check(self, info_objects, frame_width, frame_height):
        frame_area = frame_width * frame_height
        changes, current_state = [], {}
        for obj in info_objects:
            cls = obj["class"]
            position = self._bucket_position(obj["bbox"], frame_width)
            distance = self._bucket_distance(obj["area"], frame_area)
            current_state[cls] = {"position": position, "distance": distance}
            prev = self.last_spoken_state.get(cls)
            if prev is None:
                changes.append({**obj, "position": position, "distance": distance})
            elif prev["position"] != position or prev["distance"] != distance:
                changes.append({**obj, "position": position, "distance": distance})
        self.last_spoken_state = current_state
        return changes

# ============ DESCRIPTION + SPEECH ============
FRIENDLY_NAMES = {
    "car":"a car","bus":"a bus","truck":"a truck","motorcycle":"a motorcycle",
    "autorickshaw":"an auto-rickshaw","bicycle":"a bicycle","person":"a person",
    "rider":"a rider","animal":"an animal","traffic light":"a traffic light",
    "traffic sign":"a traffic sign","trailer":"a trailer","caravan":"a caravan",
    "train":"a train","vehicle fallback":"a vehicle",
}

def generate_description(obj):
    cls, urgency = obj["class"], obj.get("urgency","INFO")
    motion, position, distance = obj.get("motion","static"), obj.get("position","ahead"), obj.get("distance","medium")
    subject = FRIENDLY_NAMES.get(cls, cls)
    if urgency == "HAZARD":
        return f"Careful, {subject} is approaching from your {position}." if motion=="approaching" else f"Careful, {subject} is close on your {position}."
    distance_phrase = {"near":"close by","medium":"nearby","far":"further away"}.get(distance,"nearby")
    return f"There is {subject} {distance_phrase} on your {position}."

def speak(text_english, target_language="en-IN", speaker="priya", filename="output.wav"):
    try:
        final_text = text_english
        if target_language != "en-IN":
            r = requests.post("https://api.sarvam.ai/translate", headers=HEADERS, json={
                "input": text_english, "source_language_code": "en-IN", "target_language_code": target_language
            }, timeout=10)
            if r.status_code == 200:
                final_text = r.json()["translated_text"]
            else:
                print(f"Translation failed ({r.status_code}): {r.text} — falling back to English")
                target_language = "en-IN"
                final_text = text_english

        r = requests.post("https://api.sarvam.ai/text-to-speech", headers=HEADERS, json={
            "inputs": [final_text], "target_language_code": target_language, "speaker": speaker, "model": "bulbul:v3"
        }, timeout=10)

        if r.status_code == 200:
            audio_bytes = base64.b64decode(r.json()["audios"][0])
            with open(filename, "wb") as f:
                f.write(audio_bytes)
            os.system(f"afplay {filename}")
        else:
            print(f"TTS failed ({r.status_code}): {r.text}")
    except Exception as e:
        print(f"speak() error (non-fatal, continuing loop): {e}")

# ============ MAIN LOOP ============
def yolo_to_detections(result, class_names):
    detections = []
    for box in result.boxes:
        cls_id = int(box.cls[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        detections.append({"bbox": [x1, y1, x2, y2], "class": class_names[cls_id]})
    return detections

def main():
    print(f"Connecting to camera source: {CAMERA_SOURCE}")
    cap = cv2.VideoCapture(CAMERA_SOURCE)

    if not cap.isOpened():
        print("FAILED to open camera stream. Check the URL/IP and that your phone app is running.")
        return
    print("Camera stream opened successfully.")

    tracker = Tracker()
    change_detector = ChangeDetector()
    frame_w, frame_h = 512, 512
    frame_count = 0

    print("Starting LensVoice — press Ctrl+C to stop\n")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame — stream may have dropped. Retrying...")
                time.sleep(0.5)
                continue

            frame_count += 1
            if frame_count % SAMPLE_EVERY_N_FRAMES != 0:
                continue

            frame_resized = cv2.resize(frame, (frame_w, frame_h))
            results = model.predict(frame_resized, conf=CONF_THRESHOLD, device="mps", verbose=False)
            detections = yolo_to_detections(results[0], class_names)

            print(f"[Frame {frame_count}] {len(detections)} object(s): {[d['class'] for d in detections]}")

            if not detections:
                continue

            tracked = tracker.update(detections)
            urgency_results = classify_urgency(tracked)

            hazards = [o for o in urgency_results if o["urgency"] == "HAZARD"]
            info_objects = [o for o in urgency_results if o["urgency"] == "INFO"]

            if hazards:
                sentence = generate_description(hazards[0])
                print(f"  -> HAZARD: {sentence}")
                speak(sentence, target_language=TARGET_LANGUAGE, speaker=SPEAKER)
            else:
                changes = change_detector.check(info_objects, frame_w, frame_h)
                if changes:
                    sentence = generate_description(changes[0])
                    print(f"  -> INFO: {sentence}")
                    speak(sentence, target_language=TARGET_LANGUAGE, speaker=SPEAKER)

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        cap.release()
        print("Camera released. Goodbye.")

if __name__ == "__main__":
    main()