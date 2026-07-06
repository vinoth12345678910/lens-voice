<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://capsule-render.vercel.app/api?type=waving&color=0:00c6ff,100:0072ff&height=200&section=header&text=LensVoice&fontSize=60&fontColor=fff&animation=fadeIn">
  <img alt="LensVoice Banner" src="https://capsule-render.vercel.app/api?type=waving&color=0:00c6ff,100:0072ff&height=200&section=header&text=LensVoice&fontSize=60&fontColor=fff&animation=fadeIn">
</picture>

<p align="center">
  <strong>AI-Powered Spatial Awareness for the Visually Impaired</strong><br>
  <em>See the world through sound — on-device, real-time, in your language.</em>
</p>

<p align="center">
  <a href="#-the-problem"><img src="https://img.shields.io/badge/🧠-Why-0072ff?style=for-the-badge"/></a>
  <a href="#-architecture"><img src="https://img.shields.io/badge/⚙️-Architecture-00c6ff?style=for-the-badge"/></a>
  <a href="#-the-model"><img src="https://img.shields.io/badge/📊-Model-0072ff?style=for-the-badge"/></a>
  <a href="#-build"><img src="https://img.shields.io/badge/📱-APK-00c6ff?style=for-the-badge"/></a>
  <a href="#-tech-stack"><img src="https://img.shields.io/badge/🔥-Stack-0072ff?style=for-the-badge"/></a>
</p>

---

## 🧠 The Problem

> *"Every 6 seconds, someone in the world becomes blind."* — WHO

There are **285 million visually impaired people** globally. Navigation is their single biggest daily challenge — crossing a road, walking through a crowded market, detecting an approaching vehicle.

Existing solutions are either:
- **Hardware-locked** (expensive smart canes, camera-equipped white canes)
- **Backend-dependent** (require constant WiFi + a server — fail in the real world)
- **Silent on hazards** (GPS apps don't tell you about the auto-rickshaw speeding toward you)

**LensVoice is the alternative.** A single APK. Zero infrastructure. Your phone becomes a real-time co-pilot that speaks in your language.

---

## 🎯 The Vision

This is not a "research demo." This is the software layer for the next generation of **AI-assisted wearable glasses** — think Meta Ray-Ban, but purpose-built for accessibility. The architecture is designed from day one to be:

- **Glasses-ready:** The on-device pipeline (TFLite → Tracker → Urgency → TTS) runs at ~2fps on a phone. On dedicated glasses hardware with a lightweight NPU, it runs in real-time.
- **Language-native:** English, Tamil, Hindi at launch. Not gating accessibility behind English literacy.
- **Privately yours:** Zero cloud. Zero backend. Zero images leave the device. The only internet call is for voice synthesis, and that's optional (works offline with pre-cached voices).

> **"If a visually impaired person can't afford $3,000 smart glasses, their phone + LensVoice should be enough."**

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────┐
│  📱 LensVoice APK (100% On-Device)                       │
│                                                          │
│  Camera Frame (2 fps)                                    │
│     ↓                                                    │
│  ┌─────────────────────────────┐                         │
│  │  YOLOv6n TFLite (512×512)   │ ← best.tflite (9.4MB)  │
│  │  Confidence ≥ 0.25, NMS     │                         │
│  └─────────────┬───────────────┘                         │
│                ↓                                         │
│  ┌─────────────────────────────┐                         │
│  │  Tracker (IoU + Centroid)   │ • Object persistence     │
│  │                              │ • Motion classification │
│  │                              │   (approaching/receding)│
│  └─────────────┬───────────────┘                         │
│                ↓                                         │
│  ┌─────────────────────────────┐                         │
│  │  Urgency Classifier         │ • HAZARD: vehicles       │
│  │                              │   approaching, close    │
│  │                              │   persons              │
│  │                              │ • INFO: everything else │
│  └─────────────┬───────────────┘                         │
│          ┌─────┴─────┐                                   │
│       HAZARD       INFO                                   │
│          │           │                                    │
│          ↓           ↓                                    │
│  ┌────────────┐  ┌──────────────┐                        │
│  │ Immediate   │  │ Change       │                        │
│  │ Announce    │  │ Detector     │ ← dedup (no spam)     │
│  └──────┬─────┘  └──────┬───────┘                        │
│         └───────┬───────┘                                │
│                 ↓                                         │
│  ┌─────────────────────────────┐                         │
│  │  Description Generator       │ • "Car approaching      │
│  │                              │   from your left"      │
│  └─────────────┬───────────────┘                         │
│                ↓                                         │
│  ┌─────────────────────────────┐                         │
│  │  Sarvam AI                  │ • Translate + TTS        │
│  │  (en-IN/ta-IN/hi-IN)        │ • Hazard interrupts     │
│  │                              │ • Queue for INFO       │
│  └─────────────┬───────────────┘                         │
│                ↓                                         │
│  ┌─────────────────────────────┐                         │
│  │  🔊 Audio + Haptic          │ • Spoken announcement   │
│  │                              │ • Heavy pulse on hazard │
│  └─────────────────────────────┘                         │
└──────────────────────────────────────────────────────────┘
```

**Key design decisions:**
- **No backend.** The app is fully self-contained. YOLO inference runs on-device via `tflite_flutter` (TFLite int8 quantized). The only internet dependency is Sarvam AI for translation + TTS.
- **Hazards interrupt.** Info announcements queue. If a car is approaching while the app is describing a traffic sign, *the car wins* — urgency is not up for debate.
- **Change detection prevents spam.** The `ChangeDetector` remembers what was last said about each object class and only re-announces if position/distance changes meaningfully.
- **Tracker uses centroid fallback**, not just IoU. When a car moves fast between frames and bounding boxes barely overlap, the centroid-distance heuristic keeps tracking alive. This was a real bug fixed during development on Indian roads.

---

## 📊 The Model

| Detail | Value |
|---|---|
| **Architecture** | YOLOv6 Nano (2.5M params) |
| **Training Data** | Indian Driving Dataset (IDD) — 10,000+ annotated frames |
| **Training Time** | **11 hours** on Apple M4 GPU |
| **Classes** | 15 Indian road objects: car, bus, truck, motorcycle, autorickshaw, bicycle, person, rider, traffic light, traffic sign, animal, trailer, caravan, train, vehicle fallback |
| **Input Size** | 512×512 |
| **Export Format** | TFLite (int8 quantized, 9.4MB) |
| **Inference** | ~150-300ms per frame on mid-range Android |

### Why YOLOv6 Nano?

Because YOLOv11n is 2× slower on mobile. Because YOLOv8n is 1.5× the model size. YOLOv6 Nano hits the sweet spot between accuracy and latency for a 2fps real-time pipeline on 2024 mid-range phones. The model was trained specifically on **Indian road conditions** — where the distinction between "autorickshaw" and "car" is a life-safety question, not a taxonomy exercise.

### Why IDD?

The original training data. Indian roads are fundamentally different from Western ones: shared lanes, auto-rickshaws, stray animals, informal traffic patterns. A model trained on COCO or Cityscapes fails on Bengaluru's roads. IDD closes that gap.

---

## 🔥 Tech Stack

<div align="center">

| Layer | Technology |
|---|---|
| **Mobile Framework** | Flutter 3.x + Dart 3.x |
| **Object Detection** | Ultralytics YOLOv6n → TFLite int8 |
| **TFLite Runtime** | tflite_flutter 0.11 |
| **Image Processing** | dart image 4.x |
| **Translation + TTS** | Sarvam AI API |
| **Audio Playback** | just_audio 0.9 |
| **Persistence** | shared_preferences |
| **Text-to-Speech (fallback)** | flutter_tts 4.x |
| **Build Tool** | Gradle + Android SDK 35 |

</div>

---

## 📱 Build & Run

### Prerequisites
- Flutter 3.24+ ([install](https://docs.flutter.dev/get-started/install))
- Android SDK 35+
- Sarvam AI API key ([get one free](https://sarvam.ai))

### Clone & Build

```bash
# Clone
git clone https://github.com/vinoth12345678910/lens-voice.git
cd lens-voice/app

# Install dependencies
flutter pub get

# Build APK with your API key
flutter build apk --release --dart-define=SARVAM_API_KEY=your_key_here

# APK location:
# build/app/outputs/flutter-apk/app-release.apk
```

Install the APK on your Android phone. Open it. Grant camera permission. **That's it.** No server, no WiFi pairing, no config.

### Running from Source (Hot Reload)
```bash
flutter run --dart-define=SARVAM_API_KEY=your_key_here
```

---

## 🧪 Future Roadmap

- **Edge TPU acceleration** — Deploy to Google Coral / MediaTek NPU for glasses form factor
- **Offline TTS** — Pre-cache Sarvam voices so zero internet needed
- **Depth estimation** — Add MiDaS or similar for precise distance, not just bounding-box proxies
- **Hazard tracking** — "Car approaching from left" → "Car now on your left" → "Car passed" — full life-cycle announcements
- **S wake word** — "Hey Lens" to query what's ahead on demand

---

## 📄 License

MIT — because accessibility should not be paywalled.

---

<p align="center">
  <strong>Built for the 285 million people who navigate a sighted world every day.</strong><br>
  <em>One APK. Zero excuses.</em>
</p>

<p align="center">
  <a href="https://github.com/vinoth12345678910/lens-voice">
    <img src="https://img.shields.io/github/stars/vinoth12345678910/lens-voice?style=social"/>
  </a>
  <a href="https://github.com/vinoth12345678910/lens-voice/issues">
    <img src="https://img.shields.io/github/issues/vinoth12345678910/lens-voice?style=social"/>
  </a>
</p>
