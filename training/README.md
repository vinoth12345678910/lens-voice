# Training LensVoice's detector on an RTX 4050

Handoff notes for running the real training run on a CUDA box, because the
M4 is ~6x too slow to ever finish one.

---

## 1. Where things actually stand

Three runs exist under `runs/detect/lensvoice_runs/`. None of them converged:

| run | imgsz | epochs done | best mAP50 | why it stopped |
|---|---|---|---|---|
| `yolo26n_idd_mac_v1` | 640 | 1 | 0.042 | 90 min/epoch on MPS |
| `yolo26n_idd_mac_v2` | 512 | 10 of 30 | **0.084** | ~41 min/epoch, ~7h for 10 epochs |
| `yolo26n_idd_320_v3` | 320 | 1 of 60 | 0.032 | 15 min/epoch, abandoned |

Those checkpoints are parked in `archive/` purely as a baseline to beat.
mAP50 of 0.084 is why the old app had to run a confidence threshold of 0.05 —
that was compensating for a model that never finished training, not a tuning
choice.

**The hyperparameters were never the problem.** I checked: `cls_pw=0.0`,
`dis=6.0`, `rle=1.0` in the old `args.yaml` are YOLO26's own defaults, not bad
overrides. In v2 the loss curve was still dropping and mAP50 still climbing
steeply when it stopped (0.060 → 0.062 → 0.069 → 0.073 → 0.084 over epochs
6-10). It was simply nowhere near done.

One thing that matters for the epoch budget: **YOLO26 is an end-to-end,
NMS-free architecture.** Its one-to-one label assignment converges noticeably
slower over the first ~40 epochs than an anchor-based YOLOv8 would. Early
epochs looking bad is expected and is not a reason to stop.

---

## 2. The dataset

`idd-1/` — India Driving Dataset via Roboflow, 2.6 GB, already letterboxed to
640x640 JPEG.

| split | images | labels | instances |
|---|---|---|---|
| train | 31,032 | 31,032 | 367,706 |
| valid | 8,866 | 8,866 | 105,583 |
| test | 4,433 | 4,433 | — |

**It is already prepared — do not re-run `scripts_remap_classes.py`.** That was
a one-time merge of three near-empty classes (caravan 98, trailer 14, train 39
instances) into `vehicle fallback`, taking 15 classes down to 12. The labels in
`idd-1/` are the post-remap 12-class version; originals are in
`idd-1_labels_backup/` and do not need to be shipped to the GPU box.

Train instances per class:

| id | class | instances | |
|---|---|---:|---|
| 5 | motorcycle | 74,898 | |
| 7 | rider | 70,588 | |
| 4 | car | 65,798 | |
| 6 | person | 63,716 | |
| 1 | autorickshaw | 23,236 | |
| 10 | truck | 20,433 | |
| 11 | vehicle fallback | 15,419 | |
| 3 | bus | 13,747 | |
| 9 | traffic sign | 10,207 | |
| 0 | animal | 4,644 | thin |
| 8 | traffic light | 2,674 | thin |
| 2 | bicycle | 2,346 | thin |

32:1 imbalance between motorcycle and bicycle. Don't fight it — the four
classes that matter most for hazard warnings (motorcycle, rider, car, person)
are all in the top four. Just expect bicycle / traffic light / animal AP to
stay low and read the per-class table at the end of the run rather than only
the headline mAP.

`data.yaml` has Roboflow's `../train/images` paths, which look wrong but
resolve correctly — Ultralytics auto-corrects them to `idd-1/train/images`.
Verified. Leave it alone.

---

## 3. Prerequisites on the GPU machine

RTX 4050 Laptop is Ada (SM 8.9), **6 GB VRAM** — that's the real constraint.

- **NVIDIA driver 550 or newer.** `nvidia-smi` should report CUDA 12.4+.
- **Python 3.11 or 3.12.** Not 3.14 — the Mac venv here uses 3.14 and CUDA
  torch wheels for it are unreliable. 3.11 is the safe choice.
- **~20 GB free disk**: 2.6 GB dataset + ~8 GB torch/CUDA + checkpoints.
- **16 GB system RAM.**
- Laptop **plugged in**, Windows power mode on Performance / NVIDIA panel set
  to prefer maximum performance. A 4050 on battery drops to ~35W and training
  time roughly doubles.

Install:

```bash
python -m venv venv
# Windows: venv\Scripts\activate       Linux: source venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r training/requirements-gpu.txt
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

That last line must print `True NVIDIA GeForce RTX 4050 ...`. If it prints
`False`, pip resolved the CPU build — uninstall torch and redo it with the
`--index-url`.

**Windows only:** keep `--workers 4` or lower. The scripts already have the
`if __name__ == "__main__":` guard, which is mandatory on Windows or the
dataloader workers respawn forever.

### What to send your friend

Zip and send three things: `idd-1/` (2.6 GB), `training/`, and
`yolo26n.pt` (5 MB, the pretrained starting weights).

Skip `venv/`, `archive/`, `runs/` and `idd-1_labels_backup/` — none of them
are needed to train and they add ~2.3 GB to the transfer.

Have them verify the transfer before starting:

```bash
find idd-1/train/images -type f | wc -l     # must be 31032
find idd-1/valid/images -type f | wc -l     # must be 8866
find idd-1/train/labels -name '*.txt' -print0 | xargs -0 cat | awk '{print $1}' | sort -n | uniq -c
# highest class id must be 11, never 12/13/14
```

A class id above 11 means they got the pre-remap labels and training will crash
or silently learn garbage.

---

## 4. Run it

### Step 1 — smoke test (10 minutes, do not skip)

```bash
python training/train_gpu.py --stage smoke --batch 16
```

Two epochs on 5% of the data. It proves CUDA works **and** prints the real
it/s. Everything in the time table below is an estimate from comparable
hardware — replace it with what this actually measures:

> epoch time ≈ 1940 / (it/s at batch 16), in seconds, plus ~40-90s validation

Then try `--batch 32`. yolo26n is only 2.57M parameters and a bigger batch
uses the GPU far better — if 32 fits in 6 GB without OOM it can cut total time
by 20-30%. Use the largest batch that survives the smoke test. (`--batch -1`
lets Ultralytics autotune, but it's less predictable than just trying 32.)

### Step 2 — just train at 640

There is no app constraining the input size any more, so there is no reason to
cripple the model to match one. Train at **640**, which is also the native
size the dataset images are already letterboxed to — no resizing, no lost
detail on the small, distant objects that matter most here.

```bash
python training/train_gpu.py --stage main --imgsz 640 --epochs 150 --batch 32
```

Estimated **24-30 hours** on a 4050. It is fully resumable, so spread it over
two or three nights — re-run the identical command and it continues from
`last.pt`.

If your friend can only spare one night, `--imgsz 416 --epochs 200` lands in
roughly 14-16 hours and gives up maybe 4-6 mAP50 points. Don't go below 416.

When the React Native client eventually exists and you know what input size it
can afford, `--stage finetune --ft-imgsz 320` adapts the 640 model down at a
low LR in about 3 hours. That's a later problem.

### Step 3 — how many epochs, really

Don't pick a number, pick a stopping rule. The script sets `epochs=250`,
`patience=50`, `cos_lr=True`, `close_mosaic=25` and lets early stopping decide.

- `patience=50` is deliberately generous because of YOLO26's slow-starting
  one-to-one head. A tight patience is what would kill this run prematurely.
- Expect the best checkpoint to land somewhere around **epoch 110-180**.
- Watch `runs/detect/lensvoice_runs/<run>/results.csv`. When
  `metrics/mAP50-95(B)` gains less than ~0.002 across 30 epochs, it's done —
  anything past that is noise.
- If you cap it at 150 epochs for time, you're giving up maybe 1-2 mAP points
  versus 250. That's a fine trade.

**Realistic target**, based on nano-class models on IDD — treat as a range,
not a promise:

| | mAP50 | mAP50-95 |
|---|---|---|
| currently shipping | 0.084 | 0.049 |
| after 640 training | 0.40 - 0.50 | 0.24 - 0.30 |
| if later fine-tuned down to 320 | 0.30 - 0.38 | 0.18 - 0.23 |

That's roughly a 5x improvement on the best checkpoint in `archive/`. At that
accuracy a client can run a sane confidence threshold of ~0.25 instead of
0.05, which is what kills the false positives.

### Optional — the accuracy upgrade

If the 4050 has spare nights, run `--size s` (yolo26s, ~9M params) with the
same recipe. It's typically worth +8-12 mAP50 points over nano. The cost is
~3x inference latency on the phone. For a hazard-announce pipeline that isn't
doing 30fps tracking, that's very likely acceptable — but measure it on a real
device before committing.

---

## 5. Export when you need it

Nothing consumes the model right now, so exporting is not urgent — get the
training run finished first. When the React Native client exists:

```bash
python training/export_model.py runs/detect/lensvoice_runs/<run>/weights/best.pt \
    --format tflite --imgsz 640
```

`--format` takes `onnx` (desktop testing), `tflite` (Android / React Native via
a LiteRT binding), `coreml` (iOS) or `torchscript`. The script prints the real
input and output tensor shapes, including whether the input came back NCHW or
NHWC — write the client against what it prints, not against an assumption.

The output is `[1, 300, 6]`: `x1, y1, x2, y2, conf, class_id`, box in pixels of
the input square. YOLO26's head is end-to-end and NMS-free, so those 300 rows
are already decoded and deduplicated — the client needs no NMS, just a
confidence filter.

Quick visual check on a real photo before trusting anything:

```bash
python training/test_inference.py some_road_photo.jpg \
    -m best.onnx -s 640 -t 0.25
```

**Don't use `--int8`.** It's 3x smaller but quantising a 2.6M-parameter model
costs real recall on small and distant objects — precisely the ones a hazard
warning needs to catch early.

## 6. Class list

The 12 classes, in id order. Anything consuming the model must use exactly
this list:

```
0 animal          4 car          8  traffic light
1 autorickshaw    5 motorcycle   9  traffic sign
2 bicycle         6 person       10 truck
3 bus             7 rider        11 vehicle fallback
```

`training/test_inference.py` already carries it. The old 15-class list (with
`caravan`, `trailer`, `train`) is dead — if you ever see it in new client
code, it is wrong, and every id from 5 up will be shifted by one or more.
