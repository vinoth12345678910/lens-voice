# LensVoice — detector training

Object detector for road-scene hazard warning, trained on the India Driving
Dataset. This repo is **training only**: dataset, recipe, and the scripts to
run it on a CUDA box.

The Flutter app that used to live here has been removed — a React Native
client is planned later. Nothing in the training recipe is pinned to a
deployment input size any more, so the model is trained for maximum accuracy
and exported to whatever the client eventually needs.

## Layout

```
idd-1/         the dataset, 2.6 GB, 12 classes, ready to train (do not re-remap)
training/      everything you run
  README.md      full setup + recipe + epoch guidance  <- start here
  train_gpu.py   smoke / main / finetune stages, all resumable
  export_model.py  checkpoint -> onnx | tflite | coreml, reports tensor shapes
  test_inference.py  run a model on one image, draw boxes
  remap_classes.py   one-time 15->12 class merge, ALREADY APPLIED
  requirements-gpu.txt
yolo26n.pt     pretrained COCO weights, the starting point for training
archive/       old undertrained weights and Mac runs, kept only as a baseline
```

## Quick start on the GPU machine

```bash
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r training/requirements-gpu.txt

python training/train_gpu.py --stage smoke --batch 16   # 10 min, verifies setup
python training/train_gpu.py --stage main --imgsz 640 --epochs 150 --batch 32
```

Read `training/README.md` before starting — it covers the prerequisites, how
long the run actually takes, and how to tell when the model has converged.

## Classes (12)

`animal, autorickshaw, bicycle, bus, car, motorcycle, person, rider,
traffic light, traffic sign, truck, vehicle fallback`

Merged from the original 15: `caravan`, `trailer` and `train` had 98, 14 and
39 training instances respectively — statistically unlearnable — and were
folded into `vehicle fallback`.
