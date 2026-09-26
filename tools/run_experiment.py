#!/usr/bin/env python
"""Experiment engine for the LensVoice detection workflow.

Each experiment is a directory under experiments/<exp>/:
    config.json      input (model, data, schedule, augs) - edit this
    metrics.json     output (mAP50, mAP50-95, P, R, per-class AP, timing)
    results.csv      ultralytics per-epoch log
    val_samples/     prediction images + confusion matrix from validation

Never overwrites a completed experiment: re-running a dir refuses unless
--resume and a last.pt exists.

Usage:
    venv/bin/python tools/run_experiment.py --exp experiments/exp001
    venv/bin/python tools/run_experiment.py --exp experiments/exp001 --resume
    venv/bin/python tools/run_experiment.py --exp experiments/exp001 --smoke --fraction 0.02 --epochs 2
"""
import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime

DEFAULT_AUGS = {
    "degrees": 5.0, "scale": 0.5, "translate": 0.1,
    "fliplr": 0.5, "flipud": 0.0,
    "hsv_h": 0.015, "hsv_s": 0.7, "hsv_v": 0.4, "mosaic": 1.0,
}


def load_or_scaffold(exp_dir, overrides):
    cfg_path = os.path.join(exp_dir, "config.json")
    if os.path.exists(cfg_path):
        cfg = json.load(open(cfg_path))
    else:
        os.makedirs(exp_dir, exist_ok=True)
        cfg = {
            "model": "yolo26n.pt",
            "data": "datasets/cleaned/v1/data.yaml",
            "imgsz": 640, "epochs": 75, "batch": 32,
            "optimizer": "auto", "lr0": None, "lrf": 0.01,
            "warmup_epochs": 3.0, "cos_lr": True,
            "patience": 25, "close_mosaic": 25,
            "fraction": 1.0, "seed": 0, "deterministic": True,
            "cache": False, "amp": True, "device": 0, "workers": 4,
            "project": "experiments", "mode": "run",
            "augs": dict(DEFAULT_AUGS),
        }
        with open(cfg_path, "w") as f:
            json.dump(cfg, f, indent=2)
        print(f"scaffolded {cfg_path}")
    cfg.update(overrides)
    cfg["project"] = "experiments"
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="tiny fraction run to prove plumbing")
    ap.add_argument("--fraction", type=float, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    exp_dir = args.exp.rstrip("/")
    name = os.path.basename(exp_dir)
    overrides = {}
    if args.fraction:
        overrides["fraction"] = args.fraction
    if args.epochs:
        overrides["epochs"] = args.epochs
    if args.imgsz:
        overrides["imgsz"] = args.imgsz
    if args.batch:
        overrides["batch"] = args.batch
    if args.device:
        overrides["device"] = args.device
    if args.smoke:
        overrides.update({"fraction": 0.02, "epochs": 2, "plots": False})

    cfg = load_or_scaffold(exp_dir, overrides)
    weights_dir = os.path.join(exp_dir, "weights")
    last_pt = os.path.join(weights_dir, "last.pt")
    metrics_path = os.path.join(exp_dir, "metrics.json")
    results_csv = os.path.join(exp_dir, "results.csv")

    if os.path.exists(metrics_path):
        if not args.resume:
            sys.exit(f"{exp_dir} already has metrics.json - refusing to clobber. "
                     "Use --resume only if you want to continue a partial run.")
        print(f"checkpoint found, resuming: {last_pt}")

    import torch
    print(f"GPU available: {torch.cuda.is_available()} | "
          f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU/MPS only'}")
    if not torch.cuda.is_available() and not args.smoke:
        sys.exit("no CUDA - refusing to waste CPU hours. Run this on the GPU box.")

    from ultralytics import YOLO
    model = YOLO(cfg["model"]) if not (args.resume and os.path.exists(last_pt)) else YOLO(last_pt)
    if args.resume and os.path.exists(last_pt):
        print(f"resuming from {last_pt}")

    t0 = time.time()
    train_kwargs = dict(
        data=cfg["data"], epochs=int(cfg["epochs"]), imgsz=cfg["imgsz"],
        batch=int(cfg["batch"]), device=cfg["device"], workers=cfg["workers"],
        project=cfg["project"], name=name, exist_ok=True,
        seed=cfg["seed"], deterministic=cfg["deterministic"],
        optimizer=cfg["optimizer"], lrf=cfg["lrf"],
        warmup_epochs=cfg["warmup_epochs"], cos_lr=cfg["cos_lr"],
        patience=cfg["patience"], close_mosaic=cfg["close_mosaic"],
        fraction=cfg["fraction"], cache=cfg["cache"], amp=cfg["amp"],
        **cfg["augs"],
    )
    if cfg.get("lr0"):
        train_kwargs["lr0"] = cfg["lr0"]
    if cfg.get("plots"):
        train_kwargs["plots"] = True

    if args.resume and os.path.exists(last_pt):
        model.train(resume=True)
    else:
        model.train(**train_kwargs)
    train_seconds = time.time() - t0

    # Ultralytics saves under runs/detect/<project>/<name>/ - mirror artifacts
    # into the experiment dir so everything lives together.
    run_root = os.path.join("runs", "detect", cfg["project"], name)
    os.makedirs(os.path.join(exp_dir, "weights"), exist_ok=True)
    for f in ("results.csv", "args.yaml"):
        src = os.path.join(run_root, f)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(exp_dir, f))
    for f in ("best.pt", "last.pt"):
        src = os.path.join(run_root, "weights", f)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(exp_dir, "weights", f))

    # ---- collect metrics from the best checkpoint ----
    best = os.path.join(exp_dir, "weights", "best.pt")
    if not os.path.exists(best):
        print("no best.pt produced; skipping metric collection")
        return
    m = YOLO(best)
    out_dir = os.path.join(exp_dir, "val_samples")
    os.makedirs(out_dir, exist_ok=True)
    v = m.val(data=cfg["data"], imgsz=cfg["imgsz"], batch=8,
              device=cfg["device"], save_json=True, plots=True,
              project=os.path.abspath(out_dir), name="val",
              save=True, exist_ok=True)

    per_class = {}
    boxes = getattr(v, "box", None)
    if boxes is not None:
        names = list(v.names.values())
        for idx, ap50 in zip([int(i) for i in boxes.ap_class_index], boxes.ap50):
            per_class[names[idx]] = round(float(ap50), 4)
    metrics = {
        "experiment": name,
        "config": cfg,
        "mAP50": round(float(v.box.map50), 4),
        "mAP50-95": round(float(v.box.map), 4),
        "precision": round(float(v.box.mp), 4),
        "recall": round(float(v.box.mr), 4),
        "train_seconds": round(train_seconds, 1),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "timestamp": datetime.utcnow().isoformat(),
        "per_class_ap50": per_class,
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()