"""
LensVoice detector training, CUDA edition (written for an RTX 4050 Laptop, 6 GB).

Three stages, selected with --stage:

  smoke     2 epochs on 5% of the data. Run this FIRST. It proves the
            environment works and prints the real it/s so you can extrapolate
            how long the full run will actually take on this machine.

  main      The real run. Trains at --imgsz (default 640) for --epochs
            (default 250) with early stopping. This is the accuracy ceiling
            and it is the one you want running tonight.

  finetune  Optional, and only once a deployment target exists. Takes the
            640-trained best.pt and adapts it down to a smaller input square
            at a low LR with mosaic off, for when a phone needs to run it
            cheaply. Skip this for now — there is no app in this repo yet.

Every stage is resumable. If the laptop sleeps, crashes or you Ctrl+C, just
re-run the exact same command: it picks up from last.pt with the original
settings instead of starting over.

Usage (from the repo root, not from inside training/):
    python training/train_gpu.py --stage smoke
    python training/train_gpu.py --stage main
"""
import argparse
import os
import shutil
import sys

PROJECT = "lensvoice_runs"  # ultralytics prepends "runs/detect/" itself
DATA = "idd-1/data.yaml"


def run_dir(name):
    return os.path.join("runs", "detect", PROJECT, name)


def resume_if_possible(name):
    """Return a YOLO resumed from last.pt, or None to start fresh."""
    from ultralytics import YOLO

    last = os.path.join(run_dir(name), "weights", "last.pt")
    if os.path.exists(last):
        print(f"Found existing checkpoint, resuming: {last}")
        return YOLO(last)
    return None


def stage_smoke(args):
    from ultralytics import YOLO

    # No resume here on purpose: a smoke test is cheap, and Ultralytics
    # refuses to resume a run that already hit its epoch count.
    name = "smoke_test"
    shutil.rmtree(run_dir(name), ignore_errors=True)

    model = YOLO(f"yolo26{args.size}.pt")
    model.train(
        data=DATA,
        epochs=2,
        fraction=0.05,        # ~1.5k of the 31k train images
        imgsz=args.imgsz,
        batch=args.batch,
        device=0,
        workers=args.workers,
        project=PROJECT,
        name=name,
        exist_ok=True,
        val=True,
        plots=False,
    )


def stage_main(args):
    from ultralytics import YOLO

    name = args.name or f"yolo26{args.size}_idd_{args.imgsz}_gpu"
    resumed = resume_if_possible(name)
    if resumed is not None:
        resumed.train(resume=True)
        return

    model = YOLO(f"yolo26{args.size}.pt")
    model.train(
        data=DATA,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=0,
        workers=args.workers,
        project=PROJECT,
        name=name,
        exist_ok=True,
        seed=0,
        deterministic=True,

        # --- schedule -------------------------------------------------
        # patience=50 is deliberately generous: YOLO26 is an end-to-end
        # (NMS-free) head and its one-to-one matching converges slowly for
        # the first ~40 epochs. A tight patience kills the run before it
        # gets going, which is roughly what happened on the Mac.
        patience=50,
        cos_lr=True,
        close_mosaic=25,      # last 25 epochs see clean, un-mosaicked images

        # --- augmentation tuned for handheld phone footage of Indian roads
        degrees=5.0,          # camera roll: the phone is held by a walking user
        scale=0.5,
        translate=0.1,
        fliplr=0.5,
        flipud=0.0,           # never flip a road scene vertically
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,            # glare, dusk, overcast
        mosaic=1.0,

        cache=False,          # 31k x 640px would need ~38 GB cached; don't
        amp=True,
    )


def stage_finetune(args):
    from ultralytics import YOLO

    src = args.weights or os.path.join(
        run_dir(args.name or f"yolo26{args.size}_idd_640_gpu"), "weights", "best.pt"
    )
    target = args.ft_imgsz
    name = f"yolo26{args.size}_idd_{target}_ft"

    resumed = resume_if_possible(name)
    if resumed is not None:
        resumed.train(resume=True)
        return

    if not os.path.exists(src):
        sys.exit(f"No source weights at {src}. Run --stage main first, or pass --weights.")

    print(f"Fine-tuning {src} down to {target}x{target}")
    model = YOLO(src)
    model.train(
        data=DATA,
        epochs=40,
        imgsz=target,
        batch=args.batch * 2,  # a smaller square costs much less memory
        device=0,
        workers=args.workers,
        project=PROJECT,
        name=name,
        exist_ok=True,
        seed=0,

        lr0=0.001,            # 10x lower: adapting, not relearning
        lrf=0.01,
        cos_lr=True,
        warmup_epochs=1.0,
        patience=15,
        mosaic=0.0,           # no mosaic when fine-tuning to a target scale
        degrees=5.0,
        scale=0.3,
        cache=False,
        amp=True,
    )


STAGES = {"smoke": stage_smoke, "main": stage_main, "finetune": stage_finetune}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", required=True, choices=STAGES)
    p.add_argument("--size", default="n", choices=["n", "s"],
                   help="yolo26 variant: n (2.6M params, default) or s (~9M, more accurate, ~3x slower on-device)")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--epochs", type=int, default=250)
    p.add_argument("--batch", type=int, default=16,
                   help="16 is safe on 6 GB at 640px. Try 24, or -1 to autotune.")
    p.add_argument("--workers", type=int, default=4 if os.name == "nt" else 8,
                   help="Windows deadlocks with high worker counts; 4 is the safe ceiling there.")
    p.add_argument("--name", default=None)
    p.add_argument("--weights", default=None, help="finetune only: explicit source .pt")
    p.add_argument("--ft-imgsz", type=int, default=320,
                   help="finetune only: the input square to adapt down to (default: 320)")
    args = p.parse_args()

    import torch
    if not torch.cuda.is_available():
        sys.exit("CUDA not available — you installed the CPU build of torch. See training/README.md.")
    print(f"GPU: {torch.cuda.get_device_name(0)} "
          f"({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB), torch {torch.__version__}")

    STAGES[args.stage](args)


if __name__ == "__main__":
    # This guard is mandatory on Windows. Without it the dataloader workers
    # re-import this module and respawn forever.
    main()
