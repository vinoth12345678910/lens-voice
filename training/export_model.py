"""
Export a trained checkpoint to a deployment format, and report the real tensor
shapes so whatever consumes it can be written against facts.

There is no app in this repo right now, so nothing here is pinned to a
particular input size or tensor layout. When the React Native side exists,
export at whatever size it needs and read the shapes this prints.

    python training/export_model.py runs/detect/lensvoice_runs/<run>/weights/best.pt
    python training/export_model.py <best.pt> --format tflite --imgsz 320
    python training/export_model.py <best.pt> --format onnx --imgsz 640

Formats worth knowing:
    onnx    good for desktop testing (training/test_inference.py reads this)
    tflite  Android, and React Native via a LiteRT binding
    coreml  iOS

YOLO26 is an end-to-end, NMS-free architecture, so the output comes out
already decoded and deduplicated as [1, 300, 6] — x1, y1, x2, y2, conf,
class_id, with the box in PIXELS of the input square. No NMS post-processing
needed on the consumer side.

int8 is off by default on purpose: quantising a 2.6M-parameter model costs
real recall on small and distant objects, which are exactly the ones this
detector most needs to catch.
"""
import argparse
import os
import sys

CLASSES = [
    'animal', 'autorickshaw', 'bicycle', 'bus', 'car', 'motorcycle',
    'person', 'rider', 'traffic light', 'traffic sign', 'truck',
    'vehicle fallback',
]


def report_tflite(path):
    try:
        import tensorflow as tf
    except ImportError:
        print("\n  (install tensorflow to inspect the .tflite tensor shapes)")
        return
    interp = tf.lite.Interpreter(model_path=path)
    interp.allocate_tensors()
    i, o = interp.get_input_details()[0], interp.get_output_details()[0]
    print(f"\n  input : {list(i['shape'])}  {i['dtype'].__name__}")
    print(f"  output: {list(o['shape'])}  {o['dtype'].__name__}")
    shape = list(i["shape"])
    if len(shape) == 4 and shape[1] == 3:
        print("  layout: NCHW (channels-first) — feed three planar R,G,B blocks")
    elif len(shape) == 4 and shape[3] == 3:
        print("  layout: NHWC (channels-last) — feed interleaved RGB pixels")
    print("  pixels: scale to 0.0-1.0, RGB order")


def report_onnx(path):
    try:
        import onnxruntime as ort
    except ImportError:
        print("\n  (install onnxruntime to inspect the .onnx tensor shapes)")
        return
    s = ort.InferenceSession(path)
    for x in s.get_inputs():
        print(f"\n  input : {x.name} {x.shape} {x.type}")
    for x in s.get_outputs():
        print(f"  output: {x.name} {x.shape} {x.type}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("weights", help="path to best.pt")
    p.add_argument("--format", default="onnx",
                   choices=["onnx", "tflite", "coreml", "torchscript"])
    p.add_argument("--imgsz", type=int, default=640,
                   help="input square size to export at (default: 640)")
    p.add_argument("--int8", action="store_true",
                   help="quantise to int8 — 3x smaller, measurably worse on small objects")
    args = p.parse_args()

    if not os.path.exists(args.weights):
        sys.exit(f"No such checkpoint: {args.weights}")

    from ultralytics import YOLO
    import ultralytics
    print(f"ultralytics {ultralytics.__version__}")
    print(f"Exporting {args.weights} -> {args.format} @ {args.imgsz}x{args.imgsz}"
          f"{' (int8)' if args.int8 else ''}")

    model = YOLO(args.weights)
    path = model.export(format=args.format, imgsz=args.imgsz, int8=args.int8, nms=False)
    print(f"\nExported: {path}")

    if args.format == "tflite":
        report_tflite(path)
    elif args.format == "onnx":
        report_onnx(path)

    print(f"\n  classes ({len(CLASSES)}): {', '.join(CLASSES)}")
    print("\nSanity-check it on a real photo:")
    print(f"  python training/test_inference.py <some.jpg> -m {path} -s {args.imgsz}")


if __name__ == "__main__":
    main()
