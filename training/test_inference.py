import argparse
import numpy as np
from PIL import Image, ImageDraw
import onnxruntime as ort

# The 12-class list, matching idd-1/data.yaml after the caravan/trailer/train
# merge. Must stay in sync with data.yaml or every label will be wrong.
CLASSES = [
    'animal', 'autorickshaw', 'bicycle', 'bus', 'car', 'motorcycle',
    'person', 'rider', 'traffic light', 'traffic sign', 'truck',
    'vehicle fallback',
]

def run_inference(model_path, image_path, output_path, conf_threshold=0.25, imgsz=640):
    sess = ort.InferenceSession(model_path)
    img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = img.size

    img_resized = img.resize((imgsz, imgsz), Image.LANCZOS)
    img_array = np.array(img_resized, dtype=np.float32) / 255.0
    input_tensor = np.transpose(img_array, (2, 0, 1))[np.newaxis, ...]

    outputs = sess.run(["output0"], {"images": input_tensor})
    boxes = outputs[0][0]

    sx = orig_w / imgsz
    sy = orig_h / imgsz
    draw = ImageDraw.Draw(img)

    print(f"{'conf':>8} {'class':>18} {'x1':>6} {'y1':>6} {'x2':>6} {'y2':>6}")
    print("-" * 62)

    kept = []
    for i in range(300):
        x1, y1, x2, y2, conf, cls_id = boxes[i]
        if conf < conf_threshold:
            continue
        cls_name = CLASSES[int(cls_id)] if 0 <= int(cls_id) < len(CLASSES) else f"cls_{int(cls_id)}"
        x1o, y1o, x2o, y2o = x1*sx, y1*sy, x2*sx, y2*sy
        print(f"{conf:>8.4f} {cls_name:>18} {x1o:>6.0f} {y1o:>6.0f} {x2o:>6.0f} {y2o:>6.0f}")
        kept.append((x1o, y1o, x2o, y2o, conf, cls_name))

    for x1o, y1o, x2o, y2o, conf, cls in kept:
        draw.rectangle([x1o, y1o, x2o, y2o], outline="lime", width=3)
        draw.text((x1o, y1o - 16), f"{cls} {conf:.3f}", fill="lime")

    img.save(output_path)
    print(f"\nDetections above {conf_threshold}: {len(kept)}")
    print(f"Saved annotated image to: {output_path}")

    max_conf = boxes[:, 4].max()
    if max_conf > 0:
        best_idx = boxes[:, 4].argmax()
        best_cls = CLASSES[int(boxes[best_idx, 5])] if 0 <= int(boxes[best_idx, 5]) < len(CLASSES) else "?"
        print(f"Top detection: {best_cls} at confidence {max_conf:.4f}")
    print(f"Image size: {orig_w}x{orig_h}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a trained LensVoice detector on one image and draw the boxes")
    parser.add_argument("image", help="Path to input image (jpg/png)")
    parser.add_argument("-m", "--model", default="best.onnx", help="Path to best.onnx model file")
    parser.add_argument("-o", "--output", default="output.jpg", help="Path to save annotated output")
    parser.add_argument("-t", "--threshold", type=float, default=0.25, help="Confidence threshold (default: 0.25)")
    parser.add_argument("-s", "--imgsz", type=int, default=640, help="Input size the model was exported at (default: 640)")
    args = parser.parse_args()

    run_inference(args.model, args.image, args.output, args.threshold, args.imgsz)
