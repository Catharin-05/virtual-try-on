"""
segmentation_detector_lite_openvino.py

Same inputs, same output structure (overlay jpg + JSON) as
segmentation_detector.py and segmentation_detector_lite.py. Only the
segmentation backend changes: SapiensLiteSegmenter -> OpenVINOSapiensSegmenter
(see sapiens_openvino.py), which runs the same 1B weights through OpenVINO
with optional INT8 quantization -- the CPU-optimized path.

Install (on top of what segmentation_detector_lite.py needs):
    pip install openvino nncf

Usage:
    python segmentation_detector_lite_openvino.py --input_dir pipeline_output/orientation_frames --out sapiens_outputs
    # First run converts + quantizes (slow, one-time). Every run after that
    # loads the cached IR directly and is fast to start.
"""

import argparse
import glob
import json
import os

import cv2
import numpy as np

from sapiens_inference.segmentation import classes as GOLIATH_CLASSES, draw_segmentation_map
from sapiens_openvino import OpenVINOSapiensSegmenter


# ---------------------------------------------------------------------------
# Stage 2: YOLO person detector -- UNCHANGED
# ---------------------------------------------------------------------------
class PersonDetector:
    PERSON_CLASS_ID = 0

    def __init__(self, weights="yolov8n.onnx", device=None):
        from ultralytics import YOLO
        self.model = YOLO(weights)
        if device:
            self.model.to(device)

    def detect_people(self, img_bgr, conf=0.25):
        results = self.model.predict(img_bgr, classes=[self.PERSON_CLASS_ID],
                                      conf=conf, verbose=False)
        boxes = []
        for r in results:
            if r.boxes is None:
                continue
            for b in r.boxes:
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                boxes.append((x1, y1, x2, y2, float(b.conf[0])))
        return boxes


def pad_box(x1, y1, x2, y2, img_w, img_h, pad_ratio=0.05):
    w, h = x2 - x1, y2 - y1
    pad_w, pad_h = w * pad_ratio, h * pad_ratio
    nx1 = max(0, int(x1 - pad_w))
    ny1 = max(0, int(y1 - pad_h))
    nx2 = min(img_w, int(x2 + pad_w))
    ny2 = min(img_h, int(y2 + pad_h))
    return nx1, ny1, nx2, ny2


def summarize_parts(mask_region):
    unique, counts = np.unique(mask_region, return_counts=True)
    total = mask_region.size
    parts = {}
    for cls_idx, cnt in zip(unique, counts):
        cls_idx = int(cls_idx)
        if cls_idx == 0 or cls_idx >= len(GOLIATH_CLASSES):
            continue
        parts[GOLIATH_CLASSES[cls_idx]] = {
            "pixels": int(cnt),
            "percent_of_region": round(100.0 * cnt / total, 2),
        }
    return parts


def process_image(image, detector, segmenter, out_dir, name="image",
                   pad_ratio=0.05, person_conf=0.25):
    os.makedirs(out_dir, exist_ok=True)

    img = cv2.imread(image) if isinstance(image, str) else image
    if img is None:
        return {"name": name, "error": f"could not read image: {image}"}

    h, w = img.shape[:2]

    mask = segmenter.predict_mask(img)

    seg_color = draw_segmentation_map(mask)
    fg = mask > 0
    overlay = img.copy()
    blended = cv2.addWeighted(img, 0.5, seg_color, 0.7, 0)
    overlay[fg] = blended[fg]

    boxes = detector.detect_people(img, conf=person_conf)
    fallback = False
    if not boxes:
        boxes = [(0, 0, w, h, 1.0)]
        fallback = True

    people_results = []
    for i, (x1, y1, x2, y2, box_conf) in enumerate(boxes):
        px1, py1, px2, py2 = pad_box(x1, y1, x2, y2, w, h, pad_ratio)
        mask_region = mask[py1:py2, px1:px2]
        if mask_region.size == 0:
            continue

        cv2.rectangle(overlay, (px1, py1), (px2, py2), (0, 255, 0), 2)
        people_results.append({
            "person_index": i,
            "bbox": [int(px1), int(py1), int(px2), int(py2)],
            "detection_conf": round(box_conf, 3),
            "parts_detected": summarize_parts(mask_region),
        })

    out_img_path = os.path.join(out_dir, f"{name}_segmented.jpg")
    cv2.imwrite(out_img_path, overlay)

    summary = {
        "name": name,
        "image_size": [w, h],
        "used_fallback_full_image_box": fallback,
        "num_people": len(people_results),
        "people": people_results,
        "image_level_parts": summarize_parts(mask),
        "overlay_path": out_img_path,
        "mask": mask,
    }

    json_path = os.path.join(out_dir, f"{name}_segments.json")
    json_safe_summary = {k: v for k, v in summary.items() if k != "mask"}
    with open(json_path, "w") as f:
        json.dump(json_safe_summary, f, indent=2)
    summary["json_path"] = json_path

    return summary


def detect_human_parts(images, names=None, out_dir="sapiens_outputs",
                        detector=None, segmenter=None,
                        yolo_weights="yolov8n.onnx", pad_ratio=0.05,
                        quantize=True, calibration_dir=None):
    os.makedirs(out_dir, exist_ok=True)

    if detector is None:
        detector = PersonDetector(weights=yolo_weights)
    if segmenter is None:
        segmenter = OpenVINOSapiensSegmenter(quantize=quantize, calibration_dir=calibration_dir)

    if names is None:
        names = []
        for i, im in enumerate(images):
            if isinstance(im, str):
                names.append(os.path.splitext(os.path.basename(im))[0])
            else:
                names.append(f"image_{i}")

    results = []
    for image, name in zip(images, names):
        print(f"Processing: {name}")
        result = process_image(image, detector, segmenter, out_dir, name=name, pad_ratio=pad_ratio)
        results.append(result)
        if "error" in result:
            print(f"  -> {result['error']}")
        else:
            print(f"  -> {result['num_people']} person(s) found, saved {result['overlay_path']}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OpenVINO-optimized Sapiens 1B segmentation + YOLO breakdown")
    parser.add_argument("--images", nargs="+", help="One or more image file paths")
    parser.add_argument("--input_dir", help="Directory of images (jpg/png) to process")
    parser.add_argument("--out", default="sapiens_outputs", help="Output directory")
    parser.add_argument("--yolo-weights", default="yolov8n.onnx")
    parser.add_argument("--no-quantize", action="store_true", help="Use fp32 OpenVINO IR instead of INT8")
    parser.add_argument("--calibration_dir", default=None,
                         help="Folder of real frames to calibrate INT8 quantization on "
                              "(only used the first time; defaults to --input_dir if omitted)")
    args = parser.parse_args()

    image_paths = list(args.images) if args.images else []
    if args.input_dir:
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            image_paths.extend(glob.glob(os.path.join(args.input_dir, ext)))

    if not image_paths:
        parser.error("Provide --images and/or --input_dir with at least one image.")

    calib_dir = args.calibration_dir or args.input_dir

    detect_human_parts(image_paths, out_dir=args.out, yolo_weights=args.yolo_weights,
                        quantize=not args.no_quantize, calibration_dir=calib_dir)