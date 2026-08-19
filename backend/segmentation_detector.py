"""
segmentation_detector.py

Detects human body parts in images using:
  1. Sapiens-1B (via the `sapiens_inference` wrapper) -> 28-class body-part
     segmentation (face/neck, hair, torso, arms, legs, hands, feet, shoes,
     socks, clothing, lips, teeth, tongue, ...) run on the FULL frame.
  2. YOLO -> person bounding boxes, used AFTER segmentation to split the
     resulting mask into a per-person body-part breakdown.

IMPORTANT — why Sapiens is NOT run on YOLO crops:
The sapiens_inference library's own author found that running Sapiens on a
cropped person is worse than running it on the full frame, even with a
generous margin around the box:

    "Running Sapiens models on a cropped person produces worse results,
     even if you crop a wider rectangle around the person."
    (https://github.com/ibaiGorordo/Sapiens-Pytorch-Inference)

This is also visible directly in the library's own predictor.py:
    self.detector = None  # Detector(config.detector_config)
    # TODO: Cropping seems to make the results worse

So this script segments the whole image once (matching how the model was
trained/benchmarked) and only uses YOLO boxes afterwards, as a spatial
filter over the resulting mask, to report which parts belong to which
detected person. Semantic segmentation classes aren't instance-aware, so
if two people overlap heavily in the frame their part counts can overlap
too -- that's an inherent limitation of this approach, not a bug.

Install:
    pip install git+https://github.com/ibaiGorordo/Sapiens-Pytorch-Inference.git ultralytics
    (installing from GitHub, not the "sapiens-inferece" PyPI package, which is
     a stale release -- the GitHub main branch has fixes PyPI doesn't, e.g. a
     Python 3.11+ dataclass bug. This script auto-patches that bug too, as a
     safety net, in case you or someone downstream installs the PyPI version.)

Usage (standalone):
    python segmentation_detector.py --images img1.jpg img2.jpg --out sapiens_outputs
    python segmentation_detector.py --input_dir ./my_frames --out sapiens_outputs
"""

import argparse
import glob
import importlib.util
import json
import os
import sys

import cv2
import numpy as np
import torch


def _patch_sapiens_inference_if_needed():
    """
    sapiens-inferece==0.2.0 has a packaging bug: SapiensConfig declares
        detector_config: DetectorConfig = DetectorConfig()
    i.e. a mutable dataclass *instance* as a plain default. Python's
    dataclass machinery rejects this on Python 3.11+ with:
        ValueError: mutable default ... not allowed: use default_factory
    We don't even use SapiensConfig/SapiensPredictor in this script, but
    `sapiens_inference/__init__.py` imports them unconditionally, so the
    whole package fails to import on Python 3.11+ (including current-gen
    Colab runtimes) until this is fixed. This patches the installed file
    in place, once, the first time it's needed.
    """
    try:
        import sapiens_inference  # noqa: F401
        return
    except ModuleNotFoundError as e:
        if e.name != "sapiens_inference":
            raise  # some other missing dependency (e.g. ultralytics) -- surface it as-is
        raise ModuleNotFoundError(
            "The 'sapiens_inference' package isn't installed. Install it with:\n\n"
            "    pip install git+https://github.com/ibaiGorordo/Sapiens-Pytorch-Inference.git ultralytics\n\n"
            "(install from GitHub, not the stale 'sapiens-inferece' PyPI package)"
        ) from e
    except ValueError as e:
        if "mutable default" not in str(e):
            raise

    # Locate the package dir WITHOUT importing it (a dotted find_spec would
    # import the parent package first and hit the same crash).
    spec = importlib.util.find_spec("sapiens_inference")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("Could not locate the sapiens_inference package to patch it.")
    pkg_dir = list(spec.submodule_search_locations)[0]
    predictor_path = os.path.join(pkg_dir, "predictor.py")

    with open(predictor_path) as f:
        src = f.read()

    fixed = src.replace(
        "from dataclasses import dataclass",
        "from dataclasses import dataclass, field",
    ).replace(
        "detector_config: DetectorConfig = DetectorConfig()",
        "detector_config: DetectorConfig = field(default_factory=DetectorConfig)",
    )

    if fixed == src:
        raise RuntimeError(
            "sapiens_inference failed to import with a 'mutable default' error, "
            "but the expected buggy line in predictor.py wasn't found to "
            "auto-patch (the package version may have changed). Patch it "
            "manually: change `detector_config: DetectorConfig = DetectorConfig()` "
            "to use `field(default_factory=DetectorConfig)` in predictor.py."
        )

    with open(predictor_path, "w") as f:
        f.write(fixed)
    print(f"[segmentation_detector] Patched a Python 3.11+ compatibility bug "
          f"in sapiens_inference at: {predictor_path}")

    for mod_name in list(sys.modules):
        if mod_name.startswith("sapiens_inference"):
            del sys.modules[mod_name]
    import sapiens_inference  # noqa: F401  (retry; raises again if still broken)


_patch_sapiens_inference_if_needed()

from sapiens_inference import SapiensSegmentation, SapiensSegmentationType
from sapiens_inference.segmentation import classes as GOLIATH_CLASSES, draw_segmentation_map


# ---------------------------------------------------------------------------
# Stage 1: Sapiens body-part segmentation (full frame, no cropping)
# ---------------------------------------------------------------------------
class SapiensSegmenter:
    def __init__(self, device=None, dtype=torch.float32,
                 seg_type=SapiensSegmentationType.SEGMENTATION_1B):
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading Sapiens segmentation model onto {device} "
              f"(downloads on first run, then cached under ./models)...")
        self._segmentor = SapiensSegmentation(type=seg_type, device=device, dtype=dtype)

    def predict_mask(self, img_bgr):
        """Full-resolution HxW uint8 class-index mask for the whole image."""
        mask = self._segmentor(img_bgr)  # already resized back to img's own H,W
        return mask.astype(np.uint8)


# ---------------------------------------------------------------------------
# Stage 2: YOLO person detector (post-hoc mask slicing, not model input)
# ---------------------------------------------------------------------------
class PersonDetector:
    PERSON_CLASS_ID = 0  # COCO class id for "person"

    def __init__(self, weights="yolov8m.pt", device=None):
        from ultralytics import YOLO
        self.model = YOLO(weights)
        if device:
            self.model.to(device)

    def detect_people(self, img_bgr, conf=0.25):
        """Returns a list of (x1, y1, x2, y2, confidence) boxes, person-only."""
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
    """Expand a box slightly (to catch hair/clothing right at the edge of
    the YOLO box) when slicing the already-computed full-frame mask."""
    w, h = x2 - x1, y2 - y1
    pad_w, pad_h = w * pad_ratio, h * pad_ratio
    nx1 = max(0, int(x1 - pad_w))
    ny1 = max(0, int(y1 - pad_h))
    nx2 = min(img_w, int(x2 + pad_w))
    ny2 = min(img_h, int(y2 + pad_h))
    return nx1, ny1, nx2, ny2


def summarize_parts(mask_region):
    """Pixel counts / percentages for each non-background class in a mask region."""
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


# ---------------------------------------------------------------------------
# Per-image processing
# ---------------------------------------------------------------------------
def process_image(image, detector, segmenter, out_dir, name="image",
                   pad_ratio=0.05, person_conf=0.25):
    """
    image: file path (str) or an already-loaded BGR numpy image.
    Returns a JSON-serializable summary dict and writes an overlay + json
    to out_dir.
    """
    img = cv2.imread(image) if isinstance(image, str) else image
    if img is None:
        return {"name": name, "error": f"could not read image: {image}"}

    h, w = img.shape[:2]

    # Segment the WHOLE frame once (matches how Sapiens was trained/benchmarked).
    mask = segmenter.predict_mask(img)

    # Overlay for the whole image.
    seg_color = draw_segmentation_map(mask)
    fg = mask > 0
    overlay = img.copy()
    blended = cv2.addWeighted(img, 0.5, seg_color, 0.7, 0)
    overlay[fg] = blended[fg]

    # Locate people, then slice the mask per person for a per-person breakdown.
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
        "image_level_parts": summarize_parts(mask),  # whole-frame breakdown, instance-agnostic
        "overlay_path": out_img_path,
        "mask": mask,  # raw HxW uint8 class-index array, kept for downstream
                        # in-memory use (e.g. anthropometry.py) -- NOT written to JSON
    }

    json_path = os.path.join(out_dir, f"{name}_segments.json")
    json_safe_summary = {k: v for k, v in summary.items() if k != "mask"}
    with open(json_path, "w") as f:
        json.dump(json_safe_summary, f, indent=2)
    summary["json_path"] = json_path

    return summary


# ---------------------------------------------------------------------------
# Batch entry point
# ---------------------------------------------------------------------------
def detect_human_parts(images, names=None, out_dir="sapiens_outputs",
                        detector=None, segmenter=None,
                        yolo_weights="yolov8m.pt", pad_ratio=0.05):
    """
    images: list of file paths (str) and/or BGR numpy arrays.
    names:  optional list of output-file basenames (auto-derived if omitted).
    detector/segmenter: pass in already-loaded instances to avoid reloading
                         the models repeatedly (e.g. from a pipeline).
    """
    os.makedirs(out_dir, exist_ok=True)

    if detector is None:
        detector = PersonDetector(weights=yolo_weights)
    if segmenter is None:
        segmenter = SapiensSegmenter()

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
        result = process_image(image, detector, segmenter, out_dir,
                                name=name, pad_ratio=pad_ratio)
        results.append(result)
        if "error" in result:
            print(f"  -> {result['error']}")
        else:
            print(f"  -> {result['num_people']} person(s) found, "
                  f"saved {result['overlay_path']}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sapiens body-part segmentation + YOLO per-person breakdown")
    parser.add_argument("--images", nargs="+", help="One or more image file paths")
    parser.add_argument("--input_dir", help="Directory of images (jpg/png) to process")
    parser.add_argument("--out", default="sapiens_outputs", help="Output directory")
    parser.add_argument("--yolo-weights", default="yolov8m.pt", help="YOLO weights for person detection")
    args = parser.parse_args()

    image_paths = list(args.images) if args.images else []
    if args.input_dir:
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            image_paths.extend(glob.glob(os.path.join(args.input_dir, ext)))

    if not image_paths:
        parser.error("Provide --images and/or --input_dir with at least one image.")

    detect_human_parts(image_paths, out_dir=args.out, yolo_weights=args.yolo_weights)