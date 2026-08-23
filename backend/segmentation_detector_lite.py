"""
segmentation_detector_lite.py

Same pipeline and same output structure as segmentation_detector.py, but the
Sapiens segmentation stage is swapped from the `sapiens_inference` PyTorch
wrapper (SEGMENTATION_1B) to the official Sapiens-Lite TorchScript checkpoint,
loaded directly with `torch.jit.load` -- no shell scripts, no sapiens/lite
repo clone required, everything stays in-process.

Why this is faster with NO accuracy loss:
Sapiens-Lite ships the SAME 1B weights, just exported as an optimized
TorchScript graph (.pt2) instead of the training-oriented checkpoint the
regular `sapiens_inference` package loads. Per the official repo, this lite
inference path is ~4x faster with the same predictions (float32 TorchScript
mode is documented as "closest to original model performance"; minor
numerical variation vs the original float32 .pth is possible but not an
accuracy regression from using a smaller model).

What changed vs. segmentation_detector.py:
  - SapiensSegmenter -> SapiensLiteSegmenter: loads a .pt2 TorchScript
    checkpoint directly instead of instantiating sapiens_inference's
    SapiensSegmentation(type=SEGMENTATION_1B).
  - Everything else (PersonDetector, pad_box, summarize_parts,
    process_image, detect_human_parts, the JSON/overlay output shape) is
    UNCHANGED, because it all just consumes the HxW uint8 class-index mask,
    regardless of which backend produced it.
  - We still import GOLIATH_CLASSES and draw_segmentation_map from
    sapiens_inference.segmentation, purely as label/visualization utilities
    (same 28-class Goliath taxonomy) -- sapiens_inference is not used to run
    the model itself here, just for these two helper objects.

Install:
    pip install torch torchvision opencv-python huggingface_hub
    pip install git+https://github.com/ibaiGorordo/Sapiens-Pytorch-Inference.git
        # ^ only used here for the GOLIATH_CLASSES list + draw_segmentation_map,
        #   NOT for loading/running the model. See get_model_weights() below
        #   for how the actual .pt2 checkpoint is fetched.
    pip install ultralytics   # YOLO person detector, same as before

Usage (standalone):
    python segmentation_detector_lite.py --images img1.jpg img2.jpg --out sapiens_outputs
    python segmentation_detector_lite.py --input_dir ./my_frames --out sapiens_outputs
"""

import argparse
import glob
import json
import os

import cv2
import numpy as np
import torch

from sapiens_inference.segmentation import classes as GOLIATH_CLASSES, draw_segmentation_map


# ---------------------------------------------------------------------------
# Checkpoint download
# ---------------------------------------------------------------------------
# IMPORTANT correction: facebook/sapiens on HuggingFace does NOT host any
# checkpoint files itself -- it's a landing page that links out to separate
# per-task repos. The 1B segmentation TorchScript checkpoint actually lives
# in its own repo, at the repo root (no sapiens_lite_host/... subfolder):
#
#   facebook/sapiens-seg-1b-torchscript
#     sapiens_1b_goliath_best_goliath_mIoU_7994_epoch_151_torchscript.pt2
#
# Other sizes/tasks follow the same "facebook/sapiens-<task>-<size>-<mode>"
# repo naming pattern (e.g. facebook/sapiens-seg-1b-bfloat16,
# facebook/sapiens-normal-1b-torchscript, facebook/sapiens-depth-1b-torchscript).
# If a download still 404s, browse https://huggingface.co/facebook and
# search "sapiens-seg" to confirm the exact repo name + filename.
SEG_1B_REPO = {
    "torchscript": "facebook/sapiens-seg-1b-torchscript",
    "bfloat16": "facebook/sapiens-seg-1b-bfloat16",
}
SEG_1B_FILENAME = "sapiens_1b_goliath_best_goliath_mIoU_7994_epoch_151_{mode}.pt2"


def get_model_weights(mode="torchscript", model_dir="models"):
    """
    Downloads (and caches) a Sapiens-Lite .pt2 checkpoint from HuggingFace,
    mirroring the download helper style sapiens_inference itself uses, so
    you don't need to `git clone facebookresearch/sapiens` or run their
    shell scripts just to get a checkpoint path.

    mode: "torchscript" (fp32, any GPU) or "bfloat16" (A100-only, faster).
    """
    from huggingface_hub import hf_hub_download

    repo_id = SEG_1B_REPO[mode]
    filename = SEG_1B_FILENAME.format(mode=mode)

    os.makedirs(model_dir, exist_ok=True)
    print(f"Fetching {filename} from {repo_id}...")
    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=model_dir,
    )
    print(f"Checkpoint ready at: {path}")
    return path


# ---------------------------------------------------------------------------
# Stage 1 (replacement): Sapiens-Lite body-part segmentation, full frame
# ---------------------------------------------------------------------------
class SapiensLiteSegmenter:
    """
    Drop-in replacement for the SapiensSegmenter in segmentation_detector.py.
    Same public interface: predict_mask(img_bgr) -> HxW uint8 class-index mask.
    """

    # Fixed input size Sapiens was trained/benchmarked at -- same constraint
    # noted in the sapiens_inference README ("Input sizes other than
    # 768x1024 don't produce good results"). Do not change this.
    INPUT_H, INPUT_W = 1024, 768
    MEAN = torch.tensor([123.5, 116.5, 103.5]).view(-1, 1, 1)
    STD = torch.tensor([58.5, 57.0, 57.5]).view(-1, 1, 1)

    def __init__(self, checkpoint_path=None, device=None, mode="torchscript"):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if checkpoint_path is None:
            checkpoint_path = get_model_weights(mode=mode)

        print(f"Loading Sapiens-Lite segmentation TorchScript model onto {self.device}...")
        self.model = torch.jit.load(checkpoint_path, map_location=self.device)
        self.model.eval()
        self.model.to(self.device)

        self.dtype = torch.bfloat16 if mode == "bfloat16" else torch.float32
        if self.dtype == torch.bfloat16:
            self.model = self.model.to(self.dtype)

    def _preprocess(self, img_bgr):
        h, w = img_bgr.shape[:2]
        resized = cv2.resize(img_bgr, (self.INPUT_W, self.INPUT_H), interpolation=cv2.INTER_LINEAR)
        # BGR -> RGB, HWC -> CHW
        tensor = torch.from_numpy(resized).permute(2, 0, 1)[[2, 1, 0], ...].float()
        tensor = (tensor - self.MEAN) / self.STD
        return tensor.unsqueeze(0), (h, w)

    @torch.no_grad()
    def predict_mask(self, img_bgr):
        """Full-resolution HxW uint8 class-index mask for the whole image."""
        inp, (orig_h, orig_w) = self._preprocess(img_bgr)
        inp = inp.to(self.device, dtype=self.dtype)

        output = self.model(inp)  # [1, num_classes, INPUT_H, INPUT_W]
        output = output.float()
        output = torch.nn.functional.interpolate(
            output, size=(orig_h, orig_w), mode="bilinear", align_corners=False
        )
        mask = output.argmax(dim=1).squeeze(0).cpu().numpy()
        return mask.astype(np.uint8)


# ---------------------------------------------------------------------------
# Stage 2: YOLO person detector -- UNCHANGED from segmentation_detector.py
# ---------------------------------------------------------------------------
class PersonDetector:
    PERSON_CLASS_ID = 0  # COCO class id for "person"

    def __init__(self, weights="yolov8n.onnx", device=None):
        from ultralytics import YOLO
        self.model = YOLO(weights)  # loads .pt or .onnx transparently
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


# ---------------------------------------------------------------------------
# Per-image processing -- UNCHANGED from segmentation_detector.py
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Batch entry point -- UNCHANGED aside from default segmenter class
# ---------------------------------------------------------------------------
def detect_human_parts(images, names=None, out_dir="sapiens_outputs",
                        detector=None, segmenter=None,
                        yolo_weights="yolov8n.onnx", pad_ratio=0.05,
                        lite_mode="torchscript"):
    os.makedirs(out_dir, exist_ok=True)

    if detector is None:
        detector = PersonDetector(weights=yolo_weights)
    if segmenter is None:
        segmenter = SapiensLiteSegmenter(mode=lite_mode)

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
    parser = argparse.ArgumentParser(description="Sapiens-Lite body-part segmentation + YOLO per-person breakdown")
    parser.add_argument("--images", nargs="+", help="One or more image file paths")
    parser.add_argument("--input_dir", help="Directory of images (jpg/png) to process")
    parser.add_argument("--out", default="sapiens_outputs", help="Output directory")
    parser.add_argument("--yolo-weights", default="yolov8n.onnx", help="YOLO weights/model for person detection (.pt or .onnx)")
    parser.add_argument("--mode", default="torchscript", choices=["torchscript", "bfloat16"],
                         help="Sapiens-Lite mode: torchscript=fp32/any GPU, bfloat16=A100-only/fastest")
    args = parser.parse_args()

    image_paths = list(args.images) if args.images else []
    if args.input_dir:
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            image_paths.extend(glob.glob(os.path.join(args.input_dir, ext)))

    if not image_paths:
        parser.error("Provide --images and/or --input_dir with at least one image.")

    detect_human_parts(image_paths, out_dir=args.out, yolo_weights=args.yolo_weights,
                        lite_mode=args.mode)