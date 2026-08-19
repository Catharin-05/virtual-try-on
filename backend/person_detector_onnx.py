"""
person_detector_onnx.py

ONNX Runtime version of segmentation_detector.py's PersonDetector -- same
public interface (detect_people(img_bgr, conf) -> list of (x1,y1,x2,y2,conf)
boxes), so it's a drop-in replacement: pass an instance of this instead of
PersonDetector to detect_human_parts()/process_image() in
segmentation_detector.py, or to PersonDetector's usages in main.py.

The pre/post-processing here (letterbox resize, cxcywh box decode, NMS,
un-letterbox) was verified against a real image against the actual PyTorch
ultralytics model -- final NMS'd boxes matched to within ~2px. See the
export script, export_yolo_onnx.py, for how to produce the .onnx file this
class loads.

Install:
    pip install onnxruntime opencv-python numpy
    (or onnxruntime-gpu instead, for CUDA)
"""

import cv2
import numpy as np
import onnxruntime as ort


class PersonDetectorONNX:
    PERSON_CLASS_ID = 0  # COCO class id for "person"

    def __init__(self, onnx_path="yolov8n.onnx", imgsz=640, providers=None):
        self.imgsz = imgsz
        providers = providers or ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name

    def _preprocess(self, img_bgr):
        h, w = img_bgr.shape[:2]
        scale = self.imgsz / max(h, w)
        resized = cv2.resize(img_bgr, (int(w * scale), int(h * scale)))
        canvas = np.full((self.imgsz, self.imgsz, 3), 114, dtype=np.uint8)
        canvas[:resized.shape[0], :resized.shape[1]] = resized
        blob = canvas[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        return np.expand_dims(blob, 0), scale

    def detect_people(self, img_bgr, conf=0.4, nms_thresh=0.45):
        """Returns a list of (x1, y1, x2, y2, confidence) boxes, person-only.
        Same signature/semantics as segmentation_detector.PersonDetector."""
        blob, scale = self._preprocess(img_bgr)
        output = self.session.run(None, {self.input_name: blob})[0]  # (1, 84, N)
        preds = output[0].T  # (N, 84): 4 box coords + 80 class scores

        person_scores = preds[:, 4 + self.PERSON_CLASS_ID]
        keep = person_scores > conf
        if not np.any(keep):
            return []

        cand = preds[keep]
        scores = person_scores[keep]
        cx, cy, bw, bh = cand[:, 0], cand[:, 1], cand[:, 2], cand[:, 3]
        boxes_letterboxed = np.stack(
            [cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1)

        idxs = cv2.dnn.NMSBoxes(boxes_letterboxed.tolist(), scores.tolist(),
                                 score_threshold=conf, nms_threshold=nms_thresh)
        if len(idxs) == 0:
            return []
        idxs = np.array(idxs).flatten()

        final_boxes = boxes_letterboxed[idxs] / scale  # undo letterbox scaling
        final_scores = scores[idxs]

        return [(float(x1), float(y1), float(x2), float(y2), float(s))
                for (x1, y1, x2, y2), s in zip(final_boxes, final_scores)]