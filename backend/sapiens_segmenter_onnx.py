"""
sapiens_segmenter_onnx.py

ONNX Runtime version of segmentation_detector.py's SapiensSegmenter -- same
public interface (predict_mask(img_bgr) -> HxW uint8 class-index mask at
the original image resolution), so it's a drop-in replacement wherever
SapiensSegmenter is used.

Matches the export script you used:
    dummy_input = torch.randn(1, 3, 1024, 768, device="cpu")
    torch.onnx.export(model, dummy_input, "sapiens_1b_goliath_seg.onnx", ...)

Since that exports the raw TorchScript module unchanged (no argmax added),
the ONNX graph's output is raw per-class logits, shape (1, 28, 1024, 768) --
the argmax-over-classes step happens here, in numpy, after inference.

The preprocessing below (resize to 1024x768, RGB, ImageNet mean/std,
CHW) matches what I verified directly from the real sapiens_inference
package source earlier in this project -- but I have NOT been able to run
this against your actual .onnx file (I don't have the multi-GB checkpoint).
Run diagnose_sapiens_onnx() first (see bottom of this file) before trusting
it in the full pipeline -- it prints the real input/output shapes from your
model and a quick sanity check on the produced mask, so any mismatch with
what's assumed here shows up immediately instead of silently producing a
wrong mask.

Install:
    pip install onnxruntime opencv-python numpy
"""

import cv2
import numpy as np
import onnxruntime as ort


class SapiensSegmenterONNX:
    IMG_H, IMG_W = 1024, 768  # fixed by the export -- dynamic_axes only covers batch
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(self, onnx_path="sapiens_1b_goliath_seg.onnx", providers=None, verbose=True):
        providers = providers or ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name

        if verbose:
            in_shape = self.session.get_inputs()[0].shape
            out_shape = self.session.get_outputs()[0].shape
            print(f"[SapiensSegmenterONNX] input '{self.input_name}': shape={in_shape}")
            print(f"[SapiensSegmenterONNX] output '{self.session.get_outputs()[0].name}': shape={out_shape}")
            if len(out_shape) == 4 and isinstance(out_shape[1], int) and out_shape[1] != 28:
                print(f"[SapiensSegmenterONNX] WARNING: expected 28 classes on axis 1, "
                      f"got {out_shape[1]} -- check GOLIATH_CLASSES still lines up.")

    def predict_mask(self, img_bgr):
        """Full-resolution HxW uint8 class-index mask for the whole image."""
        original_h, original_w = img_bgr.shape[:2]

        resized = cv2.resize(img_bgr, (self.IMG_W, self.IMG_H))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        normed = (rgb - self.MEAN) / self.STD
        chw = normed.transpose(2, 0, 1)
        blob = np.expand_dims(chw, 0).astype(np.float32)

        output = self.session.run(None, {self.input_name: blob})[0]  # (1, C, H, W) raw logits

        if output.shape[1] == 1:
            # argmax was somehow already applied in the graph after all
            mask = output[0, 0].astype(np.uint8)
        else:
            mask = np.argmax(output[0], axis=0).astype(np.uint8)  # (H, W)

        return cv2.resize(mask, (original_w, original_h), interpolation=cv2.INTER_NEAREST)


def diagnose_sapiens_onnx(onnx_path="sapiens_1b_goliath_seg.onnx", test_image_path=None):
    """
    Run this FIRST, before wiring SapiensSegmenterONNX into the real
    pipeline. Prints real input/output shapes and, if given a test image,
    runs a real prediction and reports basic sanity stats (unique classes
    found, whether a person-shaped silhouette exists) so you can confirm
    the model behaves as expected before trusting it downstream.
    """
    segmenter = SapiensSegmenterONNX(onnx_path, verbose=True)

    if test_image_path is None:
        print("\nNo test image given -- pass test_image_path= to run a real "
              "prediction and see class statistics.")
        return

    img = cv2.imread(test_image_path)
    if img is None:
        print(f"Could not read {test_image_path}")
        return

    mask = segmenter.predict_mask(img)
    unique, counts = np.unique(mask, return_counts=True)
    total = mask.size
    print(f"\nMask shape: {mask.shape} (should match input image {img.shape[:2]})")
    print("Classes found (index: pixel % of image):")
    for cls_idx, cnt in sorted(zip(unique, counts), key=lambda x: -x[1]):
        print(f"  {int(cls_idx):>2}: {100*cnt/total:5.1f}%")

    non_background_pct = 100 * (mask > 0).sum() / total
    if non_background_pct < 1:
        print("\nWARNING: <1% of the image is non-background -- the model may not "
              "be detecting a person at all. Check the input image has a visible "
              "person and that preprocessing matches your export.")
    else:
        print(f"\n{non_background_pct:.1f}% of the image classified as a body part "
              f"-- looks like the model is finding a person.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Diagnose a Sapiens ONNX export before trusting it in the pipeline")
    parser.add_argument("--onnx", default="sapiens_1b_goliath_seg.onnx")
    parser.add_argument("--test-image", default=None, help="Path to a test image with a visible person")
    args = parser.parse_args()

    diagnose_sapiens_onnx(args.onnx, args.test_image)