"""
sapiens_openvino.py

CPU-optimized Sapiens-Lite segmentation for Intel CPUs. Stacks every lever
that actually helps on CPU:

  1. OpenVINO execution instead of plain PyTorch/onnxruntime CPU -- this is
     the single biggest win on Intel hardware, since OpenVINO is Intel's own
     inference runtime built around their CPU instruction sets (AVX-512,
     VNNI where available).
  2. INT8 post-training quantization on top, via NNCF, calibrated on your
     own frames -- typically another meaningful cut on top of #1, using the
     CPU's dedicated int8 instructions that fp32 can't touch.
  3. IR caching, in TWO stages: fp32 is cached immediately after conversion,
     BEFORE attempting quantization, so a slow one-time conversion is never
     lost even if quantization itself fails (e.g. runs out of memory). If
     quantization succeeds, int8 is cached separately on top. Every run after
     the first loads straight from whichever cache applies, skipping
     torch.jit.load and the conversion step entirely.
  4. Full CPU thread utilization.

Same public interface as SapiensLiteSegmenter:
    predict_mask(img_bgr) -> HxW uint8 class-index mask

Install (on top of what you already have):
    pip install openvino nncf
"""

import gc
import glob
import os

import cv2
import numpy as np
import torch

from segmentation_detector_lite import get_model_weights


class OpenVINOSapiensSegmenter:
    INPUT_H, INPUT_W = 1024, 768
    MEAN = np.array([123.5, 116.5, 103.5], dtype=np.float32).reshape(-1, 1, 1)
    STD = np.array([58.5, 57.0, 57.5], dtype=np.float32).reshape(-1, 1, 1)

    def __init__(self, quantize=True, calibration_dir=None,
                 cache_dir="models/openvino_cache", num_threads=None):
        """
        quantize: if True, attempt INT8 quantization (needs `nncf` + calibration_dir
                  with real sample frames the first time). Falls back to fp32
                  OpenVINO automatically -- not a crash -- if nncf is missing,
                  no calibration data is given, or quantization itself fails
                  (e.g. an out-of-memory error during statistics collection).
        calibration_dir: folder of real frames to calibrate quantization on.
                  Your pipeline_output/orientation_frames folder works well for this.
        cache_dir: where the converted/quantized OpenVINO IR is cached. After
                  the first successful run, subsequent runs load straight from
                  here and skip conversion (and even skip touching the .pt2 /
                  torch entirely).
        """
        import openvino as ov

        self.num_threads = num_threads or os.cpu_count()
        torch.set_num_threads(self.num_threads)
        os.environ.setdefault("OMP_NUM_THREADS", str(self.num_threads))
        os.environ.setdefault("MKL_NUM_THREADS", str(self.num_threads))

        os.makedirs(cache_dir, exist_ok=True)
        int8_path = os.path.join(cache_dir, "sapiens_1b_seg_int8.xml")
        fp32_path = os.path.join(cache_dir, "sapiens_1b_seg_fp32.xml")

        self.core = ov.Core()
        used_int8 = False

        if quantize and os.path.exists(int8_path):
            print(f"Loading cached OpenVINO IR (int8) from {int8_path} -- skipping conversion.")
            ov_model = self.core.read_model(int8_path)
            used_int8 = True

        elif os.path.exists(fp32_path):
            print(f"Loading cached OpenVINO IR (fp32) from {fp32_path} -- skipping torch conversion.")
            ov_model = self.core.read_model(fp32_path)
            if quantize:
                ov_model, used_int8 = self._try_quantize_and_cache(ov_model, calibration_dir, int8_path)

        else:
            print("No cached OpenVINO IR found -- converting from TorchScript "
                  "(one-time cost, can take a couple of minutes for a 1B model)...")
            checkpoint_path = get_model_weights(mode="torchscript")
            torch_model = torch.jit.load(checkpoint_path, map_location="cpu")
            torch_model.eval()

            dummy_input = torch.randn(1, 3, self.INPUT_H, self.INPUT_W)
            ov_model = ov.convert_model(torch_model, example_input=dummy_input)

            # Cache fp32 immediately -- if quantization fails below (e.g. OOM),
            # this conversion step is not wasted; next run resumes from here
            # instead of redoing the torch conversion.
            ov.save_model(ov_model, fp32_path)
            print(f"Cached fp32 OpenVINO IR to {fp32_path}.")

            # Free the ~4.4GB torch model before the memory-hungry quantization
            # step -- it's not needed anymore, and NNCF's statistics collection
            # needs its own significant headroom on top of the model itself.
            del torch_model
            gc.collect()

            if quantize:
                ov_model, used_int8 = self._try_quantize_and_cache(ov_model, calibration_dir, int8_path)

        config = {"INFERENCE_NUM_THREADS": str(self.num_threads), "PERFORMANCE_HINT": "LATENCY"}
        self.compiled_model = self.core.compile_model(ov_model, device_name="CPU", config=config)
        self.output_layer = self.compiled_model.output(0)
        print(f"Compiled for CPU inference ({'int8' if used_int8 else 'fp32'}).")

    def _try_quantize_and_cache(self, ov_model, calibration_dir, int8_path):
        """Attempts quantization; on ANY failure (missing deps, no calibration
        data, or a runtime error like OOM), logs it and falls back to the
        fp32 model instead of crashing the whole pipeline."""
        try:
            quantized = self._quantize(ov_model, calibration_dir)
            if quantized is ov_model:
                return ov_model, False  # _quantize itself decided to skip (see below)
            import openvino as ov
            ov.save_model(quantized, int8_path)
            print(f"Cached int8 OpenVINO IR to {int8_path}.")
            return quantized, True
        except Exception as e:
            print(f"Quantization failed ({type(e).__name__}: {e}) -- falling back to fp32 OpenVINO. "
                  f"fp32 is already cached, so this fallback is immediate on future runs too. "
                  f"If this was a memory error, try closing other applications and rerunning, "
                  f"or pass quantize=False to skip INT8 entirely.")
            gc.collect()
            return ov_model, False

    def _quantize(self, ov_model, calibration_dir):
        try:
            import nncf
        except ImportError:
            print("nncf not installed -- skipping quantization, using fp32 OpenVINO IR. "
                  "Run `pip install nncf` to enable INT8.")
            return ov_model

        calib_paths = []
        if calibration_dir:
            for ext in ("*.jpg", "*.jpeg", "*.png"):
                calib_paths.extend(glob.glob(os.path.join(calibration_dir, ext)))
        if not calib_paths:
            print("No calibration_dir / no images found in it -- skipping quantization, "
                  "using fp32 OpenVINO IR. Pass calibration_dir= pointing at real "
                  "frames (e.g. your orientation_frames folder) to enable INT8.")
            return ov_model

        print(f"Quantizing with {len(calib_paths)} calibration frame(s) from {calibration_dir}...")
        tensors = [self._preprocess(cv2.imread(p))[0] for p in calib_paths]
        dataset = nncf.Dataset(tensors, lambda x: x)
        return nncf.quantize(ov_model, dataset)

    def _preprocess(self, img_bgr):
        h, w = img_bgr.shape[:2]
        resized = cv2.resize(img_bgr, (self.INPUT_W, self.INPUT_H), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)
        chw = rgb.transpose(2, 0, 1)
        normed = (chw - self.MEAN) / self.STD
        return normed[np.newaxis, ...].astype(np.float32), (h, w)

    def predict_mask(self, img_bgr):
        """Full-resolution HxW uint8 class-index mask for the whole image."""
        inp, (orig_h, orig_w) = self._preprocess(img_bgr)
        result = self.compiled_model([inp])[self.output_layer]  # [1, num_classes, H, W]

        result_t = torch.from_numpy(result)
        result_t = torch.nn.functional.interpolate(
            result_t, size=(orig_h, orig_w), mode="bilinear", align_corners=False
        )
        mask = result_t.argmax(dim=1).squeeze(0).numpy()
        return mask.astype(np.uint8)