"""
demo_runner.py

Loads the OpenVINO-optimized Sapiens segmenter + YOLO detector ONCE, then
lets you run the pipeline on any frame folder repeatedly without reloading.
This is the fix for paying model-load time on every demo -- start this once
before your demo session, then just point it at a folder each time.

Usage:
    python demo_runner.py
    # then, when prompted, type a frame folder path (repeatable, Ctrl+C to quit):
    #   pipeline_output/orientation_frames
    #   pipeline_output/some_other_frames
    #   ...

Or non-interactive, process one folder and exit (model still only loads once):
    python demo_runner.py --input_dir pipeline_output/orientation_frames --out sapiens_outputs
"""

import argparse
import glob
import os
import time

from segmentation_detector_lite_openvino import (
    PersonDetector,
    detect_human_parts,
)
from sapiens_openvino import OpenVINOSapiensSegmenter


def run_once(input_dir, out_dir, detector, segmenter):
    image_paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        image_paths.extend(glob.glob(os.path.join(input_dir, ext)))
    if not image_paths:
        print(f"  No frames found in {input_dir}, skipping.")
        return

    start = time.perf_counter()
    detect_human_parts(image_paths, out_dir=out_dir, detector=detector, segmenter=segmenter)
    elapsed = time.perf_counter() - start
    print(f"  -> {len(image_paths)} frame(s) in {elapsed:.2f}s "
          f"({elapsed / len(image_paths):.2f}s/frame), saved to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Warm-loaded demo runner for the OpenVINO Sapiens pipeline")
    parser.add_argument("--input_dir", default=None,
                         help="If given, process this folder once and exit. "
                              "If omitted, enters interactive mode.")
    parser.add_argument("--out", default="sapiens_outputs")
    parser.add_argument("--yolo-weights", default="yolov8n.onnx")
    parser.add_argument("--no-quantize", action="store_true")
    parser.add_argument("--calibration_dir", default=None,
                         help="Real frames to calibrate INT8 on the first time (only needed once ever)")
    args = parser.parse_args()

    print("Loading models once for this session (first run also converts/quantizes -- be patient)...")
    load_start = time.perf_counter()
    detector = PersonDetector(weights=args.yolo_weights)
    segmenter = OpenVINOSapiensSegmenter(
        quantize=not args.no_quantize,
        calibration_dir=args.calibration_dir or args.input_dir,
    )
    print(f"Models loaded in {time.perf_counter() - load_start:.2f}s. Ready.\n")

    if args.input_dir:
        run_once(args.input_dir, args.out, detector, segmenter)
    else:
        print("Interactive mode -- enter a frame folder path to process it (Ctrl+C to quit).")
        try:
            while True:
                folder = input("\nFrame folder> ").strip()
                if not folder:
                    continue
                if not os.path.isdir(folder):
                    print(f"  Not a directory: {folder}")
                    continue
                run_once(folder, args.out, detector, segmenter)
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")