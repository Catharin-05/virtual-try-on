"""
main.py

End-to-end pipeline:

    video --(orientation_detector)--> 10 sampled frames, each scored as
             front / back / side_left / side_right
         --(segmentation_detector)--> YOLO person detection + Sapiens
             28-class body-part segmentation on the chosen frame(s)
         --(anthropometry)--------> chest/waist/hip/neck/thigh circumference,
             shoulder width, sleeve length, inseam, torso length -- using the
             best FRONT + SIDE frames' keypoints and masks, calibrated to
             real-world cm via the person's known height
         --(mhr_calibration)------> a 3D MHR mesh, optimized so its own
             measured dimensions match the values just extracted

That last stage (--calibrate-mesh) is the DEFAULT way to get a 3D model:
it starts from a neutral/average body and directly optimizes MHR's 45
identity params to match your measurements. It only needs the public,
ungated `mhr` package -- no GPU requirement, no repo cloning, no license
approval wait.

There is also an OPTIONAL, heavier enhancement (--build-3d-model, using
Meta's SAM 3D Body) that reconstructs each of your 4 photographed views
individually first and fuses them into a photo-informed starting identity,
which --calibrate-mesh will then use instead of the neutral default. This
can give the optimizer a head start, but requires a gated HF checkpoint, a
full local clone of the sam-3d-body repo, and (per its current code) a
CUDA GPU. It is skippable -- see body_3d.py's docstring for the full
trade-off discussion.

By default it segments only the single best FRONT, BACK, SIDE_LEFT, and
SIDE_RIGHT frames found by the orientation detector (so the expensive
Sapiens model runs at most 4 times per video). Pass --all-frames to run
segmentation on every sampled frame instead.

Each chosen frame is handed to Sapiens as a FULL, uncropped frame (YOLO is
only used afterwards to slice the resulting mask per person) -- see the
docstring in segmentation_detector.py for why.

Install:
    pip install rfdetr supervision opencv-python numpy \
                git+https://github.com/ibaiGorordo/Sapiens-Pytorch-Inference.git ultralytics \
                mhr pymomentum-cpu   # only needed for --calibrate-mesh

Usage:
    # Standard: measurements only
    python main.py --video input.mp4 --out pipeline_output --height 170

    # + a 3D model calibrated to those measurements (recommended default)
    python main.py --video input.mp4 --out pipeline_output --height 170 \
        --calibrate-mesh --mhr-assets /path/to/unzipped/mhr/assets

    # + the optional heavier SAM 3D Body starting point
    python main.py --video input.mp4 --out pipeline_output --height 170 \
        --build-3d-model --sam3d-body-repo /path/to/sam-3d-body \
        --calibrate-mesh --mhr-assets /path/to/unzipped/mhr/assets
"""

import argparse
import json
import os

from orientation_detector import analyze_video
from segmentation_detector import PersonDetector, SapiensSegmenter, detect_human_parts
from person_detector_onnx import PersonDetectorONNX
from anthropometry import estimate_measurements
from visualize_measurements import visualize_measurements

ORIENTATION_LABELS = ("front", "back", "side_left", "side_right")


def select_target_frames(orientation_results, segment_all_frames=False):
    """Pick which sampled frames get sent to the segmentation stage."""
    if segment_all_frames:
        return [r for r in orientation_results if r["label"] != "no_person"]

    best_per_label = {}
    for r in orientation_results:
        if r["label"] not in ORIENTATION_LABELS:
            continue
        if r["label"] not in best_per_label or r["score"] > best_per_label[r["label"]]["score"]:
            best_per_label[r["label"]] = r

    return list(best_per_label.values())


def pick_best_view(targets, seg_results, label_options):
    """From the (orientation_result, segmentation_result) pairs, return the
    highest-scoring pair whose label is in label_options, or (None, None)."""
    best_t, best_seg = None, None
    for t, seg in zip(targets, seg_results):
        if t["label"] not in label_options:
            continue
        if best_t is None or t["score"] > best_t["score"]:
            best_t, best_seg = t, seg
    return best_t, best_seg


def run_pipeline(video_path, out_dir="pipeline_output", num_frames=10,
                  segment_all_frames=False, yolo_weights="yolov8n.pt",
                  user_height_cm=None, build_3d_model_flag=False, sam3d_body_repo=None,
                  calibrate_mesh_flag=False, mhr_assets=None):
    os.makedirs(out_dir, exist_ok=True)
    orientation_dir = os.path.join(out_dir, "orientation_frames")
    seg_dir = os.path.join(out_dir, "segmentation")
    os.makedirs(seg_dir, exist_ok=True)

    # ---- Stage 1: orientation detection ----
    print("=== Orientation detection ===")
    orientation_results = analyze_video(video_path, orientation_dir, num_frames)

    targets = select_target_frames(orientation_results, segment_all_frames)
    if not targets:
        print("\nNo frames with a confident front/back/side orientation were "
              "found -- nothing to segment or measure.")
        return orientation_results, [], None, None, None

    # ---- Stage 2: body-part segmentation on the chosen frames ----
    print(f"\n=== Body-part segmentation on {len(targets)} frame(s) ===")
    detector = PersonDetectorONNX(onnx_path="yolov8n.onnx")
    segmenter = SapiensSegmenter()

    images = [t["frame"] for t in targets]
    names = [f"{t['label']}_frame{t['frame_idx']}" for t in targets]

    seg_results = detect_human_parts(images, names=names, out_dir=seg_dir,
                                      detector=detector, segmenter=segmenter)

    # ---- Stage 3: anthropometric measurements (needs a front view + height) ----
    measurements = None
    if user_height_cm is None:
        print("\n=== Anthropometric measurements: skipped (pass --height <cm> to enable) ===")
    else:
        print("\n=== Anthropometric measurements ===")

        # Pull out exactly the pieces the measurement code needs: the front
        # frame's pose keypoints + mask, and the side frame's mask, as plain
        # variables, then hand them to estimate_measurements().
        front_target, front_seg = pick_best_view(targets, seg_results, {"front"})
        side_target, side_seg = pick_best_view(targets, seg_results, {"side_left", "side_right"})

        if front_target is None:
            print("No confident FRONT frame found -- skipping measurements.")
        else:
            front_view = {
                "mask": front_seg.get("mask"),
                "xy": front_target.get("keypoints_xy"),
                "conf": front_target.get("keypoints_conf"),
            }
            side_view = {"mask": side_seg.get("mask")} if side_target is not None else None

            if side_view is None:
                print("No confident SIDE frame found -- circumference "
                      "measurements (chest/waist/hips/neck/thigh) will be "
                      "skipped; shoulder width/sleeve/inseam/torso still run.")

            measurements = estimate_measurements(front_view, side_view, user_height_cm)
            print(json.dumps(measurements, indent=2, default=str))

            if "error" not in measurements:
                side_frame = side_target["frame"] if side_target is not None else None
                front_viz_path, side_viz_path = visualize_measurements(
                    front_target["frame"], front_view["xy"], front_view["conf"],
                    side_frame, measurements, out_dir=out_dir, prefix="measurements")
                print(f"\nAnnotated front view -> {front_viz_path}")
                if side_viz_path:
                    print(f"Annotated side view  -> {side_viz_path}")

    # ---- Combine + save a single pipeline summary ----
    summary = []
    for t, seg in zip(targets, seg_results):
        summary.append({
            "frame_idx": t["frame_idx"],
            "orientation": t["label"],
            "orientation_score": round(t["score"], 3),
            "segmentation": seg,
        })

    # ---- Stage 4 (optional): 3D model reconstruction (Meta SAM 3D Body / MHR) ----
    body_3d_result = None
    if build_3d_model_flag:
        print("\n=== 3D model reconstruction (optional -- SAM 3D Body) ===")
        if not sam3d_body_repo:
            print("Skipped: --sam3d-body-repo <path to your local sam-3d-body clone> is required.")
        else:
            try:
                from body_3d import load_estimator, build_3d_model

                by_label = {}
                for t, seg in zip(targets, seg_results):
                    if t["label"] not in by_label or t["score"] > by_label[t["label"]]["score"]:
                        by_label[t["label"]] = t
                        by_label[t["label"] + "_seg"] = seg

                views = {}
                for label in ORIENTATION_LABELS:
                    if label in by_label:
                        t = by_label[label]
                        seg = by_label[label + "_seg"]
                        views[label] = {
                            "frame": t["frame"],
                            "keypoints_xy": t.get("keypoints_xy"),
                            "keypoints_conf": t.get("keypoints_conf"),
                            "mask": seg.get("mask"),
                            "score": t.get("score", 1.0),
                        }

                estimator = load_estimator(repo_path=sam3d_body_repo)
                body_3d_result = build_3d_model(estimator, views, out_dir=out_dir)
            except Exception as e:
                print(f"3D model reconstruction failed: {e}")
                print("This stage depends on a heavy, gated external model -- "
                      "see the docstring in body_3d.py for setup requirements "
                      "and troubleshooting.")

    # ---- Stage 5 (optional): calibrate the MHR mesh to match extracted measurements ----
    calibration_result = None
    if calibrate_mesh_flag:
        print("\n=== 3D mesh calibration (MHR, matched to your measurements) ===")
        if measurements is None or "error" in measurements:
            print("Skipped: no valid anthropometric measurements available (needs --height).")
        elif not mhr_assets:
            print("Skipped: --mhr-assets <path to unzipped MHR assets> is required.")
        else:
            try:
                from mhr_calibration import calibrate_and_export

                initial_identity = (body_3d_result["fused_identity"]
                                     if body_3d_result and "fused_identity" in body_3d_result
                                     else None)
                calibration_result = calibrate_and_export(
                    asset_folder=mhr_assets,
                    target_measurements_cm=measurements,
                    initial_identity=initial_identity,
                    out_path=os.path.join(out_dir, "body_3d_model.obj"),
                )
            except Exception as e:
                print(f"Mesh calibration failed: {e}")
                print("See the docstring in mhr_calibration.py -- in particular, the "
                      "unverified mesh-face-attribute and height-fraction assumptions "
                      "noted there may need adjusting for your installed MHR version.")

    summary_path = os.path.join(out_dir, "pipeline_summary.json")
    json_safe_summary = []
    for s in summary:
        s_copy = dict(s)
        s_copy["segmentation"] = {k: v for k, v in s["segmentation"].items() if k != "mask"}
        json_safe_summary.append(s_copy)
    with open(summary_path, "w") as f:
        json.dump({
            "frames": json_safe_summary,
            "measurements": measurements,
            "body_3d_obj_path": body_3d_result["obj_path"] if body_3d_result else None,
            "body_3d_model_path": calibration_result["obj_path"] if calibration_result else None,
        }, f, indent=2, default=str)

    print("\n=== Pipeline complete ===")
    for s in summary:
        n_people = s["segmentation"].get("num_people", 0)
        print(f"  frame {s['frame_idx']:>5}  {s['orientation']:<12} "
              f"(score={s['orientation_score']})  -> {n_people} person(s) segmented")
    print(f"\nFull summary saved -> {summary_path}")

    return orientation_results, summary, measurements, body_3d_result, calibration_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video -> orientation -> body-part segmentation -> measurements pipeline")
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--out", default="pipeline_output", help="Output directory")
    parser.add_argument("--frames", type=int, default=10, help="Number of frames to sample from the video")
    parser.add_argument("--all-frames", action="store_true",
                         help="Run segmentation on every sampled frame instead of just the best front/back/side ones")
    parser.add_argument("--yolo-weights", default="yolov8n.pt", help="YOLO weights for person detection")
    parser.add_argument("--height", type=float, default=None,
                         help="Person's real height in centimeters. Required to compute anthropometric "
                              "measurements (chest/waist/hips/neck/thigh/shoulder/sleeve/inseam/torso). "
                              "If omitted, the pipeline still runs orientation + segmentation only.")
    parser.add_argument("--build-3d-model", action="store_true",
                         help="OPTIONAL, heavier: reconstruct each view with Meta's SAM 3D Body and fuse "
                              "into a photo-informed starting shape for --calibrate-mesh. Requires "
                              "--sam3d-body-repo, a gated HF checkpoint, and (currently) a CUDA GPU. "
                              "Skip this and --calibrate-mesh alone still produces a full 3D model, "
                              "just starting from a neutral body instead of one informed by your photos.")
    parser.add_argument("--sam3d-body-repo", default=None,
                         help="Path to a local clone of https://github.com/facebookresearch/sam-3d-body "
                              "(required if --build-3d-model is set).")
    parser.add_argument("--calibrate-mesh", action="store_true",
                         help="Build a 3D MHR mesh by optimizing identity/shape parameters so the mesh's "
                              "own measured dimensions match the values from anthropometry.py. This is "
                              "the recommended way to get a 3D model -- works standalone (starting from a "
                              "neutral body), or uses --build-3d-model's fused_identity as a better "
                              "starting point if that optional stage also ran. Requires --height and "
                              "--mhr-assets. Only needs the public `mhr` package -- no GPU, no gated "
                              "access, no repo cloning.")
    parser.add_argument("--mhr-assets", default=None,
                         help="Path to the unzipped MHR asset folder (compact_v6_1.model, lod*.fbx, "
                              "corrective_*.npz) from https://github.com/facebookresearch/MHR "
                              "(required if --calibrate-mesh is set).")
    args = parser.parse_args()

    run_pipeline(args.video, args.out, args.frames, args.all_frames, args.yolo_weights,
                 args.height, args.build_3d_model, args.sam3d_body_repo,
                 args.calibrate_mesh, args.mhr_assets)