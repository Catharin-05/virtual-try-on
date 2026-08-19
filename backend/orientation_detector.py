"""
orientation_detector.py

Samples N frames (default 10) evenly across a video, runs RF-DETR's
pose/keypoint model on each frame, and estimates which frame shows the
person most exactly FRONT-facing, BACK-facing, and SIDE-facing (left or
right profile), purely from keypoint geometry (no separate classifier).

Install:
    pip install rfdetr supervision opencv-python numpy

Usage:
    python orientation_detector.py --video input.mp4 --out out_frames --frames 10

Notes / assumptions (heuristic, not a trained classifier):
  - Uses the standard COCO-17 keypoint layout that RF-DETR Keypoint (Preview)
    outputs: nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles.
  - FRONT: nose + both eyes confidently visible, nose horizontally centered
    between the shoulders, shoulders wide (torso squarely facing camera).
  - BACK: shoulders/torso confidently visible but nose+eyes are NOT
    (occluded because we're looking at the back of the head), shoulders wide.
  - SIDE: strong left/right asymmetry between eye+ear confidence on one side
    vs the other, and shoulders visually compressed (foreshortened) compared
    to a front-on stance.
  - Only the single most-confident person per frame is considered. If your
    video has multiple people, adapt `pick_primary_person()`.
  - Thresholds (CONF_THRESH, FRONT_WIDTH_REF, MIN_SCORE) are tunable -- treat
    them as a starting point and calibrate against your own footage.
"""

import argparse
import os

import cv2
import numpy as np

from rfdetr import RFDETRKeypointPreview

# ---------------------------------------------------------------------------
# COCO-17 keypoint layout
# ---------------------------------------------------------------------------
KP = {
    "nose": 0, "left_eye": 1, "right_eye": 2, "left_ear": 3, "right_ear": 4,
    "left_shoulder": 5, "right_shoulder": 6, "left_elbow": 7, "right_elbow": 8,
    "left_wrist": 9, "right_wrist": 10, "left_hip": 11, "right_hip": 12,
    "left_knee": 13, "right_knee": 14, "left_ankle": 15, "right_ankle": 16,
}

CONF_THRESH = 0.30      # min confidence to trust a single keypoint
FRONT_WIDTH_REF = 0.9   # typical normalized shoulder width when facing camera
MIN_SCORE = 0.12        # below this, we don't trust the orientation call


# ---------------------------------------------------------------------------
# 1. Frame sampling
# ---------------------------------------------------------------------------
def extract_frames(video_path, num_frames=10):
    """Return num_frames evenly spaced (frame_index, BGR image) pairs."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        raise ValueError("Could not determine frame count for this video.")

    num_frames = min(num_frames, total)
    indices = np.linspace(0, total - 1, num_frames, dtype=int)

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if ok:
            frames.append((int(idx), frame))
    cap.release()
    return frames


# ---------------------------------------------------------------------------
# 2. Pose inference
# ---------------------------------------------------------------------------
def pick_primary_person(key_points):
    """
    key_points: sv.KeyPoints result from RFDETRKeypointPreview.predict()
    Returns (xy, conf) for the detection with the highest mean confidence,
    or (None, None) if no person was detected.
    """
    if key_points is None or key_points.xy is None or len(key_points.xy) == 0:
        return None, None

    conf = key_points.confidence  # shape (num_people, 17)
    mean_conf = conf.mean(axis=1)
    best = int(np.argmax(mean_conf))
    return key_points.xy[best], conf[best]


# ---------------------------------------------------------------------------
# 3. Orientation scoring
# ---------------------------------------------------------------------------
def classify_orientation(xy, conf):
    """
    Returns (label, score, details) where label is one of:
    'front', 'back', 'side_left', 'side_right', 'undetermined'
    """
    lsh_c, rsh_c = conf[KP["left_shoulder"]], conf[KP["right_shoulder"]]
    if lsh_c < CONF_THRESH or rsh_c < CONF_THRESH:
        return "undetermined", 0.0, {"reason": "shoulders not confidently visible"}

    lsh, rsh = xy[KP["left_shoulder"]], xy[KP["right_shoulder"]]
    shoulder_width = abs(lsh[0] - rsh[0])
    shoulder_mid_x = (lsh[0] + rsh[0]) / 2.0

    # Scale-normalize shoulder width by torso height so it's roughly
    # independent of how close the person is to the camera.
    lhip_c, rhip_c = conf[KP["left_hip"]], conf[KP["right_hip"]]
    if lhip_c > CONF_THRESH and rhip_c > CONF_THRESH:
        hip_mid_y = (xy[KP["left_hip"]][1] + xy[KP["right_hip"]][1]) / 2.0
        shoulder_mid_y = (lsh[1] + rsh[1]) / 2.0
        torso_height = abs(hip_mid_y - shoulder_mid_y) + 1e-6
    else:
        torso_height = shoulder_width + 1e-6  # fallback scale reference

    norm_width = shoulder_width / torso_height
    width_factor = min(1.0, norm_width / FRONT_WIDTH_REF)   # ~1 when squarely on
    compression = max(0.0, 1.0 - width_factor)               # ~1 when turned sideways

    nose_c = conf[KP["nose"]]
    leye_c, reye_c = conf[KP["left_eye"]], conf[KP["right_eye"]]
    lear_c, rear_c = conf[KP["left_ear"]], conf[KP["right_ear"]]
    eye_avg = (leye_c + reye_c) / 2.0

    nose_x = xy[KP["nose"]][0]
    symmetry = 1.0 - min(1.0, abs(nose_x - shoulder_mid_x) / (shoulder_width / 2.0 + 1e-6))

    # --- FRONT: face clearly visible and centered, shoulders squared on ---
    front_score = nose_c * eye_avg * symmetry * width_factor
    if nose_c < CONF_THRESH or eye_avg < CONF_THRESH:
        front_score = 0.0

    # --- BACK: face NOT visible at all, but torso squarely visible ---
    face_visibility = max(nose_c, eye_avg)
    back_score = (1.0 - face_visibility) * width_factor
    if face_visibility > CONF_THRESH:
        back_score = 0.0

    # --- SIDE: strong left/right facial asymmetry + compressed shoulders ---
    left_face = (leye_c + lear_c) / 2.0
    right_face = (reye_c + rear_c) / 2.0
    asymmetry = abs(left_face - right_face)
    visible_side_conf = max(left_face, right_face)
    side_score = asymmetry * visible_side_conf * compression
    side_dir = "side_left" if left_face > right_face else "side_right"

    candidates = {
        "front": front_score,
        "back": back_score,
        side_dir: side_score,
    }
    label = max(candidates, key=candidates.get)
    score = candidates[label]

    if score < MIN_SCORE:
        label = "undetermined"

    details = {
        "front_score": front_score,
        "back_score": back_score,
        "side_score": side_score,
        "side_dir": side_dir,
        "norm_shoulder_width": norm_width,
        "symmetry": symmetry,
    }
    return label, score, details


# ---------------------------------------------------------------------------
# 4. Main pipeline
# ---------------------------------------------------------------------------
def analyze_video(video_path, out_dir, num_frames=10):
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading RF-DETR pose model...")
    model = RFDETRKeypointPreview()

    print(f"Sampling {num_frames} frames from {video_path}...")
    frames = extract_frames(video_path, num_frames)

    results = []
    for idx, frame in frames:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        key_points = model.predict(rgb, threshold=0.3)
        xy, conf = pick_primary_person(key_points)

        if xy is None:
            results.append({"frame_idx": idx, "frame": frame, "label": "no_person",
                             "score": 0.0, "details": {},
                             "keypoints_xy": None, "keypoints_conf": None})
            continue

        label, score, details = classify_orientation(xy, conf)
        results.append({"frame_idx": idx, "frame": frame, "label": label,
                         "score": score, "details": details,
                         "keypoints_xy": xy, "keypoints_conf": conf})

    # --- report per-frame results ---
    print("\nFrame-by-frame orientation:")
    for r in results:
        print(f"  frame {r['frame_idx']:>5}  ->  {r['label']:<12}  score={r['score']:.3f}")

    # --- pick the single "most exact" frame for each orientation ---
    def best_of(labels):
        candidates = [r for r in results if r["label"] in labels]
        if not candidates:
            return None
        return max(candidates, key=lambda r: r["score"])

    best_front = best_of(["front"])
    best_back = best_of(["back"])
    best_side_left = best_of(["side_left"])
    best_side_right = best_of(["side_right"])

    print("\nBest matches:")
    for name, r in [("FRONT", best_front), ("BACK", best_back),
                     ("SIDE (left profile)", best_side_left),
                     ("SIDE (right profile)", best_side_right)]:
        if r is None:
            print(f"  {name:<22}: not found in sampled frames")
        else:
            print(f"  {name:<22}: frame {r['frame_idx']} (score={r['score']:.3f})")
            out_path = os.path.join(out_dir, f"{name.split()[0].lower()}_frame{r['frame_idx']}.jpg")
            cv2.imwrite(out_path, r["frame"])
            print(f"    saved -> {out_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect front/back/side facing frames from a video.")
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--out", default="orientation_frames", help="Output directory for saved frames")
    parser.add_argument("--frames", type=int, default=10, help="Number of frames to sample")
    args = parser.parse_args()

    analyze_video(args.video, args.out, args.frames)