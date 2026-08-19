"""
anthropometry.py

Estimates body measurements from:
  - RF-DETR pose keypoints on the FRONT-facing frame (skeletal distances:
    shoulder width, sleeve length, inseam, torso length)
  - Sapiens body-part segmentation masks on the FRONT and SIDE frames
    (silhouette width/depth, used for girth measurements)
  - the person's real height in cm, supplied by the user, which calibrates
    pixels -> centimeters independently for each frame

How circumferences are estimated
---------------------------------
A single photo only gives you a body's WIDTH, not its full girth. To get a
tape-measure-like circumference we need the body's DEPTH too, which is
what the side-view frame gives us. We model each cross-section (chest,
waist, hip, neck, thigh) as an ellipse:

    width_cm  = front-view silhouette width at that body level
    depth_cm  = side-view silhouette width ("depth") at the SAME body level
    circumference = Ramanujan's ellipse-circumference approximation
                     using semi-axes a = width/2, b = depth/2

"Same body level" is found as a FRACTION of total body height (derived from
front-view keypoints for chest/waist/hip, or from the neck's own visible
extent), then re-applied to the side mask's own top/bottom -- so it still
works even if the front and side frames have different zoom/crop.

Assumptions / limitations (heuristic, not clinical-grade):
  - Requires the person's full body (head to feet) visible in BOTH the
    front and side frames for the height-based calibration to be valid.
  - The Sapiens mask includes shoes, which slightly inflates the calibrated
    "pixel height" versus bare-foot height -- this makes all cm measurements
    a little smaller than true (typically within 1-2%, shoe-heel-dependent).
  - Chest/waist/hip level heuristics (fractions of the shoulder-to-hip span)
    are common tailoring rules of thumb, not measured from the individual --
    calibrate FRACTION_* constants against ground truth if you need
    production-grade accuracy.
  - Circumference measurements require a confident SIDE view; without one
    they're skipped rather than guessed from width alone.
"""

import math

import numpy as np

from orientation_detector import KP, CONF_THRESH as KP_CONF_THRESH
from segmentation_detector import GOLIATH_CLASSES

CLASS_IDX = {name: i for i, name in enumerate(GOLIATH_CLASSES)}

# Body-level heuristics, expressed as a fraction of the shoulder-to-hip
# vertical span, measured downward from the shoulder line. Tunable.
FRACTION_CHEST_OF_TORSO = 0.20   # a bit below the armpits
FRACTION_WAIST_OF_TORSO = 0.62   # natural waist, above the hip bones
FRACTION_HIP_OF_TORSO = 0.08     # measured down from the hip keypoints, not the shoulders

# Sanity bounds for adult humans (cm). A circumference computed outside
# these is treated as a measurement FAILURE (bad/sparse segmentation for
# that body part in one view -- e.g. a side profile not showing much neck)
# and reported as None with an explanation, rather than silently returned
# as a confident-looking but physically impossible number.
PLAUSIBLE_CIRCUMFERENCE_CM = {
    "neck": (20, 55),
    "chest_bust": (55, 170),
    "waist": (45, 160),
    "hips": (55, 170),
    "thigh": (25, 95),
}


# ---------------------------------------------------------------------------
# Small geometry / mask helpers
# ---------------------------------------------------------------------------
def euclid(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def kp_ok(conf, idx):
    return conf is not None and idx < len(conf) and conf[idx] >= KP_CONF_THRESH


def ellipse_circumference_cm(width_cm, depth_cm):
    """Ramanujan's second approximation for an ellipse's perimeter."""
    if width_cm is None or depth_cm is None or width_cm <= 0 or depth_cm <= 0:
        return None
    a, b = width_cm / 2.0, depth_cm / 2.0
    h = ((a - b) ** 2) / ((a + b) ** 2)
    return math.pi * (a + b) * (1 + (3 * h) / (10 + math.sqrt(4 - 3 * h)))


def body_vertical_extent(mask):
    """(top_y, bottom_y) of the whole person silhouette (any non-background pixel)."""
    ys = np.nonzero(mask > 0)[0]
    if ys.size == 0:
        return None
    return int(ys.min()), int(ys.max())


def compute_scale_cm_per_px(mask, user_height_cm):
    """Calibrate pixels->cm for THIS mask using the person's known height."""
    extent = body_vertical_extent(mask)
    if extent is None:
        return None
    top_y, bottom_y = extent
    pixel_height = bottom_y - top_y + 1
    if pixel_height <= 0:
        return None
    return user_height_cm / pixel_height


def _row_x_extent(mask, y, class_indices):
    if y < 0 or y >= mask.shape[0]:
        return None
    row = mask[y]
    xs = np.nonzero(np.isin(row, class_indices))[0]
    if xs.size == 0:
        return None
    return int(xs.min()), int(xs.max())


def band_width_px(mask, y_center, class_indices, half_band=4, mode="max"):
    """Width of the union of class_indices across a small vertical band of
    rows around y_center. mode='max' (widest row) suits girth measurements;
    mode='median' is steadier for noisy/thin regions like the neck."""
    widths = []
    for y in range(max(0, y_center - half_band), min(mask.shape[0], y_center + half_band + 1)):
        ext = _row_x_extent(mask, y, class_indices)
        if ext is not None:
            widths.append(ext[1] - ext[0] + 1)
    if not widths:
        return None
    return max(widths) if mode == "max" else float(np.median(widths))


def narrowest_band_width(mask, y_top, y_bottom, class_indices, half_band=2):
    """Scan rows y_top..y_bottom, return the MINIMUM band width -- used for
    the neck, which tapers, so we want its narrowest point, not the jaw."""
    best = None
    for y in range(y_top, y_bottom + 1):
        w = band_width_px(mask, y, class_indices, half_band=half_band, mode="median")
        if w is not None and (best is None or w < best):
            best = w
    return best


def band_extent_at_max(mask, y_center, class_indices, half_band=4):
    """Like band_width_px(mode='max'), but also returns WHERE the widest
    row was found -- (y, x_min, x_max) -- so it can be drawn later. Uses the
    exact same widest-row selection as band_width_px, so the derived width
    (x_max - x_min + 1) always matches what band_width_px would report."""
    best = None  # (width, y, x_min, x_max)
    for y in range(max(0, y_center - half_band), min(mask.shape[0], y_center + half_band + 1)):
        ext = _row_x_extent(mask, y, class_indices)
        if ext is None:
            continue
        x_min, x_max = ext
        width = x_max - x_min + 1
        if best is None or width > best[0]:
            best = (width, y, x_min, x_max)
    return best  # or None


def narrowest_extent_over_range(mask, y_top, y_bottom, class_indices, half_band=2):
    """Scan rows y_top..y_bottom, return (width, y, x_min, x_max) for the
    row with the narrowest SMOOTHED width -- using the median width across a
    small vertical band around each candidate row, not that row's raw single
    -pixel extent. Without this smoothing, a single noisy/broken
    segmentation row (an antialiasing edge, a stray hair pixel, a class-
    boundary artifact) can hijack the result down to a physically impossible
    1-pixel-wide "neck" -- which is exactly what happened before this fix."""
    best = None  # (median_width, y, x_min, x_max)
    for y in range(y_top, y_bottom + 1):
        band = range(max(0, y - half_band), min(mask.shape[0], y + half_band + 1))
        candidates = []
        for yy in band:
            ext = _row_x_extent(mask, yy, class_indices)
            if ext is not None:
                x_min, x_max = ext
                candidates.append((x_max - x_min + 1, x_min, x_max))
        if not candidates:
            continue
        candidates.sort(key=lambda c: c[0])
        median_width, med_x_min, med_x_max = candidates[len(candidates) // 2]
        if best is None or median_width < best[0]:
            best = (median_width, y, med_x_min, med_x_max)
    return best


def fraction_of_height(mask, y_px):
    extent = body_vertical_extent(mask)
    if extent is None:
        return None
    top_y, bottom_y = extent
    if bottom_y <= top_y:
        return None
    return (y_px - top_y) / (bottom_y - top_y)


def row_at_fraction(mask, fraction):
    extent = body_vertical_extent(mask)
    if extent is None:
        return None
    top_y, bottom_y = extent
    return int(round(top_y + fraction * (bottom_y - top_y)))


# ---------------------------------------------------------------------------
# Keypoint-chain measurements (front view only)
# ---------------------------------------------------------------------------
def measure_shoulder_width(xy, conf, scale_cm_per_px):
    l, r = KP["left_shoulder"], KP["right_shoulder"]
    if not (kp_ok(conf, l) and kp_ok(conf, r)):
        return None
    return round(euclid(xy[l], xy[r]) * scale_cm_per_px, 1)


def measure_sleeve_length(xy, conf, scale_cm_per_px):
    lengths_px = []
    for side in ("left", "right"):
        sh, el, wr = KP[f"{side}_shoulder"], KP[f"{side}_elbow"], KP[f"{side}_wrist"]
        if kp_ok(conf, sh) and kp_ok(conf, el) and kp_ok(conf, wr):
            lengths_px.append(euclid(xy[sh], xy[el]) + euclid(xy[el], xy[wr]))
    if not lengths_px:
        return None
    return round((sum(lengths_px) / len(lengths_px)) * scale_cm_per_px, 1)


def measure_inseam(xy, conf, scale_cm_per_px):
    lengths_px = []
    for side in ("left", "right"):
        hip, knee, ank = KP[f"{side}_hip"], KP[f"{side}_knee"], KP[f"{side}_ankle"]
        if kp_ok(conf, hip) and kp_ok(conf, knee) and kp_ok(conf, ank):
            lengths_px.append(euclid(xy[hip], xy[knee]) + euclid(xy[knee], xy[ank]))
    if not lengths_px:
        return None
    return round((sum(lengths_px) / len(lengths_px)) * scale_cm_per_px, 1)


def measure_torso_length(xy, conf, scale_cm_per_px):
    ls, rs = KP["left_shoulder"], KP["right_shoulder"]
    lh, rh = KP["left_hip"], KP["right_hip"]
    if not all(kp_ok(conf, i) for i in (ls, rs, lh, rh)):
        return None
    shoulder_mid = ((xy[ls][0] + xy[rs][0]) / 2, (xy[ls][1] + xy[rs][1]) / 2)
    hip_mid = ((xy[lh][0] + xy[rh][0]) / 2, (xy[lh][1] + xy[rh][1]) / 2)
    return round(euclid(shoulder_mid, hip_mid) * scale_cm_per_px, 1)


# ---------------------------------------------------------------------------
# Circumference measurements (need front width + side depth)
# ---------------------------------------------------------------------------
def landmark_fractions(front_xy, front_conf, front_mask):
    """Height-fractions (0=top of head, 1=feet) for chest/waist/hip levels,
    derived from front-view keypoints, so the same relative height can be
    located in the side-view mask regardless of that frame's own crop/zoom."""
    ls, rs = KP["left_shoulder"], KP["right_shoulder"]
    lh, rh = KP["left_hip"], KP["right_hip"]
    if not all(kp_ok(front_conf, i) for i in (ls, rs, lh, rh)):
        return {}

    shoulder_y = (front_xy[ls][1] + front_xy[rs][1]) / 2
    hip_y = (front_xy[lh][1] + front_xy[rh][1]) / 2
    torso_span = hip_y - shoulder_y  # positive: shoulders sit above hips in image y

    chest_y = shoulder_y + FRACTION_CHEST_OF_TORSO * torso_span
    waist_y = shoulder_y + FRACTION_WAIST_OF_TORSO * torso_span
    hip_level_y = hip_y + FRACTION_HIP_OF_TORSO * torso_span

    fractions = {}
    for name, y in (("chest", chest_y), ("waist", waist_y), ("hip", hip_level_y)):
        frac = fraction_of_height(front_mask, y)
        if frac is not None:
            fractions[name] = min(max(frac, 0.0), 1.0)
    return fractions


def measure_circumference_at_fraction(front_mask, front_scale, side_mask, side_scale,
                                       fraction, class_indices, half_band=5):
    front_row_y = row_at_fraction(front_mask, fraction)
    side_row_y = row_at_fraction(side_mask, fraction)
    if front_row_y is None or side_row_y is None:
        return None

    front_ext = band_extent_at_max(front_mask, front_row_y, class_indices, half_band=half_band)
    side_ext = band_extent_at_max(side_mask, side_row_y, class_indices, half_band=half_band)
    if front_ext is None or side_ext is None:
        return None

    width_px, f_y, f_x_min, f_x_max = front_ext
    depth_px, s_y, s_x_min, s_x_max = side_ext

    width_cm = width_px * front_scale
    depth_cm = depth_px * side_scale
    circumference = ellipse_circumference_cm(width_cm, depth_cm)
    if circumference is None:
        return None
    return {
        "circumference_cm": round(circumference, 1),
        "width_cm": round(width_cm, 1),
        "depth_cm": round(depth_cm, 1),
        # pixel-space geometry, for drawing this measurement on the images
        "front_row": f_y, "front_x_min": f_x_min, "front_x_max": f_x_max,
        "side_row": s_y, "side_x_min": s_x_min, "side_x_max": s_x_max,
    }


def measure_neck(front_mask, front_scale, side_mask, side_scale, front_xy, front_conf):
    nose, ls, rs = KP["nose"], KP["left_shoulder"], KP["right_shoulder"]
    if not all(kp_ok(front_conf, i) for i in (nose, ls, rs)):
        return None

    nose_y = int(front_xy[nose][1])
    shoulder_y = int((front_xy[ls][1] + front_xy[rs][1]) / 2)
    if shoulder_y <= nose_y:
        return None

    neck_classes = [CLASS_IDX["Face Neck"]]
    front_ext = narrowest_extent_over_range(front_mask, nose_y, shoulder_y, neck_classes)
    if front_ext is None:
        return None
    front_width, f_y, f_x_min, f_x_max = front_ext

    frac_top = fraction_of_height(front_mask, nose_y)
    frac_bottom = fraction_of_height(front_mask, shoulder_y)
    if frac_top is None or frac_bottom is None:
        return None

    side_extent = body_vertical_extent(side_mask)
    if side_extent is None:
        return None
    s_top, s_bottom = side_extent
    side_y_top = int(s_top + frac_top * (s_bottom - s_top))
    side_y_bottom = int(s_top + frac_bottom * (s_bottom - s_top))
    side_ext = narrowest_extent_over_range(side_mask, side_y_top, side_y_bottom, neck_classes)
    if side_ext is None:
        return None
    side_depth, s_y, s_x_min, s_x_max = side_ext

    width_cm = front_width * front_scale
    depth_cm = side_depth * side_scale
    circumference = ellipse_circumference_cm(width_cm, depth_cm)
    if circumference is None:
        return None
    return {
        "circumference_cm": round(circumference, 1),
        "width_cm": round(width_cm, 1),
        "depth_cm": round(depth_cm, 1),
        "front_row": f_y, "front_x_min": f_x_min, "front_x_max": f_x_max,
        "side_row": s_y, "side_x_min": s_x_min, "side_x_max": s_x_max,
    }


def measure_thigh(front_mask, front_scale, side_mask, side_scale):
    """Thigh girth near the top of the leg (widest point, avoids knee taper),
    averaged across both legs where available. Keeps a per-leg breakdown
    (with pixel geometry) alongside the averaged circumference, for drawing."""
    legs = {}

    # Body midline -- used to split the generic "Lower Clothing" class into
    # left/right halves when full-coverage pants mean Sapiens never labels
    # any pixels as the skin-specific "Left/Right Upper Leg" classes at all
    # (common with trousers; those classes are mostly picked up on bare skin).
    fg_cols = np.nonzero((front_mask > 0).any(axis=0))[0]
    midline_x = int((fg_cols.min() + fg_cols.max()) / 2) if fg_cols.size else front_mask.shape[1] // 2

    for side_name, cls_name in (("left", "Left Upper Leg"), ("right", "Right Upper Leg")):
        leg_class = [CLASS_IDX[cls_name]]
        clothing_classes = leg_class + [CLASS_IDX["Lower Clothing"], CLASS_IDX["Apparel"]]

        ys = np.nonzero(np.isin(front_mask, leg_class))[0]
        used_clothing_fallback = False
        if ys.size == 0:
            # No skin-specific leg pixels found -- fall back to generic
            # lower-body clothing, restricted to this leg's half of the
            # body (so we don't grab the other leg or the torso hem).
            used_clothing_fallback = True
            clothing_only = np.isin(front_mask, [CLASS_IDX["Lower Clothing"], CLASS_IDX["Apparel"]])
            half_mask = np.zeros_like(clothing_only)
            if side_name == "left":
                half_mask[:, :midline_x] = True
            else:
                half_mask[:, midline_x:] = True
            ys = np.nonzero(clothing_only & half_mask)[0]
            if ys.size == 0:
                continue

        top_y, bottom_y = int(ys.min()), int(ys.max())
        sample_y = int(top_y + 0.25 * (bottom_y - top_y))

        # When using the clothing fallback, restrict the FRONT width scan to
        # this leg's half of the image too -- otherwise, if both pant legs
        # are close together, the scan can span the gap between them and
        # measure hip-to-hip width instead of one thigh's width. (The SIDE
        # view is a profile silhouette where left/right legs typically
        # overlap in depth rather than sit side-by-side in image columns,
        # so this left/right split doesn't apply there -- both legs use the
        # same unrestricted side scan, same as a real tape measure can't
        # separate near-leg from far-leg depth in profile either.)
        front_scan_mask = front_mask
        if used_clothing_fallback:
            front_scan_mask = front_mask.copy()
            if side_name == "left":
                front_scan_mask[:, midline_x:] = 0
            else:
                front_scan_mask[:, :midline_x] = 0

        front_ext = band_extent_at_max(front_scan_mask, sample_y, clothing_classes, half_band=4)
        frac = fraction_of_height(front_mask, sample_y)
        if front_ext is None or frac is None:
            continue
        front_width, f_y, f_x_min, f_x_max = front_ext

        side_row_y = row_at_fraction(side_mask, frac)
        if side_row_y is None:
            continue
        side_ext = band_extent_at_max(side_mask, side_row_y, clothing_classes, half_band=4)
        if side_ext is None:
            continue
        side_depth, s_y, s_x_min, s_x_max = side_ext

        circ = ellipse_circumference_cm(front_width * front_scale, side_depth * side_scale)
        if circ is None:
            continue

        legs[side_name] = {
            "circumference_cm": round(circ, 1),
            "width_cm": round(front_width * front_scale, 1),
            "depth_cm": round(side_depth * side_scale, 1),
            "front_row": f_y, "front_x_min": f_x_min, "front_x_max": f_x_max,
            "side_row": s_y, "side_x_min": s_x_min, "side_x_max": s_x_max,
            "used_clothing_fallback": used_clothing_fallback,
        }

    if not legs:
        return None

    avg_circ = sum(leg["circumference_cm"] for leg in legs.values()) / len(legs)
    return {
        "circumference_cm": round(avg_circ, 1),
        "note": ("average of left and right thigh circumference" if len(legs) == 2
                  else f"only the {list(legs.keys())[0]} thigh was visible/confident"),
        **legs,
    }


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------
def validate_circumference(key, entry):
    """
    Rejects a circumference result that's outside PLAUSIBLE_CIRCUMFERENCE_CM
    for that body part, replacing it with None + an explanation instead of
    silently returning a physically impossible number. This is what catches
    cases like a side-profile view barely showing the neck, where the
    underlying pixel data is genuinely too sparse across the whole scanned
    region for smoothing/band-median tricks to help (not just one noisy row).
    """
    if entry is None or entry.get("circumference_cm") is None:
        return entry
    lo, hi = PLAUSIBLE_CIRCUMFERENCE_CM[key]
    circ = entry["circumference_cm"]
    if lo <= circ <= hi:
        return entry
    return {
        "circumference_cm": None,
        "error": (f"Measured {circ} cm is outside the plausible adult range "
                  f"({lo}-{hi} cm) for {key.replace('_', ' ')} -- likely a "
                  f"segmentation failure in one of the views (e.g. that body "
                  f"part barely visible from that angle), not a real "
                  f"measurement. Raw pixel data kept below for debugging."),
        "raw_debug": entry,
    }


def estimate_measurements(front, side, user_height_cm):
    """
    front: dict with 'mask' (HxW uint8 array), 'xy' (17,2) keypoints,
           'conf' (17,) keypoint confidences -- all required.
    side:  dict with 'mask' (HxW uint8 array), or None if no confident
           side view was found. Keypoints aren't needed for the side view.
    user_height_cm: the person's real height in centimeters.

    Returns a dict of measurements in cm. Circumference entries are dicts
    with circumference_cm/width_cm/depth_cm; skeletal ones are plain floats.
    Anything that couldn't be estimated is None, with a note explaining why.
    """
    measurements = {}

    front_mask = front.get("mask") if front else None
    front_xy = front.get("xy") if front else None
    front_conf = front.get("conf") if front else None

    if front_mask is None or front_xy is None or front_conf is None:
        measurements["error"] = ("A confident FRONT-facing frame with both a "
                                  "segmentation mask and pose keypoints is required.")
        return measurements

    front_scale = compute_scale_cm_per_px(front_mask, user_height_cm)
    if front_scale is None:
        measurements["error"] = "Could not calibrate pixel scale from the front-view mask."
        return measurements

    measurements["calibration"] = {"front_cm_per_px": round(front_scale, 5)}

    # --- skeletal measurements (front view only) ---
    measurements["shoulder_width_cm"] = measure_shoulder_width(front_xy, front_conf, front_scale)
    measurements["sleeve_length_cm"] = measure_sleeve_length(front_xy, front_conf, front_scale)
    measurements["inseam_cm"] = measure_inseam(front_xy, front_conf, front_scale)
    measurements["torso_length_cm"] = measure_torso_length(front_xy, front_conf, front_scale)

    # --- circumference measurements (need a side view for depth) ---
    side_mask = side.get("mask") if side else None
    side_scale = compute_scale_cm_per_px(side_mask, user_height_cm) if side_mask is not None else None

    if side_mask is None or side_scale is None:
        measurements["warning"] = (
            "No confident side-view frame available -- chest/bust, waist, hips, "
            "neck, and thigh circumferences need a front+side pair and were "
            "skipped. Shoulder width, sleeve length, inseam, and torso length "
            "only need the front view and were still computed."
        )
        measurements["chest_bust"] = None
        measurements["waist"] = None
        measurements["hips"] = None
        measurements["neck"] = None
        measurements["thigh"] = None
        return measurements

    measurements["calibration"]["side_cm_per_px"] = round(side_scale, 5)

    # Sanity check: front_scale and side_scale are calibrated independently
    # (each from that frame's own visible head-to-toe pixel extent), which
    # is correct when the two frames were shot at different zoom/distance.
    # But if one frame's mask doesn't actually capture the full body --
    # partial crop, an occluding limb, a bad segmentation -- its scale will
    # be silently wrong, and every measurement using it inherits that error
    # with no visible signal. A large mismatch between the two independently
    # -derived scales is the cheapest available signal that something's off.
    scale_ratio = max(front_scale, side_scale) / min(front_scale, side_scale)
    if scale_ratio > 1.15:
        measurements["calibration"]["scale_mismatch_warning"] = (
            f"front_cm_per_px ({front_scale:.5f}) and side_cm_per_px ({side_scale:.5f}) "
            f"differ by {round((scale_ratio - 1) * 100)}%. This is expected if the camera "
            f"was genuinely closer/farther or zoomed differently between the two frames, "
            f"but it's also what you'd see if one frame's mask doesn't capture the full "
            f"body head-to-toe (occlusion, partial crop, bad segmentation). Worth visually "
            f"checking both frames' overlay masks before trusting these measurements."
        )

    fractions = landmark_fractions(front_xy, front_conf, front_mask)
    torso_classes = [CLASS_IDX["Torso"], CLASS_IDX["Upper Clothing"], CLASS_IDX["Apparel"]]
    lower_classes = [CLASS_IDX["Torso"], CLASS_IDX["Lower Clothing"], CLASS_IDX["Apparel"]]
    hip_classes = [CLASS_IDX["Torso"], CLASS_IDX["Lower Clothing"], CLASS_IDX["Apparel"],
                   CLASS_IDX["Left Upper Leg"], CLASS_IDX["Right Upper Leg"]]

    measurements["chest_bust"] = validate_circumference("chest_bust", (
        measure_circumference_at_fraction(front_mask, front_scale, side_mask, side_scale,
                                           fractions["chest"], torso_classes)
        if "chest" in fractions else None
    ))
    measurements["waist"] = validate_circumference("waist", (
        measure_circumference_at_fraction(front_mask, front_scale, side_mask, side_scale,
                                           fractions["waist"], lower_classes)
        if "waist" in fractions else None
    ))
    measurements["hips"] = validate_circumference("hips", (
        measure_circumference_at_fraction(front_mask, front_scale, side_mask, side_scale,
                                           fractions["hip"], hip_classes)
        if "hip" in fractions else None
    ))
    measurements["neck"] = validate_circumference("neck", measure_neck(
        front_mask, front_scale, side_mask, side_scale, front_xy, front_conf))
    measurements["thigh"] = validate_circumference("thigh", measure_thigh(
        front_mask, front_scale, side_mask, side_scale))

    return measurements