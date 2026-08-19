"""
visualize_measurements.py

Draws the anthropometric measurements from anthropometry.py on top of the
front and side frames they were computed from.

IMPORTANT distinction this module is careful to draw correctly:

  - LENGTH measurements (shoulder width, sleeve length, inseam, torso
    length) are literal straight-line, point-to-point distances. The line
    drawn on the image IS the measurement -- its on-screen length
    corresponds exactly to the cm value shown.

  - CIRCUMFERENCE measurements (chest/bust, waist, hips, neck, thigh) are
    NOT something a single photo can show as a straight line. A photo only
    gives a flat front-view WIDTH and a flat side-view DEPTH at a given
    body level; the circumference is an ESTIMATE derived from combining
    both (modeling the cross-section as an ellipse). So for these, the
    line drawn on the front image is the WIDTH component (not the full
    loop), and the line on the side image is the DEPTH component -- each
    labeled as exactly that, with the derived circumference shown
    separately as "~X cm around" so it's never confused with the length
    of the line itself.
"""

import os

import cv2

from orientation_detector import KP, CONF_THRESH

# BGR colors, one per measurement, reused consistently across both views
COLORS = {
    "shoulder": (0, 200, 255),
    "sleeve": (255, 140, 0),
    "inseam": (147, 20, 255),
    "torso": (0, 255, 0),
    "chest": (255, 80, 0),
    "waist": (0, 165, 255),
    "hips": (255, 0, 255),
    "neck": (0, 255, 255),
    "thigh": (180, 0, 180),
}

# key in the measurements dict -> (display name, color key)
CIRCUMFERENCE_KEYS = (
    ("chest_bust", "Chest/Bust", "chest"),
    ("waist", "Waist", "waist"),
    ("hips", "Hips", "hips"),
    ("neck", "Neck", "neck"),
)


def _pt(xy_row):
    return int(round(float(xy_row[0]))), int(round(float(xy_row[1])))


def _kp_ok(conf, idx):
    return conf is not None and conf[idx] >= CONF_THRESH


def _label_lines(img, lines, pos, color):
    """Draw a small stacked-text label box (one box, N lines) anchored near
    pos, clamped so the whole box stays inside the image."""
    font, scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
    sizes = [cv2.getTextSize(line, font, scale, thickness)[0] for line in lines]
    box_w = max(w for w, h in sizes) + 8
    line_h = sizes[0][1] + 8
    box_h = line_h * len(lines)

    x, y = pos
    x = max(0, min(x, img.shape[1] - box_w))
    y = max(box_h, min(y, img.shape[0] - 2))

    cv2.rectangle(img, (x, y - box_h), (x + box_w, y), (0, 0, 0), -1)
    for i, line in enumerate(lines):
        baseline_y = y - box_h + line_h * (i + 1) - 5
        cv2.putText(img, line, (x + 4, baseline_y), font, scale, color, thickness, cv2.LINE_AA)


def _draw_length_line(img, p1, p2, label_text, color, label_pos=None):
    cv2.line(img, p1, p2, color, 2)
    cv2.circle(img, p1, 4, color, -1)
    cv2.circle(img, p2, 4, color, -1)
    pos = label_pos or (max(p1[0], p2[0]) + 6, min(p1[1], p2[1]))
    _label_lines(img, [label_text], pos, color)


def _draw_flat_band(img, entry, row_key, xmin_key, xmax_key,
                     flat_key, flat_word, name, color):
    """Draws the WIDTH (front view) or DEPTH (side view) band actually
    measured, and labels it honestly as that -- plus a second line showing
    the derived circumference, clearly marked as an estimate ("~X cm
    around"), never implying the drawn line itself is that long."""
    if entry is None or row_key not in entry:
        return
    y = entry[row_key]
    p1, p2 = (entry[xmin_key], y), (entry[xmax_key], y)
    cv2.line(img, p1, p2, color, 2)
    cv2.circle(img, p1, 3, color, -1)
    cv2.circle(img, p2, 3, color, -1)

    lines = [f"{name} {flat_word}: {entry[flat_key]} cm"]
    if "circumference_cm" in entry:
        lines.append(f"~{entry['circumference_cm']} cm around")
    _label_lines(img, lines, (p2[0] + 6, y), color)


# ---------------------------------------------------------------------------
# Front view: skeleton LENGTH lines + circumference WIDTH bands
# ---------------------------------------------------------------------------
def draw_front_measurements(front_img, front_xy, front_conf, measurements):
    img = front_img.copy()

    # Shoulder width (a true length -- point to point)
    if measurements.get("shoulder_width_cm") is not None:
        l, r = KP["left_shoulder"], KP["right_shoulder"]
        if _kp_ok(front_conf, l) and _kp_ok(front_conf, r):
            p1, p2 = _pt(front_xy[l]), _pt(front_xy[r])
            c = COLORS["shoulder"]
            mid_x = (p1[0] + p2[0]) // 2
            _draw_length_line(img, p1, p2, f"Shoulder width: {measurements['shoulder_width_cm']} cm", c,
                               label_pos=(mid_x - 60, min(p1[1], p2[1]) - 12))

    # Sleeve length (a true length -- shoulder->elbow->wrist chain)
    if measurements.get("sleeve_length_cm") is not None:
        c = COLORS["sleeve"]
        label_pos = None
        for side in ("left", "right"):
            sh, el, wr = KP[f"{side}_shoulder"], KP[f"{side}_elbow"], KP[f"{side}_wrist"]
            if _kp_ok(front_conf, sh) and _kp_ok(front_conf, el) and _kp_ok(front_conf, wr):
                p_sh, p_el, p_wr = _pt(front_xy[sh]), _pt(front_xy[el]), _pt(front_xy[wr])
                cv2.line(img, p_sh, p_el, c, 2)
                cv2.line(img, p_el, p_wr, c, 2)
                cv2.circle(img, p_el, 3, c, -1)
                cv2.circle(img, p_wr, 3, c, -1)
                if label_pos is None:
                    label_pos = p_wr
        if label_pos is not None:
            _label_lines(img, [f"Sleeve length: {measurements['sleeve_length_cm']} cm"], label_pos, c)

    # Inseam (a true length -- hip->knee->ankle chain)
    if measurements.get("inseam_cm") is not None:
        c = COLORS["inseam"]
        label_pos = None
        for side in ("left", "right"):
            hip, knee, ank = KP[f"{side}_hip"], KP[f"{side}_knee"], KP[f"{side}_ankle"]
            if _kp_ok(front_conf, hip) and _kp_ok(front_conf, knee) and _kp_ok(front_conf, ank):
                p_hip, p_knee, p_ank = _pt(front_xy[hip]), _pt(front_xy[knee]), _pt(front_xy[ank])
                cv2.line(img, p_hip, p_knee, c, 2)
                cv2.line(img, p_knee, p_ank, c, 2)
                cv2.circle(img, p_knee, 3, c, -1)
                cv2.circle(img, p_ank, 3, c, -1)
                if label_pos is None:
                    label_pos = p_ank
        if label_pos is not None:
            _label_lines(img, [f"Inseam: {measurements['inseam_cm']} cm"], label_pos, c)

    # Torso length (a true length -- shoulder-mid to hip-mid)
    if measurements.get("torso_length_cm") is not None:
        ls, rs = KP["left_shoulder"], KP["right_shoulder"]
        lh, rh = KP["left_hip"], KP["right_hip"]
        if all(_kp_ok(front_conf, i) for i in (ls, rs, lh, rh)):
            sh_mid = ((front_xy[ls][0] + front_xy[rs][0]) / 2, (front_xy[ls][1] + front_xy[rs][1]) / 2)
            hip_mid = ((front_xy[lh][0] + front_xy[rh][0]) / 2, (front_xy[lh][1] + front_xy[rh][1]) / 2)
            p1, p2 = _pt(sh_mid), _pt(hip_mid)
            c = COLORS["torso"]
            _draw_length_line(img, p1, p2, f"Torso length: {measurements['torso_length_cm']} cm", c,
                               label_pos=(p2[0] + 10, (p1[1] + p2[1]) // 2))

    # Chest/waist/hips/neck: these are CIRCUMFERENCES -- the line below is
    # only the front-view WIDTH component that fed into the estimate.
    for key, name, color_key in CIRCUMFERENCE_KEYS:
        entry = measurements.get(key)
        if entry:
            _draw_flat_band(img, entry, "front_row", "front_x_min", "front_x_max",
                             "width_cm", "width", name, COLORS[color_key])

    # Thigh: also a circumference -- front-view width component per leg.
    thigh = measurements.get("thigh")
    if thigh:
        c = COLORS["thigh"]
        for side_name in ("left", "right"):
            leg = thigh.get(side_name)
            if leg:
                _draw_flat_band(img, leg, "front_row", "front_x_min", "front_x_max",
                                 "width_cm", "width", "Thigh", c)

    return img


# ---------------------------------------------------------------------------
# Side view: circumference DEPTH bands only -- no length measurements are
# computed from the side view.
# ---------------------------------------------------------------------------
def draw_side_measurements(side_img, measurements):
    img = side_img.copy()

    for key, name, color_key in CIRCUMFERENCE_KEYS:
        entry = measurements.get(key)
        if entry:
            _draw_flat_band(img, entry, "side_row", "side_x_min", "side_x_max",
                             "depth_cm", "depth", name, COLORS[color_key])

    thigh = measurements.get("thigh")
    if thigh:
        c = COLORS["thigh"]
        for side_name in ("left", "right"):
            leg = thigh.get(side_name)
            if leg:
                _draw_flat_band(img, leg, "side_row", "side_x_min", "side_x_max",
                                 "depth_cm", "depth", "Thigh", c)

    return img


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def visualize_measurements(front_img, front_xy, front_conf, side_img, measurements,
                            out_dir="pipeline_output", prefix="measurements"):
    """
    front_img, side_img: BGR numpy arrays (the original frames, not masks).
                          side_img may be None if no side view was used.
    front_xy, front_conf: (17,2) / (17,) pose keypoints for the front frame.
    measurements: the dict returned by anthropometry.estimate_measurements().

    Returns (front_annotated_path, side_annotated_path_or_None).
    """
    os.makedirs(out_dir, exist_ok=True)

    front_annotated = draw_front_measurements(front_img, front_xy, front_conf, measurements)
    front_path = os.path.join(out_dir, f"{prefix}_front.jpg")
    cv2.imwrite(front_path, front_annotated)

    side_path = None
    if side_img is not None:
        side_annotated = draw_side_measurements(side_img, measurements)
        side_path = os.path.join(out_dir, f"{prefix}_side.jpg")
        cv2.imwrite(side_path, side_annotated)

    return front_path, side_path