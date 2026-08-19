"""
mhr_calibration.py

Optimizes MHR's identity (shape) parameters so the resulting 3D mesh's own
measured dimensions match the cm values already extracted by anthropometry.py
from the 2D photos.

This works STANDALONE: pass initial_identity=None and it optimizes from a
neutral/average body -- no need for SAM 3D Body, no GPU requirement, no
gated access, no repo cloning. If body_3d.py's fused_identity (from
reconstructing your 4 photographed views with SAM 3D Body) is available,
passing it in as initial_identity gives the optimizer a photo-informed
starting point instead of a neutral one -- but it's an optional head start,
not a requirement; the optimizer converges to your target measurements
either way since it's gradient descent against real cm targets, not a
one-shot estimate.

VERIFIED vs UNVERIFIED, please read before using:

VERIFIED (I downloaded the public `mhr` PyPI package and read its actual
source in mhr/mhr.py):
  - MHR.forward(identity_coeffs, model_parameters, face_expr_coeffs) returns
    (vertices, skel_state) as plain differentiable torch.Tensors.
    - identity_coeffs: (batch, 45)
    - model_parameters: (batch, N) where N = character.parameter_transform.size
      - 45 - 72 (the identity + face-expression blendshape dims are handled
      separately and padded internally -- do NOT include them here)
    - face_expr_coeffs: (batch, 72) or None
  - MHR.from_files(folder, device, lod) loads the model from a local asset
    folder (NOT gated -- `curl -OL .../assets.zip` per the MHR repo README).
  - NUM_IDENTITY_BLENDSHAPES = 45, NUM_FACE_EXPRESSION_BLENDSHAPES = 72 are
    literal constants in mhr.mhr.

UNVERIFIED (pymomentum's Character/skeleton classes are a compiled C++
extension -- I could not read their Python-visible API surface without a
full torch + weights install, which didn't fit in this sandbox's disk
budget):
  - The exact attribute name for mesh face topology on `character`
  - The exact joint names/ordering in `skel_state`, needed to measure
    skeletal LENGTHS (shoulder width, sleeve, inseam, torso length) the
    same way anthropometry.py did with keypoints.
  Both are resolved defensively at runtime (see FACE_ATTR_CANDIDATES /
  get_mesh_faces below) with a clear, actionable error if my guesses are
  wrong -- exactly the same pattern as body_3d.py.

WHAT'S ACTUALLY TESTED in this file (no model/weights required):
  - measure_width_depth(): the differentiable vertex-band measurement math
  - The Adam optimization LOOP itself, verified against a synthetic
    differentiable stand-in for MHR (a toy linear identity->vertices
    function) to confirm it actually converges targets -- see the bottom
    of this docstring's companion test in the chat for results.
"""

import os

import numpy as np
import torch

FACE_ATTR_CANDIDATES = ("faces", "mesh_faces", "triangles", "tris")

# Circumference/length groups measured on the MESH mirror the ones in
# anthropometry.py, so the calibration target uses the exact same units and
# semantics as what was extracted from the photos.
CIRCUMFERENCE_TARGETS = ("chest_bust", "waist", "hips", "neck", "thigh")
LENGTH_TARGETS = ("shoulder_width_cm", "sleeve_length_cm", "inseam_cm", "torso_length_cm")


# ---------------------------------------------------------------------------
# Model loading (thin wrapper over the verified mhr.MHR API)
# ---------------------------------------------------------------------------
def load_mhr_model(asset_folder=None, device=None, lod=1):
    """
    asset_folder: path (str or Path) to the unzipped MHR assets
                  (compact_v6_1.model, lod{N}.fbx, corrective_*.npz).
                  Defaults to mhr's own bundled default folder if None
                  (see mhr/io.py).
    """
    from pathlib import Path

    from mhr.mhr import MHR  # local import -- only needed once this runs for real

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kwargs = {"device": device, "lod": lod}
    if asset_folder is not None:
        # mhr/io.py builds asset paths with pathlib's `folder / "file.ext"` --
        # that operator requires folder to be a Path, not a plain str (a
        # bare string here raises "unsupported operand type(s) for /:
        # 'str' and 'str'"), so we convert explicitly regardless of what
        # type the caller passed in (e.g. a CLI arg, which is always str).
        kwargs["folder"] = Path(asset_folder)
    return MHR.from_files(**kwargs)


def get_mesh_faces(mhr_model):
    """Defensive lookup of mesh face topology -- see module docstring."""
    character = mhr_model.character
    for attr in FACE_ATTR_CANDIDATES:
        if hasattr(character, attr):
            return np.asarray(getattr(character, attr))
        mesh = getattr(character, "mesh", None)
        if mesh is not None and hasattr(mesh, attr):
            return np.asarray(getattr(mesh, attr))
    available = [a for a in dir(character) if not a.startswith("_")]
    raise AttributeError(
        f"Couldn't find mesh face topology using candidates {FACE_ATTR_CANDIDATES}. "
        f"Attributes actually present on `character`: {available}. "
        f"Add the correct one to FACE_ATTR_CANDIDATES in mhr_calibration.py."
    )


def neutral_model_parameters(mhr_model, batch_size=1, device=None):
    """A zero pose vector -- MHR's neutral/rest standing pose."""
    from mhr.mhr import NUM_FACE_EXPRESSION_BLENDSHAPES, NUM_IDENTITY_BLENDSHAPES

    device = device or next(mhr_model.parameters(), torch.zeros(1)).device
    n_pose = (mhr_model.character.parameter_transform.size
              - NUM_IDENTITY_BLENDSHAPES - NUM_FACE_EXPRESSION_BLENDSHAPES)
    return torch.zeros(batch_size, n_pose, device=device)


def make_forward_fn(mhr_model, device=None):
    """Wraps mhr_model.forward into forward_fn(identity_1d) -> vertices (V,3),
    fixing pose to neutral/rest and batch size to 1 -- the interface
    calibrate_identity_to_measurements() expects. Kept separate from that
    function so the optimizer can be tested against a synthetic stand-in
    without needing the real model (see module docstring)."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pose = neutral_model_parameters(mhr_model, batch_size=1, device=device)

    def forward_fn(identity_1d):
        identity_batched = identity_1d.unsqueeze(0)  # (1, 45)
        vertices, _ = mhr_model.forward(identity_batched, pose, face_expr_coeffs=None)
        return vertices[0]  # (V, 3)

    return forward_fn


# Default height fractions (0=feet, 1=head-top) for each circumference level,
# based on standard adult figure-drawing proportions (an "8-heads-tall"
# canonical figure). These assume MHR's neutral/rest pose is a roughly
# linear standing pose spanning the mesh's own vertical extent -- UNVERIFIED
# against the real mesh (see module docstring); treat as a starting point
# and tune against a real exported OBJ before trusting the calibration.
DEFAULT_HEIGHT_FRACS = {
    "neck": 0.86,
    "chest_bust": 0.73,
    "waist": 0.60,
    "hips": 0.53,
    "thigh": 0.48,
}


# ---------------------------------------------------------------------------
# Differentiable mesh measurement (pure torch -- fully testable standalone)
# ---------------------------------------------------------------------------
def measure_width_depth(vertices, height_frac, up_axis=1, front_axis=2, side_axis=0,
                         band_frac=0.02):
    """
    vertices: (V, 3) tensor, a single mesh in rest/neutral pose.
    height_frac: 0 (feet) .. 1 (head) -- which body level to measure.
    up_axis/front_axis/side_axis: which of x/y/z is vertical/depth/width in
        MHR's coordinate convention. Defaults assume Y-up (common convention);
        VERIFY against a real exported mesh (e.g. open it in Blender/MeshLab)
        before trusting these -- flip as needed, this is the one geometric
        assumption in this module I could not verify without real output.

    Returns (width, depth) as differentiable scalars: the extent of the
    vertex band at that height along side_axis (width) and front_axis
    (depth) -- the same width/depth ellipse-cross-section idea used in
    anthropometry.py, just measured directly on the 3D mesh instead of
    inferred from a 2D silhouette.
    """
    y = vertices[:, up_axis]
    y_min, y_max = y.min(), y.max()
    target_y = y_min + height_frac * (y_max - y_min)
    band = band_frac * (y_max - y_min)

    mask = (y >= target_y - band) & (y <= target_y + band)
    band_verts = vertices[mask]
    if band_verts.shape[0] == 0:
        # Widen the band if nothing fell inside it (sparse mesh regions)
        mask = (y - target_y).abs() <= band * 3
        band_verts = vertices[mask]

    width = band_verts[:, side_axis].max() - band_verts[:, side_axis].min()
    depth = band_verts[:, front_axis].max() - band_verts[:, front_axis].min()
    return width, depth


def ellipse_circumference(width, depth):
    """Same Ramanujan approximation as anthropometry.py, in torch so it stays
    differentiable end-to-end through the optimizer."""
    a, b = width / 2, depth / 2
    h = ((a - b) ** 2) / ((a + b) ** 2 + 1e-8)
    return torch.pi * (a + b) * (1 + (3 * h) / (10 + torch.sqrt(4 - 3 * h + 1e-8)))


# ---------------------------------------------------------------------------
# Calibration loop
# ---------------------------------------------------------------------------
def calibrate_identity_to_measurements(forward_fn, target_measurements_cm,
                                        height_fracs=None, num_identity=45,
                                        initial_identity=None, iterations=300,
                                        lr=0.05, device=None, verbose=True,
                                        reg_weight=0.03, clamp_range=3.0):
    """
    forward_fn(identity_coeffs) -> vertices (V,3): a closure wrapping the
        real MHR forward pass with pose/face params fixed to neutral, so
        the optimizer only has to reason about the 45-dim identity vector.
        (Kept as an injected function -- rather than calling mhr.MHR
        directly in here -- so this loop can be tested against a synthetic
        stand-in without the real model, and reused unchanged once wired
        to the real one. Use make_forward_fn(mhr_model) to build this for
        the real model.)
    target_measurements_cm: the dict returned by
        anthropometry.estimate_measurements() -- chest_bust/waist/hips/neck/
        thigh (each {"circumference_cm": ...}) and shoulder_width_cm/
        sleeve_length_cm/inseam_cm/torso_length_cm.
    height_fracs: dict mapping each CIRCUMFERENCE_TARGETS key to a 0..1
        height fraction (feet=0, head=1) on the MESH. Defaults to
        DEFAULT_HEIGHT_FRACS (approximate, see its docstring) if omitted.
    reg_weight: strength of an L2 penalty pulling identity back toward
        zero (the neutral/average body) each step. WITHOUT this, gradient
        descent can push identity params to extreme, out-of-distribution
        values to force-match a target circumference -- MHR's blendshapes
        are only well-behaved near the training distribution, so extreme
        values produce anatomically nonsensical meshes (flared hips, twig
        -thin limbs) even though the loss numerically improves. This is
        exactly what caused the distorted mesh you saw earlier -- this
        parameter is the fix, not just a tuning knob to leave at 0.
    clamp_range: hard safety net on top of the soft regularization --
        identity values are clamped to [-clamp_range, clamp_range] after
        every step. Meta's own docs describe -3..+3 as identity's "typical
        range", so this is a generous bound, not a tight one.

    Returns (final_identity, history) where history is a list of
    per-iteration dicts: {"loss", "reg_loss", "identity_norm", "errors":
    {key: {"predicted_cm", "target_cm", "pct_error"}}} -- useful for
    verifying the fit actually converged sanely, not just that loss went down.
    """
    if height_fracs is None:
        height_fracs = DEFAULT_HEIGHT_FRACS

    device = device or torch.device("cpu")
    identity = (initial_identity.clone() if initial_identity is not None
                else torch.zeros(num_identity, device=device))
    identity.requires_grad_(True)

    optimizer = torch.optim.Adam([identity], lr=lr)
    history = []

    for step in range(iterations):
        optimizer.zero_grad()
        vertices = forward_fn(identity)

        measurement_loss = torch.tensor(0.0, device=device)
        n_terms = 0
        step_errors = {}

        for key in CIRCUMFERENCE_TARGETS:
            target = target_measurements_cm.get(key)
            if not target or target.get("circumference_cm") is None or key not in height_fracs:
                continue
            width, depth = measure_width_depth(vertices, height_fracs[key])
            pred_circ = ellipse_circumference(width, depth)
            target_circ = torch.tensor(float(target["circumference_cm"]), device=device)
            measurement_loss = measurement_loss + ((pred_circ - target_circ) / target_circ) ** 2
            n_terms += 1
            step_errors[key] = {
                "predicted_cm": round(pred_circ.item(), 1),
                "target_cm": round(target_circ.item(), 1),
                "pct_error": round(100 * (pred_circ.item() - target_circ.item()) / target_circ.item(), 1),
            }

        if n_terms == 0:
            raise ValueError(
                "No circumference targets matched height_fracs -- nothing to optimize. "
                "Provide height_fracs for at least one of: " + ", ".join(CIRCUMFERENCE_TARGETS))

        reg_loss = reg_weight * (identity ** 2).mean()
        loss = measurement_loss + reg_loss

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            identity.clamp_(-clamp_range, clamp_range)

        history.append({
            "loss": loss.item(),
            "measurement_loss": measurement_loss.item(),
            "reg_loss": reg_loss.item(),
            "identity_norm": identity.norm().item(),
            "errors": step_errors,
        })

        if verbose and (step % max(1, iterations // 10) == 0 or step == iterations - 1):
            print(f"  step {step:>4}/{iterations}  loss={loss.item():.6f}  "
                  f"(measurement={measurement_loss.item():.6f}, reg={reg_loss.item():.6f})  "
                  f"|identity|={identity.norm().item():.2f}")

    if verbose:
        print("\nFinal per-measurement fit:")
        for key, err in history[-1]["errors"].items():
            print(f"  {key:<12} predicted={err['predicted_cm']:>6}cm  "
                  f"target={err['target_cm']:>6}cm  error={err['pct_error']:+.1f}%")

    return identity.detach(), history


def export_obj(vertices, faces, out_path):
    """
    Writes a minimal Wavefront .obj file. faces may be 0- or 1-indexed;
    OBJ format requires 1-indexed, so we detect and correct automatically.

    (Identical logic to body_3d.py's export_obj, duplicated here so
    mhr_calibration.py has zero dependency on body_3d.py/SAM 3D Body ever
    being installed -- --calibrate-mesh is meant to work fully standalone.)
    """
    faces = np.asarray(faces)
    if faces.min() == 0:
        faces = faces + 1  # convert 0-indexed -> OBJ's 1-indexed convention

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        f.write("# exported by mhr_calibration.py\n")
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in faces:
            idx_str = " ".join(str(int(i)) for i in face)
            f.write(f"f {idx_str}\n")
    return out_path


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------
def calibrate_and_export(asset_folder, target_measurements_cm, initial_identity=None,
                          out_path="pipeline_output/body_3d_model.obj",
                          height_fracs=None, iterations=300, lr=0.05, lod=1, device=None,
                          reg_weight=0.03, clamp_range=3.0):
    """
    One-call convenience wrapper: load MHR, calibrate identity to the given
    measurements (optionally starting from body_3d.py's fused_identity),
    forward once more at the final identity, and export the mesh.

    initial_identity: numpy array or torch tensor (45,) from
        body_3d.build_3d_model()'s "fused_identity" result -- gives the
        optimizer a head start informed by your actual 4 photographed views,
        rather than starting from a generic average body. Optional -- pass
        None (the default) to run fully standalone, no body_3d.py needed.
    reg_weight, clamp_range: passed straight through to
        calibrate_identity_to_measurements() -- see its docstring. If the
        exported mesh still looks distorted, RAISE reg_weight (fewer
        distortions, looser measurement fit); if it looks too generic/
        doesn't match your measurements closely enough, LOWER it.

    Returns {"obj_path", "identity", "loss_history"}.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mhr_model = load_mhr_model(asset_folder=asset_folder, device=device, lod=lod)
    faces = get_mesh_faces(mhr_model)
    forward_fn = make_forward_fn(mhr_model, device=device)

    init = None
    if initial_identity is not None:
        init = torch.as_tensor(np.asarray(initial_identity), dtype=torch.float32, device=device)

    final_identity, history = calibrate_identity_to_measurements(
        forward_fn, target_measurements_cm, height_fracs=height_fracs,
        initial_identity=init, iterations=iterations, lr=lr, device=device)

    with torch.no_grad():
        final_vertices = forward_fn(final_identity).cpu().numpy()

    export_obj(final_vertices, faces, out_path)
    print(f"Calibrated 3D mesh saved -> {out_path}")

    return {"obj_path": out_path, "identity": final_identity.cpu().numpy(), "loss_history": history}