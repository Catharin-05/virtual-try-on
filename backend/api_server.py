"""
api_server.py

FastAPI wrapper around main.py's run_pipeline(), so the frontend can:
  1. POST a video + height -> get a job_id back immediately
  2. poll GET /api/scan/{job_id}/status while it processes in the background
  3. once done, GET /api/scan/{job_id}/model.obj and /api/scan/{job_id}/measurements

Run:
    pip install fastapi uvicorn python-multipart
    uvicorn api_server:app --reload --port 8000

Config (env vars):
    MHR_ASSETS_PATH   -- path to the unzipped MHR asset folder (default: ./assets)
    JOBS_DIR          -- where per-job video/output folders are stored (default: ./jobs)
    FRONTEND_ORIGIN   -- allowed CORS origin for the frontend dev server
                         (default: http://localhost:5173, Vite's default port)

NOTE on scaling: jobs are tracked in an in-memory dict and run in a
background thread within this single process. That's fine for local/dev
use, but a server restart loses in-flight job state, and heavy models
mean only one job realistically runs at a time anyway. For production,
swap JOBS for a persistent store (Redis/DB) and the thread for a real
task queue (Celery/RQ) -- the HTTP interface below wouldn't need to change.
"""

import os
import shutil
import threading
import traceback
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from main import run_pipeline

MHR_ASSETS_PATH = os.environ.get("MHR_ASSETS_PATH", "./assets")
JOBS_DIR = Path(os.environ.get("JOBS_DIR", "./jobs"))
JOBS_DIR.mkdir(exist_ok=True)
FRONTEND_ORIGIN = os.environ.get("FRONTEND_ORIGIN", "http://localhost:5173")

# Optional server-side bypass: skip SAM 3D Body entirely and use a
# pre-generated OBJ (+ optionally its .npy identity vector) as the starting
# point for --calibrate-mesh instead. This is a fixed operator-configured
# fallback (e.g. a generic body mesh prepared once), not something an
# end-user uploads per-scan -- see the input_3d_obj/input_identity bypass
# block in main.py's Stage 4.
INPUT_3D_OBJ_PATH = os.environ.get("INPUT_3D_OBJ_PATH") or None
INPUT_IDENTITY_PATH = os.environ.get("INPUT_IDENTITY_PATH") or None

app = FastAPI(title="Virtual Fitting Room API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_ORIGIN],
    allow_methods=["*"],
    allow_headers=["*"],
)

# job_id -> {"status": "processing"|"done"|"error", "message": str, ...}
JOBS = {}


def _run_job(job_id: str, video_path: Path, height_cm: float):
    job_dir = JOBS_DIR / job_id
    try:
        JOBS[job_id]["message"] = "Finding front/back/side views…"

        _, _summary, measurements, _body_3d_result, calibration_result = run_pipeline(
            video_path=str(video_path),
            out_dir=str(job_dir),
            num_frames=10,
            user_height_cm=height_cm,
            calibrate_mesh_flag=True,
            mhr_assets=MHR_ASSETS_PATH,
        )

        if calibration_result is None or not os.path.exists(calibration_result["obj_path"]):
            raise RuntimeError(
                "Pipeline finished but no 3D model was produced -- check the "
                "server logs for the mesh calibration stage (it needs a "
                "confident front view and MHR assets to be reachable).")

        JOBS[job_id].update({
            "status": "done",
            "message": "Done",
            "obj_path": calibration_result["obj_path"],
            "measurements": measurements,
        })
    except Exception as e:
        traceback.print_exc()
        JOBS[job_id].update({"status": "error", "message": str(e)})


@app.post("/api/scan")
async def start_scan(video: UploadFile = File(...), height_cm: float = Form(...)):
    if not (100 <= height_cm <= 230):
        raise HTTPException(400, "height_cm should be a realistic adult height in centimeters.")

    job_id = uuid.uuid4().hex[:12]
    job_dir = JOBS_DIR / job_id
    upload_dir = job_dir / "input_video"
    upload_dir.mkdir(parents=True, exist_ok=True)

    video_path = upload_dir / (video.filename or "upload.mp4")
    with open(video_path, "wb") as f:
        shutil.copyfileobj(video.file, f)

    JOBS[job_id] = {"status": "processing", "message": "Queued…"}
    thread = threading.Thread(target=_run_job, args=(job_id, video_path, height_cm), daemon=True)
    thread.start()

    return {"job_id": job_id}


@app.get("/api/scan/{job_id}/status")
async def get_status(job_id: str):
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "Unknown job_id")
    return {"status": job["status"], "message": job.get("message", "")}


@app.get("/api/scan/{job_id}/model.obj")
async def get_model(job_id: str):
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "Unknown job_id")
    if job["status"] != "done":
        raise HTTPException(409, "Model isn't ready yet")
    return FileResponse(job["obj_path"], media_type="text/plain", filename="model.obj")


@app.get("/api/scan/{job_id}/measurements")
async def get_measurements(job_id: str):
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "Unknown job_id")
    if job["status"] != "done":
        raise HTTPException(409, "Measurements aren't ready yet")
    return JSONResponse(job["measurements"])


@app.get("/api/health")
async def health():
    return {"ok": True}