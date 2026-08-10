import argparse
import os
import re
import shutil
import threading
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse

from task_runtime import create_runtime


BASE_DIR = Path(__file__).resolve().parent
WORKSPACE = Path(os.environ.get("RK_CV_WORKSPACE", BASE_DIR / "workspace"))
UPLOADS = WORKSPACE / "uploads"
OUTPUTS = WORKSPACE / "outputs"
UPLOADS.mkdir(parents=True, exist_ok=True)
OUTPUTS.mkdir(parents=True, exist_ok=True)

runtime = None
runtime_lock = threading.Lock()
config_lock = threading.Lock()
runtime_config = {"overlay_alpha": 0.5}
last_preview = None
preview_lock = threading.Lock()
analysis_state = {
    "is_processing": False,
    "progress": 0,
    "current_file": "",
    "error": "",
}

app = FastAPI(
    title="reComputer RK-CV DeepLabV3 API",
    version="1.1.0",
    description="RK3576/RK3588 DeepLabV3 semantic-segmentation service",
)


def safe_filename(filename, allowed=None):
    if not filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    name = Path(filename).name
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,254}", name):
        raise HTTPException(status_code=400, detail="Invalid filename")
    if allowed and Path(name).suffix.lower() not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Allowed extensions: {sorted(allowed)}",
        )
    return name


def validate_overlay_alpha(value):
    try:
        alpha = float(value)
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=400,
            detail="overlay_alpha must be a number between 0 and 1",
        ) from exc
    if not 0 <= alpha <= 1:
        raise HTTPException(
            status_code=400,
            detail="overlay_alpha must be between 0 and 1",
        )
    return alpha


def set_preview(image):
    global last_preview
    if image is None:
        return
    if isinstance(image, bytes):
        encoded = image
    else:
        ok, data = cv2.imencode(".jpg", image)
        if not ok:
            return
        encoded = data.tobytes()
    with preview_lock:
        last_preview = encoded


def get_preview():
    with preview_lock:
        if last_preview is not None:
            return last_preview
    image = np.zeros((360, 640, 3), dtype=np.uint8)
    cv2.putText(
        image,
        "Upload an image to start",
        (135, 185),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 230, 118),
        2,
    )
    return cv2.imencode(".jpg", image)[1].tobytes()


def run_prediction(image, params):
    with runtime_lock:
        result, preview = runtime.predict(image, params)
    set_preview(preview)
    return result


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "model": runtime.name if runtime else None,
        "platform": runtime.platform if runtime else None,
        "input_kind": runtime.input_kind if runtime else None,
        "model_ready": runtime is not None,
        "models": runtime.model_names if runtime else [],
    }


@app.get("/api/config")
async def get_config():
    with config_lock:
        return dict(runtime_config)


@app.post("/api/config")
async def update_config(value: dict):
    if "overlay_alpha" not in value:
        raise HTTPException(
            status_code=400,
            detail="overlay_alpha is required",
        )
    alpha = validate_overlay_alpha(value["overlay_alpha"])
    with config_lock:
        runtime_config["overlay_alpha"] = alpha
        return dict(runtime_config)


@app.post("/api/models/deeplabv3/predict")
async def predict(
    file: UploadFile = File(...),
    overlay_alpha: Optional[float] = Form(None),
):
    if runtime is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty upload")
    image = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image")

    with config_lock:
        params = dict(runtime_config)
    if overlay_alpha is not None:
        params["overlay_alpha"] = validate_overlay_alpha(overlay_alpha)

    started = time.perf_counter()
    try:
        result = run_prediction(image, params)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {
        "success": True,
        "model": runtime.name,
        "inference_time": round(time.perf_counter() - started, 4),
        "result": result,
    }


@app.get("/api/video_feed")
async def video_feed():
    def generate():
        while True:
            yield (
                b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                + get_preview()
                + b"\r\n"
            )
            time.sleep(0.1)

    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.post("/api/video/upload")
async def upload_video(file: UploadFile = File(...)):
    filename = safe_filename(file.filename, {".mp4"})
    path = UPLOADS / filename
    with path.open("wb") as target:
        shutil.copyfileobj(file.file, target)
    return {"status": "uploaded", "filename": filename}


def analyze_video_file(input_path, output_path):
    analysis_state.update(is_processing=True, progress=0, error="")
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        analysis_state.update(is_processing=False, error="Cannot open video")
        return
    total = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    frame_index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            with config_lock:
                params = dict(runtime_config)
            _, preview = runtime.predict(frame, params)
            writer.write(preview)
            set_preview(preview)
            frame_index += 1
            analysis_state["progress"] = int(frame_index * 100 / total)
        analysis_state["progress"] = 100
    except Exception as exc:
        analysis_state["error"] = str(exc)
    finally:
        cap.release()
        writer.release()
        analysis_state["is_processing"] = False


@app.post("/api/video/analyze")
async def analyze_video(filename: str = Form(...)):
    filename = safe_filename(filename, {".mp4"})
    source = UPLOADS / filename
    if not source.exists():
        raise HTTPException(status_code=404, detail="Video not found")
    if analysis_state["is_processing"]:
        raise HTTPException(
            status_code=409,
            detail="Another video is being processed",
        )
    output = OUTPUTS / f"{source.stem}_result.mp4"
    analysis_state["current_file"] = filename
    threading.Thread(
        target=analyze_video_file,
        args=(source, output),
        daemon=True,
    ).start()
    return {"status": "started", "output": output.name}


@app.get("/api/video/status")
async def video_status():
    return dict(analysis_state)


@app.get("/api/video/list")
async def video_list():
    return {
        "uploads": sorted(path.name for path in UPLOADS.glob("*.mp4")),
        "outputs": sorted(path.name for path in OUTPUTS.glob("*.mp4")),
    }


@app.get("/api/video/download/{filename}")
async def video_download(filename: str):
    filename = safe_filename(filename, {".mp4"})
    path = OUTPUTS / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path, media_type="video/mp4", filename=filename)


@app.get("/", response_class=HTMLResponse)
async def index():
    return """<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>DeepLabV3 Semantic Segmentation</title>
  <style>
    body { font-family: sans-serif; background: #111; color: #eee; max-width: 960px; margin: 30px auto; }
    .panel { background: #222; padding: 18px; border-radius: 8px; margin-bottom: 16px; }
    button { padding: 10px 18px; background: #00e676; border: 0; cursor: pointer; }
    input[type=file] { margin: 10px 0; }
    img { max-width: 100%; border: 1px solid #444; }
    pre { background: #181818; padding: 12px; white-space: pre-wrap; }
  </style>
</head>
<body>
  <h1>DeepLabV3 Semantic Segmentation</h1>
  <p><a href="/docs" style="color:#00e676">OpenAPI documentation</a></p>
  <div class="panel">
    <input id="file" type="file" accept="image/*"><br>
    <label>Mask opacity: <span id="alphaValue">0.50</span></label>
    <input id="alpha" type="range" min="0" max="1" step="0.05" value="0.5">
    <button onclick="run()">Run segmentation</button>
    <pre id="result">Ready</pre>
  </div>
  <img src="/api/video_feed" alt="Segmentation preview">
  <script>
    const alpha = document.getElementById("alpha");
    alpha.oninput = () => {
      document.getElementById("alphaValue").innerText =
        Number(alpha.value).toFixed(2);
    };
    async function run() {
      const file = document.getElementById("file");
      const result = document.getElementById("result");
      if (!file.files.length) {
        result.innerText = "Please select an image.";
        return;
      }
      const form = new FormData();
      form.append("file", file.files[0]);
      form.append("overlay_alpha", alpha.value);
      result.innerText = "Running inference...";
      const response = await fetch("/api/models/deeplabv3/predict", {
        method: "POST",
        body: form
      });
      result.innerText = JSON.stringify(await response.json(), null, 2);
    }
  </script>
</body>
</html>"""


def main():
    parser = argparse.ArgumentParser(
        description="DeepLabV3 Web service on Rockchip NPU"
    )
    parser.add_argument(
        "--platform",
        choices=["rk3576", "rk3588"],
        required=True,
        help="Target Rockchip platform",
    )
    parser.add_argument(
        "--model_path",
        default="model/deeplabv3.rknn",
        help="DeepLabV3 RKNN model path",
    )
    parser.add_argument(
        "--sample_path",
        default="model/test.jpg",
        help="Warm-up preview image path",
    )
    parser.add_argument(
        "--overlay_alpha",
        type=float,
        default=0.5,
        help="Initial mask opacity between 0 and 1",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="FastAPI listen address",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="FastAPI listen port",
    )
    args = parser.parse_args()

    alpha = validate_overlay_alpha(args.overlay_alpha)
    with config_lock:
        runtime_config["overlay_alpha"] = alpha

    global runtime
    runtime = create_runtime(
        args.platform,
        Path(args.model_path),
        Path(args.sample_path),
    )
    try:
        runtime.warmup_preview(set_preview, dict(runtime_config))
    except Exception as exc:
        print(f"Sample warmup skipped: {exc}", flush=True)
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
        log_config=None,
    )


if __name__ == "__main__":
    main()
