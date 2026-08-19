import argparse
import json
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
runtime_config = {"min_score": 0.0, "max_plates": 8, "min_text_length": 6, "plate_layout": "auto"}
last_preview = None
preview_lock = threading.Lock()
latest_result = {"plates": [], "count": 0, "source_mode": "waiting", "inference_ms": 0.0}
result_lock = threading.Lock()
analysis_state = {
    "is_processing": False,
    "progress": 0,
    "current_file": "",
    "output": "",
    "error": "",
}
analysis_lock = threading.Lock()
source_state = {
    "mode": "web",
    "source": "",
    "is_running": False,
    "frames": 0,
    "inference_ms": 0.0,
    "error": "",
}
source_lock = threading.Lock()

app = FastAPI(title="reComputer RK-CV LPRNet API", version="1.0.0")


def safe_filename(filename, allowed=None):
    if not filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    name = Path(filename).name
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,254}", name):
        raise HTTPException(status_code=400, detail="Use an ASCII filename containing letters, numbers, dots, dashes, or underscores")
    if allowed and Path(name).suffix.lower() not in allowed:
        raise HTTPException(status_code=400, detail=f"Allowed extensions: {sorted(allowed)}")
    return name


def config_snapshot():
    with config_lock:
        return dict(runtime_config)


def set_preview(image):
    global last_preview
    if image is None:
        return
    if isinstance(image, bytes):
        encoded = image
    else:
        ok, data = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
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
    cv2.putText(image, "Waiting for camera, video, or upload", (52, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 230, 118), 2)
    return cv2.imencode(".jpg", image)[1].tobytes()


def run_prediction(image, params=None):
    if params is None:
        params = config_snapshot()
    started = time.perf_counter()
    with runtime_lock:
        result, preview = runtime.predict(image, params)
    inference_ms = round((time.perf_counter() - started) * 1000, 2)
    set_preview(preview)
    snapshot = dict(result)
    snapshot["inference_ms"] = inference_ms
    snapshot["updated_at"] = time.time()
    with result_lock:
        latest_result.clear()
        latest_result.update(snapshot)
    return result, inference_ms


@app.get("/api/health")
async def health():
    with source_lock:
        source = dict(source_state)
    return {
        "status": "ok",
        "model": runtime.name if runtime else None,
        "platform": runtime.platform if runtime else None,
        "input_kind": runtime.input_kind if runtime else None,
        "model_ready": runtime is not None,
        "models": runtime.model_names if runtime else [],
        "source": source,
    }


@app.get("/api/results/latest")
async def results_latest():
    with result_lock:
        return dict(latest_result)


@app.get("/api/config")
async def get_config():
    return config_snapshot()


@app.post("/api/config")
async def update_config(value: dict):
    with config_lock:
        if "min_score" in value:
            min_score = float(value["min_score"])
            if not 0 <= min_score <= 1:
                raise HTTPException(status_code=400, detail="min_score must be between 0 and 1")
            runtime_config["min_score"] = min_score
        if "max_plates" in value:
            max_plates = int(value["max_plates"])
            if not 1 <= max_plates <= 32:
                raise HTTPException(status_code=400, detail="max_plates must be between 1 and 32")
            runtime_config["max_plates"] = max_plates
        if "min_text_length" in value:
            min_text_length = int(value["min_text_length"])
            if not 1 <= min_text_length <= 8:
                raise HTTPException(status_code=400, detail="min_text_length must be between 1 and 8")
            runtime_config["min_text_length"] = min_text_length
        if "plate_layout" in value:
            plate_layout = str(value["plate_layout"])
            if plate_layout not in {"auto", "chinese", "international"}:
                raise HTTPException(status_code=400, detail="plate_layout must be auto, chinese, or international")
            runtime_config["plate_layout"] = plate_layout
        return dict(runtime_config)


@app.post("/api/models/lprnet/predict")
async def predict(
    file: UploadFile = File(...),
    min_score: Optional[float] = Form(None),
    max_plates: Optional[int] = Form(None),
    min_text_length: Optional[int] = Form(None),
    plate_layout: Optional[str] = Form(None),
    manual_box: Optional[str] = Form(None),
    whole_image: bool = Form(False),
):
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty upload")
    image = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="The uploaded file is not a decodable image")
    params = config_snapshot()
    if min_score is not None:
        params["min_score"] = float(min_score)
    if max_plates is not None:
        params["max_plates"] = int(max_plates)
    if min_text_length is not None:
        params["min_text_length"] = int(min_text_length)
    if plate_layout is not None:
        params["plate_layout"] = plate_layout
    if manual_box:
        try:
            params["manual_box"] = json.loads(manual_box)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail="manual_box must be JSON [x1, y1, x2, y2]") from exc
    params["whole_image"] = whole_image
    try:
        result, inference_ms = run_prediction(image, params)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {
        "success": True,
        "model": "lprnet",
        "source": "uploaded image",
        "inference_time": round(inference_ms / 1000, 4),
        "result": result,
    }


@app.get("/api/video_feed")
async def video_feed():
    def generate():
        while True:
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + get_preview() + b"\r\n"
            time.sleep(0.1)

    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.post("/api/video/upload")
async def upload_video(file: UploadFile = File(...)):
    filename = safe_filename(file.filename, {".mp4"})
    path = UPLOADS / filename
    with path.open("wb") as target:
        shutil.copyfileobj(file.file, target)
    return {"status": "uploaded", "filename": filename}


def analyze_video_file(input_path, output_path):
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        with analysis_lock:
            analysis_state.update(is_processing=False, error="Cannot open video")
        return
    total = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        cap.release()
        with analysis_lock:
            analysis_state.update(is_processing=False, error="Cannot create output video")
        return
    frame_index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            _, preview = None, None
            with runtime_lock:
                result, preview = runtime.predict(frame, config_snapshot())
            set_preview(preview)
            snapshot = dict(result)
            snapshot["updated_at"] = time.time()
            with result_lock:
                latest_result.clear()
                latest_result.update(snapshot)
            writer.write(preview if preview is not None else frame)
            frame_index += 1
            with analysis_lock:
                analysis_state["progress"] = int(frame_index * 100 / total)
        with analysis_lock:
            analysis_state["progress"] = 100
    except Exception as exc:
        with analysis_lock:
            analysis_state["error"] = str(exc)
    finally:
        cap.release()
        writer.release()
        with analysis_lock:
            analysis_state["is_processing"] = False


@app.post("/api/video/analyze")
async def analyze_video(filename: str = Form(...)):
    filename = safe_filename(filename, {".mp4"})
    source = UPLOADS / filename
    if not source.exists():
        raise HTTPException(status_code=404, detail="Video not found")
    with analysis_lock:
        if analysis_state["is_processing"]:
            raise HTTPException(status_code=409, detail="Another video is being processed")
        output = OUTPUTS / f"{source.stem}_result.mp4"
        analysis_state.update(
            is_processing=True,
            progress=0,
            current_file=filename,
            output=output.name,
            error="",
        )
    threading.Thread(target=analyze_video_file, args=(source, output), daemon=True).start()
    return {"status": "started", "output": output.name}


@app.get("/api/video/status")
async def video_status():
    with analysis_lock:
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


def process_realtime_source(video, camera_id):
    is_video = video is not None
    source = str(video) if is_video else f"/dev/video{camera_id}"
    mode = "video" if is_video else "camera"
    with source_lock:
        source_state.update(mode=mode, source=source, is_running=False, frames=0, inference_ms=0.0, error="")
    cap = cv2.VideoCapture(str(video) if is_video else camera_id)
    if not is_video:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        error = f"Cannot open {mode} source: {source}"
        with source_lock:
            source_state["error"] = error
        print(error, flush=True)
        return

    source_fps = cap.get(cv2.CAP_PROP_FPS) if is_video else 0
    frame_interval = 1.0 / source_fps if source_fps and source_fps > 0 else 0
    with source_lock:
        source_state["is_running"] = True
    print(f"Analyzing {mode} source: {source}", flush=True)
    try:
        while True:
            loop_started = time.perf_counter()
            ok, frame = cap.read()
            if not ok:
                if is_video:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    time.sleep(0.05)
                    continue
                with source_lock:
                    source_state["error"] = f"Lost camera source: {source}"
                break
            try:
                _, inference_ms = run_prediction(frame)
            except Exception as exc:
                with source_lock:
                    source_state["error"] = str(exc)
                print(f"Realtime inference failed: {exc}", flush=True)
                break
            with source_lock:
                source_state["frames"] += 1
                source_state["inference_ms"] = inference_ms
            if frame_interval:
                remaining = frame_interval - (time.perf_counter() - loop_started)
                if remaining > 0:
                    time.sleep(remaining)
    finally:
        cap.release()
        with source_lock:
            source_state["is_running"] = False


HTML_PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>LPRNet Live Recognition</title>
  <style>
    :root { color-scheme: dark; --green:#00e676; --panel:#171a1f; --muted:#9ca3af; --border:#30363d; }
    * { box-sizing:border-box; }
    body { margin:0; background:#0d1117; color:#f0f3f6; font:15px/1.5 system-ui,sans-serif; }
    header { padding:22px max(20px,calc((100% - 1400px)/2)); border-bottom:1px solid var(--border); }
    h1 { margin:0 0 4px; font-size:24px; } h2 { margin:0 0 14px; font-size:17px; }
    main { max-width:1400px; margin:auto; padding:20px; display:grid; grid-template-columns:minmax(0,2fr) minmax(300px,1fr); gap:18px; }
    .card { background:var(--panel); border:1px solid var(--border); border-radius:12px; padding:16px; margin-bottom:18px; }
    #stream { display:block; width:100%; min-height:260px; background:#000; border-radius:8px; object-fit:contain; }
    .status { color:var(--muted); } .good { color:var(--green); }
    .plate { border-left:4px solid var(--green); background:#20252c; border-radius:6px; padding:10px 12px; margin:8px 0; }
    .plate.rejected { border-left-color:#ffaa00; }
    .plate strong { display:block; font-size:20px; letter-spacing:1px; color:#fff; }
    button,.button { display:inline-block; cursor:pointer; border:0; border-radius:6px; padding:9px 14px; color:#06130b; background:var(--green); font-weight:700; text-decoration:none; }
    button:disabled { opacity:.5; cursor:not-allowed; } input[type=number] { width:90px; } input[type=text] { width:220px; } input[type=number],input[type=text],select { padding:7px; background:#0d1117; color:#fff; border:1px solid var(--border); border-radius:5px; }
    .row { display:flex; gap:10px; align-items:center; flex-wrap:wrap; margin:9px 0; }
    .filename { color:var(--muted); max-width:260px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
    progress { width:100%; height:15px; accent-color:var(--green); }
    code { color:#7ee787; } a { color:var(--green); } pre { white-space:pre-wrap; overflow:auto; color:#c9d1d9; }
    @media(max-width:850px) { main { grid-template-columns:1fr; } }
  </style>
</head>
<body>
<header><h1>LPRNet Live License Plate Recognition</h1><div class="status">Rockchip NPU inference · camera, local video, image, and uploaded video</div></header>
<main>
  <section>
    <div class="card">
      <h2>Live annotated preview</h2>
      <img id="stream" src="/api/video_feed" alt="Live annotated license plate stream">
      <div id="sourceStatus" class="status">Loading source status...</div>
    </div>
    <div class="card">
      <h2>Analyze an image</h2>
      <div class="row"><label class="button" for="imageFile">Choose image</label><span id="imageName" class="filename">No image selected</span></div>
      <input id="imageFile" type="file" accept="image/*" hidden>
      <div class="row"><label><input id="wholeImage" type="checkbox"> The image is already a cropped license plate</label></div>
      <div class="row"><label>Manual box (optional) <input id="manualBox" type="text" placeholder="x1,y1,x2,y2"></label></div>
      <button id="imageButton" onclick="analyzeImage()">Run image recognition</button>
      <pre id="imageResponse"></pre>
    </div>
    <div class="card">
      <h2>Upload and analyze an MP4 video</h2>
      <div class="row"><label class="button" for="videoFile">Choose MP4</label><span id="videoName" class="filename">No video selected</span></div>
      <input id="videoFile" type="file" accept="video/mp4,.mp4" hidden>
      <button id="videoButton" onclick="analyzeVideo()">Upload and start analysis</button>
      <div class="row"><progress id="progress" value="0" max="100"></progress><span id="progressText">Idle</span></div>
      <a id="download" style="display:none" class="button">Download analyzed video</a>
    </div>
  </section>
  <aside>
    <div class="card"><h2>Recognized plates</h2><div id="plates" class="status">No plate recognized yet.</div></div>
    <div class="card">
      <h2>Runtime settings</h2>
      <div class="row"><label>Minimum score <input id="minScore" type="number" min="0" max="1" step="0.01" value="0"></label></div>
      <div class="row"><label>Maximum plates <input id="maxPlates" type="number" min="1" max="32" value="8"></label></div>
      <div class="row"><label>Minimum text length <input id="minTextLength" type="number" min="1" max="8" value="6"></label></div>
      <div class="row"><label>Plate layout <select id="plateLayout"><option value="auto">Auto</option><option value="chinese">Chinese plate</option><option value="international">International / light plate</option></select></label></div>
      <button onclick="saveConfig()">Apply settings</button><div id="configStatus" class="status"></div>
    </div>
    <div class="card"><h2>API</h2><p><a href="/docs">OpenAPI documentation</a></p><code>GET /api/video_feed</code><br><code>GET /api/results/latest</code><br><code>POST /api/models/lprnet/predict</code></div>
  </aside>
</main>
<script>
const imageFile=document.getElementById('imageFile'),videoFile=document.getElementById('videoFile');
imageFile.addEventListener('change',()=>document.getElementById('imageName').textContent=imageFile.files[0]?.name||'No image selected');
videoFile.addEventListener('change',()=>document.getElementById('videoName').textContent=videoFile.files[0]?.name||'No video selected');
const escapeHtml=value=>String(value).replace(/[&<>"']/g,char=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#039;'}[char]));
async function jsonRequest(url,options){const response=await fetch(url,options);const body=await response.json();if(!response.ok)throw new Error(body.detail||JSON.stringify(body));return body;}
async function analyzeImage(){
  if(!imageFile.files.length){document.getElementById('imageResponse').textContent='Choose an image first.';return;}
  const button=document.getElementById('imageButton');button.disabled=true;
  try{const data=new FormData();data.append('file',imageFile.files[0]);data.append('whole_image',document.getElementById('wholeImage').checked);data.append('plate_layout',document.getElementById('plateLayout').value);const manual=document.getElementById('manualBox').value.trim();if(manual){const values=manual.split(',').map(Number);if(values.length!==4||values.some(value=>!Number.isFinite(value)))throw new Error('Manual box must contain x1,y1,x2,y2');data.append('manual_box',JSON.stringify(values));}const body=await jsonRequest('/api/models/lprnet/predict',{method:'POST',body:data});document.getElementById('imageResponse').textContent=JSON.stringify(body,null,2);}
  catch(error){document.getElementById('imageResponse').textContent='Error: '+error.message;}finally{button.disabled=false;}
}
async function analyzeVideo(){
  if(!videoFile.files.length){document.getElementById('progressText').textContent='Choose an MP4 first.';return;}
  const button=document.getElementById('videoButton');button.disabled=true;document.getElementById('download').style.display='none';
  try{const upload=new FormData();upload.append('file',videoFile.files[0]);const saved=await jsonRequest('/api/video/upload',{method:'POST',body:upload});const request=new FormData();request.append('filename',saved.filename);await jsonRequest('/api/video/analyze',{method:'POST',body:request});document.getElementById('progressText').textContent='Analysis started';}
  catch(error){document.getElementById('progressText').textContent='Error: '+error.message;button.disabled=false;}
}
async function saveConfig(){
  try{const value={min_score:Number(document.getElementById('minScore').value),max_plates:Number(document.getElementById('maxPlates').value),min_text_length:Number(document.getElementById('minTextLength').value),plate_layout:document.getElementById('plateLayout').value};const body=await jsonRequest('/api/config',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(value)});document.getElementById('configStatus').textContent='Saved: '+JSON.stringify(body);}
  catch(error){document.getElementById('configStatus').textContent='Error: '+error.message;}
}
async function refresh(){
  try{
    const [health,result,job]=await Promise.all([jsonRequest('/api/health'),jsonRequest('/api/results/latest'),jsonRequest('/api/video/status')]);
    const source=health.source||{};document.getElementById('sourceStatus').textContent=`Source: ${source.mode||'web'} ${source.source||''} · ${source.is_running?'running':'idle'} · ${source.inference_ms||result.inference_ms||0} ms`;
    const candidates=result.candidates||result.plates||[];const warnings=result.warnings||[];document.getElementById('plates').innerHTML=(candidates.length?candidates.map((plate,index)=>`<div class="plate ${plate.accepted===false?'rejected':''}"><strong>#${index+1} ${escapeHtml(plate.text||'Unrecognized candidate')}</strong><span>Box: [${plate.box.join(', ')}]</span><br><span>Style: ${escapeHtml(plate.style||'unknown')} · recognizer: ${escapeHtml(plate.recognizer||'lprnet')} · score: ${plate.recognition_score}</span>${plate.reject_reason?`<br><span>${escapeHtml(plate.reject_reason)}</span>`:''}</div>`).join(''):'No plate candidate located in the latest frame.')+(warnings.length?`<div class="status">${warnings.map(escapeHtml).join('<br>')}</div>`:'');
    document.getElementById('progress').value=job.progress||0;document.getElementById('progressText').textContent=job.error?`Error: ${job.error}`:(job.is_processing?`Processing ${job.current_file}: ${job.progress}%`:(job.output&&job.progress===100?'Analysis complete':'Idle'));
    const download=document.getElementById('download');if(!job.is_processing&&job.output&&job.progress===100){download.href='/api/video/download/'+encodeURIComponent(job.output);download.style.display='inline-block';document.getElementById('videoButton').disabled=false;}
  }catch(error){document.getElementById('sourceStatus').textContent='Service status error: '+error.message;}
}
setInterval(refresh,750);refresh();
</script>
</body></html>"""


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML_PAGE


def main():
    parser = argparse.ArgumentParser(description="LPRNet Web and API service")
    parser.add_argument("--platform", choices=["rk3576", "rk3588"], required=True)
    parser.add_argument("--model_dir", default="model")
    parser.add_argument("--camera_id", type=int, default=-1, help="Camera ID; -1 enables Web-upload-only mode")
    parser.add_argument("--video", "--video_path", dest="video", help="Local video path; overrides camera_id and loops continuously")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    global runtime
    runtime = create_runtime(args.platform, Path(args.model_dir))
    try:
        runtime.warmup_preview(set_preview)
    except Exception as exc:
        print(f"Sample warmup skipped: {exc}", flush=True)
    if args.video or args.camera_id >= 0:
        threading.Thread(target=process_realtime_source, args=(args.video, args.camera_id), daemon=True).start()
    else:
        print("Running in Web-upload-only mode (camera_id=-1)", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_config=None)


if __name__ == "__main__":
    main()
