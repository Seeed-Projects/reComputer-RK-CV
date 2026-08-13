import argparse
import json
import os
import re
import shutil
import tempfile
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
runtime_config = {
    "threshold": 0.25,
    "topk": 5,
    "point_coords": None,
    "point_labels": None,
}
last_preview = None
preview_lock = threading.Lock()
analysis_state = {"is_processing": False, "progress": 0, "current_file": "", "error": ""}
source_state = {
    "mode": "web",
    "source": "",
    "is_running": False,
    "frames": 0,
    "inference_ms": 0.0,
    "error": "",
}
source_lock = threading.Lock()

app = FastAPI(title="reComputer RK-CV Standard Model API", version="1.0.0")


def validate_prompt(point_coords, point_labels):
    if point_coords is None and point_labels is None:
        return None, None
    try:
        coords = np.asarray(point_coords, dtype=np.float32)
        labels = np.asarray(point_labels, dtype=np.int32)
    except (TypeError, ValueError) as exc:
        raise ValueError("point_coords and point_labels must be numeric arrays") from exc
    if coords.shape != (2, 2) or labels.shape != (2,):
        raise ValueError("MobileSAM requires exactly two coordinates and two labels")
    if not np.isfinite(coords).all() or (coords < 0).any():
        raise ValueError("point coordinates must be finite, non-negative values")
    label_values = labels.tolist()
    if label_values != [2, 3] and any(label not in (-1, 0, 1) for label in label_values):
        raise ValueError("use labels [2,3] for a box, or -1/0/1 for point prompts")
    if label_values == [2, 3] and (
        coords[0, 0] >= coords[1, 0] or coords[0, 1] >= coords[1, 1]
    ):
        raise ValueError("box coordinates must be ordered from top-left to bottom-right")
    if label_values != [2, 3] and all(label == -1 for label in label_values):
        raise ValueError("at least one foreground or background point is required")
    return coords.astype(float).tolist(), label_values


def safe_filename(filename, allowed=None):
    if not filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    name = Path(filename).name
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,254}", name):
        raise HTTPException(status_code=400, detail="Invalid filename")
    if allowed and Path(name).suffix.lower() not in allowed:
        raise HTTPException(status_code=400, detail=f"Allowed extensions: {sorted(allowed)}")
    return name


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
    cv2.putText(image, "Waiting for camera, video, or upload", (55, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 230, 118), 2)
    return cv2.imencode(".jpg", image)[1].tobytes()


def decode_uploaded(contents, kind):
    if kind == "image":
        image = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image")
        return image
    return contents


def run_prediction(payload, params):
    with runtime_lock:
        result, preview = runtime.predict(payload, params)
    set_preview(preview)
    return result


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


@app.get("/api/config")
async def get_config():
    with config_lock:
        return dict(runtime_config)


@app.post("/api/config")
async def update_config(value: dict):
    with config_lock:
        if "threshold" in value:
            threshold = float(value["threshold"])
            if not 0 <= threshold <= 1:
                raise HTTPException(status_code=400, detail="threshold must be between 0 and 1")
            runtime_config["threshold"] = threshold
        if "topk" in value:
            topk = int(value["topk"])
            if not 1 <= topk <= 100:
                raise HTTPException(status_code=400, detail="topk must be between 1 and 100")
            runtime_config["topk"] = topk
        if "point_coords" in value or "point_labels" in value:
            try:
                coords, labels = validate_prompt(
                    value.get("point_coords"), value.get("point_labels")
                )
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            runtime_config["point_coords"] = coords
            runtime_config["point_labels"] = labels
        return dict(runtime_config)


@app.post("/api/models/{model_name}/predict")
async def predict(
    model_name: str,
    file: Optional[UploadFile] = File(None),
    text: Optional[str] = Form(None),
    point_coords: Optional[str] = Form(None),
    point_labels: Optional[str] = Form(None),
    threshold: Optional[float] = Form(None),
    topk: Optional[int] = Form(None),
):
    if runtime is None or model_name != runtime.name:
        raise HTTPException(status_code=404, detail="Unknown model")
    with config_lock:
        params = dict(runtime_config)
    if threshold is not None:
        params["threshold"] = float(threshold)
    if topk is not None:
        params["topk"] = int(topk)
    if text is not None:
        params["text"] = text
    if point_coords:
        try:
            params["point_coords"] = json.loads(point_coords)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail="point_coords must be JSON") from exc
    if point_labels:
        try:
            params["point_labels"] = json.loads(point_labels)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail="point_labels must be JSON") from exc
    if "point_coords" in params or "point_labels" in params:
        try:
            params["point_coords"], params["point_labels"] = validate_prompt(
                params.get("point_coords"), params.get("point_labels")
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    if runtime.input_kind == "text":
        if not text:
            raise HTTPException(status_code=400, detail="text is required")
        payload = text
    else:
        if file is None:
            raise HTTPException(status_code=400, detail="file is required")
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Empty upload")
        payload = decode_uploaded(contents, runtime.input_kind)

    started = time.perf_counter()
    try:
        result = run_prediction(payload, params)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"success": True, "model": runtime.name, "inference_time": round(time.perf_counter() - started, 4), "result": result}


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
        analysis_state.update(is_processing=False, error="Cannot open video")
        return
    total = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        cap.release()
        analysis_state.update(is_processing=False, error="Cannot create output video")
        return
    frame_index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            with config_lock:
                params = dict(runtime_config)
            with runtime_lock:
                _, preview = runtime.predict(frame, params)
            writer.write(preview if preview is not None else frame)
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


def process_realtime_source(video, camera_id):
    """Continuously process a camera or local video for Web preview."""
    is_video = video is not None
    source = str(video) if is_video else f"/dev/video{camera_id}"
    mode = "video" if is_video else "camera"
    with source_lock:
        source_state.update(
            mode=mode,
            source=source,
            is_running=False,
            frames=0,
            inference_ms=0.0,
            error="",
        )

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
            with config_lock:
                params = dict(runtime_config)
            inference_started = time.perf_counter()
            try:
                run_prediction(frame, params)
            except Exception as exc:
                with source_lock:
                    source_state["error"] = str(exc)
                print(f"Realtime inference failed: {exc}", flush=True)
                break
            inference_ms = (time.perf_counter() - inference_started) * 1000
            with source_lock:
                source_state["frames"] += 1
                source_state["inference_ms"] = round(inference_ms, 2)
            if frame_interval:
                remaining = frame_interval - (time.perf_counter() - loop_started)
                if remaining > 0:
                    time.sleep(remaining)
    finally:
        cap.release()
        with source_lock:
            source_state["is_running"] = False


@app.post("/api/video/analyze")
async def analyze_video(filename: str = Form(...)):
    if runtime.input_kind != "image":
        raise HTTPException(status_code=400, detail="Video analysis is only available for image models")
    filename = safe_filename(filename, {".mp4"})
    source = UPLOADS / filename
    if not source.exists():
        raise HTTPException(status_code=404, detail="Video not found")
    if analysis_state["is_processing"]:
        raise HTTPException(status_code=409, detail="Another video is being processed")
    output = OUTPUTS / f"{source.stem}_result.mp4"
    analysis_state.update(
        is_processing=True,
        progress=0,
        current_file=filename,
        error="",
    )
    threading.Thread(target=analyze_video_file, args=(source, output), daemon=True).start()
    return {"status": "started", "output": output.name}


@app.get("/api/video/status")
async def video_status():
    return dict(analysis_state)


@app.get("/api/video/list")
async def video_list():
    return {"uploads": sorted(p.name for p in UPLOADS.glob("*.mp4")), "outputs": sorted(p.name for p in OUTPUTS.glob("*.mp4"))}


@app.get("/api/video/download/{filename}")
async def video_download(filename: str):
    filename = safe_filename(filename, {".mp4"})
    path = OUTPUTS / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path, media_type="video/mp4", filename=filename)


@app.get("/", response_class=HTMLResponse)
async def index():
    task = runtime.name if runtime else "model"
    html = """<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>MobileSAM interactive segmentation</title>
<style>
body{font-family:system-ui,sans-serif;background:#111;color:#eee;max-width:1080px;margin:24px auto;padding:0 16px}
a{color:#00e676}.toolbar{display:flex;gap:8px;flex-wrap:wrap;margin:12px 0}.toolbar button,label{padding:9px 14px;border:1px solid #444;border-radius:6px;background:#222;color:#eee;cursor:pointer}.toolbar button.active{background:#00a853;border-color:#00e676}.toolbar button.primary{background:#00c853;color:#071b0d;border:0}.viewer{position:relative;display:inline-block;max-width:100%;line-height:0;background:#222}.viewer img{display:block;max-width:100%;height:auto}.viewer canvas{position:absolute;inset:0;width:100%;height:100%;cursor:crosshair;touch-action:none}.hint{color:#bbb;line-height:1.5}.status{display:grid;grid-template-columns:1fr 1fr;gap:12px}pre{background:#222;padding:12px;white-space:pre-wrap;overflow:auto;min-height:56px}@media(max-width:700px){.status{grid-template-columns:1fr}}
</style></head><body>
<h1>MobileSAM interactive segmentation</h1>
<p class="hint">Drag a green box around a target, or add foreground/background points. The prompt is applied immediately to camera and video frames. With no custom prompt, the full image is used.</p>
<div class="toolbar">
  <button id="boxMode" class="active" onclick="setMode('box')">Box / 框选</button>
  <button id="fgMode" onclick="setMode('foreground')">Foreground point / 前景点</button>
  <button id="bgMode" onclick="setMode('background')">Background point / 背景点</button>
  <button onclick="useFullImage()">Full image / 全画面</button>
</div>
<div class="viewer"><img id="preview" src="/api/video_feed"><canvas id="promptCanvas"></canvas></div>
<div class="toolbar">
  <label>Upload image / 上传图片 <input id="file" type="file" accept="image/*"></label>
  <button class="primary" onclick="runUpload()">Run uploaded image / 分析上传图片</button>
  <a href="/docs">OpenAPI</a>
</div>
<p class="hint">Green point = foreground, red point = background. The converted decoder accepts two prompt slots; a single point is automatically padded.</p>
<div class="status"><pre id="promptStatus">Loading prompt...</pre><pre id="source">Loading source status...</pre></div><pre id="result">Ready</pre>
<script>
const task='__TASK__', img=document.getElementById('preview'), canvas=document.getElementById('promptCanvas'), ctx=canvas.getContext('2d');
let mode='box', dragging=false, start=null, draft=null, points=[], activePrompt=null;
function setMode(next){mode=next;if(next==='box')points=[];document.querySelectorAll('.toolbar button[id]').forEach(b=>b.classList.remove('active'));document.getElementById(next==='box'?'boxMode':next==='foreground'?'fgMode':'bgMode').classList.add('active');draw();}
function imagePoint(event){const r=canvas.getBoundingClientRect(),w=img.naturalWidth||canvas.width,h=img.naturalHeight||canvas.height;return [Math.max(0,Math.min(w-1,(event.clientX-r.left)*w/r.width)),Math.max(0,Math.min(h-1,(event.clientY-r.top)*h/r.height))];}
function canvasPoint(p){const w=img.naturalWidth||1,h=img.naturalHeight||1;return [p[0]*canvas.width/w,p[1]*canvas.height/h];}
function resizeCanvas(){const r=img.getBoundingClientRect();const w=Math.max(1,Math.round(r.width)),h=Math.max(1,Math.round(r.height));if(canvas.width!==w||canvas.height!==h){canvas.width=w;canvas.height=h}draw();}
function draw(){ctx.clearRect(0,0,canvas.width,canvas.height);ctx.lineWidth=3;ctx.font='14px sans-serif';if(draft){let a=canvasPoint(draft[0]),b=canvasPoint(draft[1]);ctx.strokeStyle='#00e676';ctx.strokeRect(a[0],a[1],b[0]-a[0],b[1]-a[1]);}if(activePrompt&&activePrompt.point_coords){let c=activePrompt.point_coords,l=activePrompt.point_labels;if(l[0]===2&&l[1]===3){let a=canvasPoint(c[0]),b=canvasPoint(c[1]);ctx.strokeStyle='#00e676';ctx.strokeRect(a[0],a[1],b[0]-a[0],b[1]-a[1]);}else{c.forEach((p,i)=>{if(l[i]===-1)return;let q=canvasPoint(p);ctx.beginPath();ctx.arc(q[0],q[1],7,0,Math.PI*2);ctx.fillStyle=l[i]===1?'#00e676':'#ff3d00';ctx.fill();ctx.strokeStyle='#fff';ctx.stroke();});}}}
async function savePrompt(coords,labels){let r=await fetch('/api/config',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({point_coords:coords,point_labels:labels})});let body=await r.json();if(!r.ok)throw new Error(body.detail||'Cannot update prompt');activePrompt={point_coords:body.point_coords,point_labels:body.point_labels};showPrompt();draw();}
async function useFullImage(){points=[];draft=null;await savePrompt(null,null);}
function showPrompt(){document.getElementById('promptStatus').textContent=activePrompt&&activePrompt.point_coords?'Prompt: '+JSON.stringify(activePrompt):'Prompt: full image / 全画面';}
canvas.addEventListener('pointerdown',e=>{if(mode==='box'){dragging=true;start=imagePoint(e);draft=[start,start];canvas.setPointerCapture(e.pointerId);}else{let p=imagePoint(e),label=mode==='foreground'?1:0;points.push({p,label});if(points.length>2)points.shift();let coords=points.map(x=>x.p),labels=points.map(x=>x.label);if(coords.length===1){coords.push(coords[0]);labels.push(-1);}savePrompt(coords,labels).catch(showError);}});
canvas.addEventListener('pointermove',e=>{if(dragging){draft=[start,imagePoint(e)];draw();}});
canvas.addEventListener('pointerup',e=>{if(!dragging)return;dragging=false;points=[];let end=imagePoint(e),x1=Math.min(start[0],end[0]),y1=Math.min(start[1],end[1]),x2=Math.max(start[0],end[0]),y2=Math.max(start[1],end[1]);draft=null;if(x2-x1<3||y2-y1<3){draw();return;}savePrompt([[x1,y1],[x2,y2]],[2,3]).catch(showError);});
function showError(e){document.getElementById('result').textContent='Error: '+e.message;}
async function runUpload(){let file=document.getElementById('file');if(!file.files.length){showError(new Error('Choose an image first'));return;}let f=new FormData();f.append('file',file.files[0]);let r=await fetch('/api/models/'+task+'/predict',{method:'POST',body:f}),body=await r.json();document.getElementById('result').textContent=JSON.stringify(body,null,2);if(r.ok)img.src='/api/video_feed?ts='+Date.now();}
async function refresh(){try{let [health,config]=await Promise.all([fetch('/api/health').then(r=>r.json()),fetch('/api/config').then(r=>r.json())]);document.getElementById('source').textContent=JSON.stringify(health.source,null,2);activePrompt={point_coords:config.point_coords,point_labels:config.point_labels};showPrompt();draw();}catch(e){showError(e)}}
document.getElementById('file').addEventListener('change',e=>{if(e.target.files.length)img.src=URL.createObjectURL(e.target.files[0]);});new ResizeObserver(resizeCanvas).observe(img);img.addEventListener('load',resizeCanvas);setInterval(refresh,1000);refresh();
</script></body></html>"""
    return html.replace("__TASK__", task)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--platform", choices=["rk3576", "rk3588"], required=True)
    parser.add_argument("--model_dir", default="model")
    parser.add_argument(
        "--camera_id",
        type=int,
        default=-1,
        help="Camera device ID; -1 enables Web-upload-only mode",
    )
    parser.add_argument(
        "--video",
        "--video_path",
        dest="video",
        help="Local video path; overrides camera_id and loops continuously",
    )
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
        threading.Thread(
            target=process_realtime_source,
            args=(args.video, args.camera_id),
            daemon=True,
        ).start()
    else:
        print("Running in Web-upload-only mode (camera_id=-1)", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_config=None)


if __name__ == "__main__":
    main()
