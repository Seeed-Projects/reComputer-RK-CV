import argparse
import json
import threading
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, Response

from task_runtime import create_runtime


runtime = None
runtime_lock = threading.Lock()
state_lock = threading.Lock()
runtime_config = {
    "det_threshold": 0.3,
    "box_threshold": 0.6,
    "unclip_ratio": 1.5,
    "drop_score": 0.5,
    "max_results": 100,
}
latest_result = {
    "text": "",
    "lines": [],
    "count": 0,
    "detected_count": 0,
    "processed_count": 0,
}
latest_image = None

app = FastAPI(
    title="reComputer RK-CV PPOCR API",
    description="End-to-end text detection and recognition for still images.",
    version="1.0.0",
)


def config_snapshot():
    with state_lock:
        return dict(runtime_config)


def encode_image(image, suffix=".jpg"):
    extension = suffix.lower() if suffix.lower() in {".jpg", ".jpeg", ".png"} else ".jpg"
    ok, encoded = cv2.imencode(extension, image)
    if not ok:
        raise ValueError("Failed to encode result image")
    return encoded.tobytes()


def decode_image(contents):
    if not contents:
        raise HTTPException(status_code=400, detail="Empty upload")
    image = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="The uploaded file is not a valid image")
    return image


def load_local_image(path):
    image_path = Path(path).expanduser().resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot decode image: {image_path}")
    return image, image_path


def run_prediction(image, params):
    global latest_image
    started = time.perf_counter()
    with runtime_lock:
        result, preview = runtime.predict(image, params)
    inference_time = round(time.perf_counter() - started, 4)
    with state_lock:
        latest_result.clear()
        latest_result.update(result)
        latest_image = encode_image(preview)
    return result, preview, inference_time


def request_params(det_threshold, box_threshold, unclip_ratio, drop_score, max_results, include_crops):
    params = config_snapshot()
    overrides = {
        "det_threshold": det_threshold,
        "box_threshold": box_threshold,
        "unclip_ratio": unclip_ratio,
        "drop_score": drop_score,
        "max_results": max_results,
    }
    for key, value in overrides.items():
        if value is not None:
            params[key] = value
    validate_config(params)
    params["include_crops"] = include_crops
    return params


def validate_config(config):
    for key in ("det_threshold", "box_threshold", "drop_score"):
        value = float(config[key])
        if not 0 <= value <= 1:
            raise ValueError(f"{key} must be between 0 and 1")
        config[key] = value
    config["unclip_ratio"] = float(config["unclip_ratio"])
    if not 0.1 <= config["unclip_ratio"] <= 5:
        raise ValueError("unclip_ratio must be between 0.1 and 5")
    config["max_results"] = int(config["max_results"])
    if not 1 <= config["max_results"] <= 1000:
        raise ValueError("max_results must be between 1 and 1000")


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "model": runtime.name if runtime else None,
        "platform": runtime.platform if runtime else None,
        "input_kind": "image",
        "model_ready": runtime is not None,
        "models": runtime.model_names if runtime else [],
        "supports": {"image": True, "video": False, "camera": False},
    }


@app.get("/api/config")
async def get_config():
    return config_snapshot()


@app.post("/api/config")
async def update_config(value: dict):
    candidate = config_snapshot()
    for key in candidate:
        if key in value:
            candidate[key] = value[key]
    try:
        validate_config(candidate)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    with state_lock:
        runtime_config.update(candidate)
        return dict(runtime_config)


@app.post("/api/models/ppocr/predict")
async def predict(
    file: UploadFile = File(...),
    det_threshold: Optional[float] = Form(None),
    box_threshold: Optional[float] = Form(None),
    unclip_ratio: Optional[float] = Form(None),
    drop_score: Optional[float] = Form(None),
    max_results: Optional[int] = Form(None),
    include_crops: bool = Form(False),
):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        raise HTTPException(status_code=400, detail="Supported image types: JPG, PNG, BMP, WEBP")
    image = decode_image(await file.read())
    try:
        params = request_params(
            det_threshold,
            box_threshold,
            unclip_ratio,
            drop_score,
            max_results,
            include_crops,
        )
        result, _, inference_time = run_prediction(image, params)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {
        "success": True,
        "model": "ppocr",
        "inference_time": inference_time,
        "result": result,
    }


@app.get("/api/results/latest")
async def results_latest():
    with state_lock:
        return dict(latest_result)


@app.get("/api/results/latest/image")
async def latest_result_image():
    with state_lock:
        payload = latest_image
    if payload is None:
        raise HTTPException(status_code=404, detail="No result image is available")
    return Response(payload, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML_PAGE


HTML_PAGE = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>PPOCR End-to-End OCR</title>
  <style>
    :root{color-scheme:dark;--bg:#091017;--panel:#121b24;--line:#2b3947;--text:#edf4fa;--muted:#98a7b7;--green:#00e676}
    *{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:16px/1.45 system-ui,sans-serif}
    header,main{max-width:1500px;margin:auto;padding:24px}header{border-bottom:1px solid var(--line)}h1,h2,p{margin-top:0}
    .grid{display:grid;grid-template-columns:minmax(0,2fr) minmax(330px,1fr);gap:20px}.card{background:var(--panel);border:1px solid var(--line);border-radius:16px;padding:20px}
    .controls{display:grid;grid-template-columns:repeat(5,minmax(120px,1fr));gap:12px;margin:18px 0}label{color:var(--muted);font-size:13px}input{width:100%;margin-top:5px;padding:10px;background:#091017;color:var(--text);border:1px solid var(--line);border-radius:8px}
    button{padding:12px 20px;border:0;border-radius:9px;background:var(--green);color:#03150d;font-weight:750;cursor:pointer}button:disabled{opacity:.5;cursor:wait}
    #resultImage{display:none;width:100%;max-height:72vh;object-fit:contain;background:#05090d;border-radius:10px}.muted{color:var(--muted)}
    .summary{white-space:pre-wrap;background:#091017;border-radius:10px;padding:14px;min-height:64px}.line{display:grid;grid-template-columns:110px 1fr;gap:12px;padding:12px 0;border-bottom:1px solid var(--line)}
    .line img{width:110px;height:60px;object-fit:contain;background:white;border-radius:6px}.text{font-size:18px;word-break:break-all}.meta{font-size:13px;color:var(--muted)}a{color:var(--green)}
    @media(max-width:900px){.grid{grid-template-columns:1fr}.controls{grid-template-columns:1fr 1fr}}
  </style>
</head>
<body>
  <header><h1>PPOCR End-to-End OCR</h1><p class="muted">Upload a still image. PPOCR-Det locates text, perspective correction crops every region, and PPOCR-Rec recognizes the content.</p></header>
  <main>
    <section class="card">
      <input id="file" type="file" accept="image/jpeg,image/png,image/bmp,image/webp">
      <div class="controls">
        <label>Detection threshold<input id="detThreshold" type="number" min="0" max="1" step="0.05" value="0.3"></label>
        <label>Box threshold<input id="boxThreshold" type="number" min="0" max="1" step="0.05" value="0.6"></label>
        <label>Unclip ratio<input id="unclipRatio" type="number" min="0.1" max="5" step="0.1" value="1.5"></label>
        <label>Recognition score<input id="dropScore" type="number" min="0" max="1" step="0.05" value="0.5"></label>
        <label>Maximum regions<input id="maxResults" type="number" min="1" max="1000" value="100"></label>
      </div>
      <button id="run" onclick="runOCR()">Analyze image</button>
      <span id="status" class="muted"> Select an image to begin.</span>
      <p class="muted" style="margin-top:14px">API documentation: <a href="/docs">/docs</a></p>
    </section>
    <div class="grid" style="margin-top:20px">
      <section class="card"><h2>Annotated image</h2><img id="resultImage" alt="Annotated OCR result"><p id="emptyPreview" class="muted">The result image will appear here.</p></section>
      <section class="card"><h2>Recognized text</h2><div id="summary" class="summary">No result yet.</div><div id="lines"></div></section>
    </div>
  </main>
  <script>
    const el=id=>document.getElementById(id);
    function escapeHtml(value){return String(value).replace(/[&<>'"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]));}
    async function runOCR(){
      const file=el('file').files[0];if(!file){el('status').textContent=' Choose an image first.';return;}
      const data=new FormData();data.append('file',file);data.append('det_threshold',el('detThreshold').value);data.append('box_threshold',el('boxThreshold').value);data.append('unclip_ratio',el('unclipRatio').value);data.append('drop_score',el('dropScore').value);data.append('max_results',el('maxResults').value);data.append('include_crops','true');
      el('run').disabled=true;el('status').textContent=' Processing...';
      try{
        const response=await fetch('/api/models/ppocr/predict',{method:'POST',body:data});const body=await response.json();if(!response.ok)throw new Error(body.detail||'Request failed');
        const result=body.result;el('summary').textContent=result.text||'No text passed the recognition score.';
        el('lines').innerHTML=result.lines.map(line=>`<div class="line">${line.crop_image?`<img src="${line.crop_image}" alt="Text crop ${line.index}">`:''}<div><div class="text">${escapeHtml(line.text||'(empty)')}</div><div class="meta">#${line.index} · confidence ${line.confidence.toFixed(4)} · ${line.accepted?'accepted':'below score'}<br>box ${escapeHtml(JSON.stringify(line.box))}</div></div></div>`).join('');
        el('resultImage').src='/api/results/latest/image?t='+Date.now();el('resultImage').style.display='block';el('emptyPreview').style.display='none';
        const timing=result.timing_ms;el('status').textContent=` ${result.count}/${result.detected_count} accepted · Det ${timing.detection} ms · Rec ${timing.recognition} ms · Total ${timing.total} ms`;
      }catch(error){el('status').textContent=' '+error.message;}finally{el('run').disabled=false;}
    }
  </script>
</body></html>"""


def save_result(path, image):
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    suffix = output.suffix.lower() if output.suffix else ".jpg"
    if not output.suffix:
        output = output.with_suffix(suffix)
    encoded = encode_image(image, suffix)
    output.write_bytes(encoded)
    return output


def main():
    parser = argparse.ArgumentParser(description="PPOCR Det+Rec Web and image service")
    parser.add_argument("--platform", choices=["rk3576", "rk3588"], required=True)
    parser.add_argument("--model_dir", default="model")
    parser.add_argument("--image", help="Analyze a local image before starting the Web service")
    parser.add_argument("--output", help="Save the annotated --image result as JPG or PNG")
    parser.add_argument("--no_server", action="store_true", help="Exit after processing --image")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if args.output and not args.image:
        parser.error("--output requires --image")
    if args.no_server and not args.image:
        parser.error("--no_server requires --image")

    global runtime
    runtime = create_runtime(args.platform, Path(args.model_dir))
    sample = args.image or str(Path(args.model_dir) / "test.jpg")
    try:
        image, image_path = load_local_image(sample)
        params = config_snapshot()
        params["include_crops"] = False
        result, preview, inference_time = run_prediction(image, params)
        print(json.dumps({"image": str(image_path), "inference_time": inference_time, "result": result}, ensure_ascii=False, indent=2), flush=True)
        if args.output:
            print(f"Annotated image: {save_result(args.output, preview)}", flush=True)
    except Exception as exc:
        if args.image:
            runtime.release()
            raise SystemExit(f"Image analysis failed: {exc}") from exc
        print(f"Sample warmup skipped: {exc}", flush=True)

    if args.no_server:
        runtime.release()
        return
    try:
        uvicorn.run(app, host=args.host, port=args.port, log_config=None)
    finally:
        runtime.release()


if __name__ == "__main__":
    main()
