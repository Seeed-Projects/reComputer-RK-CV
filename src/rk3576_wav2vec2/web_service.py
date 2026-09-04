import argparse
import asyncio
import json
import os
import re
import threading
import time
import uuid
from pathlib import Path
from typing import List, Optional

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from task_runtime import create_runtime


BASE_DIR = Path(__file__).resolve().parent
WEB_DIR = BASE_DIR / "web"
WORKSPACE = Path(os.environ.get("RK_CV_WORKSPACE", BASE_DIR / "workspace"))
OUTPUTS = WORKSPACE / "outputs"
OUTPUTS.mkdir(parents=True, exist_ok=True)
MAX_UPLOAD_BYTES = 64 * 1024 * 1024

runtime = None
runtime_lock = threading.Lock()
app = FastAPI(title="reComputer RK AI Model API", version="2.0.0")


def parse_prompts(value):
    if not value:
        return None
    value = value.strip()
    if value.startswith("["):
        try:
            prompts = json.loads(value)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail="prompts must be a JSON array or newline-separated text") from exc
        if not isinstance(prompts, list) or not all(isinstance(item, str) for item in prompts):
            raise HTTPException(status_code=400, detail="prompts must be an array of strings")
        return [item.strip() for item in prompts if item.strip()]
    return [item.strip() for item in re.split(r"[|\n]", value) if item.strip()]


async def read_upload(file):
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="The uploaded file is empty")
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="The uploaded file exceeds the 64 MB limit")
    return contents


def save_audio(result):
    audio = result.pop("_audio_bytes", None)
    if audio is None:
        return result
    filename = f"{uuid.uuid4().hex}.wav"
    (OUTPUTS / filename).write_bytes(audio)
    result["audio_url"] = f"/api/audio/{filename}"
    result["audio_filename"] = filename
    files = sorted(OUTPUTS.glob("*.wav"), key=lambda path: path.stat().st_mtime, reverse=True)
    for path in files[20:]:
        path.unlink(missing_ok=True)
    return result


def run_predict(payload, params):
    with runtime_lock:
        return save_audio(runtime.predict(payload, params))


@app.get("/")
async def index():
    page = WEB_DIR / "index.html"
    if not page.exists():
        raise HTTPException(status_code=500, detail="Web UI is missing")
    return FileResponse(page, media_type="text/html")


@app.get("/style.css", include_in_schema=False)
async def stylesheet():
    return FileResponse(WEB_DIR / "style.css", media_type="text/css")


@app.get("/audio.js", include_in_schema=False)
async def audio_helpers():
    path = WEB_DIR / "audio.js"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Audio helpers are not used by this application")
    return FileResponse(path, media_type="application/javascript")


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "model_ready": runtime is not None,
        "model": runtime.name if runtime else None,
        "platform": runtime.platform if runtime else None,
        "input_kind": runtime.input_kind if runtime else None,
        "models": runtime.model_names if runtime else [],
        "capabilities": runtime.describe() if runtime else {},
    }


@app.get("/api/config")
async def get_config():
    if runtime is None:
        raise HTTPException(status_code=503, detail="Model is not ready")
    return runtime.get_config()


@app.post("/api/config")
async def update_config(value: dict):
    if runtime is None:
        raise HTTPException(status_code=503, detail="Model is not ready")
    try:
        return runtime.update_config(value)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/models/{model_name}/predict")
async def predict(
    model_name: str,
    file: Optional[UploadFile] = File(None),
    text: Optional[str] = Form(None),
    prompts: Optional[str] = Form(None),
    topk: Optional[int] = Form(None),
    threshold: Optional[float] = Form(None),
    hop_seconds: Optional[float] = Form(None),
    beam_width: Optional[int] = Form(None),
):
    if runtime is None or model_name != runtime.name:
        raise HTTPException(status_code=404, detail="Unknown model")
    params = runtime.get_config()
    if prompts is not None:
        params["prompts"] = parse_prompts(prompts)
    if topk is not None:
        params["topk"] = topk
    if threshold is not None:
        params["threshold"] = threshold
    if hop_seconds is not None:
        params["hop_seconds"] = hop_seconds
    if beam_width is not None:
        params["beam_width"] = beam_width

    if runtime.input_kind == "text":
        if not text or not text.strip():
            raise HTTPException(status_code=400, detail="The text field is required")
        payload = text.strip()
    else:
        if file is None:
            raise HTTPException(status_code=400, detail="The file field is required")
        payload = await read_upload(file)

    started = time.perf_counter()
    try:
        result = await asyncio.to_thread(run_predict, payload, params)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {
        "success": True,
        "model": runtime.name,
        "platform": runtime.platform,
        "inference_time_ms": round((time.perf_counter() - started) * 1000, 2),
        "result": result,
    }


@app.post("/api/models/clip/retrieve")
async def clip_retrieve(
    files: List[UploadFile] = File(...),
    text: str = Form(...),
    topk: Optional[int] = Form(None),
):
    if runtime is None or runtime.name != "clip" or not hasattr(runtime, "retrieve"):
        raise HTTPException(status_code=404, detail="CLIP retrieval is not available")
    if not text.strip():
        raise HTTPException(status_code=400, detail="The text field is required")
    if not files or len(files) > 32:
        raise HTTPException(status_code=400, detail="Upload between 1 and 32 images")
    images = []
    for file in files:
        images.append((Path(file.filename or "image").name, await read_upload(file)))
    started = time.perf_counter()
    try:
        with runtime_lock:
            result = runtime.retrieve(images, text.strip(), topk)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "success": True,
        "model": runtime.name,
        "platform": runtime.platform,
        "inference_time_ms": round((time.perf_counter() - started) * 1000, 2),
        "result": result,
    }


@app.get("/api/audio/{filename}")
async def audio_result(filename: str):
    if not re.fullmatch(r"[0-9a-f]{32}\.wav", filename):
        raise HTTPException(status_code=400, detail="Invalid audio filename")
    path = OUTPUTS / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Audio result not found")
    return FileResponse(path, media_type="audio/wav", filename=filename)


@app.websocket("/api/models/zipformer/stream")
async def zipformer_stream(websocket: WebSocket):
    await websocket.accept()
    if runtime is None or runtime.name != "zipformer" or not hasattr(runtime, "create_stream_session"):
        await websocket.send_json({"type": "error", "message": "Zipformer streaming is not available"})
        await websocket.close(code=1008)
        return
    session = runtime.create_stream_session()
    await websocket.send_json({"type": "ready", "sample_rate": 16000, "format": "pcm_s16le"})
    try:
        while True:
            message = await websocket.receive()
            if message.get("bytes") is not None:
                with runtime_lock:
                    result = session.feed_pcm16(message["bytes"], final=False)
                if result.get("processed"):
                    await websocket.send_json({"type": "partial", "result": result})
            elif message.get("text") is not None:
                command = json.loads(message["text"])
                if command.get("action") == "stop":
                    with runtime_lock:
                        result = session.feed_pcm16(b"", final=True)
                    await websocket.send_json({"type": "final", "result": result})
                    await websocket.close()
                    return
                if command.get("action") == "reset":
                    session = runtime.create_stream_session()
                    await websocket.send_json({"type": "ready", "sample_rate": 16000, "format": "pcm_s16le"})
    except WebSocketDisconnect:
        return
    except Exception as exc:
        await websocket.send_json({"type": "error", "message": str(exc)})
        await websocket.close(code=1011)


def main():
    parser = argparse.ArgumentParser(description="Run a reComputer RK AI web service")
    parser.add_argument("--platform", choices=["rk3576", "rk3588"], required=True)
    parser.add_argument("--model_dir", default="model")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    global runtime
    runtime = create_runtime(args.platform, Path(args.model_dir))
    uvicorn.run(app, host=args.host, port=args.port, log_config=None)


if __name__ == "__main__":
    main()
