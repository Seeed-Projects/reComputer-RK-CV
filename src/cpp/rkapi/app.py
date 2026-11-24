import asyncio
import json
import logging
import os
import struct
import tempfile
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from time import perf_counter
from typing import Literal

from anyio import CancelScope
import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, Response, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import MutableHeaders

from .runtime import load_runtime
from .backends.base import SourceUnavailable, UnsupportedOperation
from .cameras import CameraPreview, cameras

MAX_UPLOAD = int(os.getenv("RK_MAX_UPLOAD_MB", "128")) * 1024 * 1024
MAX_PIXELS = 8192 * 8192
log = logging.getLogger("rkapi")


class RequestTiming:
    """Measure ingress before FastAPI parses multipart data, without buffering it."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        state = scope["rk_timing"] = {"start": perf_counter(), "receive_wait_ms": 0.0}

        async def timed_receive():
            started = perf_counter()
            message = await receive()
            state["receive_wait_ms"] += (perf_counter() - started) * 1000
            return message

        async def timed_send(message):
            if message["type"] == "http.response.start":
                now = perf_counter()
                metrics = {"server_total": (now - state["start"]) * 1000,
                           "receive_wait": state["receive_wait_ms"]}
                if "handler_start" in state:
                    metrics["request_parse"] = (state["handler_start"] - state["start"]) * 1000
                MutableHeaders(scope=message).append(
                    "Server-Timing", ", ".join(f"{name};dur={value:.3f}" for name, value in metrics.items()))
            await send(message)

        await self.app(scope, timed_receive, timed_send)


def prediction_response(result):
    started = perf_counter()
    response = JSONResponse(result)
    response.headers["Server-Timing"] = f"response_encode;dur={(perf_counter() - started) * 1000:.3f}"
    response.headers["Cache-Control"] = "no-store"
    return response


def preview_packet(result, jpeg=b"", candidates=b""):
    metadata = json.dumps(result, separators=(",", ":")).encode()
    return struct.pack("<III", len(jpeg), len(metadata), len(candidates)) + jpeg + metadata + candidates


class Configuration(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    obj_thresh: float | None = Field(default=None, ge=0, le=1)
    nms_thresh: float | None = Field(default=None, ge=0, le=1)


@asynccontextmanager
async def lifespan(app):
    # One process owns the NPU contexts; concurrent requests use its native mutex.
    app.state.runtime = await run_in_threadpool(load_runtime)
    try:
        yield
    finally:
        runtime = app.state.runtime
        await run_in_threadpool(runtime.close)


app = FastAPI(title="Rockchip CV API", version="0.3.0", lifespan=lifespan)
app.add_middleware(RequestTiming)


@app.get("/health")
def health():
    rt = app.state.runtime
    response = {"status": "ok", "model": rt.model_info, "backend": rt.description,
                "live": rt.source_status()}
    if rt.demo_video:
        response["demo_video"] = True
    return response


@app.get("/api/models")
def models():
    """Describe the model loaded by this image; this is not a remote model registry."""
    return {"models": [app.state.runtime.model_info]}


@app.get("/api/metadata")
def metadata():
    """Describe how to interpret predictions from the loaded model."""
    rt = app.state.runtime
    return {"schema_version": 1, "model": rt.model_info, **rt.prediction_metadata}


@app.get("/api/config")
def get_config():
    rt = app.state.runtime
    with rt.config_lock:
        return dict(rt.config)


@app.post("/api/config")
def set_config(config: Configuration):
    rt = app.state.runtime
    values = config.model_dump(exclude_none=True)
    rt.configure(values)
    return {"status": "success"}


async def save_upload(upload, target):
    size = 0
    while chunk := await upload.read(1024 * 1024):
        size += len(chunk)
        if size > MAX_UPLOAD:
            raise HTTPException(413, "upload exceeds RK_MAX_UPLOAD_MB")
        target.write(chunk)
    if size == 0:
        raise HTTPException(400, "empty upload")
    target.flush()


def video_frame(path, timestamp):
    capture = cv2.VideoCapture(path)
    try:
        if not capture.isOpened():
            raise HTTPException(400, "cannot open uploaded video")
        if timestamp > 0:
            if not capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000):
                raise HTTPException(400, "video does not support seeking")
        ok, image = capture.read()
        if not ok:
            raise HTTPException(400, "timestamp is outside the video or frame cannot be decoded")
        return image
    finally:
        capture.release()


@app.post("/api/models/{model_id}/predict", openapi_extra={"parameters": [
    {"name": "model_id", "in": "path", "required": True, "schema": {"type": "string"},
     "description": "Loaded model ID or family from /api/models"}]})
@app.post("/api/predict")
async def predict(request: Request, file: UploadFile | None = File(None), video: UploadFile | None = File(None),
                  timestamp: float = Form(0, ge=0, allow_inf_nan=False), realtime: bool = Form(False),
                  conf: float | None = Form(None, ge=0, le=1, allow_inf_nan=False),
                  iou: float | None = Form(None, ge=0, le=1, allow_inf_nan=False)):
    request.scope["rk_timing"]["handler_start"] = perf_counter()
    rt = app.state.runtime
    model_id = request.path_params.get("model_id")
    if model_id and model_id not in rt.aliases:
        raise HTTPException(404, "model is not loaded; see /api/models")
    conf, iou = rt.thresholds(conf, iou)
    if file is not None and video is not None:
        raise HTTPException(400, "provide either file or video")
    if realtime and (file is not None or video is not None):
        raise HTTPException(400, "realtime cannot be combined with an upload")
    try:
        if file is None and video is None:
            result = await run_in_threadpool(rt.predict_realtime, conf, iou)
            result["model"] = rt.model_info
            return prediction_response(result)
        upload = file if file is not None else video
        # Generated names live outside the source tree and are removed on all paths.
        with tempfile.NamedTemporaryFile(suffix=".media") as temporary:
            started = perf_counter()
            await save_upload(upload, temporary)
            upload_store_ms = (perf_counter() - started) * 1000
            decode_started = perf_counter()
            if file is not None:
                def decode():
                    return cv2.imdecode(np.fromfile(temporary.name, dtype=np.uint8), cv2.IMREAD_COLOR)
                image = await run_in_threadpool(decode)
                source = "image"
            else:
                image = await run_in_threadpool(video_frame, temporary.name, timestamp)
                source = f"video frame at {timestamp:g}s"
            decode_ms = (perf_counter() - decode_started) * 1000
            if image is None or image.size == 0:
                raise HTTPException(400, "cannot decode image or video frame")
            height, width = image.shape[:2]
            if width > 8192 or height > 8192 or width * height > MAX_PIXELS:
                raise HTTPException(413, "image dimensions exceed 8192 x 8192")
            process_started = perf_counter()
            result = await run_in_threadpool(rt.image, image, conf, iou, source)
            result["model"] = rt.model_info
            result.setdefault("timing", {}).update({"upload_store_ms": upload_store_ms, "decode_ms": decode_ms,
                                     "process_ms": (perf_counter() - process_started) * 1000})
            return prediction_response(result)
    except SourceUnavailable as error:
        raise HTTPException(503, str(error)) from error
    except UnsupportedOperation as error:
        raise HTTPException(501, str(error)) from error
    except HTTPException:
        raise
    except (RuntimeError, cv2.error, ValueError) as error:
        log.exception("prediction failed")
        raise HTTPException(400, str(error)) from error
    finally:
        for upload in (file, video):
            if upload is not None:
                await upload.close()


class LiveResponse(StreamingResponse):
    """Hold one live-source lease until the HTTP stream disconnects."""

    def __init__(self, runtime, content, **kwargs):
        super().__init__(content, **kwargs)
        self.runtime = runtime

    async def __call__(self, scope, receive, send):
        acquired = False
        try:
            with CancelScope(shield=True):
                await run_in_threadpool(self.runtime.acquire_live)
                acquired = True
            await super().__call__(scope, receive, send)
        except SourceUnavailable as error:
            raise HTTPException(503, str(error)) from error
        except UnsupportedOperation as error:
            raise HTTPException(501, str(error)) from error
        finally:
            if acquired:
                with CancelScope(shield=True):
                    try:
                        await self.body_iterator.aclose()
                    finally:
                        await run_in_threadpool(self.runtime.release_live)


@app.get("/api/demo/cameras", include_in_schema=False)
def demo_cameras():
    return {"cameras": cameras()}


@app.post("/api/demo/frame", include_in_schema=False)
async def demo_frame(request: Request):
    payload = bytearray()
    async for chunk in request.stream():
        if len(payload) + len(chunk) > min(MAX_UPLOAD, 4 * 1024 * 1024):
            raise HTTPException(413, "preview frame is too large")
        payload.extend(chunk)
    if not payload:
        raise HTTPException(400, "empty preview frame")
    rt = app.state.runtime

    def process():
        started = perf_counter()
        image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(400, "cannot decode preview frame")
        decode_ms = (perf_counter() - started) * 1000
        conf, iou = rt.thresholds(None, None)
        result, candidates = rt.preview_image(image, conf, iou, "uploaded video")
        result["timing"]["decode_ms"] = decode_ms
        return preview_packet(result, candidates=candidates)

    try:
        packet = await run_in_threadpool(process)
    except (RuntimeError, cv2.error, ValueError) as error:
        raise HTTPException(400, str(error)) from error
    return Response(packet, media_type="application/octet-stream", headers={"Cache-Control": "no-store"})


@app.get("/api/demo/cameras/{device}/frames", include_in_schema=False)
async def demo_camera_frames(device: str):
    devices = await run_in_threadpool(cameras)
    camera = next((camera for camera in devices if camera["id"] == device), None)
    if camera is None:
        raise HTTPException(404, "This board camera is unavailable")
    preview = CameraPreview(camera, app.state.runtime)

    async def frames():
        capture = None
        pending = deque()
        try:
            while True:
                if capture is None and len(pending) < preview.runtime.preview_workers:
                    capture = asyncio.create_task(run_in_threadpool(preview.read))
                if pending and pending[0].done():
                    result, jpeg, candidates = pending.popleft().result()
                    yield preview_packet(result, jpeg, candidates)
                elif capture is not None and capture.done():
                    image = capture.result()
                    capture = None
                    pending.append(asyncio.create_task(run_in_threadpool(preview.frame, image)))
                else:
                    waiting = [capture] if capture is not None else []
                    if pending:
                        waiting.append(pending[0])
                    await asyncio.wait(waiting, return_when=asyncio.FIRST_COMPLETED)
        except (RuntimeError, cv2.error, ValueError) as error:
            yield preview_packet({"error": str(error)})
        finally:
            with CancelScope(shield=True):
                if capture is not None:
                    pending.append(capture)
                await asyncio.gather(*pending, return_exceptions=True)

    return LiveResponse(preview, frames(), media_type="application/octet-stream",
                        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})


@app.get("/api/video_feed")
async def video_feed():
    rt = app.state.runtime
    async def frames():
        previous = None
        while True:
            try:
                state, jpeg = await run_in_threadpool(rt.preview)
            except (SourceUnavailable, UnsupportedOperation):
                break
            raw = state["result"]
            if state["error"]:
                break
            if raw is not None and raw["frame"] != previous:
                previous = raw["frame"]
                annotated = await run_in_threadpool(rt.annotate, jpeg, raw)
                yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " +
                       str(len(annotated)).encode() + b"\r\n\r\n" + annotated + b"\r\n")
            elif state["ended"]:
                break
            await asyncio.sleep(.05)

    return LiveResponse(rt, frames(), media_type="multipart/x-mixed-replace; boundary=frame",
                             headers={"Cache-Control": "no-store"})


def demo_runtime():
    rt = app.state.runtime
    if rt.demo_video is None:
        raise HTTPException(404, "set RK_LOOP_VIDEO=1 for the local MP4 demo")
    return rt


@app.get("/api/demo/video", include_in_schema=False)
def demo_video():
    return FileResponse(demo_runtime().demo_video, media_type="video/mp4",
                        headers={"Cache-Control": "no-cache"})


@app.get("/api/demo/photos/{name}.jpg", include_in_schema=False)
def demo_photo(name: Literal["dog", "banana", "tennis-ball", "boats"]):
    path = Path("/opt/demo/photos") / f"{name}.jpg"
    if not path.is_file():
        raise HTTPException(404, "this model image does not include that example photo")
    return FileResponse(path, media_type="image/jpeg")


@app.get("/api/demo/results", include_in_schema=False)
async def demo_results():
    rt = demo_runtime()

    async def events():
        previous = 0
        while True:
            try:
                result = await run_in_threadpool(rt.demo_result, previous)
            except SourceUnavailable:
                break
            if result is not None:
                previous = result["frame"]
                yield "data: " + json.dumps(result, separators=(",", ":")) + "\n\n"
            await asyncio.sleep(.015)

    return LiveResponse(rt, events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})


@app.get("/api/demo/candidates", include_in_schema=False)
async def demo_candidates():
    rt = demo_runtime()
    if not rt.description.get("live", {}).get("candidate_stream"):
        raise HTTPException(404, "the loaded model does not publish detection candidates")

    async def frames():
        previous = 0
        while True:
            try:
                packet = await run_in_threadpool(rt.demo_candidates, previous)
            except SourceUnavailable:
                break
            if packet is None:
                await asyncio.sleep(.008)
                continue
            previous, data = packet
            yield data

    return LiveResponse(rt, frames(), media_type="application/octet-stream",
                             headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(Path(__file__).with_name("demo.html"), media_type="text/html")


@app.get("/demo", include_in_schema=False)
def demo(request: Request):
    return RedirectResponse(request.url_for("index"), status_code=308)
