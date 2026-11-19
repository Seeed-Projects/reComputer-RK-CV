"""YOLO26 RKNN3 service. UI and endpoint conventions follow rk3588_yolov8."""
import argparse
from contextlib import asynccontextmanager
import os
from pathlib import Path
import shutil
import tempfile
import threading
import time
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field
import uvicorn

from py_utils.yolo26 import YOLO26, draw

ROOT = Path(__file__).resolve().parent
WORKSPACE = Path(os.environ.get('WORKSPACE', str(ROOT / 'workspace')))
UPLOADS, OUTPUTS = WORKSPACE / 'uploads', WORKSPACE / 'outputs'
settings_lock = threading.Lock()
thresholds = (0.25, 0.45)
engine = None
preview = None
options = None


def config():
    with settings_lock:
        return thresholds


def file_in(directory, filename):
    if not filename or filename != Path(filename).name or '\\' in filename:
        raise HTTPException(400, 'Invalid filename')
    path = directory / filename
    if path.is_symlink():
        raise HTTPException(400, 'Symbolic links are not supported')
    return path


class Preview:
    def __init__(self, source):
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f'Cannot open video source: {source}')
        self.is_file = isinstance(source, str) and Path(source).is_file()
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.period = 1 / fps if self.is_file and 0 < fps < 240 else 0
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.raw = self.jpeg = None
        self.frames = 0
        self.error = ''
        self.thread = threading.Thread(target=self.run, daemon=True)

    def run(self):
        try:
            while not self.stop_event.is_set():
                start = time.perf_counter()
                ok, frame = self.cap.read()
                if not ok:
                    if self.is_file:
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ok, frame = self.cap.read()
                    if not ok:
                        raise RuntimeError('Video source has no decodable frames')
                predictions, timing = engine.detect(frame, *config())
                annotated = draw(frame.copy(), predictions)
                text_scale = max(1.0, frame.shape[1] / 1280)
                cv2.putText(annotated, f"YOLO26 RK1820 | runtime {timing['runtime_ms']:.1f} ms",
                            (round(20 * text_scale), round(35 * text_scale)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7 * text_scale, (0, 230, 118), max(2, round(2 * text_scale)))
                # Bound preview encoding cost for high resolution cameras/videos.
                h, w = annotated.shape[:2]
                if w > 1280:
                    annotated = cv2.resize(annotated, (1280, round(h * 1280 / w)))
                ok, data = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if not ok:
                    raise RuntimeError('JPEG encoding failed')
                with self.lock:
                    self.raw, self.jpeg = frame, data.tobytes()
                    self.frames += 1
                self.stop_event.wait(max(0, self.period - (time.perf_counter() - start)))
        except Exception as exc:
            self.error = str(exc)
            print(f'Preview failed: {exc}', flush=True)
        finally:
            self.cap.release()

    def raw_frame(self):
        with self.lock:
            return None if self.raw is None else self.raw.copy()

    def stop(self):
        self.stop_event.set()
        self.thread.join()


class VideoAnalyzer:
    def __init__(self):
        self.lock = threading.Lock()
        self.state = dict(is_processing=False, progress=0, current_video='', error='')
        self.stop_event = threading.Event()
        self.thread = None

    def status(self):
        with self.lock:
            return dict(self.state)

    def update(self, **kwargs):
        with self.lock:
            self.state.update(kwargs)

    def start(self, source, destination):
        with self.lock:
            if self.state['is_processing']:
                raise HTTPException(409, 'A video is already being analyzed')
            self.state.update(is_processing=True, progress=0, current_video=source.name, error='')
            self.stop_event.clear()
            self.thread = threading.Thread(target=self.run, args=(source, destination), daemon=True)
            self.thread.start()

    def run(self, source, destination):
        cap, writer = cv2.VideoCapture(str(source)), None
        partial = destination.with_name(destination.stem + '.partial.mp4')
        try:
            if not cap.isOpened():
                raise RuntimeError('Cannot open uploaded video')
            w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps, count = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if count <= 0 or not 0 < fps < 1000:
                raise RuntimeError('Invalid video metadata')
            writer = cv2.VideoWriter(str(partial), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            if not writer.isOpened():
                raise RuntimeError('Cannot create output video')
            done = 0
            while not self.stop_event.is_set():
                ok, frame = cap.read()
                if not ok:
                    break
                predictions, _ = engine.detect(frame, *config())
                writer.write(draw(frame, predictions))
                done += 1
                self.update(progress=min(99, int(done * 100 / count)))
            if self.stop_event.is_set():
                raise RuntimeError('Analysis stopped')
            if done == 0:
                raise RuntimeError('No frames decoded')
            writer.release()
            writer = None
            partial.replace(destination)
            self.update(progress=100)
        except Exception as exc:
            self.update(error=str(exc))
        finally:
            cap.release()
            if writer:
                writer.release()
            partial.unlink(missing_ok=True)
            self.update(is_processing=False)

    def stop(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join()


analyzer = VideoAnalyzer()


@asynccontextmanager
async def lifespan(app):
    global engine, preview
    UPLOADS.mkdir(parents=True, exist_ok=True)
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    sample = ROOT / 'video/test.mp4'
    if sample.is_file() and not (UPLOADS / 'test.mp4').exists():
        shutil.copyfile(sample, UPLOADS / 'test.mp4')
    try:
        engine = YOLO26(options.model_path, options.weight_path, options.class_path, options.core_mask)
        if options.video_path or options.rtsp or options.camera_id >= 0:
            preview = Preview(options.video_path or options.rtsp or options.camera_id)
            preview.thread.start()
        yield
    finally:
        if preview:
            preview.stop()
        analyzer.stop()
        if engine:
            engine.release()
        engine = None


app = FastAPI(title='reComputer RK-CV · YOLO26 · RKNN3 / RK1820', lifespan=lifespan)


@app.get('/')
def index():
    return FileResponse(ROOT / 'index.html')


@app.get('/api/health')
def health():
    healthy = engine is not None and not (preview and preview.error)
    data = dict(status='ok' if healthy else 'error', model='yolo26n', accelerator='rk1820',
                host='rk3588', runtime='rknn3-toolkit-lite 1.1.0',
                inference_count=engine.calls if engine else 0,
                runtime_ms=round(engine.last_inference_ms, 2) if engine else None,
                total_ms=round(engine.last_total_ms, 2) if engine else None,
                preview_frames=preview.frames if preview else 0,
                preview_error=preview.error if preview else '')
    return JSONResponse(data, status_code=200 if healthy else 503)


class Config(BaseModel):
    obj_thresh: float = Field(0.25, ge=0, le=1, allow_inf_nan=False)
    nms_thresh: float = Field(0.45, ge=0, le=1, allow_inf_nan=False)


@app.get('/api/config')
def get_config():
    conf, iou = config()
    return dict(obj_thresh=conf, nms_thresh=iou)


@app.post('/api/config')
def update_config(value: Config):
    global thresholds
    with settings_lock:
        thresholds = (value.obj_thresh, value.nms_thresh)
    return dict(status='success')


@app.get('/api/video_feed')
def video_feed():
    if preview is None:
        raise HTTPException(409, 'Preview disabled; start with --video_path, --rtsp or --camera_id')
    def generate():
        last = None
        while not preview.stop_event.is_set() and not preview.error:
            with preview.lock:
                frame = preview.jpeg
            if frame is not None and frame is not last:
                last = frame
                yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'
            time.sleep(0.01)
    return StreamingResponse(generate(), media_type='multipart/x-mixed-replace; boundary=frame')


@app.get('/api/snapshot')
def snapshot():
    if preview:
        with preview.lock:
            if preview.jpeg:
                return Response(preview.jpeg, media_type='image/jpeg')
    raise HTTPException(503, 'No preview frame available')


@app.post('/api/models/yolo26/predict')
def predict(file: Optional[UploadFile] = File(None), video: Optional[UploadFile] = File(None),
            timestamp: float = Form(0, ge=0), realtime: bool = Form(False),
            conf: Optional[float] = Form(None, ge=0, le=1), iou: Optional[float] = Form(None, ge=0, le=1)):
    if engine is None:
        raise HTTPException(503, 'Model not initialized')
    frame, source = None, 'realtime camera frame'
    if file:
        frame = cv2.imdecode(np.frombuffer(file.file.read(), dtype=np.uint8), cv2.IMREAD_COLOR)
        source = 'uploaded image'
        if frame is None:
            raise HTTPException(400, 'Invalid image')
    elif video:
        with tempfile.NamedTemporaryFile(suffix='.mp4') as tmp:
            shutil.copyfileobj(video.file, tmp)
            tmp.flush()
            cap = cv2.VideoCapture(tmp.name)
            try:
                cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
                ok, frame = cap.read()
                if not ok:
                    raise HTTPException(400, 'No video frame at this timestamp')
            finally:
                cap.release()
        source = f'video frame at {timestamp}s'
    elif preview:
        frame = preview.raw_frame()
    if frame is None:
        raise HTTPException(503, 'No input frame available')
    default_conf, default_iou = config()
    predictions, timing = engine.detect(frame, conf if conf is not None else default_conf,
                                        iou if iou is not None else default_iou)
    return dict(success=True, source=source, predictions=predictions,
                image=dict(width=frame.shape[1], height=frame.shape[0]), timing=timing)


@app.post('/api/video/upload')
def upload_video(file: UploadFile = File(...)):
    path = file_in(UPLOADS, file.filename)
    if path.suffix.lower() not in ('.mp4', '.avi', '.mov', '.mkv'):
        raise HTTPException(400, 'Unsupported video extension')
    if analyzer.status()['is_processing'] and analyzer.status()['current_video'] == path.name:
        raise HTTPException(409, 'This video is being analyzed')
    # Atomic replacement prevents a reader seeing a partially uploaded file.
    with tempfile.NamedTemporaryFile(dir=UPLOADS, delete=False) as tmp:
        temp = Path(tmp.name)
        try:
            shutil.copyfileobj(file.file, tmp)
        except Exception:
            temp.unlink(missing_ok=True)
            raise
    temp.replace(path)
    return dict(filename=path.name, status='uploaded')


@app.get('/api/video/list')
def list_videos():
    return dict(uploads=sorted(p.name for p in UPLOADS.iterdir() if p.is_file()),
                outputs=sorted(p.name for p in OUTPUTS.glob('*.mp4') if '.partial.' not in p.name))


@app.post('/api/video/analyze')
def analyze_video(filename: str = Form(...)):
    path = file_in(UPLOADS, filename)
    if not path.is_file():
        raise HTTPException(404, 'Video not found')
    name = path.stem + '_results.mp4'
    analyzer.start(path, OUTPUTS / name)
    return dict(status='started', output_filename=name)


@app.get('/api/video/status')
def video_status():
    return analyzer.status()


@app.get('/api/video/download/{filename}')
def download_video(filename: str):
    path = file_in(OUTPUTS, filename)
    if not path.is_file() or '.partial.' in filename:
        raise HTTPException(404, 'Result not found')
    return FileResponse(path, filename=path.name, media_type='video/mp4')


def main():
    global options
    parser = argparse.ArgumentParser(description='YOLO26 on RK1820 PCIe + RK3588')
    parser.add_argument('--model_path', default=str(ROOT / 'model/yolo26n.rknn'))
    parser.add_argument('--weight_path', default=str(ROOT / 'model/yolo26n.weight'))
    parser.add_argument('--class_path', default=str(ROOT / 'model/coco_80_labels_list.txt'))
    parser.add_argument('--core_mask', type=lambda n: int(n, 0), default=1)
    parser.add_argument('--video_path', '--video', default=None)
    parser.add_argument('--camera_id', type=int, default=-1)
    parser.add_argument('--rtsp', default=None)
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8000)
    options = parser.parse_args()
    uvicorn.run(app, host=options.host, port=options.port)


if __name__ == '__main__':
    main()
