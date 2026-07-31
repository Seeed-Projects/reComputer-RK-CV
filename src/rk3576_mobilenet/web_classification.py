import os
import sys
import cv2
import argparse
import time
import numpy as np
import threading
import re
from fastapi import FastAPI, Response, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
import uvicorn
import shutil
from pathlib import Path
from typing import Optional

# 导入共享工具
from py_utils.mobilenet_utils import MobileNet_helper

# 尝试导入RKNN-Toolkit-Lite2
try:
    from rknnlite.api import RKNNLite
    RKNN_LITE_AVAILABLE = True
except ImportError:
    RKNN_LITE_AVAILABLE = False
    print("Warning: RKNN-Toolkit-Lite2 not available, using fallback")

# --- 应用路径与视频分析核心组件 ---
BASE_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = Path(os.environ.get("RK_CV_WORKSPACE", BASE_DIR / "workspace"))
UPLOAD_DIR = WORKSPACE_DIR / "uploads"
OUTPUT_DIR = WORKSPACE_DIR / "outputs"
ALLOWED_MEDIA_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".mp4"}
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def safe_media_filename(filename: str) -> str:
    """Reject path traversal and unsupported upload types."""
    if not filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    safe_name = Path(filename).name
    if (
        safe_name in {".", ".."}
        or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,254}", safe_name)
        or Path(safe_name).suffix.lower() not in ALLOWED_MEDIA_EXTENSIONS
    ):
        raise HTTPException(status_code=400, detail="Unsupported file type")
    return safe_name

class VideoAnalyzer:
    def __init__(self, model=None, mn_helper=None):
        self.model = model
        self.mn_helper = mn_helper
        self.is_processing = False
        self.progress = 0
        self.current_video = ""
        self.error_msg = ""
        self._stop_event = threading.Event()
        self._thread = None
        self._state_lock = threading.Lock()

    def set_engine(self, model, mn_helper):
        self.model = model
        self.mn_helper = mn_helper

    def start_analysis(self, input_path, output_path):
        with self._state_lock:
            if self.is_processing:
                return False
            self.is_processing = True
            self._stop_event.clear()

            # 判断是视频还是图片
            ext = os.path.splitext(input_path)[1].lower()
            if ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                self._thread = threading.Thread(target=self._process_image, args=(input_path, output_path))
            else:
                self._thread = threading.Thread(target=self._process_video, args=(input_path, output_path))

            self._thread.daemon = True
            self._thread.start()
            return True

    def _process_image(self, input_path, output_path):
        self.progress = 0
        self.error_msg = ""
        self.current_video = os.path.basename(input_path)

        try:
            frame = cv2.imread(input_path)
            if frame is None:
                self.error_msg = f"Error: Cannot open image {input_path}"
                self.is_processing = False
                return

            if self.model and self.mn_helper:
                processed_img = self.mn_helper.preprocess(frame)
                outputs = self.model.run(processed_img)

                if outputs is not None:
                    current_conf = det_config.get()
                    topk_classes, topk_scores = self.mn_helper.get_topk(outputs, topk=5, conf_thresh=current_conf)
                    self.mn_helper.draw_topk(frame, topk_classes, topk_scores)

            cv2.imwrite(output_path, frame)
            self.progress = 100
        except Exception as e:
            self.error_msg = f"Process error: {str(e)}"
        finally:
            self.is_processing = False

    def _process_video(self, input_path, output_path):
        self.progress = 0
        self.error_msg = ""
        self.current_video = os.path.basename(input_path)

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            self.error_msg = f"Error: Cannot open video {input_path}"
            self.is_processing = False
            return

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames <= 0:
            self.error_msg = "Error: Invalid total frames"
            self.is_processing = False
            cap.release()
            return

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_idx = 0
        try:
            while not self._stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    break

                # 推理流程
                if self.model and self.mn_helper:
                    processed_img = self.mn_helper.preprocess(frame)
                    outputs = self.model.run(processed_img)

                    if outputs is not None:
                        current_conf = det_config.get()
                        topk_classes, topk_scores = self.mn_helper.get_topk(outputs, topk=5, conf_thresh=current_conf)
                        self.mn_helper.draw_topk(frame, topk_classes, topk_scores)

                out.write(frame)
                frame_idx += 1
                self.progress = int((frame_idx / total_frames) * 100)

        except Exception as e:
            self.error_msg = f"Process error: {str(e)}"
        finally:
            cap.release()
            out.release()
            self.is_processing = False
            if not self.error_msg:
                self.progress = 100

    def stop(self):
        self._stop_event.set()

video_analyzer = VideoAnalyzer()

class DetectionConfig:
    def __init__(self):
        self.conf_thresh = 0.0
        self.lock = threading.Lock()

    def update(self, conf_thresh):
        with self.lock:
            self.conf_thresh = conf_thresh

    def get(self):
        with self.lock:
            return self.conf_thresh

det_config = DetectionConfig()

# --- FastAPI 核心组件 ---
app = FastAPI(
    title="reComputer RK-CV MobileNet API",
    version="1.0.0",
    description="RK3576 MobileNet image and video classification service",
)

@app.get("/api/config")
async def get_config():
    conf = det_config.get()
    return {"conf_thresh": conf}

@app.post("/api/config")
async def update_config(config: dict):
    conf_thresh = float(config.get("conf_thresh", 0.0))
    if not 0.0 <= conf_thresh <= 1.0:
        raise HTTPException(status_code=422, detail="conf_thresh must be between 0 and 1")
    det_config.update(conf_thresh)
    return {"status": "success"}


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "model": "mobilenet",
        "platform": "rk3576",
        "model_ready": _global_model is not None,
    }

class FrameBuffer:
    def __init__(self):
        self.frame = None
        self.raw_frame = None
        self.lock = threading.Lock()

    def set_frame(self, frame, raw_frame=None):
        with self.lock:
            self.frame = frame
            if raw_frame is not None:
                self.raw_frame = raw_frame

    def get_frame(self):
        with self.lock:
            return self.frame

    def get_raw_frame(self):
        with self.lock:
            return self.raw_frame.copy() if self.raw_frame is not None else None

frame_buffer = FrameBuffer()

@app.get("/api/video_feed")
async def video_feed():
    def generate():
        while True:
            frame = frame_buffer.get_frame()
            if frame is not None:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.03)
    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/")
async def index():
    import sys
    # 判断是否为纯 Web 模式 (通过启动参数或环境变量判断)
    is_web_only = False
    if '--camera_id' in sys.argv:
        idx = sys.argv.index('--camera_id')
        if idx + 1 < len(sys.argv) and sys.argv[idx+1] == '-1':
            is_web_only = True

    if is_web_only:
        return Response(content="""
        <html>
          <head>
            <title>reComputer RK-CV Classification Web Analytics</title>
            <style>
              body { background-color: #1a1a1a; color: white; text-align: center; font-family: sans-serif; margin: 0; padding: 20px; }
              .container { max-width: 800px; margin: 0 auto; }
              .upload-box { margin: 20px auto; padding: 30px; border: 2px dashed #00e676; border-radius: 10px; background: #2a2a2a; }
              .status-box { margin: 20px auto; padding: 20px; background: #333; border-radius: 10px; display: none; }
              .result-box { margin: 20px auto; display: none; }
              .btn { background: #00e676; color: black; border: none; padding: 10px 20px; cursor: pointer; border-radius: 5px; font-weight: bold; margin-top: 10px; }
              .btn:disabled { background: #555; cursor: not-allowed; }
              h1 { color: #00e676; }
              #progress-bar { width: 100%; height: 20px; background: #444; border-radius: 10px; margin-top: 10px; overflow: hidden; }
              #progress-fill { height: 100%; background: #00e676; width: 0%; transition: width 0.3s; }
            </style>
          </head>
          <body>
            <div class="container">
              <h1>MobileNet Image & Video Analytics</h1>
              <p>Upload a video (MP4) or an image (JPG/PNG) for offline classification analysis.</p>

              <div class="upload-box">
                <input type="file" id="fileInput" accept="video/mp4,image/jpeg,image/png,image/bmp">
                <br><br>
                <button class="btn" id="uploadBtn" onclick="uploadAndAnalyze()">Upload & Analyze</button>
              </div>

              <div class="status-box" id="statusBox">
                <h3 id="statusText">Processing...</h3>
                <div id="progress-bar"><div id="progress-fill"></div></div>
              </div>

              <div class="result-box" id="resultBox">
                <h3>Analysis Complete!</h3>
                <a id="downloadLink" href="#" class="btn" download>Download Result</a>
                <br><br>
                <div id="previewArea"></div>
              </div>
            </div>

            <script>
              let pollInterval;

              async function uploadAndAnalyze() {
                const fileInput = document.getElementById('fileInput');
                if (!fileInput.files.length) {
                  alert('Please select a file first.');
                  return;
                }

                const file = fileInput.files[0];
                const uploadBtn = document.getElementById('uploadBtn');
                uploadBtn.disabled = true;
                uploadBtn.innerText = 'Uploading...';

                // 1. Upload
                const formData = new FormData();
                formData.append('file', file);

                try {
                  const uploadRes = await fetch('/api/video/upload', {
                    method: 'POST',
                    body: formData
                  });
                  const uploadData = await uploadRes.json();

                  // 2. Start Analyze
                  const analyzeFormData = new FormData();
                  analyzeFormData.append('filename', uploadData.filename);

                  const analyzeRes = await fetch('/api/video/analyze', {
                    method: 'POST',
                    body: analyzeFormData
                  });
                  const analyzeData = await analyzeRes.json();

                  if (analyzeData.status === 'started') {
                    document.getElementById('statusBox').style.display = 'block';
                    document.getElementById('resultBox').style.display = 'none';
                    pollStatus(analyzeData.output);
                  } else {
                    alert('Error: ' + analyzeData.message);
                    uploadBtn.disabled = false;
                    uploadBtn.innerText = 'Upload & Analyze';
                  }
                } catch (e) {
                  alert('Error during upload: ' + e);
                  uploadBtn.disabled = false;
                  uploadBtn.innerText = 'Upload & Analyze';
                }
              }

              function pollStatus(outputFilename) {
                pollInterval = setInterval(async () => {
                  const res = await fetch('/api/video/status');
                  const data = await res.json();

                  if (data.error) {
                    clearInterval(pollInterval);
                    document.getElementById('statusText').innerText = 'Error: ' + data.error;
                    document.getElementById('uploadBtn').disabled = false;
                    document.getElementById('uploadBtn').innerText = 'Upload & Analyze';
                    return;
                  }

                  document.getElementById('progress-fill').style.width = data.progress + '%';
                  document.getElementById('statusText').innerText = `Processing: ${data.progress}%`;

                  if (!data.is_processing && data.progress === 100) {
                    clearInterval(pollInterval);
                    showResult(outputFilename);
                  }
                }, 1000);
              }

              function showResult(filename) {
                document.getElementById('statusBox').style.display = 'none';
                document.getElementById('resultBox').style.display = 'block';

                const downloadUrl = `/api/video/download/${filename}`;
                document.getElementById('downloadLink').href = downloadUrl;

                const previewArea = document.getElementById('previewArea');
                const ext = filename.split('.').pop().toLowerCase();

                if (['jpg', 'jpeg', 'png', 'bmp'].includes(ext)) {
                    previewArea.innerHTML = `<img src="${downloadUrl}" style="max-width:100%; border-radius:10px;">`;
                } else {
                    previewArea.innerHTML = `<video src="${downloadUrl}" controls style="max-width:100%; border-radius:10px;"></video>`;
                }

                const uploadBtn = document.getElementById('uploadBtn');
                uploadBtn.disabled = false;
                uploadBtn.innerText = 'Upload & Analyze New File';
              }
            </script>
          </body>
        </html>
        """, media_type="text/html")

    # 如果不是纯 Web 模式，展示实时视频流
    return Response(content="""
    <html>
      <head>
        <title>reComputer RK-CV Classification Web Preview</title>
        <style>
          body { background-color: #1a1a1a; color: white; text-align: center; font-family: sans-serif; margin: 0; padding: 20px; }
          .container { max-width: 1200px; margin: 0 auto; }
          .video-box { margin: 20px auto; display: inline-block; border: 5px solid #333; border-radius: 10px; overflow: hidden; background: #000; width: 100%; max-width: 800px; }
          .controls { background: #2a2a2a; padding: 20px; border-radius: 10px; display: inline-block; text-align: left; min-width: 400px; vertical-align: top; margin: 10px; }
          .control-group { margin-bottom: 15px; }
          .control-group label { display: block; margin-bottom: 5px; font-weight: bold; }
          .slider-container { display: flex; align-items: center; gap: 15px; }
          input[type=range] { flex-grow: 1; cursor: pointer; }
          .value-display { min-width: 50px; font-family: monospace; background: #444; padding: 2px 8px; border-radius: 4px; text-align: center; }
          h1 { color: #00e676; }
        </style>
      </head>
      <body>
        <div class="container">
          <h1>reComputer RK-CV Real-time MobileNet Classification</h1>
          <div class="video-box">
            <img src="/api/video_feed" style="max-width: 100%; height: auto;">
          </div>

          <div class="controls">
            <div class="control-group">
              <label>Confidence Threshold (置信度过滤阈值)</label>
              <div class="slider-container">
                <input type="range" id="confSlider" min="0.0" max="1.0" step="0.01" value="0.0">
                <span id="confValue" class="value-display">0.00</span>
              </div>
            </div>
          </div>

          <p style="color: #888; margin-top: 20px;">Streaming via FastAPI + MJPEG | Port: 8000</p>
        </div>

        <script>
          const confSlider = document.getElementById('confSlider');
          const confValue = document.getElementById('confValue');

          function updateConfig() {
            const conf_thresh = parseFloat(confSlider.value);
            confValue.innerText = conf_thresh.toFixed(2);

            fetch('/api/config', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ conf_thresh })
            });
          }

          confSlider.oninput = updateConfig;

          // 初始化获取当前值
          fetch('/api/config').then(res => res.json()).then(data => {
            confSlider.value = data.conf_thresh;
            confValue.innerText = data.conf_thresh.toFixed(2);
          });
        </script>
      </body>
    </html>
    """, media_type="text/html")

# --- 视频分析 API ---
@app.post("/api/video/upload")
async def upload_video(file: UploadFile = File(...)):
    filename = safe_media_filename(file.filename or "")
    file_path = UPLOAD_DIR / filename
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {"filename": filename, "status": "uploaded"}

@app.get("/api/video/list")
async def list_videos():
    uploads = sorted(os.listdir(UPLOAD_DIR))
    outputs = sorted(os.listdir(OUTPUT_DIR))
    return {"uploads": uploads, "outputs": outputs}

@app.post("/api/video/analyze")
async def analyze_video(filename: str = Form(...)):
    filename = safe_media_filename(filename)
    input_path = UPLOAD_DIR / filename
    if not os.path.exists(input_path):
        raise HTTPException(status_code=404, detail="File not found")

    ext = os.path.splitext(filename)[1].lower()
    is_image = ext in ['.jpg', '.jpeg', '.png', '.bmp']

    if is_image:
        img = cv2.imread(input_path)
        if img is None:
            raise HTTPException(status_code=400, detail="Cannot open image file")
        width = img.shape[1]
        height = img.shape[0]
    else:
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Cannot open video file")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

    name_base = os.path.splitext(filename)[0]
    output_filename = f"{name_base}_{width}x{height}_results{ext}"
    output_path = OUTPUT_DIR / output_filename

    success = video_analyzer.start_analysis(str(input_path), str(output_path))
    if success:
        return {"status": "started", "output": output_filename}
    else:
        return {"status": "error", "message": "Already processing another file"}

@app.get("/api/video/status")
async def get_analysis_status():
    return {
        "is_processing": video_analyzer.is_processing,
        "progress": video_analyzer.progress,
        "current_video": video_analyzer.current_video,
        "error": video_analyzer.error_msg
    }

@app.get("/api/video/download/{filename}")
async def download_video(filename: str):
    filename = safe_media_filename(filename)
    file_path = OUTPUT_DIR / filename
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    ext = os.path.splitext(filename)[1].lower()
    media_type = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.bmp': 'image/bmp',
        '.mp4': 'video/mp4',
    }[ext]
    return FileResponse(file_path, media_type=media_type, filename=filename)

# 全局变量用于在 API 接口中访问模型和辅助类
_global_model = None
_global_mn_helper = None

@app.post("/api/models/mobilenet/predict")
async def predict(
    file: Optional[UploadFile] = File(None),
    video: Optional[UploadFile] = File(None),
    timestamp: Optional[float] = Form(None),
    conf: Optional[float] = Form(None),
    topk: int = Form(5),
):
    if _global_model is None or _global_mn_helper is None:
        raise HTTPException(status_code=503, detail="Model not initialized")

    try:
        if file is not None and video is not None:
            raise HTTPException(status_code=422, detail="Provide either file or video, not both")
        if timestamp is not None and timestamp < 0:
            raise HTTPException(status_code=422, detail="timestamp must be non-negative")
        if conf is not None and not 0.0 <= conf <= 1.0:
            raise HTTPException(status_code=422, detail="conf must be between 0 and 1")
        if not 1 <= topk <= 20:
            raise HTTPException(status_code=422, detail="topk must be between 1 and 20")

        img = None
        source_info = ""

        if file:
            contents = await file.read()
            nparr = np.frombuffer(contents, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                raise HTTPException(status_code=400, detail="Invalid image file")
            source_info = "uploaded image"

        elif video:
            import tempfile
            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                    tmp.write(await video.read())
                    tmp_path = tmp.name

                cap = cv2.VideoCapture(tmp_path)
                if cap.isOpened():
                    if timestamp is not None:
                        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
                    ret, frame = cap.read()
                    if ret:
                        img = frame
                        source_info = f"video frame at {timestamp if timestamp is not None else 0}s"
                    cap.release()
                if img is None:
                    raise HTTPException(status_code=400, detail="Invalid video file or timestamp")
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    os.unlink(tmp_path)

        if img is None:
            img = frame_buffer.get_raw_frame()
            source_info = "realtime camera frame"

        if img is None:
            raise HTTPException(
                status_code=400,
                detail="No valid input source found (image, video, or realtime frame)",
            )

        h, w = img.shape[:2]

        input_img = _global_mn_helper.preprocess(img)
        outputs = _global_model.run(input_img)
        if outputs is None:
            raise HTTPException(status_code=500, detail="RKNN inference failed")

        current_conf = det_config.get() if conf is None else conf
        topk_classes, topk_scores = _global_mn_helper.get_topk(
            outputs, topk=topk, conf_thresh=current_conf
        )

        predictions = []
        for cls, score in zip(topk_classes, topk_scores):
            predictions.append({
                "class": cls,
                "confidence": float(score)
            })

        return {
            "success": True,
            "model": "mobilenet",
            "source": source_info,
            "predictions": predictions,
            "image": {
                "width": w,
                "height": h
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        return {"success": False, "message": str(e)}


class RKNNLiteModel:
    def __init__(self, model_path):
        if not RKNN_LITE_AVAILABLE:
            raise ImportError("RKNN-Toolkit-Lite2 is not available")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"RKNN model file not found: {model_path}")
        self.rknn_lite = RKNNLite()
        self.lock = threading.Lock()
        print(f'Loading RKNN model from {model_path}...', flush=True)
        sys.stdout.flush()
        ret = self.rknn_lite.load_rknn(model_path)
        if ret != 0:
            raise Exception(f"Load RKNN model failed with error code: {ret}")
        print('Initializing runtime...', flush=True)
        sys.stdout.flush()
        ret = self.rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1)
        if ret != 0:
            raise Exception(f"Init runtime failed with error code: {ret}")
        print('RKNN model loaded successfully', flush=True)
        sys.stdout.flush()

    def run(self, inputs):
        try:
            if len(inputs.shape) == 3:
                inputs = np.expand_dims(inputs, axis=0)
            if inputs.dtype != np.uint8:
                inputs = inputs.astype(np.uint8)
            with self.lock:
                return self.rknn_lite.inference(inputs=[inputs])
        except Exception as e:
            print(f"Inference error: {e}")
            return None

    def release(self):
        if hasattr(self, 'rknn_lite'):
            with self.lock:
                self.rknn_lite.release()


def main():
    parser = argparse.ArgumentParser(description='Web MobileNet classification on RK3576')
    parser.add_argument(
        '--model_path',
        type=str,
        default=str(BASE_DIR / 'model' / 'rk3576_mobilenet_v2.rknn'),
        help='RKNN model path',
    )
    parser.add_argument('--camera_id', type=int, default=1, help='Camera device ID (default: 1 for /dev/video1)')
    parser.add_argument('--video_path', type=str, help='Path to video file (overrides camera_id)')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Web server host')
    parser.add_argument('--port', type=int, default=8000, help='Web server port')
    args = parser.parse_args()

    if not RKNN_LITE_AVAILABLE:
        print("Error: RKNN-Toolkit-Lite2 is not available.")
        return

    global _global_model, _global_mn_helper

    # 初始化模型与辅助类
    model = RKNNLiteModel(args.model_path)
    mn_helper = MobileNet_helper(class_label_path=str(BASE_DIR / 'model' / 'synset.txt'))

    _global_model = model
    _global_mn_helper = mn_helper

    # 为视频分析器设置引擎
    video_analyzer.set_engine(model, mn_helper)

    # 启动 Web 服务器线程
    def run_fastapi():
        print(f"Web Preview started at http://{args.host}:{args.port}", flush=True)
        uvicorn.run(app, host=args.host, port=args.port, log_level="info", log_config=None)

    web_thread = threading.Thread(target=run_fastapi, daemon=True)
    web_thread.start()

    # 如果 camera_id 为 -1，则进入纯 Web 模式，不打开摄像头，仅提供 API 和 Web 服务
    if args.camera_id == -1 and not args.video_path:
        print("Running in Web-only mode (camera_id=-1). Waiting for file uploads...", flush=True)
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("Interrupted by user")
        finally:
            model.release()
        return

    # 打开视频流
    if args.video_path:
        cap = cv2.VideoCapture(args.video_path)
    else:
        cap = cv2.VideoCapture(args.camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print(f"Error: Cannot open video source (ID: {args.camera_id if not args.video_path else args.video_path})")
        # 如果摄像头打不开，我们让程序进入一个死循环维持 Web 运行（以便用户继续使用视频上传分析功能）
        while True:
            time.sleep(1)
        return

    fps_counter = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                if args.video_path:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break

            # 推理流程
            processed_img = mn_helper.preprocess(frame)
            start_time = time.time()
            outputs = model.run(processed_img)
            inference_time = time.time() - start_time

            if outputs is not None:
                current_conf = det_config.get()
                topk_classes, topk_scores = mn_helper.get_topk(outputs, topk=5, conf_thresh=current_conf)
                mn_helper.draw_topk(frame, topk_classes, topk_scores)

            inf_fps = 1.0 / inference_time if inference_time > 0 else 0
            fps_counter = 0.9 * fps_counter + 0.1 * inf_fps if fps_counter > 0 else inf_fps
            cv2.putText(frame, f'NPU FPS: {fps_counter:.1f}', (frame.shape[1] - 250, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            # 更新 Web 帧缓冲区
            _, buffer = cv2.imencode('.jpg', frame)
            frame_buffer.set_frame(buffer.tobytes(), frame)

            time.sleep(0.01) # 降低 CPU 占用

    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        cap.release()
        model.release()

if __name__ == '__main__':
    main()
