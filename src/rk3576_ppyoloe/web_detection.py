import os
import sys
import cv2
import argparse
import time
import numpy as np
import threading
from fastapi import FastAPI, Response, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
import uvicorn
import shutil
import re
from pathlib import Path
from typing import Optional, List

# 导入共享工具
from py_utils.coco_utils import COCO_test_helper, post_process, draw, CLASSES

# 尝试导入RKNN-Toolkit-Lite2
try:
    from rknnlite.api import RKNNLite
    RKNN_LITE_AVAILABLE = True
except ImportError:
    RKNN_LITE_AVAILABLE = False
    print("Warning: RKNN-Toolkit-Lite2 not available, using fallback")

# 常量定义
OBJ_THRESH = 0.25
NMS_THRESH = 0.45
IMG_SIZE = (640, 640)  # (width, height)

# 默认类别定义 (COCO 80类)
DEFAULT_CLASSES = ("person", "bicycle", "car","motorbike ","aeroplane ","bus ","train","truck ","boat","traffic light",
           "fire hydrant","stop sign ","parking meter","bench","bird","cat","dog ","horse ","sheep","cow","elephant",
           "bear","zebra ","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite",
           "baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork","knife ",
           "spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza ","donut","cake","chair","sofa",
           "pottedplant","bed","diningtable","toilet ","tvmonitor","laptop	","mouse	","remote ","keyboard ","cell phone","microwave ",
           "oven ","toaster","sink","refrigerator ","book","clock","vase","scissors ","teddy bear ","hair drier", "toothbrush ")

CLASSES = DEFAULT_CLASSES

def load_classes(path):
    """
    从文件加载类别，支持双引号和逗号分隔的格式
    例如: "person", "bicycle", "car"
    """
    global CLASSES
    if not path or not os.path.exists(path):
        CLASSES = DEFAULT_CLASSES
        return

    try:
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            # 简单的解析逻辑：移除换行，按逗号分割，去除空格和双引号
            import re
            # 匹配双引号内的内容
            items = re.findall(r'"([^"]*)"', content)
            if items:
                CLASSES = tuple(items)
                print(f"Successfully loaded {len(CLASSES)} classes from {path}")
            else:
                # 备选方案：如果没匹配到双引号，尝试按逗号分割
                items = [item.strip().strip('"') for item in re.split(r'[,\r\n]+', content) if item.strip()]
                if items:
                    CLASSES = tuple(items)
                    print(f"Loaded {len(CLASSES)} classes from {path} (fallback parsing)")
                else:
                    print(f"Warning: No classes found in {path}, using default COCO classes")
                    CLASSES = DEFAULT_CLASSES
    except Exception as e:
        print(f"Error loading classes from {path}: {e}. Using default COCO classes")
        CLASSES = DEFAULT_CLASSES

# 动态配置参数
class DetectionConfig:
    def __init__(self):
        self.obj_thresh = 0.25
        self.nms_thresh = 0.45
        self.lock = threading.Lock()

    def update(self, obj_thresh, nms_thresh):
        with self.lock:
            self.obj_thresh = obj_thresh
            self.nms_thresh = nms_thresh

    def get(self):
        with self.lock:
            return self.obj_thresh, self.nms_thresh

det_config = DetectionConfig()

# --- 视频分析核心组件 ---
BASE_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = Path(os.environ.get("RK_CV_WORKSPACE", BASE_DIR / "workspace"))
UPLOAD_DIR = WORKSPACE_DIR / "uploads"
OUTPUT_DIR = WORKSPACE_DIR / "outputs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def safe_video_filename(filename: str) -> str:
    """Reject path traversal and unsupported video uploads."""
    if not filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    safe_name = Path(filename).name
    if (
        safe_name in {".", ".."}
        or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,254}", safe_name)
        or Path(safe_name).suffix.lower() != ".mp4"
    ):
        raise HTTPException(status_code=400, detail="Only MP4 video files are supported")
    return safe_name

class VideoAnalyzer:
    def __init__(self, model=None, co_helper=None):
        self.model = model
        self.co_helper = co_helper
        self.is_processing = False
        self.progress = 0
        self.current_video = ""
        self.error_msg = ""
        self._stop_event = threading.Event()
        self._thread = None

    def set_engine(self, model, co_helper):
        self.model = model
        self.co_helper = co_helper

    def start_analysis(self, input_path, output_path):
        if self.is_processing:
            return False
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._process_video, args=(input_path, output_path))
        self._thread.daemon = True
        self._thread.start()
        return True

    def _process_video(self, input_path, output_path):
        self.is_processing = True
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
        local_helper = COCO_test_helper(enable_letter_box=True)

        frame_idx = 0
        try:
            while not self._stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    break

                # 推理流程
                if self.model and self.co_helper:
                    processed_img = preprocess_frame(frame, local_helper)
                    outputs = self.model.run(processed_img)

                    if outputs is not None:
                        obj, nms = det_config.get()
                        boxes, classes, scores = post_process_with_thresh(outputs, obj, nms)
                        if boxes is not None:
                            real_boxes = local_helper.get_real_box(boxes)
                            draw(frame, real_boxes, scores, classes)

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

# --- FastAPI 核心组件 ---
app = FastAPI(
    title="reComputer RK-CV PP-YOLOE API",
    version="1.0.0",
    description="RK3588/RK3576 PP-YOLOE object detection service",
)

@app.get("/api/config")
async def get_config():
    obj, nms = det_config.get()
    return {"obj_thresh": obj, "nms_thresh": nms}

@app.post("/api/config")
async def update_config(config: dict):
    det_config.update(config.get("obj_thresh", 0.25), config.get("nms_thresh", 0.45))
    return {"status": "success"}

# --- 视频分析 API ---
@app.post("/api/video/upload")
async def upload_video(file: UploadFile = File(...)):
    filename = safe_video_filename(file.filename)
    file_path = UPLOAD_DIR / filename
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {"filename": filename, "status": "uploaded"}

@app.get("/api/video/list")
async def list_videos():
    uploads = os.listdir(UPLOAD_DIR)
    outputs = os.listdir(OUTPUT_DIR)
    return {"uploads": uploads, "outputs": outputs}

@app.post("/api/video/analyze")
async def analyze_video(filename: str = Form(...)):
    filename = safe_video_filename(filename)
    input_path = str(UPLOAD_DIR / filename)
    if not os.path.exists(input_path):
        raise HTTPException(status_code=404, detail="Video not found")

    # 获取视频分辨率以生成文件名
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise HTTPException(status_code=400, detail="Cannot open video file")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    # 格式化输出文件名：原名_宽x高_results.mp4
    name_base = os.path.splitext(filename)[0]
    output_filename = f"{name_base}_{width}x{height}_results.mp4"
    output_path = str(OUTPUT_DIR / output_filename)

    success = video_analyzer.start_analysis(input_path, output_path)
    if success:
        return {"status": "started", "output": output_filename}
    else:
        return {"status": "error", "message": "Already processing another video"}

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
    filename = safe_video_filename(filename)
    file_path = OUTPUT_DIR / filename
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path, media_type='video/mp4', filename=filename)

# 全局变量用于在 API 接口中访问模型和辅助类
_global_model = None
_global_co_helper = None


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "model": "ppyoloe",
        "platform": getattr(_global_model, "platform", None),
        "model_path": os.path.basename(getattr(_global_model, "model_path", "")),
        "model_ready": _global_model is not None,
    }

@app.post("/api/models/ppyoloe/predict")
async def predict(
    file: Optional[UploadFile] = File(None),
    video: Optional[UploadFile] = File(None),
    timestamp: Optional[float] = Form(None),
    realtime: Optional[bool] = Form(False),
    conf: Optional[float] = Form(None),
    iou: Optional[float] = Form(None)
):
    if _global_model is None or _global_co_helper is None:
        return {"success": False, "message": "Model not initialized"}

    try:
        img = None
        source_info = ""

        # 1. 优先级：如果有上传图片
        if file:
            contents = await file.read()
            nparr = np.frombuffer(contents, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            source_info = "uploaded image"

        # 2. 优先级：如果有上传视频且有时间戳
        elif video:
            # 保存临时视频文件
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                tmp.write(await video.read())
                tmp_path = tmp.name

            cap = cv2.VideoCapture(tmp_path)
            if cap.isOpened():
                if timestamp is not None:
                    # 跳转到指定时间 (毫秒)
                    cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
                ret, frame = cap.read()
                if ret:
                    img = frame
                    source_info = f"video frame at {timestamp if timestamp else 0}s"
                cap.release()
            os.unlink(tmp_path)

        # 3. 优先级：realtime 参数或没有任何文件上传，使用摄像头当前帧
        if img is None:
            img = frame_buffer.get_raw_frame()
            source_info = "realtime camera frame"

        if img is None:
            return {"success": False, "message": "No valid input source found (image, video, or camera)"}

        h, w = img.shape[:2]

        # 预处理
        request_helper = COCO_test_helper(enable_letter_box=True)
        input_img = preprocess_frame(img, request_helper)

        # 推理
        outputs = _global_model.run(input_img)

        # 使用请求参数或全局配置
        current_obj_thresh, current_nms_thresh = det_config.get()
        target_conf = conf if conf is not None else current_obj_thresh
        target_iou = iou if iou is not None else current_nms_thresh

        # 后处理
        boxes, classes, scores = post_process_with_thresh(outputs, target_conf, target_iou)

        predictions = []
        if boxes is not None:
            real_boxes = request_helper.get_real_box(boxes)
            for box, score, cl in zip(real_boxes, scores, classes):
                predictions.append({
                    "class": CLASSES[cl],
                    "confidence": float(score),
                    "box": {
                        "x1": int(box[0]),
                        "y1": int(box[1]),
                        "x2": int(box[2]),
                        "y2": int(box[3])
                    }
                })

        return {
            "success": True,
            "source": source_info,
            "predictions": predictions,
            "image": {
                "width": w,
                "height": h
            }
        }
    except Exception as e:
        return {"success": False, "message": str(e)}

class FrameBuffer:
    def __init__(self):
        self.frame = None
        self.raw_frame = None  # 新增：保存原始 BGR 帧用于 API 推理
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
    # 判断是否为纯 Web 模式 (通过启动参数判断)
    is_web_only = False
    if '--camera_id' in sys.argv:
        idx = sys.argv.index('--camera_id')
        if idx + 1 < len(sys.argv) and sys.argv[idx+1] == '-1':
            is_web_only = True

    if is_web_only:
        return Response(content="""
        <html>
          <head>
            <title>reComputer RK-CV PP-YOLOE Web Analytics</title>
            <style>
              body { background-color: #1a1a1a; color: white; text-align: center; font-family: sans-serif; margin: 0; padding: 20px; }
              .container { max-width: 800px; margin: 0 auto; }
              .video-analysis { text-align: left; background: #2a2a2a; padding: 20px; border-radius: 10px; margin: 10px; }
              .btn { background: #00e676; color: #000; border: none; padding: 8px 16px; border-radius: 4px; cursor: pointer; font-weight: bold; margin: 5px; }
              .btn:hover { background: #00c853; }
              .btn:disabled { background: #555; cursor: not-allowed; }
              .progress-container { width: 100%; background: #444; border-radius: 10px; margin: 15px 0; height: 20px; position: relative; overflow: hidden; }
              .progress-bar { height: 100%; background: #00e676; width: 0%; transition: 0.3s; }
              .progress-text { position: absolute; width: 100%; text-align: center; top: 0; left: 0; line-height: 20px; font-size: 12px; font-weight: bold; color: #fff; text-shadow: 1px 1px 2px #000; }
              table { width: 100%; border-collapse: collapse; margin-top: 15px; }
              th, td { text-align: left; padding: 10px; border-bottom: 1px solid #444; }
              th { color: #888; }
              h1 { color: #00e676; }
              .control-group { margin-bottom: 15px; }
              .control-group label { display: block; margin-bottom: 5px; font-weight: bold; }
            </style>
          </head>
          <body>
            <div class="container">
              <h1>PP-YOLOE Offline Video Analytics</h1>

              <div class="video-analysis">
                <h3>Analyze Local Video</h3>
                <div class="control-group">
                  <label>Upload New Video (.mp4)</label>
                  <input type="file" id="videoUpload" accept=".mp4">
                  <button class="btn" onclick="uploadVideo()">Upload</button>
                </div>

                <div id="processingArea" style="display: none;">
                  <p id="statusText">Processing: <span id="currentFileName">-</span></p>
                  <div class="progress-container">
                    <div id="progressBar" class="progress-bar"></div>
                    <div id="progressText" class="progress-text">0%</div>
                  </div>
                  <p id="errorText" style="color: #ff5252;"></p>
                </div>

                <div class="control-group">
                  <label>File Management</label>
                  <button class="btn" onclick="refreshFileList()">Refresh List</button>
                  <table>
                    <thead>
                      <tr><th>File Name</th><th>Action</th></tr>
                    </thead>
                    <tbody id="fileTableBody">
                      <!-- Files will be listed here -->
                    </tbody>
                  </table>
                </div>
              </div>
            </div>

            <script>
              let pollInterval = null;

              async function uploadVideo() {
                const fileInput = document.getElementById('videoUpload');
                if (!fileInput.files.length) {
                  alert('Please select a video file first');
                  return;
                }

                const formData = new FormData();
                formData.append('file', fileInput.files[0]);

                try {
                  const res = await fetch('/api/video/upload', {
                    method: 'POST',
                    body: formData
                  });
                  const data = await res.json();

                  if (data.filename) {
                    startAnalysis(data.filename);
                  }
                } catch (e) {
                  alert('Upload failed: ' + e);
                }
              }

              async function startAnalysis(filename) {
                const formData = new FormData();
                formData.append('filename', filename);

                try {
                  const res = await fetch('/api/video/analyze', {
                    method: 'POST',
                    body: formData
                  });
                  const data = await res.json();

                  if (data.status === 'started') {
                    document.getElementById('processingArea').style.display = 'block';
                    document.getElementById('currentFileName').innerText = filename;
                    document.getElementById('errorText').innerText = '';

                    if (pollInterval) clearInterval(pollInterval);
                    pollInterval = setInterval(checkStatus, 1000);
                  } else {
                    alert(data.message || 'Failed to start analysis');
                  }
                } catch (e) {
                  alert('Analysis request failed: ' + e);
                }
              }

              async function checkStatus() {
                try {
                  const res = await fetch('/api/video/status');
                  const data = await res.json();

                  if (data.error) {
                    document.getElementById('errorText').innerText = data.error;
                    clearInterval(pollInterval);
                    return;
                  }

                  const progress = data.progress;
                  document.getElementById('progressBar').style.width = progress + '%';
                  document.getElementById('progressText').innerText = progress + '%';

                  if (!data.is_processing && progress === 100) {
                    clearInterval(pollInterval);
                    setTimeout(() => {
                      document.getElementById('processingArea').style.display = 'none';
                      refreshFileList();
                    }, 2000);
                  }
                } catch (e) {
                  console.error('Status check failed:', e);
                }
              }

              async function refreshFileList() {
                try {
                  const res = await fetch('/api/video/list');
                  const data = await res.json();

                  const tbody = document.getElementById('fileTableBody');
                  tbody.innerHTML = '';

                  data.uploads.forEach(f => {
                      const tr = document.createElement('tr');
                      tr.innerHTML = `
                          <td>${f} (Original)</td>
                          <td><button class="btn" onclick="startAnalysis('${f}')">Analyze</button></td>
                      `;
                      tbody.appendChild(tr);
                  });

                  data.outputs.forEach(f => {
                      const tr = document.createElement('tr');
                      tr.innerHTML = `
                          <td>${f} (Analyzed)</td>
                          <td><button class="btn" onclick="window.open('/api/video/download/${f}')">Download</button></td>
                      `;
                      tbody.appendChild(tr);
                  });
                } catch (e) {
                  console.error('Failed to refresh file list:', e);
                }
              }

              // Initial load
              refreshFileList();
            </script>
          </body>
        </html>
        """, media_type="text/html")

    # 如果不是纯 Web 模式，展示实时视频流
    return Response(content="""
    <html>
      <head>
        <title>reComputer RK-CV Real-time PP-YOLOE Detection</title>
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
          <h1>PP-YOLOE Real-time Detection</h1>

          <div class="video-box">
            <img id="streamImg" src="/api/video_feed" style="max-width: 100%; height: auto;">
          </div>

          <div class="controls">
            <div class="control-group">
              <label>Confidence Threshold (置信度阈值)</label>
              <div class="slider-container">
                <input type="range" id="confSlider" min="0.01" max="1.0" step="0.01" value="0.25">
                <span id="confValue" class="value-display">0.25</span>
              </div>
            </div>

            <div class="control-group">
              <label>IOU Threshold (NMS 阈值)</label>
              <div class="slider-container">
                <input type="range" id="iouSlider" min="0.01" max="1.0" step="0.01" value="0.45">
                <span id="iouValue" class="value-display">0.45</span>
              </div>
            </div>
          </div>

          <p style="color: #888; margin-top: 20px;">Streaming via FastAPI + MJPEG | Port: 8000</p>
        </div>

        <script>
          const confSlider = document.getElementById('confSlider');
          const confValue = document.getElementById('confValue');
          const iouSlider = document.getElementById('iouSlider');
          const iouValue = document.getElementById('iouValue');

          function updateConfig() {
            const obj_thresh = parseFloat(confSlider.value);
            const nms_thresh = parseFloat(iouSlider.value);

            confValue.innerText = obj_thresh.toFixed(2);
            iouValue.innerText = nms_thresh.toFixed(2);

            fetch('/api/config', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ obj_thresh, nms_thresh })
            });
          }

          confSlider.oninput = updateConfig;
          iouSlider.oninput = updateConfig;

          // Initialize
          fetch('/api/config').then(res => res.json()).then(data => {
            confSlider.value = data.obj_thresh;
            confValue.innerText = data.obj_thresh.toFixed(2);
            iouSlider.value = data.nms_thresh;
            iouValue.innerText = data.nms_thresh.toFixed(2);
          });
        </script>
      </body>
    </html>
    """, media_type="text/html")

def run_fastapi(host, port):
    # 打印所有注册的路由，用于调试 404 问题
    print("\n" + "="*50, flush=True)
    print("Registered Routes:", flush=True)
    for route in app.routes:
        if hasattr(route, "methods"):
            print(f"Path: {route.path:35} | Methods: {route.methods}", flush=True)
    print("="*50 + "\n", flush=True)
    sys.stdout.flush()

    # 将 log_level 改为 info 以便查看请求日志
    uvicorn.run(app, host=host, port=port, log_level="info", log_config=None)

# --- 推理逻辑 ---

def post_process_with_thresh(outputs, obj_thresh, nms_thresh):
    """Decode PP-YOLOE DFL outputs and apply class-aware NMS."""
    if outputs is None:
        return None, None, None

    if len(outputs) < 6 or len(outputs) % 3 != 0:
        raise ValueError(f"Unexpected PP-YOLOE output count: {len(outputs)}")
    output_per_branch = len(outputs) // 3
    decoded_boxes, class_scores = [], []
    for branch in range(3):
        base = branch * output_per_branch
        decoded_boxes.append(box_process(outputs[base]))
        class_scores.append(outputs[base + 1])

    def sp_flatten(_in):
        ch = _in.shape[1]
        _in = _in.transpose(0, 2, 3, 1)
        return _in.reshape(-1, ch)

    boxes = np.concatenate([sp_flatten(value) for value in decoded_boxes])
    class_scores = np.concatenate([sp_flatten(value) for value in class_scores])
    classes = np.argmax(class_scores, axis=-1)
    scores = class_scores[np.arange(class_scores.shape[0]), classes]
    keep = scores >= obj_thresh
    boxes, classes, scores = boxes[keep], classes[keep], scores[keep]
    if scores.size == 0:
        return None, None, None

    kept = []
    for class_id in np.unique(classes):
        indices = np.where(classes == class_id)[0]
        kept.extend(indices[nms_boxes(boxes[indices], scores[indices], nms_thresh)].tolist())
    if not kept:
        return None, None, None
    kept = np.asarray(kept, dtype=np.int64)
    return boxes[kept], classes[kept].astype(np.int64), scores[kept]

def nms_boxes(boxes, scores, threshold):
    x1, y1, x2, y2 = boxes.T
    areas = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        index = order[0]
        keep.append(index)
        xx1 = np.maximum(x1[index], x1[order[1:]])
        yy1 = np.maximum(y1[index], y1[order[1:]])
        xx2 = np.minimum(x2[index], x2[order[1:]])
        yy2 = np.minimum(y2[index], y2[order[1:]])
        intersection = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        union = areas[index] + areas[order[1:]] - intersection + 1e-6
        order = order[np.where(intersection / union <= threshold)[0] + 1]
    return np.asarray(keep, dtype=np.int64)

def box_process(position):
    grid_h, grid_w = position.shape[2:4]
    col, row = np.meshgrid(np.arange(0, grid_w), np.arange(0, grid_h))
    col = col.reshape(1, 1, grid_h, grid_w)
    row = row.reshape(1, 1, grid_h, grid_w)
    grid = np.concatenate((col, row), axis=1)
    stride = np.array([IMG_SIZE[1]//grid_h, IMG_SIZE[0]//grid_w]).reshape(1,2,1,1)

    n, c, h, w = position.shape
    bins = c // 4
    logits = position.reshape(n, 4, bins, h, w)
    exp_logits = np.exp(logits - np.max(logits, axis=2, keepdims=True))
    probabilities = exp_logits / np.sum(exp_logits, axis=2, keepdims=True)
    distances = np.sum(
        probabilities * np.arange(bins, dtype=np.float32).reshape(1, 1, bins, 1, 1),
        axis=2,
    )
    top_left = grid + 0.5 - distances[:, :2]
    bottom_right = grid + 0.5 + distances[:, 2:4]
    xyxy = np.concatenate((top_left * stride, bottom_right * stride), axis=1)
    return xyxy

# removed duplicated and unused post-processing functions

def post_process(input_data):
    obj, nms = det_config.get()
    return post_process_with_thresh(input_data, obj, nms)

def draw(image, boxes, scores, classes):
    for box, score, cl in zip(boxes, scores, classes):
        x1, y1, x2, y2 = [int(_b) for _b in box]
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(image, '{0} {1:.2f}'.format(CLASSES[cl], score),
                        (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

class RKNNLiteModel:
    def __init__(self, model_path, platform):
        if not RKNN_LITE_AVAILABLE:
            raise ImportError("RKNN-Toolkit-Lite2 is not available")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"RKNN model file not found: {model_path}")
        self.model_path = model_path
        self.platform = platform
        self.inference_lock = threading.Lock()
        self.rknn_lite = RKNNLite()
        print(f'Loading RKNN model from {model_path}...', flush=True)
        sys.stdout.flush()
        ret = self.rknn_lite.load_rknn(model_path)
        if ret != 0:
            raise Exception(f"Load RKNN model failed with error code: {ret}")
        print('Initializing runtime...', flush=True)
        sys.stdout.flush()
        core_mask = (
            RKNNLite.NPU_CORE_0_1
            if platform == "rk3576"
            else RKNNLite.NPU_CORE_0_1_2
        )
        ret = self.rknn_lite.init_runtime(core_mask=core_mask)
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
            with self.inference_lock:
                return self.rknn_lite.inference(inputs=[inputs])
        except Exception as e:
            print(f"Inference error: {e}")
            return None

    def release(self):
        if hasattr(self, 'rknn_lite'):
            self.rknn_lite.release()

def preprocess_frame(frame, co_helper):
    img = co_helper.letter_box(im=frame.copy(), new_shape=(IMG_SIZE[1], IMG_SIZE[0]), pad_color=(0,0,0))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def main():
    parser = argparse.ArgumentParser(description='Web PP-YOLOE detection on Rockchip NPU')
    parser.add_argument('--model_path', type=str, required=True, help='RKNN model path')
    parser.add_argument('--platform', choices=['rk3588', 'rk3576'], required=True, help='Target Rockchip platform')
    parser.add_argument('--camera_id', type=int, default=1, help='Camera device ID (default: 1 for /dev/video1)')
    parser.add_argument('--video_path', type=str, help='Path to video file (overrides camera_id)')
    parser.add_argument('--class_path', type=str, help='Path to class_config.txt file for dynamic category loading')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Web server host')
    parser.add_argument('--port', type=int, default=8000, help='Web server port')
    args = parser.parse_args()

    if not RKNN_LITE_AVAILABLE:
        print("Error: RKNN-Toolkit-Lite2 is not available.")
        return

    # 加载自定义类别
    if args.class_path:
        load_classes(args.class_path)

    global _global_model, _global_co_helper
    # 初始化模型
    model = RKNNLiteModel(args.model_path, args.platform)
    co_helper = COCO_test_helper(enable_letter_box=True)

    # 导出模型为全局变量并设置分析器
    _global_model = model
    _global_co_helper = co_helper
    video_analyzer.set_engine(model, co_helper)

    # 启动 Web 服务器线程
    web_thread = threading.Thread(target=run_fastapi, args=(args.host, args.port), daemon=True)
    web_thread.start()
    print(f"Web Preview started at http://{args.host}:{args.port}", flush=True)
    sys.stdout.flush()

    # 如果 camera_id 为 -1，则进入纯 Web 模式（视频文件分析模式）
    if args.camera_id == -1:
        print("Running in Video Analysis Mode. Access Web UI to process local videos.", flush=True)
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("Interrupted by user")
        finally:
            model.release()
        return

    # 打开视频源
    if args.video_path:
        cap = cv2.VideoCapture(args.video_path)
    else:
        cap = cv2.VideoCapture(args.camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print(f"Error: Cannot open video source (ID: {args.camera_id if not args.video_path else args.video_path})")
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
            processed_img = preprocess_frame(frame, co_helper)
            start_time = time.time()
            outputs = model.run(processed_img)
            inference_time = time.time() - start_time

            if outputs is not None:
                boxes, classes, scores = post_process(outputs)
                if boxes is not None:
                    real_boxes = co_helper.get_real_box(boxes)
                    draw(frame, real_boxes, scores, classes)

            # 计算并显示 FPS
            inf_fps = 1.0 / inference_time if inference_time > 0 else 0
            fps_counter = 0.9 * fps_counter + 0.1 * inf_fps if fps_counter > 0 else inf_fps
            cv2.putText(frame, f'NPU FPS: {fps_counter:.1f}', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # 更新 Web 帧缓冲区
            _, buffer = cv2.imencode('.jpg', frame)
            frame_buffer.set_frame(buffer.tobytes(), frame)

            # 降低 CPU 占用
            time.sleep(0.01)

    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        cap.release()
        model.release()

if __name__ == '__main__':
    main()
