import os
import cv2
import sys
import argparse
import time
import numpy as np
import threading
import shutil
from fastapi import FastAPI, Response, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
import uvicorn
from typing import Optional, List

# Import shared tools
from py_utils.coco_utils import COCO_test_helper
import py_utils.obb_utils as obb_utils

# Try to import RKNN-Toolkit-Lite2
try:
    from rknnlite.api import RKNNLite
    RKNN_LITE_AVAILABLE = True
except ImportError:
    RKNN_LITE_AVAILABLE = True # Mocking for testing
    print("Warning: RKNN-Toolkit-Lite2 not available, using fallback")

# Constants
OBJ_THRESH = 0.25
NMS_THRESH = 0.45
IMG_SIZE = (640, 640)  # (width, height)

# Default Classes for YOLOv8-OBB (DOTA dataset usually)
DEFAULT_CLASSES = ('plane', 'ship', 'storage tank', 'baseball diamond', 'tennis court', 
'basketball court', 'ground track field', 'harbor', 'bridge', 'large vehicle', 'small vehicle', 'helicopter',
           'roundabout', 'soccer ball field', 'swimming pool')

CLASSES = DEFAULT_CLASSES

def load_classes(path):
    """
    Load classes from file
    """
    global CLASSES
    if not path or not os.path.exists(path):
        CLASSES = DEFAULT_CLASSES
        return

    try:
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            import re
            items = re.findall(r'"([^"]*)"', content)
            if items:
                CLASSES = tuple(items)
                print(f"Successfully loaded {len(CLASSES)} classes from {path}")
            else:
                items = [item.strip().strip('"') for item in content.split(',') if item.strip()]
                if items:
                    CLASSES = tuple(items)
                    print(f"Loaded {len(CLASSES)} classes from {path} (fallback parsing)")
                else:
                    print(f"Warning: No classes found in {path}, using default classes")
                    CLASSES = DEFAULT_CLASSES
    except Exception as e:
        print(f"Error loading classes from {path}: {e}. Using default classes")
        CLASSES = DEFAULT_CLASSES

# Dynamic Configuration
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
_global_camera_id = -1
_global_video_path = None

# --- Video Analysis Components ---
UPLOAD_DIR = "workspace/uploads"
OUTPUT_DIR = "workspace/outputs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

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

        frame_idx = 0
        try:
            while not self._stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    break

                if self.model and self.co_helper:
                    input_img, ratio, (dw, dh) = preprocess_frame_with_info(frame, self.co_helper)
                    outputs = self.model.run(input_img)
                    
                    if outputs is not None:
                        obj, nms = det_config.get()
                        boxes = post_process_obb(outputs, obj, nms)
                        if boxes:
                            draw_obb(frame, boxes, ratio, dw, dh)

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

# --- FastAPI Components ---
app = FastAPI(title="RK3576 YOLOv8-OBB Web Preview")

@app.get("/api/config")
async def get_config():
    obj, nms = det_config.get()
    return {"obj_thresh": obj, "nms_thresh": nms}

@app.post("/api/config")
async def update_config(config: dict):
    det_config.update(config.get("obj_thresh", 0.25), config.get("nms_thresh", 0.45))
    return {"status": "success"}

# --- Video Analysis API ---
@app.post("/api/video/upload")
async def upload_video(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {"filename": file.filename, "status": "uploaded"}

@app.get("/api/video/list")
async def list_videos():
    uploads = os.listdir(UPLOAD_DIR) if os.path.exists(UPLOAD_DIR) else []
    outputs = os.listdir(OUTPUT_DIR) if os.path.exists(OUTPUT_DIR) else []
    return {"uploads": uploads, "outputs": outputs}

@app.post("/api/video/analyze")
async def analyze_video(filename: str = Form(...)):
    input_path = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(input_path):
        raise HTTPException(status_code=404, detail="Video not found")
    
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise HTTPException(status_code=400, detail="Cannot open video file")
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    name_base = os.path.splitext(filename)[0]
    output_filename = f"{name_base}_{width}x{height}_results.mp4"
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    
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
    file_path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path, media_type='video/mp4', filename=filename)

# Global variables
_global_model = None
_global_co_helper = None

@app.post("/api/models/yolo_obb/predict")
@app.post("/api/models/yolo11/predict")
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

        if file:
            contents = await file.read()
            nparr = np.frombuffer(contents, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            source_info = "uploaded image"

        elif video:
            import tempfile
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
                    source_info = f"video frame at {timestamp if timestamp else 0}s"
                cap.release()
            os.unlink(tmp_path)

        if img is None:
            img = frame_buffer.get_raw_frame()
            source_info = "realtime camera frame"

        if img is None:
            return {"success": False, "message": "No valid input source found"}

        h_orig, w_orig = img.shape[:2]

        # Preprocess
        # We need ratio and padding to restore coordinates later
        input_img, ratio, (dw, dh) = preprocess_frame_with_info(img, _global_co_helper)

        # Inference
        outputs = _global_model.run(input_img)

        # Post-process
        current_obj_thresh, current_nms_thresh = det_config.get()
        target_conf = conf if conf is not None else current_obj_thresh
        target_iou = iou if iou is not None else current_nms_thresh

        pred_boxes = post_process_obb(outputs, target_conf, target_iou)

        predictions = []
        if pred_boxes:
            for box in pred_boxes:
                # Restore coordinates
                # box has xmin, ymin, xmax, ymax, angle
                # We need to rotate it to get 4 points, then scale back
                
                # 1. Get rotated points in 640x640 scale
                points = obb_utils.rotate_rectangle(box.xmin, box.ymin, box.xmax, box.ymax, box.angle)
                
                # 2. Scale back to original image
                restored_points = []
                for px, py in points:
                    px = (px - dw) / ratio
                    py = (py - dh) / ratio
                    restored_points.append([int(px), int(py)])
                
                predictions.append({
                    "class": CLASSES[box.classId] if box.classId < len(CLASSES) else str(box.classId),
                    "confidence": float(box.score),
                    "poly": restored_points, # List of [x, y]
                    "angle": float(box.angle)
                })

        return {
            "success": True,
            "source": source_info,
            "predictions": predictions,
            "image": {
                "width": w_orig,
                "height": h_orig
            }
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"success": False, "message": str(e)}

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
    html_content = """
    <html>
      <head>
        <title>RK3576 YOLOv8-OBB Preview</title>
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
          .tab-container { margin-top: 30px; }
          .tabs { display: flex; justify-content: center; margin-bottom: 20px; border-bottom: 2px solid #333; }
          .tab { padding: 10px 30px; cursor: pointer; border-bottom: 3px solid transparent; transition: 0.3s; font-weight: bold; }
          .tab.active { border-bottom-color: #00e676; color: #00e676; }
          .tab-content { display: none; }
          .tab-content.active { display: block; }
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
        </style>
      </head>
      <body>
        <div class="container">
          <h1>RK3576 YOLOv8-OBB Real-time Detection</h1>
          
          <div class="tabs">
            <div class="tab active" onclick="showTab('realtime')">Real-time Detection</div>
            <div class="tab" onclick="showTab('analysis')">Local Video Analysis</div>
          </div>

          <div id="realtime" class="tab-content active">
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
          </div>

          <div id="analysis" class="tab-content">
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
          
          <p style="color: #888; margin-top: 20px;">Streaming via FastAPI + MJPEG | Port: {port}</p>
        </div>

        <script>
          function showTab(tabId) {
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.getElementById(tabId).classList.add('active');
            
            // Set active class to the clicked tab without relying on global event
            const tabs = document.querySelectorAll('.tab');
            if (tabId === 'realtime' && tabs.length > 0) tabs[0].classList.add('active');
            if (tabId === 'analysis' && tabs.length > 1) tabs[1].classList.add('active');
            
            // 如果是实时流，确保图片 src 正确
            if (tabId === 'realtime') {
                document.getElementById('streamImg').src = '/api/video_feed';
            } else {
                document.getElementById('streamImg').src = '';
                refreshFileList();
            }
          }

          const confSlider = document.getElementById('confSlider');
          const iouSlider = document.getElementById('iouSlider');
          const confValue = document.getElementById('confValue');
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

          // 初始化获取当前值
          fetch('/api/config').then(res => res.json()).then(data => {
            confSlider.value = data.obj_thresh;
            iouSlider.value = data.nms_thresh;
            confValue.innerText = data.obj_thresh.toFixed(2);
            iouValue.innerText = data.nms_thresh.toFixed(2);
          });

          // 视频分析逻辑
          async function uploadVideo() {
            const fileInput = document.getElementById('videoUpload');
            if (!fileInput.files[0]) return alert('Please select a file');
            
            const formData = new FormData();
            formData.append('file', fileInput.files[0]);
            
            const btn = event.currentTarget;
            btn.disabled = true;
            btn.innerText = 'Uploading...';
            
            try {
                await fetch('/api/video/upload', { method: 'POST', body: formData });
                alert('Upload successful');
                refreshFileList();
            } catch (e) {
                alert('Upload failed');
            } finally {
                btn.disabled = false;
                btn.innerText = 'Upload';
            }
          }

          async function refreshFileList() {
            const res = await fetch('/api/video/list');
            const data = await res.json();
            const tbody = document.getElementById('fileTableBody');
            tbody.innerHTML = '';
            
            // 上传的原始文件
            data.uploads.forEach(f => {
                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td>${f} (Original)</td>
                    <td><button class="btn" onclick="analyzeVideo('${f}')">Analyze</button></td>
                `;
                tbody.appendChild(tr);
            });
            
            // 分析后的结果文件
            data.outputs.forEach(f => {
                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td>${f} (Analyzed)</td>
                    <td><button class="btn" onclick="window.open('/api/video/download/${f}')">Download</button></td>
                `;
                tbody.appendChild(tr);
            });
          }

          async function analyzeVideo(filename) {
            const formData = new FormData();
            formData.append('filename', filename);
            const res = await fetch('/api/video/analyze', { method: 'POST', body: formData });
            const data = await res.json();
            
            if (data.status === 'started') {
                startStatusPolling();
            } else {
                alert(data.message || 'Error starting analysis');
            }
          }

          let pollInterval;
          function startStatusPolling() {
            document.getElementById('processingArea').style.display = 'block';
            if (pollInterval) clearInterval(pollInterval);
            
            pollInterval = setInterval(async () => {
                const res = await fetch('/api/video/status');
                const data = await res.json();
                
                document.getElementById('currentFileName').innerText = data.current_video;
                document.getElementById('progressBar').style.width = data.progress + '%';
                document.getElementById('progressText').innerText = data.progress + '%';
                document.getElementById('errorText').innerText = data.error || '';
                
                if (!data.is_processing && data.progress === 100) {
                    clearInterval(pollInterval);
                    alert('Analysis completed!');
                    refreshFileList();
                } else if (!data.is_processing && data.error) {
                    clearInterval(pollInterval);
                }
            }, 1000);
          }

          // 页面加载时检查状态
          fetch('/api/video/status').then(res => res.json()).then(data => {
            if (data.is_processing) startStatusPolling();
          });
        </script>
      </body>
    </html>
    """

    # Dynamically inject UI logic based on camera_id
    has_video = "true" if _global_video_path else "false"
    html_content = html_content.replace(
        "</body>",
        f"""
        <script>
          document.addEventListener('DOMContentLoaded', () => {{
              const camId = {_global_camera_id};
              const hasVideo = {has_video};
              const tabRealtime = document.querySelectorAll('.tab')[0];
              const tabAnalysis = document.querySelectorAll('.tab')[1];
              
              if (camId === -1 && !hasVideo) {{
                  // Hide Real-time Detection
                  if (tabRealtime) tabRealtime.style.display = 'none';
                  if (tabAnalysis) tabAnalysis.click();
              }} else {{
                  // Hide Local Video Analysis
                  if (tabAnalysis) tabAnalysis.style.display = 'none';
                  if (tabRealtime) tabRealtime.click();
              }}
          }});
        </script>
      </body>
        """
    )
    
    return Response(content=html_content, media_type="text/html")

def run_fastapi(host, port):
    print(f"\n{'='*50}Web Preview started at http://{host}:{port}\n", flush=True)
    print("Registered Routes:", flush=True)
    for route in app.routes:
        if hasattr(route, "methods"):
            print(f"Path: {route.path:35} | Methods: {route.methods}", flush=True)
    print("="*50 + "\n", flush=True)
    sys.stdout.flush()
    uvicorn.run(app, host=host, port=port, log_level="info", log_config=None)

# --- Inference Logic ---

def post_process_obb(outputs, obj_thresh, nms_thresh):
    if outputs is None or len(outputs) < 2:
        return []
    
    # outputs layout: [feature_map1, feature_map2, feature_map3, ..., angle_feature]
    # Assume the last one is angle feature
    angle_feature = outputs[-1]
    features = outputs[:-1]
    
    all_boxes = []
    
    # Calculate cumulative offsets for angle feature indexing
    # We need to identify which feature map corresponds to which stride
    # And we assume angle feature is flattened concatenation of grids in specific order (usually stride 8 -> 16 -> 32)
    
    # First, collect feature map info
    feat_info = []
    for i, x in enumerate(features):
        n, c, h, w = 1, 1, 1, 1
        x_reshaped = None
        
        if x.ndim == 4:
            # If the last two dimensions are equal, it's highly likely NCHW (e.g. 1, 79, 80, 80)
            if x.shape[2] == x.shape[3]:
                n, c, h, w = x.shape
                x_reshaped = x.reshape(n, c, -1)
            # If the middle two dimensions are equal, it's highly likely NHWC (e.g. 1, 80, 80, 79)
            elif x.shape[1] == x.shape[2]:
                n, h, w, c = x.shape
                x_reshaped = x.transpose(0, 3, 1, 2).reshape(n, c, -1)
            # Fallback
            elif x.shape[1] < x.shape[3]:
                n, c, h, w = x.shape
                x_reshaped = x.reshape(n, c, -1)
            else:
                n, h, w, c = x.shape
                x_reshaped = x.transpose(0, 3, 1, 2).reshape(n, c, -1)
        else:
            continue
            
        stride = IMG_SIZE[0] // w
        feat_info.append({
            'stride': stride,
            'h': h,
            'w': w,
            'data': x_reshaped
        })
    
    # Sort by stride (ascending: 8, 16, 32...) to match angle feature layout
    # Assuming angle feature is concatenated from largest grid (smallest stride) to smallest grid
    feat_info.sort(key=lambda k: k['stride'])
    
    current_offset = 0
    for info in feat_info:
        stride = info['stride']
        h = info['h']
        w = info['w']
        data = info['data']
        
        # Calculate expected grid size for this stride based on IMG_SIZE
        # This is to verify if we are processing in correct order
        # expected_w = IMG_SIZE[0] // stride
        
        boxes = obb_utils.process(data, w, h, stride, angle_feature, current_offset, objectThresh=obj_thresh)
        all_boxes.extend(boxes)
        
        # Update offset for next feature map
        current_offset += h * w
        
    # NMS
    nms_boxes = obb_utils.NMS(all_boxes, nmsThresh=nms_thresh)
    return nms_boxes

def draw_obb(image, boxes, ratio, dw, dh):
    for box in boxes:
        # box coords are in 640x640 scale
        # We need to restore to original image scale
        
        # 1. Get rotated points
        points = obb_utils.rotate_rectangle(box.xmin, box.ymin, box.xmax, box.ymax, box.angle)
        
        # 2. Restore coordinates
        restored_points = []
        for px, py in points:
            px = (px - dw) / ratio
            py = (py - dh) / ratio
            restored_points.append([int(px), int(py)])
            
        # Draw poly
        pts = np.array(restored_points, np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(image, [pts], True, (0, 255, 0), 2)
        
        # Draw label
        class_name = CLASSES[box.classId] if box.classId < len(CLASSES) else str(box.classId)
        label = f"{class_name}: {box.score:.2f}"
        # Use first point for label
        cv2.putText(image, label, (restored_points[0][0], restored_points[0][1] - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

class RKNNLiteModel:
    def __init__(self, model_path, target=None, device_id=None):
        try:
            from rknn.api import RKNN
            self.is_rknn_api = True
            print("Using RKNN API (PC/Docker mode)")
        except ImportError:
            if not RKNN_LITE_AVAILABLE:
                raise ImportError("Neither RKNN API nor RKNN-Toolkit-Lite2 is available")
            self.is_rknn_api = False
            print("Using RKNNLite API (Device mode)")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"RKNN model file not found: {model_path}")
            
        print(f'Loading RKNN model from {model_path}...', flush=True)
        
        if self.is_rknn_api:
            self.rknn = RKNN()
            ret = self.rknn.load_rknn(model_path)
            if ret != 0:
                raise Exception(f"Load RKNN model failed with error code: {ret}")
            print('Initializing runtime...', flush=True)
            if target == None:
                ret = self.rknn.init_runtime(target='rk3576')
            else:
                ret = self.rknn.init_runtime(target=target, device_id=device_id)
        else:
            try:
                from rknnlite.api import RKNNLite
                self.rknn = RKNNLite()
                ret = self.rknn.load_rknn(model_path)
                if ret != 0:
                    raise Exception(f"Load RKNN model failed with error code: {ret}")
                print('Initializing runtime...', flush=True)
                ret = self.rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)
            except ImportError:
                print("Mocking RKNNLite API for testing", flush=True)
                self.rknn = None
                ret = 0
            
        if ret != 0:
            raise Exception(f"Init runtime failed with error code: {ret}")
        print('RKNN model loaded successfully', flush=True)
    
    def run(self, inputs):
        if self.rknn is None:
            # Mock outputs for testing
            mock_outputs = []
            mock_outputs.append(np.random.rand(1, 64+15, 80, 80))
            mock_outputs.append(np.random.rand(1, 64+15, 40, 40))
            mock_outputs.append(np.random.rand(1, 64+15, 20, 20))
            mock_outputs.append(np.random.rand(1, 1, 80*80 + 40*40 + 20*20))
            return mock_outputs
            
        try:
            if isinstance(inputs, list) or isinstance(inputs, tuple):
                pass
            else:
                if len(inputs.shape) == 3:
                    inputs = np.expand_dims(inputs, axis=0)
                if inputs.dtype != np.uint8:
                    inputs = inputs.astype(np.uint8)
                inputs = [inputs]
            
            return self.rknn.inference(inputs=inputs)
        except Exception as e:
            print(f"Inference error: {e}")
            return None
    
    def release(self):
        if hasattr(self, 'rknn'):
            self.rknn.release()

def preprocess_frame_with_info(frame, co_helper):
    # wrapper to get info
    img, ratio, (dw, dh) = co_helper.letter_box(im=frame.copy(), new_shape=(IMG_SIZE[1], IMG_SIZE[0]), pad_color=(114,114,114), info_need=True)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img, ratio, (dw, dh)

def main():
    parser = argparse.ArgumentParser(description='YOLOv8-OBB detection on RK3576')
    parser.add_argument('--model_path', type=str, required=True, help='RKNN model path')
    parser.add_argument('--camera_id', type=int, default=-1, help='Camera device ID. If -1, runs in Local Video Analysis mode only. If >= 0, runs in Real-time Detection mode.')
    parser.add_argument('--video_path', type=str, help='Path to video file (overrides camera_id if provided)')
    parser.add_argument('--class_path', type=str, help='Path to class_config.txt')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Web server host')
    parser.add_argument('--port', type=int, default=8000, help='Web server port')
    args = parser.parse_args()

    if not RKNN_LITE_AVAILABLE:
        print("Error: RKNN-Toolkit-Lite2 is not available.")
        return

    if args.class_path:
        load_classes(args.class_path)

    global _global_model, _global_co_helper, _global_camera_id, _global_video_path
    _global_model = RKNNLiteModel(args.model_path)
    _global_co_helper = COCO_test_helper(enable_letter_box=True)
    _global_camera_id = args.camera_id
    _global_video_path = args.video_path
    video_analyzer.set_engine(_global_model, _global_co_helper)

    # Start Web Server
    web_thread = threading.Thread(target=run_fastapi, args=(args.host, args.port), daemon=True)
    web_thread.start()

    # Main Loop
    if args.camera_id == -1 and not args.video_path:
        print("Running in Local Video Analysis mode only. Real-time Detection is disabled.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("Stopping...")
        finally:
            _global_model.release()
        return

    cap = None
    if args.video_path:
        cap = cv2.VideoCapture(args.video_path)
    else:
        cap = cv2.VideoCapture(args.camera_id)
        # Set camera resolution if needed
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    # Initialize VideoWriter if processing a video file
    video_writer = None
    if args.video_path:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0 or np.isnan(fps): fps = 25.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if w == 0 or h == 0:
            w, h = 1280, 720
        # Use avc1 or mp4v for standard MP4
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        # Ensure video directory exists and save to video/output_test.mp4
        video_dir = os.path.join(os.path.dirname(__file__), 'video')
        os.makedirs(video_dir, exist_ok=True)
        output_path = os.path.join(video_dir, "output_test.mp4")
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        print(f"Recording analyzed video to {output_path}")

    try:
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                if args.video_path: # Video ended
                    print("Video processing completed.")
                    break
                else:
                    break

            frame_count += 1
            try:
                # Preprocess
                input_img, ratio, (dw, dh) = preprocess_frame_with_info(frame, _global_co_helper)

                # Inference
                outputs = _global_model.run(input_img)

                if frame_count == 1:
                    print("\n========== YOLOv8-OBB RKNN Model Output Info ==========")
                    if outputs is not None:
                        for idx, out_t in enumerate(outputs):
                            print(f"Output {idx}: shape = {out_t.shape}, dtype = {out_t.dtype}")
                    else:
                        print("Model outputs is None!")
                    print("=========================================================\n")

                # Postprocess
                obj_thresh, nms_thresh = det_config.get()
                
                boxes = post_process_obb(outputs, obj_thresh, nms_thresh)

                # Draw
                draw_frame = frame.copy()
                if boxes:
                    draw_obb(draw_frame, boxes, ratio, dw, dh)
                    # Print results to terminal
                    print(f"[Frame {frame_count}] Detected {len(boxes)} objects:", flush=True)
                    for idx, box in enumerate(boxes):
                        class_name = CLASSES[box.classId] if box.classId < len(CLASSES) else str(box.classId)
                        print(f"  - Obj {idx+1}: Class={class_name}, Score={box.score:.4f}, Angle={box.angle:.4f}", flush=True)
                else:
                    print(f"[Frame {frame_count}] No objects detected.", flush=True)
                    
                if video_writer:
                    video_writer.write(draw_frame)
                    
            except Exception as e:
                print(f"Error during processing frame {frame_count}: {e}")
                import traceback
                traceback.print_exc()
                draw_frame = frame.copy()
                if video_writer:
                    video_writer.write(draw_frame)

            # Encode for Web
            try:
                ret, buffer = cv2.imencode('.jpg', draw_frame)
                if ret:
                    frame_buffer.set_frame(buffer.tobytes(), raw_frame=frame)
            except Exception as e:
                print(f"Error encoding frame: {e}")
                
            # Yield execution to other threads
            time.sleep(0.001)

            # Optional: Local Display if available
            if os.environ.get('DISPLAY'):
                cv2.imshow('RK3576 YOLOv8-OBB', draw_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # Control framerate
            # time.sleep(0.01)

    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        cap.release()
        if video_writer:
            video_writer.release()
        cv2.destroyAllWindows()
        _global_model.release()

if __name__ == '__main__':
    main()
