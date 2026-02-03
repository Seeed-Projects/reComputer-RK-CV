import os
import cv2
import sys
import argparse
import time
import numpy as np
import threading
from fastapi import FastAPI, Response, UploadFile, File, Form
from fastapi.responses import StreamingResponse
import uvicorn
from typing import Optional

# 导入共享工具
from py_utils.coco_utils import COCO_test_helper

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

CLASSES = ("person", "bicycle", "car","motorbike ","aeroplane ","bus ","train","truck ","boat","traffic light",
           "fire hydrant","stop sign ","parking meter","bench","bird","cat","dog ","horse ","sheep","cow","elephant",
           "bear","zebra ","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite",
           "baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork","knife ",
           "spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza ","donut","cake","chair","sofa",
           "pottedplant","bed","diningtable","toilet ","tvmonitor","laptop	","mouse	","remote ","keyboard ","cell phone","microwave ",
           "oven ","toaster","sink","refrigerator ","book","clock","vase","scissors ","teddy bear ","hair drier", "toothbrush ")

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

# --- FastAPI 核心组件 ---
app = FastAPI(title="reComputer RK-CV Combined Preview (RK3576)")

@app.get("/api/config")
async def get_config():
    obj, nms = det_config.get()
    return {"obj_thresh": obj, "nms_thresh": nms}

@app.post("/api/config")
async def update_config(config: dict):
    det_config.update(config.get("obj_thresh", 0.25), config.get("nms_thresh", 0.45))
    return {"status": "success"}

# 全局变量用于在 API 接口中访问模型和辅助类
_global_model = None
_global_co_helper = None

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
        input_img = preprocess_frame(img, _global_co_helper)

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
            for box, score, cl in zip(boxes, scores, classes):
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
    return Response(content="""
    <html>
      <head>
        <title>reComputer RK-CV Web Preview</title>
        <style>
          body { background-color: #1a1a1a; color: white; text-align: center; font-family: sans-serif; margin: 0; padding: 20px; }
          .container { max-width: 1000px; margin: 0 auto; }
          .video-box { margin: 20px auto; display: inline-block; border: 5px solid #333; border-radius: 10px; overflow: hidden; background: #000; }
          .controls { background: #2a2a2a; padding: 20px; border-radius: 10px; display: inline-block; text-align: left; min-width: 400px; }
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
          <h1>reComputer RK-CV Real-time Detection</h1>
          <div class="video-box">
            <img src="/api/video_feed" style="max-width: 100%; height: auto;">
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

# --- 原有推理逻辑 ---

# 为了支持单次推理的自定义阈值，我们提取核心逻辑
def post_process_with_thresh(outputs, obj_thresh, nms_thresh):
    """
    后处理逻辑，接受显式的阈值参数
    """
    if outputs is None:
        return None, None, None
    
    boxes, scores, classes_conf = [], [], []
    defualt_branch = 3
    pair_per_branch = len(outputs) // defualt_branch
    for i in range(defualt_branch):
        boxes.append(box_process(outputs[pair_per_branch * i]))
        classes_conf.append(outputs[pair_per_branch * i + 1])
        scores.append(np.ones_like(outputs[pair_per_branch * i + 1][:, :1, :, :], dtype=np.float32))

    def sp_flatten(_in):
        ch = _in.shape[1]
        _in = _in.transpose(0, 2, 3, 1)
        return _in.reshape(-1, ch)

    boxes = np.concatenate([sp_flatten(_v) for _v in boxes])
    classes_conf = np.concatenate([sp_flatten(_v) for _v in classes_conf])
    scores = np.concatenate([sp_flatten(_v) for _v in scores])

    # filter_boxes logic inline
    scores_flat = scores.reshape(-1)
    class_max_score = np.max(classes_conf, axis=-1)
    classes = np.argmax(classes_conf, axis=-1)
    _class_pos = np.where(class_max_score * scores_flat >= obj_thresh)
    
    scores = (class_max_score * scores_flat)[_class_pos]
    boxes = boxes[_class_pos]
    classes = classes[_class_pos]

    if len(classes) == 0:
        return None, None, None

    nboxes, nclasses, nscores = [], [], []
    for c in set(classes):
        inds = np.where(classes == c)[0]
        b = boxes[inds]
        s = scores[inds]
        
        # NMS core logic
        x = b[:, 0]
        y = b[:, 1]
        w = b[:, 2] - b[:, 0]
        h = b[:, 3] - b[:, 1]
        areas = w * h
        order = s.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x[i], x[order[1:]])
            yy1 = np.maximum(y[i], y[order[1:]])
            xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
            yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])
            w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
            h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
            inter = w1 * h1
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds_nms = np.where(ovr <= nms_thresh)[0]
            order = order[inds_nms + 1]
        
        if len(keep) > 0:
            nboxes.append(b[keep])
            nclasses.append(np.full(len(keep), c))
            nscores.append(s[keep])

    if not nboxes:
        return None, None, None
        
    return np.concatenate(nboxes), np.concatenate(nclasses), np.concatenate(nscores)

def filter_boxes(boxes, box_confidences, box_class_probs):
    """Filter boxes with object threshold. (Legacy wrapper using config)"""
    box_confidences = box_confidences.reshape(-1)
    class_max_score = np.max(box_class_probs, axis=-1)
    classes = np.argmax(box_class_probs, axis=-1)
    _class_pos = np.where(class_max_score* box_confidences >= det_config.obj_thresh)
    scores = (class_max_score* box_confidences)[_class_pos]
    boxes = boxes[_class_pos]
    classes = classes[_class_pos]
    return boxes, classes, scores

def nms_boxes(boxes, scores):
    """NMS wrapper using config"""
    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    areas = w * h
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x[i], x[order[1:]])
        yy1 = np.maximum(y[i], y[order[1:]])
        xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
        yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])
        w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
        h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
        inter = w1 * h1
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= det_config.nms_thresh)[0]
        order = order[inds + 1]
    return np.array(keep)

def dfl(position):
    n, c, h, w = position.shape
    p_num = 4
    mc = c // p_num
    y = position.reshape(n, p_num, mc, h, w)
    y_exp = np.exp(y - np.max(y, axis=2, keepdims=True))
    y_softmax = y_exp / np.sum(y_exp, axis=2, keepdims=True)
    acc_metrix = np.arange(mc).reshape(1, 1, mc, 1, 1).astype(np.float32)
    y = (y_softmax * acc_metrix).sum(2)
    return y

def box_process(position):
    grid_h, grid_w = position.shape[2:4]
    col, row = np.meshgrid(np.arange(0, grid_w), np.arange(0, grid_h))
    col = col.reshape(1, 1, grid_h, grid_w)
    row = row.reshape(1, 1, grid_h, grid_w)
    grid = np.concatenate((col, row), axis=1)
    stride = np.array([IMG_SIZE[1]//grid_h, IMG_SIZE[0]//grid_w]).reshape(1,2,1,1)
    position = dfl(position)
    box_xy  = grid +0.5 -position[:,0:2,:,:]
    box_xy2 = grid +0.5 +position[:,2:4,:,:]
    xyxy = np.concatenate((box_xy*stride, box_xy2*stride), axis=1)
    return xyxy

def post_process(input_data):
    obj, nms = det_config.get()
    return post_process_with_thresh(input_data, obj, nms)

def draw(image, boxes, scores, classes):
    for box, score, cl in zip(boxes, scores, classes):
        top, left, right, bottom = [int(_b) for _b in box]
        cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
        cv2.putText(image, '{0} {1:.2f}'.format(CLASSES[cl], score),
                    (top, left - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

class RKNNLiteModel:
    def __init__(self, model_path):
        if not RKNN_LITE_AVAILABLE:
            raise ImportError("RKNN-Toolkit-Lite2 is not available")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"RKNN model file not found: {model_path}")
        self.rknn_lite = RKNNLite()
        print(f'Loading RKNN model from {model_path}...', flush=True)
        sys.stdout.flush()
        ret = self.rknn_lite.load_rknn(model_path)
        if ret != 0:
            raise Exception(f"Load RKNN model failed with error code: {ret}")
        print('Initializing runtime...', flush=True)
        sys.stdout.flush()
        ret = self.rknn_lite.init_runtime()
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
    parser = argparse.ArgumentParser(description='Real-time object detection on RK3576')
    parser.add_argument('--model_path', type=str, required=True, help='RKNN model path')
    parser.add_argument('--camera_id', type=int, default=1, help='Camera device ID (default: 1 for /dev/video1)')
    parser.add_argument('--video_path', type=str, help='Path to video file (overrides camera_id)')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Web server host')
    parser.add_argument('--port', type=int, default=8000, help='Web server port')
    parser.add_argument('--no_gui', action='store_true', help='Disable local GUI window')
    args = parser.parse_args()

    if not RKNN_LITE_AVAILABLE:
        print("Error: RKNN-Toolkit-Lite2 is not available.")
        return

    # 启动 Web 服务器线程
    web_thread = threading.Thread(target=run_fastapi, args=(args.host, args.port), daemon=True)
    web_thread.start()
    print(f"Web Preview started at http://{args.host}:{args.port}", flush=True)
    sys.stdout.flush()

    global _global_model, _global_co_helper
    # 初始化模型
    model = RKNNLiteModel(args.model_path)
    _global_model = model
    co_helper = COCO_test_helper(enable_letter_box=True)
    _global_co_helper = co_helper

    # 打开视频源
    if args.video_path:
        cap = cv2.VideoCapture(args.video_path)
    else:
        cap = cv2.VideoCapture(args.camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    if not cap.isOpened():
        print("Error: Cannot open video source")
        return

    # GUI 状态
    show_gui = True
    window_name = 'RK3576 Real-time Detection'
    if args.no_gui:
        print("GUI disabled by --no_gui argument. Running in Web-only mode.")
        show_gui = False
    elif os.environ.get('DISPLAY') is None:
        print("No DISPLAY detected, running in Web-only mode.")
        show_gui = False
    else:
        try:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 1280, 720)
        except Exception as e:
            print(f"Failed to initialize GUI window: {e}. Falling back to Web-only mode.")
            show_gui = False

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
                    draw(frame, co_helper.get_real_box(boxes), scores, classes)

            # 计算并显示 FPS
            inf_fps = 1.0 / inference_time if inference_time > 0 else 0
            fps_counter = 0.9 * fps_counter + 0.1 * inf_fps if fps_counter > 0 else inf_fps
            cv2.putText(frame, f'NPU FPS: {fps_counter:.1f}', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # 更新 Web 帧缓冲区
            _, buffer = cv2.imencode('.jpg', frame)
            frame_buffer.set_frame(buffer.tobytes(), frame)

            # 本地显示 (仅当有屏幕时)
            if show_gui:
                try:
                    cv2.imshow(window_name, frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                except Exception:
                    show_gui = False
                    print("GUI display failed during runtime, switching to Web-only mode.")
            else:
                time.sleep(0.01)

    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        model.release()

if __name__ == '__main__':
    main()
