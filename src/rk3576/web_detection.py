import os
import cv2
import sys
import argparse
import time
import numpy as np
from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse
import uvicorn
import threading

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
IMG_SIZE = (640, 640)
CLASSES = ("person", "bicycle", "car","motorbike ","aeroplane ","bus ","train","truck ","boat","traffic light",
           "fire hydrant","stop sign ","parking meter","bench","bird","cat","dog ","horse ","sheep","cow","elephant",
           "bear","zebra ","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite",
           "baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork","knife ",
           "spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza ","donut","cake","chair","sofa",
           "pottedplant","bed","diningtable","toilet ","tvmonitor","laptop	","mouse	","remote ","keyboard ","cell phone","microwave ",
           "oven ","toaster","sink","refrigerator ","book","clock","vase","scissors ","teddy bear ","hair drier", "toothbrush ")

app = FastAPI(title="reComputer RK3576 Web Preview")

def filter_boxes(boxes, box_confidences, box_class_probs):
    """Filter boxes with object threshold."""
    box_confidences = box_confidences.reshape(-1)
    class_max_score = np.max(box_class_probs, axis=-1)
    classes = np.argmax(box_class_probs, axis=-1)
    _class_pos = np.where(class_max_score * box_confidences >= OBJ_THRESH)
    scores = (class_max_score * box_confidences)[_class_pos]
    boxes = boxes[_class_pos]
    classes = classes[_class_pos]
    return boxes, classes, scores

def nms_boxes(boxes, scores):
    """Suppress non-maximal boxes."""
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
        inds = np.where(ovr <= NMS_THRESH)[0]
        order = order[inds + 1]
    return keep

def dfl(position):
    # Distribution Focal Loss (DFL)
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
    box_xy  = grid + 0.5 - position[:,0:2,:,:]
    box_xy2 = grid + 0.5 + position[:,2:4,:,:]
    xyxy = np.concatenate((box_xy*stride, box_xy2*stride), axis=1)
    return xyxy

def post_process(input_data):
    if input_data is None:
        return None, None, None
    boxes, scores, classes_conf = [], [], []
    defualt_branch = 3
    pair_per_branch = len(input_data) // defualt_branch
    for i in range(defualt_branch):
        boxes.append(box_process(input_data[pair_per_branch*i]))
        classes_conf.append(input_data[pair_per_branch*i+1])
        scores.append(np.ones_like(input_data[pair_per_branch*i+1][:,:1,:,:], dtype=np.float32))

    def sp_flatten(_in):
        ch = _in.shape[1]
        _in = _in.transpose(0, 2, 3, 1)
        return _in.reshape(-1, ch)

    boxes = [sp_flatten(_v) for _v in boxes]
    classes_conf = [sp_flatten(_v) for _v in classes_conf]
    scores = [sp_flatten(_v) for _v in scores]
    boxes = np.concatenate(boxes)
    classes_conf = np.concatenate(classes_conf)
    scores = np.concatenate(scores)
    boxes, classes, scores = filter_boxes(boxes, scores, classes_conf)
    nboxes, nclasses, nscores = [], [], []
    for c in set(classes):
        inds = np.where(classes == c)
        b = boxes[inds]
        c = classes[inds]
        s = scores[inds]
        keep = nms_boxes(b, s)
        if len(keep) != 0:
            nboxes.append(b[keep])
            nclasses.append(c[keep])
            nscores.append(s[keep])
    if not nclasses and not nscores:
        return None, None, None
    boxes = np.concatenate(nboxes)
    classes = np.concatenate(nclasses)
    scores = np.concatenate(nscores)
    return boxes, classes, scores

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
        print('Loading RKNN model...')
        ret = self.rknn_lite.load_rknn(model_path)
        if ret != 0:
            raise Exception(f"Load RKNN model failed with error code: {ret}")
        print('Initializing runtime...')
        # RK3576 使用默认初始化
        ret = self.rknn_lite.init_runtime()
        if ret != 0:
            raise Exception(f"Init runtime failed with error code: {ret}")

    def run(self, input_data):
        return self.rknn_lite.inference(inputs=[input_data])

class VideoStreamer:
    def __init__(self, model_path, source):
        self.model = RKNNLiteModel(model_path)
        self.cap = cv2.VideoCapture(source)
        self.co_helper = COCO_test_helper(enable_letter_box=True)
        self.fps_counter = 0
        self.inference_time = 0
        self.lock = threading.Lock()

    def generate_frames(self):
        while True:
            success, frame = self.cap.read()
            if not success:
                if isinstance(self.cap, cv2.VideoCapture):
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break
            
            # 预处理
            img = self.co_helper.letter_box(im=frame.copy(), new_shape=(IMG_SIZE[1], IMG_SIZE[0]), pad_color=(0,0,0))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            input_data = np.expand_dims(img, axis=0)
            
            # 推理
            t1 = time.time()
            outputs = self.model.run(input_data)
            self.inference_time = time.time() - t1
            
            if self.inference_time > 0:
                inf_fps = 1.0 / self.inference_time
                self.fps_counter = 0.9 * self.fps_counter + 0.1 * inf_fps if self.fps_counter > 0 else inf_fps

            # 后处理
            boxes, classes, scores = post_process(outputs)
            
            if boxes is not None:
                draw(frame, self.co_helper.get_real_box(boxes), scores, classes)
            
            # 绘制信息
            cv2.putText(frame, f'NPU FPS: {self.fps_counter:.1f}', (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f'Inference: {self.inference_time*1000:.1f}ms', (20, 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

streamer = None

@app.get("/api/video_feed")
async def video_feed():
    return StreamingResponse(streamer.generate_frames(), 
                            media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/")
async def index():
    return Response(content="""
    <html>
      <head><title>reComputer RK3576 Web Preview</title></head>
      <body style="background-color: #1a1a1a; color: white; text-align: center; font-family: sans-serif;">
        <h1>reComputer RK3576 Real-time Detection</h1>
        <div style="margin: 20px auto; display: inline-block; border: 5px solid #333; border-radius: 10px; overflow: hidden;">
          <img src="/api/video_feed" style="max-width: 100%; height: auto;">
        </div>
        <p>Streaming via FastAPI + MJPEG</p>
      </body>
    </html>
    """, media_type="text/html")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RK3576 YOLO Web Detection")
    parser.add_argument('--model_path', type=str, default='model/yolo11n.rknn', help='path to rknn model')
    parser.add_argument('--source', type=str, default='1', help='camera id or video path')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='host address')
    parser.add_argument('--port', type=int, default=8000, help='port number')
    args = parser.parse_args()

    if args.source.isdigit():
        source = int(args.source)
    else:
        source = args.source

    streamer = VideoStreamer(args.model_path, source)
    uvicorn.run(app, host=args.host, port=args.port)
