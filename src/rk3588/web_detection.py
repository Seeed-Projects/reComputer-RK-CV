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

app = FastAPI(title="reComputer RK-CV Web Preview")

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
        # RK3588 使用全核模式
        ret = self.rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)
        if ret != 0:
            raise Exception(f"Init runtime failed with error code: {ret}")

    def run(self, input_data):
        return self.rknn_lite.inference(inputs=[input_data])

def filter_boxes(boxes, box_confidences, box_class_probs):
    box_confidences = box_confidences.reshape(-1)
    class_max_score = np.max(box_class_probs, axis=-1)
    classes = np.argmax(box_class_probs, axis=-1)
    _class_pos = np.where(class_max_score * box_confidences >= OBJ_THRESH)
    scores = (class_max_score * box_confidences)[_class_pos]
    boxes = boxes[_class_pos]
    classes = classes[_class_pos]
    return boxes, classes, scores

def nms_boxes(boxes, scores):
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

def draw(image, boxes, scores, classes):
    for box, score, cl in zip(boxes, scores, classes):
        top, left, right, bottom = [int(_b) for _b in box]
        cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
        cv2.putText(image, '{0} {1:.2f}'.format(CLASSES[cl], score),
                    (top, left - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

class VideoStreamer:
    def __init__(self, model_path, source):
        self.model = RKNNLiteModel(model_path)
        self.cap = cv2.VideoCapture(source)
        self.co_helper = COCO_test_helper(IMG_SIZE)
        self.fps_counter = 0
        self.inference_time = 0
        self.lock = threading.Lock()

    def generate_frames(self):
        while True:
            success, frame = self.cap.read()
            if not success:
                # 视频文件循环播放
                if isinstance(self.cap, cv2.VideoCapture):
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break
            
            # 预处理
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, IMG_SIZE)
            input_data = np.expand_dims(img, axis=0)
            
            # 推理
            t1 = time.time()
            outputs = self.model.run(input_data)
            self.inference_time = time.time() - t1
            
            # 计算推理 FPS
            if self.inference_time > 0:
                inf_fps = 1.0 / self.inference_time
                self.fps_counter = 0.9 * self.fps_counter + 0.1 * inf_fps if self.fps_counter > 0 else inf_fps

            # 后处理
            input0_data = outputs[0].reshape([3, -1, IMG_SIZE[0] // 8, IMG_SIZE[1] // 8])
            input1_data = outputs[1].reshape([3, -1, IMG_SIZE[0] // 16, IMG_SIZE[1] // 16])
            input2_data = outputs[2].reshape([3, -1, IMG_SIZE[0] // 32, IMG_SIZE[1] // 32])
            
            input_data_list = [input0_data, input1_data, input2_data]
            boxes, classes, scores = self.co_helper.post_process(input_data_list)
            
            if boxes is not None:
                draw(frame, self.co_helper.get_real_box(boxes), scores, classes)
            
            # 绘制信息
            cv2.putText(frame, f'NPU FPS: {self.fps_counter:.1f}', (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f'Inference: {self.inference_time*1000:.1f}ms', (20, 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # 编码为 JPEG
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
      <head><title>reComputer RK-CV Web Preview</title></head>
      <body style="background-color: #1a1a1a; color: white; text-align: center; font-family: sans-serif;">
        <h1>reComputer RK-CV Real-time Detection</h1>
        <div style="margin: 20px auto; display: inline-block; border: 5px solid #333; border-radius: 10px; overflow: hidden;">
          <img src="/api/video_feed" style="max-width: 100%; height: auto;">
        </div>
        <p>Streaming via FastAPI + MJPEG</p>
      </body>
    </html>
    """, media_type="text/html")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RK3588 YOLO Web Detection")
    parser.add_argument('--model_path', type=str, default='model/yolo11n.rknn', help='path to rknn model')
    parser.add_argument('--source', type=str, default='0', help='camera id or video path')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='host address')
    parser.add_argument('--port', type=int, default=8000, help='port number')
    args = parser.parse_args()

    # 处理 source 参数
    if args.source.isdigit():
        source = int(args.source)
    else:
        source = args.source

    streamer = VideoStreamer(args.model_path, source)
    uvicorn.run(app, host=args.host, port=args.port)
