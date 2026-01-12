import os
import cv2
import sys
import argparse
import time
import numpy as np

# add path
realpath = os.path.abspath(__file__)
_sep = os.path.sep
realpath = realpath.split(_sep)
#sys.path.append(os.path.join(realpath[0]+_sep, *realpath[1:realpath.index('rknn_model_zoo')+1]))

from py_utils.coco_utils import COCO_test_helper

# 导入RKNN-Toolkit-Lite2
try:
    from rknnlite.api import RKNNLite
    RKNN_LITE_AVAILABLE = True
except ImportError:
    RKNN_LITE_AVAILABLE = False
    print("Warning: RKNN-Toolkit-Lite2 not available, using fallback")

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

def filter_boxes(boxes, box_confidences, box_class_probs):
    """Filter boxes with object threshold."""
    box_confidences = box_confidences.reshape(-1)
    candidate, class_num = box_class_probs.shape

    class_max_score = np.max(box_class_probs, axis=-1)
    classes = np.argmax(box_class_probs, axis=-1)

    _class_pos = np.where(class_max_score* box_confidences >= OBJ_THRESH)
    scores = (class_max_score* box_confidences)[_class_pos]

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
    keep = np.array(keep)
    return keep

def dfl(position):
    # Distribution Focal Loss (DFL)
    # 移除torch依赖，使用numpy实现
    n, c, h, w = position.shape
    p_num = 4
    mc = c // p_num
    
    # 使用numpy实现softmax
    y = position.reshape(n, p_num, mc, h, w)
    # 计算softmax
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
    # 添加None检查
    if input_data is None:
        return None, None, None
        
    boxes, scores, classes_conf = [], [], []
    defualt_branch=3
    pair_per_branch = len(input_data)//defualt_branch
    
    for i in range(defualt_branch):
        boxes.append(box_process(input_data[pair_per_branch*i]))
        classes_conf.append(input_data[pair_per_branch*i+1])
        scores.append(np.ones_like(input_data[pair_per_branch*i+1][:,:1,:,:], dtype=np.float32))

    def sp_flatten(_in):
        ch = _in.shape[1]
        _in = _in.transpose(0,2,3,1)
        return _in.reshape(-1, ch)

    boxes = [sp_flatten(_v) for _v in boxes]
    classes_conf = [sp_flatten(_v) for _v in classes_conf]
    scores = [sp_flatten(_v) for _v in scores]

    boxes = np.concatenate(boxes)
    classes_conf = np.concatenate(classes_conf)
    scores = np.concatenate(scores)

    # filter according to threshold
    boxes, classes, scores = filter_boxes(boxes, scores, classes_conf)

    # nms
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
        # 移除控制台打印，这是提升性能的关键
        # print("%s @ (%d %d %d %d) %.3f" % (CLASSES[cl], top, left, right, bottom, score))
        cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
        cv2.putText(image, '{0} {1:.2f}'.format(CLASSES[cl], score),
                    (top, left - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

class RKNNLiteModel:
    """使用RKNN-Toolkit-Lite2的模型容器"""
    def __init__(self, model_path):
        if not RKNN_LITE_AVAILABLE:
            raise ImportError("RKNN-Toolkit-Lite2 is not available")
        
        # 检查模型文件是否存在
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"RKNN model file not found: {model_path}")
            
        self.rknn_lite = RKNNLite()
        
        # 加载RKNN模型
        print(f'Loading RKNN model from {model_path}...')
        ret = self.rknn_lite.load_rknn(model_path)
        if ret != 0:
            raise Exception(f"Load RKNN model failed with error code: {ret}")
        
        # 初始化运行时环境
        print('Initializing runtime...')
        # RK3588 has 3 NPU cores. 
        # Using NPU_CORE_0_1_2 can utilize all 3 cores for better performance
        ret = self.rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)
        if ret != 0:
            raise Exception(f"Init runtime environment failed with error code: {ret}")
        
        print('RKNN model loaded successfully')
    
    def run(self, inputs):
        """运行推理"""
        try:
            # 确保输入是4维的 (batch, height, width, channels)
            if len(inputs.shape) == 3:
                inputs = np.expand_dims(inputs, axis=0)  # 从(H,W,C)变为(1,H,W,C)
            
            # 确保数据类型正确
            if inputs.dtype != np.uint8:
                inputs = inputs.astype(np.uint8)
                
            outputs = self.rknn_lite.inference(inputs=[inputs])
            return outputs
        except Exception as e:
            print(f"Inference error: {e}")
            return None
    
    def release(self):
        """释放资源"""
        if hasattr(self, 'rknn_lite'):
            self.rknn_lite.release()

def setup_model(model_path):
    """初始化RKNN模型"""
    if model_path.endswith('.rknn'):
        if not RKNN_LITE_AVAILABLE:
            raise ImportError("RKNN-Toolkit-Lite2 is not installed. Please install it first.")
        
        model = RKNNLiteModel(model_path)
        print('Model-{} is RKNN model, starting camera inference'.format(model_path))
        return model, 'rknn'
    else:
        raise ValueError("Only RKNN model is supported for on-board inference")

def preprocess_frame(frame, co_helper):
    """预处理摄像头帧"""
    # Due to rga init with (0,0,0), we using pad_color (0,0,0) instead of (114, 114, 114)
    pad_color = (0,0,0)
    img = co_helper.letter_box(im=frame.copy(), new_shape=(IMG_SIZE[1], IMG_SIZE[0]), pad_color=(0,0,0))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def main():
    parser = argparse.ArgumentParser(description='Real-time object detection on RK3588')
    parser.add_argument('--model_path', type=str, required=True, help='RKNN model path')
    parser.add_argument('--camera_id', type=int, default=1, help='Camera device ID (default: 1 for /dev/video1)')
    parser.add_argument('--video_path', type=str, help='Path to video file (overrides camera_id)')
    parser.add_argument('--fps', type=int, default=30, help='Target FPS for display')
    args = parser.parse_args()

    # 检查RKNN-Lite2是否可用
    if not RKNN_LITE_AVAILABLE:
        print("Error: RKNN-Toolkit-Lite2 is not available.")
        print("Please install it using: sudo apt install python3-rknnlite2")
        return

    # 初始化模型
    model, platform = setup_model(args.model_path)
    co_helper = COCO_test_helper(enable_letter_box=True)

    # 打开视频源
    if args.video_path:
        if not os.path.exists(args.video_path):
            print(f"Error: Video file not found: {args.video_path}")
            return
        print(f"Opening video file: {args.video_path}")
        cap = cv2.VideoCapture(args.video_path)
    else:
        # 打开摄像头
        cap = cv2.VideoCapture(args.camera_id)
        if not cap.isOpened():
            print("Error: Cannot open camera /dev/video{}".format(args.camera_id))
            # 尝试其他摄像头
            for i in [0, 2, 3]:
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    print(f"Using camera /dev/video{i}")
                    break
            else:
                print("Error: No camera found")
                return
        
        # 设置摄像头分辨率（仅对摄像头有效）
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    print("Starting real-time detection... Press 'q' to quit, 's' to save current frame")

    # 初始化预览窗口
    window_name = 'RK3588 Real-time Detection'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)
    
    fps_counter = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                if args.video_path:
                    print("End of video file")
                else:
                    print("Error: Cannot read frame from camera")
                break

            # 预处理
            processed_img = preprocess_frame(frame, co_helper)
            input_data = processed_img

            # 推理
            start_inference_time = time.time()
            outputs = model.run(input_data)
            inference_time = time.time() - start_inference_time
            
            # 计算基于推理时间的理论 FPS
            if inference_time > 0:
                inf_fps = 1.0 / inference_time
            else:
                inf_fps = 0
            
            # 使用滑动平均使推理 FPS 显示更平滑
            fps_counter = 0.9 * fps_counter + 0.1 * inf_fps if fps_counter > 0 else inf_fps
            
            # 检查推理结果
            if outputs is None:
                print("Inference failed, skipping frame")
                continue

            # 后处理
            boxes, classes, scores = post_process(outputs)

            # 绘制结果
            if boxes is not None:
                draw(frame, co_helper.get_real_box(boxes), scores, classes)

            # 在画面上显示信息
            cv2.putText(frame, f'Inference FPS: {fps_counter:.1f}', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f'Inference: {inference_time*1000:.1f}ms', (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            if boxes is not None:
                cv2.putText(frame, f'Objects: {len(boxes)}', (10, 110), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(frame, 'Objects: 0', (10, 110), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # 显示结果
            cv2.imshow(window_name, frame)

            # 按键处理
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # 保存当前帧
                timestamp = int(time.time())
                filename = f'capture_{timestamp}.jpg'
                cv2.imwrite(filename, frame)
                print(f"Frame saved as {filename}")

    except KeyboardInterrupt:
        print("Interrupted by user")
    except Exception as e:
        print(f"Error during processing: {e}")

    finally:
        # 释放资源
        cap.release()
        cv2.destroyAllWindows()
        model.release()
        print("Resources released")

if __name__ == '__main__':
    main()