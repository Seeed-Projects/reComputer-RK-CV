from copy import copy
import os
import cv2
import numpy as np
import json

IMG_SIZE = (640, 640)
ANCHORS = None

CLASSES = ("person", "bicycle", "car","motorbike ","aeroplane ","bus ","train","truck ","boat","traffic light",
           "fire hydrant","stop sign ","parking meter","bench","bird","cat","dog ","horse ","sheep","cow","elephant",
           "bear","zebra ","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite",
           "baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork","knife ",
           "spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza ","donut","cake","chair","sofa",
           "pottedplant","bed","diningtable","toilet ","tvmonitor","laptop\t","mouse\t","remote ","keyboard ","cell phone","microwave ",
           "oven ","toaster","sink","refrigerator ","book","clock","vase","scissors ","teddy bear ","hair drier", "toothbrush ")

class Letter_Box_Info():
    def __init__(self, shape, new_shape, w_ratio, h_ratio, dw, dh, pad_color) -> None:
        self.origin_shape = shape
        self.new_shape = new_shape
        self.w_ratio = w_ratio
        self.h_ratio = h_ratio
        self.dw = dw
        self.dh = dh
        self.pad_color = pad_color

class COCO_test_helper():
    def __init__(self, enable_letter_box = False) -> None:
        self.record_list = []
        self.enable_ltter_box = enable_letter_box
        if self.enable_ltter_box is True:
            self.letter_box_info = None
        else:
            self.letter_box_info = None

    def letter_box(self, im, new_shape, pad_color=(0,0,0), info_need=False):
        # Resize and pad image while meeting stride-multiple constraints
        shape = im.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

        # Compute padding
        ratio = r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=pad_color)  # add border

        if self.enable_ltter_box is True:
            self.letter_box_info = Letter_Box_Info(shape, new_shape, ratio, ratio, dw, dh, pad_color)
        if info_need is True:
            return im, ratio, (dw, dh)
        else:
            return im

    def get_real_box(self, box, in_format='xyxy'):
        bbox = copy(box)
        if self.enable_ltter_box == True:
            if in_format=='xyxy':
                # YOLOv7 box_process 输出的是标准 xyxy 坐标
                bbox[:,0] -= self.letter_box_info.dw
                bbox[:,0] /= self.letter_box_info.w_ratio
                bbox[:,0] = np.clip(bbox[:,0], 0, self.letter_box_info.origin_shape[1])

                bbox[:,1] -= self.letter_box_info.dh
                bbox[:,1] /= self.letter_box_info.h_ratio
                bbox[:,1] = np.clip(bbox[:,1], 0, self.letter_box_info.origin_shape[0])

                bbox[:,2] -= self.letter_box_info.dw
                bbox[:,2] /= self.letter_box_info.w_ratio
                bbox[:,2] = np.clip(bbox[:,2], 0, self.letter_box_info.origin_shape[1])

                bbox[:,3] -= self.letter_box_info.dh
                bbox[:,3] /= self.letter_box_info.h_ratio
                bbox[:,3] = np.clip(bbox[:,3], 0, self.letter_box_info.origin_shape[0])

        return bbox

def get_real_box(src_shape, box, pad_color=0):
    """
    根据 letter_box 的反向操作计算回原图的坐标
    """
    # YOLOv7 输出的坐标是 [x1, y1, x2, y2]
    x1, y1, x2, y2 = box

    # 计算 ratio 和 pad
    ratio = min(IMG_SIZE[0] / src_shape[0], IMG_SIZE[1] / src_shape[1])
    new_unpad = int(round(src_shape[1] * ratio)), int(round(src_shape[0] * ratio))
    dw, dh = IMG_SIZE[1] - new_unpad[0], IMG_SIZE[0] - new_unpad[1]
    pad_w, pad_h = dw / 2, dh / 2

    # 反算坐标
    x1 = (x1 - pad_w) / ratio
    y1 = (y1 - pad_h) / ratio
    x2 = (x2 - pad_w) / ratio
    y2 = (y2 - pad_h) / ratio

    x1 = max(0, min(x1, src_shape[1] - 1))
    y1 = max(0, min(y1, src_shape[0] - 1))
    x2 = max(0, min(x2, src_shape[1] - 1))
    y2 = max(0, min(y2, src_shape[0] - 1))

    return [int(x1), int(y1), int(x2), int(y2)]

def load_anchors(path):
    global ANCHORS
    with open(path, 'r', encoding='utf-8') as f:
        values = [float(_v.strip()) for _v in f.readlines() if _v.strip()]
    ANCHORS = np.array(values, dtype=np.float32).reshape(3, -1, 2).tolist()
    return ANCHORS

def filter_boxes(boxes, box_confidences, box_class_probs, obj_thresh):
    """Filter boxes with object threshold."""
    box_confidences = box_confidences.reshape(-1)
    candidate, class_num = box_class_probs.shape
    class_max_score = np.max(box_class_probs, axis=-1)
    classes = np.argmax(box_class_probs, axis=-1)

    if class_num == 1:
        _class_pos = np.where(box_confidences >= obj_thresh)
        scores = box_confidences[_class_pos]
    else:
        _class_pos = np.where(class_max_score * box_confidences >= obj_thresh)
        scores = (class_max_score * box_confidences)[_class_pos]

    boxes = boxes[_class_pos]
    classes = classes[_class_pos]

    return boxes, classes, scores

def nms_boxes(boxes, scores, nms_thresh):
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

        # 修复除零错误：分母加一个很小的值防止除零
        ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        inds = np.where(ovr <= nms_thresh)[0]
        order = order[inds + 1]
    return np.array(keep)

def box_process(position, anchors):
    grid_h, grid_w = position.shape[2:4]
    col, row = np.meshgrid(np.arange(0, grid_w), np.arange(0, grid_h))
    col = col.reshape(1, 1, grid_h, grid_w)
    row = row.reshape(1, 1, grid_h, grid_w)
    grid = np.concatenate((col, row), axis=1)
    stride = np.array([IMG_SIZE[1]//grid_h, IMG_SIZE[0]//grid_w]).reshape(1,2,1,1)
    anchors = np.array(anchors, dtype=np.float32).reshape(-1, 2, 1, 1)
    grid = np.concatenate((col.repeat(len(anchors), axis=0), row.repeat(len(anchors), axis=0)), axis=1)
    box_xy = position[:, :2, :, :] * 2 - 0.5
    box_wh = np.power(position[:, 2:4, :, :] * 2, 2) * anchors
    box_xy = (box_xy + grid) * stride
    box = np.concatenate((box_xy, box_wh), axis=1)
    xyxy = np.copy(box)
    xyxy[:, 0, :, :] = box[:, 0, :, :] - box[:, 2, :, :] / 2
    xyxy[:, 1, :, :] = box[:, 1, :, :] - box[:, 3, :, :] / 2
    xyxy[:, 2, :, :] = box[:, 0, :, :] + box[:, 2, :, :] / 2
    xyxy[:, 3, :, :] = box[:, 1, :, :] + box[:, 3, :, :] / 2
    return xyxy

def post_process(input_data, obj_thresh=0.25, nms_thresh=0.45, anchors=None):
    if input_data is None:
        return None, None, None
    anchors = anchors or ANCHORS
    if anchors is None:
        raise ValueError("YOLOv7 anchors not loaded")

    boxes, scores, classes_conf = [], [], []
    input_data = [_in.reshape([len(anchors[0]), -1] + list(_in.shape[-2:])) for _in in input_data]
    for i in range(len(input_data)):
        boxes.append(box_process(input_data[i][:, :4, :, :], anchors[i]))
        scores.append(input_data[i][:, 4:5, :, :])
        classes_conf.append(input_data[i][:, 5:, :, :])

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
    boxes, classes, scores = filter_boxes(boxes, scores, classes_conf, obj_thresh)

    if len(classes) == 0:
        return None, None, None

    # nms
    nboxes, nclasses, nscores = [], [], []
    for c in set(classes):
        inds = np.where(classes == c)[0]
        b = boxes[inds]
        c_arr = classes[inds]
        s = scores[inds]
        keep = nms_boxes(b, s, nms_thresh)

        if len(keep) != 0:
            nboxes.append(b[keep])
            nclasses.append(c_arr[keep])
            nscores.append(s[keep])

    if not nclasses:
        return None, None, None

    boxes = np.concatenate(nboxes)
    classes = np.concatenate(nclasses)
    scores = np.concatenate(nscores)

    return boxes, classes, scores

def draw(image, boxes, scores, classes):
    for box, score, cl in zip(boxes, scores, classes):
        left, top, right, bottom = [int(_b) for _b in box]
        # 修复：原代码画框时误把 left, top 和 right, bottom 的坐标对应反了，应当是 (left, top), (right, bottom)
        cv2.rectangle(image, (left, top), (right, bottom), (255, 0, 0), 2)
        cv2.putText(image, '{0} {1:.2f}'.format(CLASSES[cl], score),
                    (left, top - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
