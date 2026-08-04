from copy import copy
import os
import cv2
import numpy as np
import json

IMG_SIZE = (640, 640)

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
                # 修复 YOLOv6 返回坐标错乱问题：
                # yolo v6 的 box_process 输出的是 (x1, y1, x2, y2)
                # 而之前这里错当成了 (y1, x1, y2, x2) 进行坐标偏移修正，这会导致边界框全部跑到画面之外（变成0或极大值）
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
    # YOLOv6 输出的坐标是 [x1, y1, x2, y2]
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

def filter_boxes(boxes, box_confidences, box_class_probs, obj_thresh):
    """Filter boxes with object threshold."""
    box_confidences = box_confidences.reshape(-1)
    class_max_score = np.max(box_class_probs, axis=-1)
    classes = np.argmax(box_class_probs, axis=-1)

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

def dfl(position):
    # Distribution Focal Loss (DFL) - numpy implementation
    n, c, h, w = position.shape
    p_num = 4
    mc = c // p_num
    y = position.reshape(n, p_num, mc, h, w)

    # 原版 torch 实现： y = y.softmax(2)
    # Numpy 的 softmax，必须确保分母不会变成 NaN
    y_exp = np.exp(y - np.max(y, axis=2, keepdims=True))
    y_softmax = y_exp / np.sum(y_exp, axis=2, keepdims=True)

    # 原版 torch 实现：acc_metrix = torch.tensor(range(mc)).float().reshape(1,1,mc,1,1)
    acc_metrix = np.arange(mc, dtype=np.float32).reshape(1, 1, mc, 1, 1)
    # 原版 torch 实现：y = (y*acc_metrix).sum(2)
    y = (y_softmax * acc_metrix).sum(axis=2)
    return y

def box_process(position):
    grid_h, grid_w = position.shape[2:4]
    col, row = np.meshgrid(np.arange(0, grid_w), np.arange(0, grid_h))
    col = col.reshape(1, 1, grid_h, grid_w)
    row = row.reshape(1, 1, grid_h, grid_w)
    grid = np.concatenate((col, row), axis=1)
    stride = np.array([IMG_SIZE[1]//grid_h, IMG_SIZE[0]//grid_w]).reshape(1,2,1,1)

    if position.shape[1] == 4:
        box_xy  = grid + 0.5 - position[:,0:2,:,:]
        box_xy2 = grid + 0.5 + position[:,2:4,:,:]
        xyxy = np.concatenate((box_xy*stride, box_xy2*stride), axis=1)
    else:
        position_decoded = dfl(position)
        # 修复：原版代码中，如果进入了 else 分支，实际上也是减去和加上 0:2 和 2:4。
        # 但是！如果模型输出的 4 个坐标本来就是相对中心的 (left, top, right, bottom)，
        # 那么 x1 (left) 和 y1 (top) 是需要被中心点减去的，
        # 而 x2 (right) 和 y2 (bottom) 是需要加上中心点的。
        # 上述逻辑没有问题，问题可能出在 dfl 返回的 tensor 布局上。
        # 这里保证与案例完全一致：
        box_xy  = grid + 0.5 - position_decoded[:,0:2,:,:]
        box_xy2 = grid + 0.5 + position_decoded[:,2:4,:,:]
        xyxy = np.concatenate((box_xy*stride, box_xy2*stride), axis=1)

    return xyxy

def post_process(input_data, obj_thresh=0.25, nms_thresh=0.45):
    if input_data is None:
        return None, None, None

    boxes, scores, classes_conf = [], [], []

    # 检查是否为 9 个输出的模型 (box, cls, obj_score 各 3 个尺度)
    if len(input_data) == 9:
        for i in range(3):
            # i*3   : box_xyxy (1, 4, H, W)
            # i*3+1 : class_probs (1, 80, H, W)
            # i*3+2 : obj_scores (1, 1, H, W)
            box_tensor = input_data[i*3]
            cls_tensor = input_data[i*3+1]
            obj_tensor = input_data[i*3+2]

            boxes.append(box_process(box_tensor))
            classes_conf.append(cls_tensor)
            scores.append(obj_tensor)
    else:
        # 兼容旧版本只有 6 个输出的情况
        default_branch = 3
        pair_per_branch = len(input_data) // default_branch
        for i in range(default_branch):
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
