import os
import cv2
import numpy as np

IMG_SIZE = (640, 640)

CLASSES = ("person", "bicycle", "car","motorbike ","aeroplane ","bus ","train","truck ","boat","traffic light",
           "fire hydrant","stop sign ","parking meter","bench","bird","cat","dog ","horse ","sheep","cow","elephant",
           "bear","zebra ","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite",
           "baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork","knife ",
           "spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza ","donut","cake","chair","sofa",
           "pottedplant","bed","diningtable","toilet ","tvmonitor","laptop\t","mouse\t","remote ","keyboard ","cell phone","microwave ",
           "oven ","toaster","sink","refrigerator ","book","clock","vase","scissors ","teddy bear ","hair drier", "toothbrush ")

class Colors:
    def __init__(self):
        hexs = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
                '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb(f'#{c}') for c in hexs]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))

class SegHelper:
    def __init__(self, enable_letter_box=True):
        self.enable_letter_box = enable_letter_box
        self.scale_ratio = 1.0
        self.pad_x = 0
        self.pad_y = 0

    def letterbox_resize(self, img, new_shape=(640, 640), bg_color=114):
        shape = img.shape[:2] 
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        self.scale_ratio = r

        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]

        dw /= 2
        dh /= 2

        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

        self.pad_x = left
        self.pad_y = top

        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(bg_color, bg_color, bg_color))
        return img

    def get_real_box(self, box):
        x1 = (box[0] - self.pad_x) / self.scale_ratio
        y1 = (box[1] - self.pad_y) / self.scale_ratio
        x2 = (box[2] - self.pad_x) / self.scale_ratio
        y2 = (box[3] - self.pad_y) / self.scale_ratio
        return [int(x1), int(y1), int(x2), int(y2)]

    def get_real_mask(self, mask_img, orig_shape):
        h, w = orig_shape[:2]
        unpad_h = int(IMG_SIZE[0] - 2 * self.pad_y)
        unpad_w = int(IMG_SIZE[1] - 2 * self.pad_x)
        
        # 安全切片，防止越界
        unpad_h = min(unpad_h, mask_img.shape[0] - self.pad_y)
        unpad_w = min(unpad_w, mask_img.shape[1] - self.pad_x)
        
        mask_unpad = mask_img[self.pad_y : self.pad_y + unpad_h, self.pad_x : self.pad_x + unpad_w]
        
        if mask_unpad.size == 0:
            return np.zeros((h, w), dtype=np.uint8)
            
        mask_real = cv2.resize(mask_unpad.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        return mask_real

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def filter_boxes(boxes, box_confidences, box_class_probs, seg_part, obj_thresh):
    box_confidences = box_confidences.reshape(-1)
    class_max_score = np.max(box_class_probs, axis=-1)
    classes = np.argmax(box_class_probs, axis=-1)

    _class_pos = np.where(class_max_score * box_confidences >= obj_thresh)
    scores = (class_max_score * box_confidences)[_class_pos]

    boxes = boxes[_class_pos]
    classes = classes[_class_pos]
    seg_part = (seg_part * box_confidences.reshape(-1, 1))[_class_pos]

    return boxes, classes, scores, seg_part

def dfl(position):
    n, c, h, w = position.shape
    p_num = 4
    mc = c // p_num
    y = position.reshape(n, p_num, mc, h, w)
    
    y_max = np.max(y, axis=2, keepdims=True)
    y_exp = np.exp(y - y_max)
    y_softmax = y_exp / np.sum(y_exp, axis=2, keepdims=True)
    
    acc_matrix = np.arange(mc, dtype=np.float32).reshape(1, 1, mc, 1, 1)
    y = np.sum(y_softmax * acc_matrix, axis=2)
    return y

def box_process(position):
    grid_h, grid_w = position.shape[2:4]
    col, row = np.meshgrid(np.arange(0, grid_w), np.arange(0, grid_h))
    col = col.reshape(1, 1, grid_h, grid_w)
    row = row.reshape(1, 1, grid_h, grid_w)
    grid = np.concatenate((col, row), axis=1)
    stride = np.array([IMG_SIZE[1]//grid_h, IMG_SIZE[0]//grid_w]).reshape(1, 2, 1, 1)

    position = dfl(position)
    box_xy  = grid + 0.5 - position[:, 0:2, :, :]
    box_xy2 = grid + 0.5 + position[:, 2:4, :, :]
    xyxy = np.concatenate((box_xy*stride, box_xy2*stride), axis=1)
    return xyxy

def crop_mask(masks, boxes):
    n, h, w = masks.shape
    x1, y1, x2, y2 = np.split(boxes[:, :, None], 4, 1) 
    r = np.arange(w, dtype=x1.dtype)[None, None, :] 
    c = np.arange(h, dtype=x1.dtype)[None, :, None] 
    return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))

def resize_masks(masks, target_shape):
    if len(masks.shape) == 2:
        masks = masks[None, :, :]
    n, h, w = masks.shape
    
    masks = masks.transpose((1, 2, 0))
    masks = cv2.resize(masks, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_LINEAR)
    if len(masks.shape) == 2:
        masks = masks[:, :, None]
    masks = masks.transpose((2, 0, 1))
    return masks

def nms_boxes(boxes, scores, iou_thresh):
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
        inds = np.where(ovr <= iou_thresh)[0]
        order = order[inds + 1]
    return np.array(keep)

def post_process_seg(input_data, obj_thresh=0.25, nms_thresh=0.45):
    proto = input_data[-1]
    boxes, scores, classes_conf, seg_part = [], [], [], []
    default_branch = 3
    pair_per_branch = len(input_data) // default_branch

    for i in range(default_branch):
        boxes.append(box_process(input_data[pair_per_branch*i]))
        classes_conf.append(input_data[pair_per_branch*i+1])
        scores.append(np.ones_like(input_data[pair_per_branch*i+1][:, :1, :, :], dtype=np.float32))
        seg_part.append(input_data[pair_per_branch*i+3])

    def sp_flatten(_in):
        ch = _in.shape[1]
        _in = _in.transpose(0, 2, 3, 1)
        return _in.reshape(-1, ch)

    boxes = np.concatenate([sp_flatten(_v) for _v in boxes])
    classes_conf = np.concatenate([sp_flatten(_v) for _v in classes_conf])
    scores = np.concatenate([sp_flatten(_v) for _v in scores])
    seg_part = np.concatenate([sp_flatten(_v) for _v in seg_part])

    boxes, classes, scores, seg_part = filter_boxes(boxes, scores, classes_conf, seg_part, obj_thresh)

    if len(boxes) == 0:
        return None, None, None, None

    sort_idx = np.argsort(scores)[::-1]
    boxes = boxes[sort_idx]
    classes = classes[sort_idx]
    scores = scores[sort_idx]
    seg_part = seg_part[sort_idx]

    keep = nms_boxes(boxes, scores, nms_thresh)
    
    MAX_DETECT = 300
    keep = keep[:MAX_DETECT]
    
    boxes = boxes[keep]
    classes = classes[keep]
    scores = scores[keep]
    seg_part = seg_part[keep]

    if len(classes) == 0:
        return None, None, None, None

    ph, pw = proto.shape[-2:]
    proto = proto.reshape(seg_part.shape[-1], -1)
    seg_img = np.matmul(seg_part, proto)
    seg_img = sigmoid(seg_img)
    seg_img = seg_img.reshape(-1, ph, pw)

    seg_threshold = 0.5
    seg_img = resize_masks(seg_img, IMG_SIZE)
    seg_img = crop_mask(seg_img, boxes)
    seg_img = seg_img > seg_threshold

    return boxes, classes, scores, seg_img

def draw_seg(image, boxes, classes, scores, seg_img, seg_helper):
    color = Colors()
    orig_shape = image.shape
    h, w = orig_shape[:2]
    
    mask_overlay = np.zeros_like(image)
    
    for i in range(len(boxes)):
        box = boxes[i]
        cl = classes[i]
        score = scores[i]
        seg = seg_img[i]
        
        real_box = seg_helper.get_real_box(box)
        top, left, right, bottom = real_box
        
        cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
        cv2.putText(image, '{0} {1:.2f}'.format(CLASSES[cl], score),
                    (top, left - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    
        real_mask = seg_helper.get_real_mask(seg, orig_shape)
        c = color(cl)
        mask_overlay[real_mask > 0] = c
        
    image = cv2.addWeighted(image, 1.0, mask_overlay, 0.5, 0)
    return image
