import numpy as np
import math
import cv2
try:
    from shapely.geometry import Polygon
    SHAPELY_AVAILABLE = True
except ImportError:
    SHAPELY_AVAILABLE = False
    print("Warning: shapely not available, OBB NMS may fail or use fallback")

class DetectBox:
    def __init__(self, classId, score, xmin, ymin, xmax, ymax, angle):
        self.classId = classId
        self.score = score
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.angle = angle

def rotate_rectangle(x1, y1, x2, y2, a):
    # Calculate center point
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    # Rotate each vertex
    # Note: angle 'a' is in radians
    x1_new = int((x1 - cx) * math.cos(a) - (y1 - cy) * math.sin(a) + cx)
    y1_new = int((x1 - cx) * math.sin(a) + (y1 - cy) * math.cos(a) + cy)

    x2_new = int((x2 - cx) * math.cos(a) - (y2 - cy) * math.sin(a) + cx)
    y2_new = int((x2 - cx) * math.sin(a) + (y2 - cy) * math.cos(a) + cy)

    x3_new = int((x1 - cx) * math.cos(a) - (y2 - cy) * math.sin(a) + cx)
    y3_new = int((x1 - cx) * math.sin(a) + (y2 - cy) * math.cos(a) + cy)

    x4_new = int((x2 - cx) * math.cos(a) - (y1 - cy) * math.sin(a) + cx)
    y4_new = int((x2 - cx) * math.sin(a) + (y1 - cy) * math.cos(a) + cy)

    return [(x1_new, y1_new), (x3_new, y3_new), (x2_new, y2_new), (x4_new, y4_new)]

def intersection(g, p):
    if not SHAPELY_AVAILABLE:
        return 0 # Fallback or error
    g = np.asarray(g)
    p = np.asarray(p)
    g = Polygon(g[:8].reshape((4, 2)))
    p = Polygon(p[:8].reshape((4, 2)))
    if not g.is_valid or not p.is_valid:
        return 0
    inter = Polygon(g).intersection(Polygon(p)).area
    union = g.area + p.area - inter
    if union == 0:
        return 0
    else:
        return inter/union

def NMS(detectResult, nmsThresh=0.4):
    predBoxs = []
    sort_detectboxs = sorted(detectResult, key=lambda x: x.score, reverse=True)
    for i in range(len(sort_detectboxs)):
        xmin1 = sort_detectboxs[i].xmin
        ymin1 = sort_detectboxs[i].ymin
        xmax1 = sort_detectboxs[i].xmax
        ymax1 = sort_detectboxs[i].ymax
        classId = sort_detectboxs[i].classId
        angle = sort_detectboxs[i].angle
        p1 = rotate_rectangle(xmin1, ymin1, xmax1, ymax1, angle)
        p1 = np.array(p1).reshape(-1)
        
        if sort_detectboxs[i].classId != -1:
            predBoxs.append(sort_detectboxs[i])
            for j in range(i + 1, len(sort_detectboxs), 1):
                if classId == sort_detectboxs[j].classId:
                    xmin2 = sort_detectboxs[j].xmin
                    ymin2 = sort_detectboxs[j].ymin
                    xmax2 = sort_detectboxs[j].xmax
                    ymax2 = sort_detectboxs[j].ymax
                    angle2 = sort_detectboxs[j].angle
                    p2 = rotate_rectangle(xmin2, ymin2, xmax2, ymax2, angle2)
                    p2 = np.array(p2).reshape(-1)
                    iou = intersection(p1, p2)
                    if iou > nmsThresh:
                        sort_detectboxs[j].classId = -1
    return predBoxs

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x, axis=-1):
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def process(out, model_w, model_h, stride, angle_feature, index, objectThresh=0.5, scale_w=1, scale_h=1):
    # out: feature map [1, 64+nc, h, w] -> transposed to [1, h, w, 64+nc] in logic? 
    # Note: The original code logic is extremely hard-coded and confusing regarding shapes.
    # The 'out' passed here seems to be the raw output from RKNN which is NCHW or NHWC depending on config.
    # Original code: feature=x.reshape(1,79,-1) where 79 = 64 (reg) + 15 (cls)
    # And then calls process(feature, ...)
    
    # We need to assume the input 'out' here matches what 'feature' was in original code.
    # Original: feature = x.reshape(1, 79, -1)
    
    # Let's verify shape assumptions.
    # In original code:
    # xywh = out[:, :64, :]
    # conf = sigmoid(out[:, 64:, :])
    
    # So 'out' is expected to be [1, channels, grid_cells_flat]
    
    class_num = out.shape[1] - 64 # Derive class num dynamically if possible
    
    angle_feature = angle_feature.reshape(-1)
    xywh = out[:, :64, :]
    conf = sigmoid(out[:, 64:, :])
    
    detected_boxes = []
    conf = conf.reshape(-1)
    
    # Grid size
    grid_len = model_h * model_w
    
    # Iterate over all grid cells * classes? 
    # Original: for ik in range(model_h*model_w*class_num):
    # This loop is very inefficient in Python. Ideally should be vectorized.
    # But for now, we stick to the original logic to ensure correctness.
    
    # Optimization: Only iterate where conf > thresh
    valid_indices = np.where(conf > objectThresh)[0]
    
    for ik in valid_indices:
        # Decode index
        # ik = c * (model_w * model_h) + h * model_w + w
        # So:
        w = ik % model_w
        h = (ik % (model_w * model_h)) // model_w
        c = ik // (model_w * model_h)
        
        # DFL decoding
        xywh_ = xywh[0, :, (h * model_w) + w] # [64]
        xywh_ = xywh_.reshape(1, 4, 16, 1)
        data = np.array([i for i in range(16)]).reshape(1, 1, 16, 1)
        xywh_ = softmax(xywh_, 2)
        xywh_ = np.multiply(data, xywh_)
        xywh_ = np.sum(xywh_, axis=2, keepdims=True).reshape(-1)
        
        xywh_add = xywh_[:2] + xywh_[2:]
        xywh_sub = (xywh_[2:] - xywh_[:2]) / 2
        
        # Angle decoding
        angle_index = index + (h * model_w) + w
        if angle_index >= len(angle_feature):
            continue # Safety check
            
        angle_feature_ = (angle_feature[angle_index] - 0.25) * 3.1415927410125732
        angle_feature_cos = math.cos(angle_feature_)
        angle_feature_sin = math.sin(angle_feature_)
        
        xy_mul1 = xywh_sub[0] * angle_feature_cos
        xy_mul2 = xywh_sub[1] * angle_feature_sin
        xy_mul3 = xywh_sub[0] * angle_feature_sin
        xy_mul4 = xywh_sub[1] * angle_feature_cos
        
        # xy = xy_mul1 - xy_mul2, xy_mul3 + xy_mul4
        
        cx = (xy_mul1 - xy_mul2) + w + 0.5
        cy = (xy_mul3 + xy_mul4) + h + 0.5
        width = xywh_add[0]
        height = xywh_add[1]
        
        # Scale back to model input size
        cx = cx * stride
        cy = cy * stride
        width = width * stride
        height = height * stride
        
        # Scale to original image size (if scale_w/h provided)
        # Note: Original code applied scale_w/h to xmin/max.
        # xmin = (cx - width/2) * scale_w
        # ymin = (cy - height/2) * scale_h
        # ...
        
        # We return box in model input scale, let outer loop handle resizing if needed,
        # or apply scale here.
        xmin = (cx - width / 2) * scale_w
        ymin = (cy - height / 2) * scale_h
        xmax = (cx + width / 2) * scale_w
        ymax = (cy + height / 2) * scale_h
        
        box = DetectBox(c, conf[ik], xmin, ymin, xmax, ymax, angle_feature_)
        detected_boxes.append(box)
        
    return detected_boxes
