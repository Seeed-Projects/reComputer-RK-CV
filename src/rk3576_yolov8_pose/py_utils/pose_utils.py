import numpy as np
import cv2

CLASSES = ['person']

pose_palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102], [230, 230, 0], [255, 153, 255],
                         [153, 204, 255], [255, 102, 255], [255, 51, 255], [102, 178, 255], [51, 153, 255],
                         [255, 153, 153], [255, 102, 102], [255, 51, 51], [153, 255, 153], [102, 255, 102],
                         [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0], [255, 255, 255]],dtype=np.uint8)
kpt_color  = pose_palette[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]
skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8], 
            [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]
limb_color = pose_palette[[9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]]

class DetectBox:
    def __init__(self, classId, score, xmin, ymin, xmax, ymax, keypoint):
        self.classId = classId
        self.score = score
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.keypoint = keypoint

class PoseHelper:
    def __init__(self, img_size=(640, 640)):
        self.img_size = img_size
        self.aspect_ratio = 1.0
        self.offset_x = 0
        self.offset_y = 0

    def letterbox_resize(self, image, bg_color=56):
        target_width, target_height = self.img_size
        image_height, image_width, _ = image.shape

        self.aspect_ratio = min(target_width / image_width, target_height / image_height)
        new_width = int(image_width * self.aspect_ratio)
        new_height = int(image_height * self.aspect_ratio)

        image_resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

        result_image = np.ones((target_height, target_width, 3), dtype=np.uint8) * bg_color
        self.offset_x = (target_width - new_width) // 2
        self.offset_y = (target_height - new_height) // 2
        result_image[self.offset_y:self.offset_y + new_height, self.offset_x:self.offset_x + new_width] = image_resized
        return result_image

    def get_real_box(self, box):
        # 还原回原图坐标
        xmin = (box.xmin - self.offset_x) / self.aspect_ratio
        ymin = (box.ymin - self.offset_y) / self.aspect_ratio
        xmax = (box.xmax - self.offset_x) / self.aspect_ratio
        ymax = (box.ymax - self.offset_y) / self.aspect_ratio
        
        # 边界保护
        return [int(max(0, xmin)), int(max(0, ymin)), int(max(0, xmax)), int(max(0, ymax))]
        
    def get_real_keypoint(self, keypoint):
        real_kp = keypoint.copy()
        real_kp[..., 0] = (real_kp[..., 0] - self.offset_x) / self.aspect_ratio
        real_kp[..., 1] = (real_kp[..., 1] - self.offset_y) / self.aspect_ratio
        return real_kp

def IOU(xmin1, ymin1, xmax1, ymax1, xmin2, ymin2, xmax2, ymax2):
    xmin = max(xmin1, xmin2)
    ymin = max(ymin1, ymin2)
    xmax = min(xmax1, xmax2)
    ymax = min(ymax1, ymax2)

    innerWidth = xmax - xmin
    innerHeight = ymax - ymin

    innerWidth = innerWidth if innerWidth > 0 else 0
    innerHeight = innerHeight if innerHeight > 0 else 0

    innerArea = innerWidth * innerHeight

    area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
    area2 = (xmax2 - xmin2) * (ymax2 - ymin2)

    total = area1 + area2 - innerArea

    return innerArea / total

def NMS(detectResult, nmsThresh=0.45):
    predBoxs = []
    sort_detectboxs = sorted(detectResult, key=lambda x: x.score, reverse=True)

    for i in range(len(sort_detectboxs)):
        xmin1 = sort_detectboxs[i].xmin
        ymin1 = sort_detectboxs[i].ymin
        xmax1 = sort_detectboxs[i].xmax
        ymax1 = sort_detectboxs[i].ymax
        classId = sort_detectboxs[i].classId

        if sort_detectboxs[i].classId != -1:
            predBoxs.append(sort_detectboxs[i])
            for j in range(i + 1, len(sort_detectboxs), 1):
                if classId == sort_detectboxs[j].classId:
                    xmin2 = sort_detectboxs[j].xmin
                    ymin2 = sort_detectboxs[j].ymin
                    xmax2 = sort_detectboxs[j].xmax
                    ymax2 = sort_detectboxs[j].ymax
                    iou = IOU(xmin1, ymin1, xmax1, ymax1, xmin2, ymin2, xmax2, ymax2)
                    if iou > nmsThresh:
                        sort_detectboxs[j].classId = -1
    return predBoxs

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x, axis=-1):
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def process(out, keypoints, index, model_w, model_h, stride, objectThresh=0.25):
    xywh = out[:, :64, :]
    conf = sigmoid(out[:, 64:65, :])  # class confidence
    out_boxes = []
    
    for h in range(model_h):
        for w in range(model_w):
            if conf[0, 0, (h*model_w)+w] > objectThresh:
                xywh_ = xywh[0, :, (h*model_w)+w]
                xywh_ = xywh_.reshape(1, 4, 16, 1)
                data = np.array([i for i in range(16)]).reshape(1, 1, 16, 1)
                xywh_ = softmax(xywh_, 2)
                xywh_ = np.multiply(data, xywh_)
                xywh_ = np.sum(xywh_, axis=2, keepdims=True).reshape(-1)

                xywh_temp = xywh_.copy()
                xywh_temp[0] = (w + 0.5) - xywh_[0]
                xywh_temp[1] = (h + 0.5) - xywh_[1]
                xywh_temp[2] = (w + 0.5) + xywh_[2]
                xywh_temp[3] = (h + 0.5) + xywh_[3]

                xywh_[0] = ((xywh_temp[0] + xywh_temp[2]) / 2)
                xywh_[1] = ((xywh_temp[1] + xywh_temp[3]) / 2)
                xywh_[2] = (xywh_temp[2] - xywh_temp[0])
                xywh_[3] = (xywh_temp[3] - xywh_temp[1])
                xywh_ = xywh_ * stride

                xmin = (xywh_[0] - xywh_[2] / 2) 
                ymin = (xywh_[1] - xywh_[3] / 2) 
                xmax = (xywh_[0] + xywh_[2] / 2) 
                ymax = (xywh_[1] + xywh_[3] / 2) 
                
                # Retrieve keypoint from the combined keypoint tensor (Output 3)
                # Output 3 shape: (1, 17, 3, 8400)
                # Global index calculation:
                # 80x80 grid: index 0 to 6400
                # 40x40 grid: index 6400 to 8000
                # 20x20 grid: index 8000 to 8400
                global_idx = index + (h * model_w) + w
                
                # Check the dimensionality of keypoints tensor
                if len(keypoints.shape) == 4:
                    # Format: (1, 17, 3, 8400) -> slice out the last dimension
                    keypoint = keypoints[0, :, :, global_idx]
                elif len(keypoints.shape) == 3:
                    # Format: (1, 51, 8400) -> slice out the last dimension and reshape
                    keypoint = keypoints[0, :, global_idx].reshape(17, 3)
                else:
                    # Fallback
                    keypoint = keypoints[..., global_idx].reshape(-1, 3)
                
                # Note: RKNN exported model outputs keypoints as [x, y, conf] usually
                # Sometimes it needs to be scaled by stride, sometimes it's already scaled.
                # Standard yolov8-pose keypoints in the latest export format are relative to the feature map or image.
                # Assuming it's already absolute or needs stride scaling based on typical rknn exports.
                # Let's keep it as is, if it's too small, it might need * stride or * 2
                
                box = DetectBox(0, float(conf[0, 0, (h*model_w)+w]), xmin, ymin, xmax, ymax, keypoint)
                out_boxes.append(box)

    return out_boxes

def post_process_pose(outputs, obj_thresh=0.25, nms_thresh=0.45):
    boxes = []
    outputs = list(outputs)  # Convert to list to allow modification
    
    # Reshape if outputs are 4D (1, C, H, W) to (1, C, H*W)
    for i in range(len(outputs)):
        # Only reshape bbox/cls outputs (which have H == W), do not reshape keypoints
        if len(outputs[i].shape) == 4 and outputs[i].shape[2] == outputs[i].shape[3]:
            outputs[i] = outputs[i].reshape(outputs[i].shape[0], outputs[i].shape[1], -1)

    # User provided shapes:
    # Output 0 shape: (1, 65, 80, 80)  -> Reshaped: (1, 65, 6400) (bbox + cls)
    # Output 1 shape: (1, 65, 40, 40)  -> Reshaped: (1, 65, 1600) (bbox + cls)
    # Output 2 shape: (1, 65, 20, 20)  -> Reshaped: (1, 65, 400)  (bbox + cls)
    # Output 3 shape: (1, 17, 3, 8400) -> Keypoints for all scales combined (6400+1600+400=8400)
    
    if len(outputs) == 4:
        # process stride 8 (80x80)
        boxes.extend(process(outputs[0], outputs[3], 0, 80, 80, 8, objectThresh=obj_thresh))
        # process stride 16 (40x40)
        boxes.extend(process(outputs[1], outputs[3], 6400, 40, 40, 16, objectThresh=obj_thresh))
        # process stride 32 (20x20)
        boxes.extend(process(outputs[2], outputs[3], 8000, 20, 20, 32, objectThresh=obj_thresh))
    else:
        print(f"WARNING: Unexpected number of outputs: {len(outputs)}. Cannot proceed with current post-processing.")
        return []
    
    # NMS
    pred_boxes = NMS(boxes, nmsThresh=nms_thresh)
    return pred_boxes

def draw_pose(image, boxes, pose_helper):
    for box in boxes:
        # 画检测框
        real_box = pose_helper.get_real_box(box)
        cv2.rectangle(image, (real_box[0], real_box[1]), (real_box[2], real_box[3]), (255, 0, 0), 2)
        
        # 写置信度
        text = f"{CLASSES[box.classId]} {box.score:.2f}"
        cv2.putText(image, text, (real_box[0], real_box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # 还原关键点到原图尺寸
        real_kp = pose_helper.get_real_keypoint(box.keypoint)
        
        # 画骨架连线
        for i, sk in enumerate(skeleton):
            # 注意：skeleton 里的索引是从 1 开始的，所以要减 1
            idx1, idx2 = sk[0] - 1, sk[1] - 1
            x1, y1, conf1 = real_kp[idx1]
            x2, y2, conf2 = real_kp[idx2]
            
            # 如果两个关键点的置信度都大于 0.5，则画连线
            if conf1 > 0.5 and conf2 > 0.5:
                # 使用绿线 (BGR: 0, 255, 0)
                color = (0, 255, 0)
                cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        
        # 画关键点
        for i in range(17):
            x, y, conf = real_kp[i]
            if conf > 0.5:
                # 关键点也使用绿色 (BGR: 0, 255, 0)
                color = (0, 255, 0)
                cv2.circle(image, (int(x), int(y)), 4, color, -1)
                
    return image
