import numpy as np
import cv2
import os

def load_classes(path):
    """
    从文件加载类别标签
    """
    if not os.path.exists(path):
        return ["Class_" + str(i) for i in range(1000)]
    with open(path, 'r', encoding='utf-8') as f:
        labels = [l.strip() for l in f.readlines()]
    return labels

def softmax(x):
    """
    计算 Softmax
    """
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

class MobileNet_helper:
    def __init__(self, img_size=(224, 224), class_label_path='model/synset.txt'):
        self.img_size = img_size
        self.classes = load_classes(class_label_path)

    def preprocess(self, frame):
        """
        预处理逻辑：缩放并保留 OpenCV BGR 通道顺序。

        该 RKNN 模型沿用 rknn_model_zoo MobileNet 官方示例的输入约定。
        """
        return cv2.resize(frame, self.img_size)

    def get_topk(self, outputs, topk=5, conf_thresh=0.0):
        """
        后处理：获取 Top-K 分类结果，并可通过置信度阈值进行过滤
        """
        if outputs is None or len(outputs) == 0:
            return [], []

        scores = softmax(outputs[0].flatten())
        topk_idx = np.argsort(scores)[::-1][:topk]

        topk_scores = scores[topk_idx]

        # 应用置信度阈值过滤
        valid_idx = topk_scores >= conf_thresh
        topk_idx = topk_idx[valid_idx]
        topk_scores = topk_scores[valid_idx]

        topk_classes = [self.classes[i] if i < len(self.classes) else f"Class {i}" for i in topk_idx]

        return topk_classes, topk_scores

    def draw_topk(self, image, topk_classes, topk_scores):
        """
        在图像上绘制 Top-K 结果
        """
        y_offset = 30
        for i, (label, score) in enumerate(zip(topk_classes, topk_scores)):
            text = f"{label}: {score:.3f}"
            color = (0, 255, 0) if i == 0 else (200, 200, 200)
            cv2.putText(image, text, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            y_offset += 30
        return image
